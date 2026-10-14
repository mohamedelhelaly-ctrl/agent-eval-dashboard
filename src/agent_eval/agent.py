import copy
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone

from agent_eval.answer import parse_final_answer
from agent_eval.config import DB_META_PATH, get_as_of_date, render_prompt
from agent_eval.llm import LLMUnavailable, chat
from agent_eval.records import RunRecord, StepRecord, ToolCallRecord
from agent_eval.schemas import AgentConfig, LLMResult, ModelSpec
from agent_eval.tools.specs import TOOL_SPECS
from agent_eval.tools.toolbox import Toolbox


# The fingerprint of the database this run used, from data/db_meta.json.
def _db_sha256() -> str:
    meta = json.loads(DB_META_PATH.read_text(encoding="utf-8"))
    if "content_sha256" not in meta:
        raise ValueError(f"'content_sha256' missing from {DB_META_PATH}")
    return meta["content_sha256"]


# Parse a tool call's arguments the way Toolbox.call reads them: empty text means {}, and anything that is
# not a JSON object comes back as None (the toolbox reports the problem to the model).
def _parse_arguments(raw: str) -> dict | None:
    raw = raw or ""  # a missing value must not raise
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


# Copy the LLM result into a StepRecord. A token count the provider did not report is stored as 0.
def _step_record(index: int, result: LLMResult) -> StepRecord:
    return StepRecord(
        index=index, content=result.content, finish_reason=result.finish_reason,
        prompt_tokens=result.prompt_tokens or 0, completion_tokens=result.completion_tokens or 0,
        cached_tokens=result.cached_tokens or 0, cost_usd=result.cost_usd,
        reported_cost_usd=result.reported_cost_usd, latency_s=result.latency_s, attempts=result.attempts,
        retry_wait_s=result.retry_wait_s, response_id=result.response_id, served_model=result.served_model,
        provider=result.provider,
    )


# Give every tool call an id. A call with an empty id gets "call_{step}_{n}", and the same id is written into
# the assistant message that goes back to the model, so the tool message and the call it answers still match.
def _tool_calls_with_ids(result: LLMResult, step_no: int) -> tuple[list[tuple[str, object]], dict]:
    assistant_message = copy.deepcopy(result.assistant_message)
    sent = assistant_message.get("tool_calls") or []
    calls = []
    for n, tc in enumerate(result.tool_calls, start=1):
        call_id = tc.id or f"call_{step_no}_{n}"
        if not tc.id and n - 1 < len(sent):
            sent[n - 1]["id"] = call_id
        calls.append((call_id, tc))
    return calls, assistant_message


# Run one agent on one question: ask the model, run the tools it requests, feed the results back, and repeat
# until it answers or a limit is hit. Returns a RunRecord. Only LLMUnavailable is caught (it ends the run as
# "infra_error"); every other exception propagates. No file writes, no scoring, no printing.
def run_agent(client, agent: AgentConfig, spec: ModelSpec, toolbox: Toolbox, question: str, *,
              question_id: str, repeat: int = 0, chat_fn=chat) -> RunRecord:

    # ===== 1. Setup: build the starting conversation and the variables the loop fills in. =====
    system_prompt = render_prompt(agent.prompt)
    # Read these now, before any model call: a bad db_meta.json or a missing date must fail the run early,
    # not after tokens have been spent.
    db_sha256 = _db_sha256()
    as_of = get_as_of_date()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    steps: list[StepRecord] = []
    status, final_text, parsed, error_message = None, None, None, None


    # ===== 2. The agent loop: one pass per model call, at most agent.max_steps passes. =====
    for step_no in range(1, agent.max_steps + 1):

        # --- 2a. Ask the model. Only LLMUnavailable is caught: it ends the run, keeping the steps so far. ---
        try:
            result = chat_fn(client, spec, messages, tools=TOOL_SPECS, temperature=agent.temperature)
        except LLMUnavailable as exc:
            status, error_message = "infra_error", str(exc)  # the steps recorded so far are kept
            break

        # --- 2b. Record the step and add the model's reply to the conversation. ---
        calls, assistant_message = _tool_calls_with_ids(result, step_no)
        messages.append(assistant_message)
        step = _step_record(step_no, result)
        steps.append(step)

        # --- 2c. No tool calls means the model gave its final answer: parse it and finish. ---
        if not result.tool_calls:
            final_text, status = result.content, "answered"
            parsed = parse_final_answer(final_text)
            break

        # --- 2d. Check the limits before running any tool. If one is reached, record the requested calls
        # as not executed and stop. The cost cap is checked first, so it wins when both limits apply.
        # The cap is checked after a step, so a run can overshoot it by one step. ---
        if sum(s.effective_cost_usd for s in steps) >= agent.max_run_cost_usd:
            status = "cost_cap"
        elif step_no == agent.max_steps:
            status = "max_steps"
        if status:
            step.tool_calls = [
                ToolCallRecord(id=call_id, name=tc.name, arguments_raw=tc.arguments,
                               arguments=_parse_arguments(tc.arguments), result=None, is_error=False,
                               executed=False)
                for call_id, tc in calls
            ]
            break

        # --- 2e. No limit hit: run every requested tool, record each call, and send the results back
        # to the model as tool messages for the next pass. ---
        records = []
        for call_id, tc in calls:

            t_call = time.monotonic()
            output = toolbox.call(tc.name, tc.arguments)
            duration = time.monotonic() - t_call

            records.append(ToolCallRecord(
                id=call_id, name=tc.name, arguments_raw=tc.arguments, arguments=_parse_arguments(tc.arguments),
                result=output, is_error=isinstance(output, dict) and "error" in output, duration_s=duration))

            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": json.dumps(output, ensure_ascii=False, default=str)})

        step.tool_calls = records

    # ===== 3. Parsed-answer fields: copy the parse result, or note that the run ended with no final answer. =====
    if parsed is not None:
        parse_fields = dict(parse_ok=parsed.ok, answer=parsed.answer, answerable=parsed.answerable,
                            clean_format=parsed.clean_format, parse_problem=parsed.problem)
    else:  # the run ended without a final message
        parse_fields = dict(parse_ok=False, answer=None, answerable=None, clean_format=False,
                            parse_problem=f"no final answer: run ended with status {status}")

    # ===== 4. Build the RunRecord: identity, outcome, the recorded steps, and a snapshot of the exact
    # configuration (agent, prompt, database, temperature) so the run can be understood later. =====
    return RunRecord(
        run_id=uuid.uuid4().hex,
        agent_name=agent.name,
        model_id=spec.id,
        prompt_name=agent.prompt,
        question_id=question_id,
        repeat=repeat,
        status=status,
        started_at=started_at,
        wall_time_s=time.monotonic() - t0,
        final_text=final_text,
        **parse_fields,
        steps=steps,
        error_message=error_message,
        config_snapshot={
            "agent": agent.model_dump(),
            "model_id": spec.id,
            "provider": spec.provider,
            "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "temperature_requested": agent.temperature,
            # A model that accepts no temperature runs at its own default, so nothing was sent.
            "temperature_sent": agent.temperature if spec.supports_temperature else None,
            "as_of_date": as_of,
            "db_sha256": db_sha256,
        },
    )
