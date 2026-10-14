import copy
import hashlib
import json
from datetime import datetime

import pytest

from agent_eval.agent import run_agent
from agent_eval.config import DB_META_PATH, render_prompt
from agent_eval.llm import LLMError, LLMUnavailable
from agent_eval.records import RunRecord
from agent_eval.schemas import AgentConfig, LLMResult, ModelSpec, ToolCall
from agent_eval.tools.specs import TOOL_SPECS
from agent_eval.tools.toolbox import Toolbox

QUESTION = "How many customers does the database have?"
GOOD_ANSWER = '{"answer": 500, "answerable": true}'


# ---------- builders ----------

def make_agent(**overrides):
    return AgentConfig(**{"name": "test_agent", "model": "ling", "prompt": "minimal", **overrides})


def make_spec(**overrides):
    values = dict(id="vendor/m1", display_name="M1", tier="budget", price_in_per_m=1.0, price_out_per_m=1.0,
                  provider="novita")
    return ModelSpec(**{**values, **overrides})


def call(name="run_sql", arguments='{"query": "SELECT COUNT(*) FROM customers"}', id="call_a"):
    return ToolCall(id=id, name=name, arguments=arguments)


def res(content=None, calls=(), cost_usd=0.001, reported=None, **overrides):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                 for c in calls]
    values = dict(content=content, tool_calls=list(calls), assistant_message=message,
                  finish_reason="tool_calls" if calls else "stop", prompt_tokens=100, completion_tokens=10,
                  cached_tokens=0, cost_usd=cost_usd, reported_cost_usd=reported, latency_s=0.5,
                  started_at="2026-01-01T00:00:00+00:00")
    return LLMResult(**{**values, **overrides})


class ScriptedChat:
    """A fake chat_fn: returns prepared results in order (or raises prepared exceptions) and records each call."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []  # one dict per call: a snapshot of the messages, the tools and the temperature

    def __call__(self, client, spec, messages, tools=None, temperature=0.0):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "temperature": temperature,
                           "spec": spec})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeToolbox:
    """Counts calls and returns canned results, so a test can prove a tool was (not) run."""

    def __init__(self, **results):
        self.results, self.calls = results, []

    def call(self, name, arguments=None):
        self.calls.append((name, arguments))
        return self.results.get(name, {"ok": True})


def go(chat, toolbox=None, agent=None, spec=None, **kwargs):
    return run_agent(None, agent or make_agent(), spec or make_spec(), toolbox or FakeToolbox(), QUESTION,
                     question_id="q01", chat_fn=chat, **kwargs)


# ---------- answered ----------

def test_direct_answer_without_tools():
    chat = ScriptedChat(res(content=GOOD_ANSWER))
    r = go(chat)
    assert r.status == "answered" and r.final_text == GOOD_ANSWER
    assert (r.parse_ok, r.answer, r.answerable, r.clean_format, r.parse_problem) == (True, 500, True, True, None)
    assert (r.n_steps, r.n_tool_calls, r.error_message) == (1, 0, None)
    assert len(chat.calls) == 1


def test_first_messages_are_the_rendered_system_prompt_and_the_question():
    chat = ScriptedChat(res(content=GOOD_ANSWER))
    go(chat)
    system, user = chat.calls[0]["messages"]
    assert system == {"role": "system", "content": render_prompt("minimal")}
    assert "Today's date is 2026-01-01" in system["content"]
    assert user == {"role": "user", "content": QUESTION}


def test_chat_receives_the_tool_specs_and_agent_temperature():
    chat = ScriptedChat(res(content=GOOD_ANSWER))
    go(chat, agent=make_agent(temperature=0.7))
    assert chat.calls[0]["tools"] is TOOL_SPECS and chat.calls[0]["temperature"] == 0.7
    assert chat.calls[0]["spec"].id == "vendor/m1"


def test_one_tool_call_then_answer_with_the_real_database():
    chat = ScriptedChat(res(calls=[call()]), res(content=GOOD_ANSWER))
    r = go(chat, toolbox=Toolbox())
    assert (r.status, r.n_steps, r.n_tool_calls, r.n_tool_errors) == ("answered", 2, 1, 0)
    record = r.steps[0].tool_calls[0]
    assert record.result["rows"] == [[500]] and record.executed and not record.is_error
    assert record.arguments == {"query": "SELECT COUNT(*) FROM customers"} and record.duration_s >= 0
    # The second call sees: system, user, the assistant message, and a tool message for that exact call.
    second = chat.calls[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["id"] == "call_a" == second[3]["tool_call_id"]
    assert json.loads(second[3]["content"])["rows"] == [[500]]


def test_several_tool_calls_in_one_step_each_get_a_tool_message():
    chat = ScriptedChat(res(calls=[call("list_tables", "{}", "a"), call("list_tables", "{}", "b")]),
                        res(content=GOOD_ANSWER))
    toolbox = FakeToolbox()
    r = go(chat, toolbox=toolbox)
    tool_messages = [m for m in chat.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    assert len(toolbox.calls) == 2 and r.n_tool_calls == 2


def test_tool_message_keeps_non_ascii_text_readable():
    chat = ScriptedChat(res(calls=[call("list_tables", "{}")]), res(content=GOOD_ANSWER))
    go(chat, toolbox=FakeToolbox(list_tables={"note": "café"}))
    assert "café" in chat.calls[1]["messages"][3]["content"]


# ---------- tool errors ----------

def test_tool_error_is_fed_back_and_the_loop_continues():
    chat = ScriptedChat(res(calls=[call("describe_table", '{"table": "nope"}')]), res(content=GOOD_ANSWER))
    r = go(chat, toolbox=Toolbox())
    record = r.steps[0].tool_calls[0]
    assert record.is_error and "Unknown table" in record.result["error"]
    assert (r.status, r.n_tool_calls, r.n_tool_errors) == ("answered", 1, 1)
    assert "Unknown table" in chat.calls[1]["messages"][3]["content"]


def test_malformed_arguments_are_recorded_exactly_with_is_error_and_no_parsed_arguments():
    chat = ScriptedChat(res(calls=[call("run_sql", "{bad json")]), res(content=GOOD_ANSWER))
    r = go(chat, toolbox=Toolbox())
    record = r.steps[0].tool_calls[0]
    assert record.arguments_raw == "{bad json" and record.arguments is None
    assert record.is_error and record.executed and "not valid JSON" in record.result["error"]
    assert r.n_tool_errors == 1 and r.status == "answered"


def test_valid_json_that_is_not_an_object_has_no_parsed_arguments():
    chat = ScriptedChat(res(calls=[call("run_sql", "[1, 2]")]), res(content=GOOD_ANSWER))
    record = go(chat, toolbox=Toolbox()).steps[0].tool_calls[0]
    assert record.arguments is None and record.is_error


def test_empty_arguments_mean_an_empty_object():
    chat = ScriptedChat(res(calls=[call("list_tables", "")]), res(content=GOOD_ANSWER))
    record = go(chat, toolbox=Toolbox()).steps[0].tool_calls[0]
    assert record.arguments_raw == "" and record.arguments == {} and not record.is_error


def test_is_error_follows_the_error_key_of_the_result():
    chat = ScriptedChat(res(calls=[call("a", "{}", "1"), call("b", "{}", "2")]), res(content=GOOD_ANSWER))
    r = go(chat, toolbox=FakeToolbox(a={"error": "boom"}, b={"rows": []}))
    assert [t.is_error for t in r.steps[0].tool_calls] == [True, False]


def test_empty_tool_call_id_gets_one_generated_id_used_everywhere():
    chat = ScriptedChat(res(calls=[call("list_tables", "{}", ""), call("list_tables", "{}", "")]),
                        res(content=GOOD_ANSWER))
    r = go(chat)
    ids = [t.id for t in r.steps[0].tool_calls]
    assert ids == ["call_1_1", "call_1_2"]
    second = chat.calls[1]["messages"]
    assert [c["id"] for c in second[2]["tool_calls"]] == ids
    assert [m["tool_call_id"] for m in second if m["role"] == "tool"] == ids


# ---------- limits ----------

def test_max_steps_does_not_execute_the_last_steps_tool_calls():
    chat = ScriptedChat(res(calls=[call("list_tables", "{}", "a")]),
                        res(calls=[call("list_tables", "{}", "b"), call("run_sql", "{bad", "c")]))
    toolbox = FakeToolbox()
    r = go(chat, toolbox=toolbox, agent=make_agent(max_steps=2))
    assert r.status == "max_steps" and r.n_steps == 2 and len(chat.calls) == 2
    first, last = r.steps[0].tool_calls, r.steps[1].tool_calls
    assert [t.executed for t in first] == [True] and [t.executed for t in last] == [False, False]
    assert all(t.result is None and not t.is_error for t in last)
    assert [t.arguments_raw for t in last] == ["{}", "{bad"] and [t.arguments for t in last] == [{}, None]
    assert r.n_tool_calls == 1 and r.n_tool_errors == 0
    assert toolbox.calls == [("list_tables", "{}")]  # the skipped calls never reached the toolbox
    assert (r.final_text, r.parse_ok, r.answer, r.answerable, r.clean_format) == (None, False, None, None, False)
    assert "max_steps" in r.parse_problem


def test_max_steps_of_one_stops_at_the_first_tool_request():
    toolbox = FakeToolbox()
    r = go(ScriptedChat(res(calls=[call()])), toolbox=toolbox, agent=make_agent(max_steps=1))
    assert r.status == "max_steps" and toolbox.calls == [] and r.n_tool_calls == 0


def test_a_final_answer_on_the_last_allowed_step_is_still_answered():
    chat = ScriptedChat(res(calls=[call("list_tables", "{}")]), res(content=GOOD_ANSWER))
    assert go(chat, agent=make_agent(max_steps=2)).status == "answered"


def test_cost_cap_stops_the_loop_and_skips_the_tool_calls():
    toolbox = FakeToolbox()
    chat = ScriptedChat(res(calls=[call()], cost_usd=0.02), res(content=GOOD_ANSWER))
    r = go(chat, toolbox=toolbox, agent=make_agent(max_run_cost_usd=0.01))
    assert r.status == "cost_cap" and r.n_steps == 1 and len(chat.calls) == 1
    assert toolbox.calls == [] and r.n_tool_calls == 0 and r.steps[0].tool_calls[0].executed is False
    assert (r.final_text, r.parse_ok) == (None, False) and "cost_cap" in r.parse_problem


def test_cost_cap_uses_the_reported_cost_over_the_computed_cost():
    # Computed cost is tiny but the provider reports a big charge: the cap uses the reported one.
    r = go(ScriptedChat(res(calls=[call()], cost_usd=0.0001, reported=0.05)), agent=make_agent(max_run_cost_usd=0.01))
    assert r.status == "cost_cap" and r.total_cost_usd == 0.05


def test_a_big_computed_cost_does_not_trip_the_cap_when_a_small_reported_cost_exists():
    chat = ScriptedChat(res(calls=[call("list_tables", "{}")], cost_usd=0.5, reported=0.001), res(content=GOOD_ANSWER))
    r = go(chat, agent=make_agent(max_run_cost_usd=0.01))
    assert r.status == "answered" and r.total_cost_usd == pytest.approx(0.002)  # 0.001 reported, plus 0.001 computed on the final step


def test_cost_cap_is_reached_at_exactly_the_cap():
    r = go(ScriptedChat(res(calls=[call()], reported=0.01)), agent=make_agent(max_run_cost_usd=0.01))
    assert r.status == "cost_cap"


def test_cost_adds_up_across_steps():
    chat = ScriptedChat(res(calls=[call("list_tables", "{}", "a")], cost_usd=0.006),
                        res(calls=[call("list_tables", "{}", "b")], cost_usd=0.006), res(content=GOOD_ANSWER))
    toolbox = FakeToolbox()
    r = go(chat, toolbox=toolbox, agent=make_agent(max_run_cost_usd=0.01))
    assert r.status == "cost_cap" and r.n_steps == 2
    assert [t.executed for s in r.steps for t in s.tool_calls] == [True, False] and len(toolbox.calls) == 1


def test_cost_cap_wins_over_max_steps_when_both_apply():
    r = go(ScriptedChat(res(calls=[call()], cost_usd=1.0)), agent=make_agent(max_steps=1, max_run_cost_usd=0.01))
    assert r.status == "cost_cap"


def test_a_final_answer_that_goes_over_the_cap_is_still_answered():
    r = go(ScriptedChat(res(content=GOOD_ANSWER, cost_usd=5.0)), agent=make_agent(max_run_cost_usd=0.01))
    assert r.status == "answered" and r.total_cost_usd == 5.0


# ---------- infra errors and exceptions ----------

def test_llm_unavailable_gives_infra_error_and_keeps_the_partial_steps():
    chat = ScriptedChat(res(calls=[call("list_tables", "{}")]), LLMUnavailable("vendor/m1 unavailable after 4 attempts"))
    r = go(chat)
    assert r.status == "infra_error" and "unavailable after 4 attempts" in r.error_message
    assert r.n_steps == 1 and r.steps[0].tool_calls[0].executed is True
    assert (r.final_text, r.parse_ok, r.answer, r.clean_format) == (None, False, None, False)
    assert "infra_error" in r.parse_problem


def test_llm_unavailable_on_the_first_step_leaves_no_steps():
    r = go(ScriptedChat(LLMUnavailable("down")))
    assert r.status == "infra_error" and r.steps == [] and r.error_message == "down" and r.total_cost_usd == 0


def test_other_exceptions_propagate_unchanged():
    with pytest.raises(RuntimeError, match="boom"):
        go(ScriptedChat(RuntimeError("boom")))
    with pytest.raises(ValueError, match="bad request"):
        go(ScriptedChat(res(calls=[call()]), ValueError("bad request")))


def test_other_llm_errors_are_not_swallowed():
    with pytest.raises(LLMError) as e:
        go(ScriptedChat(LLMError("no choices")))
    assert type(e.value) is LLMError


# ---------- the final answer ----------

def test_unparseable_final_answer_is_answered_with_parse_ok_false():
    r = go(ScriptedChat(res(content="I think it is about 500.")))
    assert r.status == "answered" and r.final_text == "I think it is about 500."
    assert (r.parse_ok, r.answer, r.answerable, r.clean_format) == (False, None, None, False)
    assert r.parse_problem


def test_a_fenced_answer_parses_but_is_not_clean():
    r = go(ScriptedChat(res(content="```json\n" + GOOD_ANSWER + "\n```")))
    assert (r.status, r.parse_ok, r.answer, r.clean_format) == ("answered", True, 500, False)


def test_an_empty_final_message_is_answered_but_not_parsed():
    r = go(ScriptedChat(res(content=None)))
    assert r.status == "answered" and r.final_text is None and r.parse_ok is False and r.parse_problem


def test_unanswerable_answer_is_recorded():
    r = go(ScriptedChat(res(content='{"answer": null, "answerable": false}')))
    assert (r.parse_ok, r.answer, r.answerable) == (True, None, False)


# ---------- the step records ----------

def test_step_record_copies_the_llm_result():
    result = res(content=GOOD_ANSWER, prompt_tokens=123, completion_tokens=45, cached_tokens=100, cost_usd=0.003,
                 reported=0.004, latency_s=1.5, attempts=2, retry_wait_s=1.0, response_id="gen-1",
                 served_model="vendor/m1-x", provider="Novita", finish_reason="stop")
    s = go(ScriptedChat(result)).steps[0]
    assert (s.index, s.content, s.finish_reason) == (1, GOOD_ANSWER, "stop")
    assert (s.prompt_tokens, s.completion_tokens, s.cached_tokens) == (123, 45, 100)
    assert (s.cost_usd, s.reported_cost_usd, s.cost_source, s.effective_cost_usd) == (0.003, 0.004, "reported", 0.004)
    assert (s.latency_s, s.attempts, s.retry_wait_s) == (1.5, 2, 1.0)
    assert (s.response_id, s.served_model, s.provider) == ("gen-1", "vendor/m1-x", "Novita")


def test_missing_token_counts_are_stored_as_zero_and_a_missing_cost_stays_none():
    result = res(content=GOOD_ANSWER, prompt_tokens=None, completion_tokens=None, cached_tokens=None, cost_usd=None)
    s = go(ScriptedChat(result)).steps[0]
    assert (s.prompt_tokens, s.completion_tokens, s.cached_tokens) == (0, 0, 0)
    assert s.cost_usd is None and s.cost_source == "none"


def test_step_indexes_start_at_one_and_totals_add_up():
    chat = ScriptedChat(res(calls=[call("list_tables", "{}")], retry_wait_s=1.0, attempts=2),
                        res(content=GOOD_ANSWER, prompt_tokens=50))
    r = go(chat)
    assert [s.index for s in r.steps] == [1, 2]
    assert (r.total_prompt_tokens, r.total_attempts, r.total_retry_wait_s) == (150, 3, 1.0)
    assert r.total_cost_usd == pytest.approx(0.002) and r.total_model_latency_s == pytest.approx(1.0)


# ---------- config snapshot and identity ----------

def test_config_snapshot_has_the_expected_keys_and_values():
    agent, spec = make_agent(temperature=0.3), make_spec()
    r = go(ScriptedChat(res(content=GOOD_ANSWER)), agent=agent, spec=spec)
    snap = r.config_snapshot
    assert set(snap) == {"agent", "model_id", "provider", "prompt_sha256", "temperature_requested",
                         "temperature_sent", "as_of_date", "db_sha256"}
    assert snap["agent"] == agent.model_dump()
    assert (snap["model_id"], snap["provider"]) == ("vendor/m1", "novita")
    assert snap["prompt_sha256"] == hashlib.sha256(render_prompt("minimal").encode("utf-8")).hexdigest()
    assert snap["as_of_date"] == "2026-01-01"
    assert snap["db_sha256"] == json.loads(DB_META_PATH.read_text(encoding="utf-8"))["content_sha256"]
    assert len(snap["db_sha256"]) == 64 and len(snap["prompt_sha256"]) == 64
    assert (snap["temperature_requested"], snap["temperature_sent"]) == (0.3, 0.3)


def test_temperature_sent_is_none_when_the_model_takes_no_temperature():
    chat = ScriptedChat(res(content=GOOD_ANSWER))
    r = go(chat, agent=make_agent(temperature=0.7), spec=make_spec(supports_temperature=False))
    assert r.config_snapshot["temperature_requested"] == 0.7 and r.config_snapshot["temperature_sent"] is None
    assert chat.calls[0]["temperature"] == 0.7  # chat() itself decides to leave it out of the request


def test_identity_fields_and_a_fresh_run_id_each_time():
    a = go(ScriptedChat(res(content=GOOD_ANSWER)), repeat=3)
    b = go(ScriptedChat(res(content=GOOD_ANSWER)))
    assert (a.agent_name, a.model_id, a.prompt_name, a.question_id, a.repeat) == (
        "test_agent", "vendor/m1", "minimal", "q01", 3)
    assert b.repeat == 0 and a.run_id != b.run_id and len(a.run_id) == 32 and int(a.run_id, 16) >= 0
    assert datetime.fromisoformat(a.started_at).utcoffset().total_seconds() == 0
    assert a.wall_time_s >= 0 and a.schema_version == 1


def test_the_record_round_trips_through_json():
    r = go(ScriptedChat(res(calls=[call("list_tables", "{}")]), res(content=GOOD_ANSWER)), toolbox=Toolbox())
    assert RunRecord.from_json_line(r.to_json_line()) == r


def test_run_agent_does_not_change_its_inputs():
    agent, spec = make_agent(), make_spec()
    before = (agent.model_dump(), spec.model_dump())
    go(ScriptedChat(res(content=GOOD_ANSWER)), agent=agent, spec=spec)
    assert (agent.model_dump(), spec.model_dump()) == before


def test_a_bad_db_meta_fails_before_any_model_call(monkeypatch):
    def broken():
        raise ValueError("'content_sha256' missing from db_meta.json")
    monkeypatch.setattr("agent_eval.agent._db_sha256", broken)
    chat = ScriptedChat(res(content=GOOD_ANSWER))
    with pytest.raises(ValueError, match="content_sha256"):
        go(chat)
    assert chat.calls == []  # no model call, so no tokens were spent
