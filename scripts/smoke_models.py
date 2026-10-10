"""Manual smoke test: ask each model one question with the real tools, for two turns at most.

Makes real, billed API calls (ling is free). Run from the project root with the package importable:
    PYTHONPATH=src .venv/bin/python scripts/smoke_models.py [--models ling solar]
"""
import argparse
import json
import os
from datetime import datetime, timezone

from agent_eval.config import PROJECT_ROOT, RESULTS_DIR, render_prompt
from agent_eval.llm import chat, make_client
from agent_eval.schemas import load_models
from agent_eval.tools.specs import TOOL_SPECS
from agent_eval.tools.toolbox import Toolbox

QUESTION = "How many customers does the database have?"
COST_TOLERANCE = 0.10


# Safety net: strip the API key out of any text before it is printed or saved to disk.
def scrub(text: str) -> str:
    """Remove the API key from any text before it is printed or saved."""
    key = os.environ.get("OPENROUTER_API_KEY")
    return text.replace(key, "[redacted]") if key else text


# Run one model through the two-turn test: ask the question, execute any tool calls it requests,
# send the tool results back, and get one follow-up reply. Returns the raw record of both turns.
def run_model(client, spec, toolbox) -> dict:
    messages = [{"role": "system", "content": render_prompt("minimal")}, {"role": "user", "content": QUESTION}]
    turn1 = chat(client, spec, messages, tools=TOOL_SPECS)
    record = {"turn1": turn1.model_dump(), "tool_results": [], "turn2": None}
    if not turn1.tool_calls:
        return record  # answered directly; nothing to feed back
    messages.append(turn1.assistant_message)
    for tc in turn1.tool_calls:
        result = toolbox.call(tc.name, tc.arguments)
        record["tool_results"].append({"id": tc.id, "name": tc.name, "result": result})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    # One follow-up only: if turn 2 asks for more tools, they are recorded but not run.
    record["turn2"] = chat(client, spec, messages, tools=TOOL_SPECS).model_dump()
    return record


# Add up one numeric field (tokens, cost...) across turns. Gives None if any turn lacks it,
# so a missing value never looks like a smaller total.
def total(turns, field):
    """Sum a field across turns; None if any turn lacks it (an understated total would mislead)."""
    values = [t[field] for t in turns]
    return None if any(v is None for v in values) else sum(values)


# Turn one model's raw record into a flat summary row: identifiers, what each turn did,
# summed latency/tokens/cost, and whether our computed cost disagrees with the reported cost by >10%.
def summarize(key: str, record: dict) -> dict:
    if "error" in record:
        return {"model": key, "error": record["error"]}
    turns = [t for t in (record["turn1"], record["turn2"]) if t]
    tokens = [total(turns, "prompt_tokens"), total(turns, "completion_tokens")]
    computed, reported = total(turns, "cost_usd"), total(turns, "reported_cost_usd")
    flag = False
    if computed is not None and reported is not None:
        flag = computed > 0 if reported == 0 else abs(computed - reported) / abs(reported) > COST_TOLERANCE
    t1, t2 = record["turn1"], record["turn2"]
    return {
        "model": key,
        "served_model": t1["served_model"],
        "provider": t1["provider"],
        "turn1": f"{t1['finish_reason']} [{', '.join(tc['name'] for tc in t1['tool_calls'])}]",
        "turn2": None if t2 is None else (
            f"{t2['finish_reason']} | " + (t2["content"] or "").replace("\n", " ")[:80]
            + (f" tools:{[tc['name'] for tc in t2['tool_calls']]}" if t2["tool_calls"] else "")),
        "latency_s": sum(t["latency_s"] for t in turns),
        "tokens": None if None in tokens else sum(tokens),
        "computed_cost": computed,
        "reported_cost": reported,
        "cost_mismatch": flag,
    }


# Format a value for the table; None (not reported) is shown as "-".
def fmt(v, spec=""):
    return "-" if v is None else format(v, spec)


# Print the summary rows as an aligned text table, with one ERROR row for any model that failed.
def print_table(rows):
    header = ["model", "served_model", "provider", "turn 1", "turn 2", "latency", "tokens", "computed $", "reported $", "flag"]
    body = []
    for r in rows:
        if "error" in r:
            body.append([r["model"], "ERROR", "", r["error"], "", "", "", "", "", ""])
            continue
        body.append([r["model"], fmt(r["served_model"]), fmt(r["provider"]), r["turn1"], fmt(r["turn2"]),
                     f"{r['latency_s']:.1f}s", fmt(r["tokens"]), fmt(r["computed_cost"], ".6f"),
                     fmt(r["reported_cost"], ".6f"), "COST DIFF >10%" if r["cost_mismatch"] else ""])
    widths = [max(len(str(row[i])) for row in [header] + body) for i in range(len(header))]
    for row in [header] + body:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())


# Entry point: read --models, validate the keys, run each model (one failure never stops the rest),
# print the table, and save the full results as timestamped JSON under results/smoke/.
def main():
    models = load_models()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(models), metavar="KEY",
                        help=f"model keys from models.yaml (default: all). Valid: {', '.join(models)}")
    args = parser.parse_args()
    unknown = [k for k in args.models if k not in models]
    if unknown:
        parser.error(f"unknown model(s) {unknown}. Valid models: {', '.join(models)}")

    client, toolbox = make_client(), Toolbox()
    started = datetime.now(timezone.utc)
    records, rows = {}, []
    for key in args.models:
        try:
            records[key] = run_model(client, models[key], toolbox)
        except Exception as e:  # one model failing must not stop the others
            records[key] = {"error": scrub(f"{type(e).__name__}: {e}")}
        rows.append(summarize(key, records[key]))

    print_table(rows)
    out = RESULTS_DIR / "smoke" / f"smoke_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"started_at": started.isoformat(), "question": QUESTION, "models": args.models,
               "summary": rows, "records": records}
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
