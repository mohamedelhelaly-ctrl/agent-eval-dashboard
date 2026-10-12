import json

import pytest

from agent_eval.records import SCHEMA_VERSION, RunRecord, StepRecord, ToolCallRecord


def tool_call(**overrides):
    values = dict(id="call_1", name="run_sql", arguments_raw='{"query": "SELECT 1"}', arguments={"query": "SELECT 1"},
                  result={"columns": ["1"], "rows": [[1]]}, is_error=False)
    return ToolCallRecord(**{**values, **overrides})


def step(**overrides):
    values = dict(index=1, content="hello", finish_reason="stop", prompt_tokens=100, completion_tokens=10,
                  cached_tokens=0, cost_usd=0.001, reported_cost_usd=None, latency_s=1.0)
    return StepRecord(**{**values, **overrides})


def run(**overrides):
    values = dict(run_id="r1", agent_name="ling__minimal", model_id="inclusionai/ling-3.0-flash-sante",
                  prompt_name="minimal", question_id="q01", repeat=0, status="answered",
                  started_at="2026-01-01T12:00:00+00:00", wall_time_s=3.2, final_text='{"answer": 500, "answerable": true}',
                  parse_ok=True, answer=500, answerable=True, clean_format=True, parse_problem=None,
                  steps=[step()], config_snapshot={"temperature_sent": 0.0})
    return RunRecord(**{**values, **overrides})


def full_run():
    return run(
        final_text="Réponse: café     \U0001F600",
        answer="café",
        steps=[
            step(index=1, content=None, finish_reason="tool_calls", tool_calls=[
                tool_call(),
                tool_call(id="call_2", arguments_raw="{bad", arguments=None, result={"error": "bad json"}, is_error=True),
                tool_call(id="call_3", result=None, executed=False),
            ], reported_cost_usd=0.002, response_id="gen-1", served_model="m", provider="Novita"),
            step(index=2, attempts=2, retry_wait_s=1.0, latency_s=2.5),
        ],
        config_snapshot={"agent": {"name": "x", "temperature": 0.0}, "prompt_sha256": "ab", "temperature_sent": None,
                         "as_of_date": "2026-01-01", "db_sha256": "cd"},
    )


# ---------- JSON line ----------

def test_json_round_trip():
    original = full_run()
    assert RunRecord.from_json_line(original.to_json_line()) == original


@pytest.mark.parametrize("answer", [500, 7.5, 1.0, "42", "", None])
def test_round_trip_keeps_the_answer_type(answer):
    back = RunRecord.from_json_line(run(answer=answer).to_json_line())
    assert back.answer == answer and type(back.answer) is type(answer)


def test_to_json_line_is_one_line_with_no_newline():
    line = full_run().to_json_line()
    assert "\n" not in line and "\r" not in line and not line.endswith("\n")
    assert line.isascii()
    assert len(line.splitlines()) == 1  # U+2028 / U+2029 in final_text must not split it
    json.loads(line)  # valid JSON


def test_line_includes_the_computed_values_for_readability():
    data = json.loads(run().to_json_line())
    for key in ("total_cost_usd", "n_steps", "n_tool_calls", "n_tool_errors", "total_prompt_tokens"):
        assert key in data
    assert {"effective_cost_usd", "cost_source"} <= set(data["steps"][0])


def test_computed_values_in_a_line_are_ignored_and_recalculated():
    data = json.loads(full_run().to_json_line())
    data["total_cost_usd"], data["n_steps"] = 999, 999
    data["steps"][0]["effective_cost_usd"], data["steps"][0]["cost_source"] = 999, "wrong"
    back = RunRecord.from_json_line(json.dumps(data))
    assert back == full_run() and back.n_steps == 2 and back.steps[0].cost_source == "reported"


@pytest.mark.parametrize("line", ["", "not json", "[1, 2]", '"text"', "null", "{}", '{"run_id": "r1"}'])
def test_from_json_line_raises_value_error_on_bad_lines(line):
    with pytest.raises(ValueError):
        RunRecord.from_json_line(line)


def test_schema_version_is_stored():
    assert run().schema_version == SCHEMA_VERSION == 1
    assert json.loads(run().to_json_line())["schema_version"] == 1


# ---------- validation ----------

def test_extra_field_is_rejected_on_every_class():
    with pytest.raises(ValueError):
        run(surprise=1)
    with pytest.raises(ValueError):
        step(surprise=1)
    with pytest.raises(ValueError):
        tool_call(surprise=1)


def test_extra_field_in_a_line_is_rejected():
    data = json.loads(run().to_json_line())
    data["surprise"] = 1
    with pytest.raises(ValueError):
        RunRecord.from_json_line(json.dumps(data))


@pytest.mark.parametrize("status", ["done", "ANSWERED", "", None, "error"])
def test_invalid_status_is_rejected(status):
    with pytest.raises(ValueError):
        run(status=status)


@pytest.mark.parametrize("status", ["answered", "max_steps", "cost_cap", "infra_error"])
def test_every_valid_status_is_accepted(status):
    assert run(status=status).status == status


@pytest.mark.parametrize("bad", [
    dict(index=0), dict(prompt_tokens=-1), dict(completion_tokens=-1), dict(cached_tokens=-1),
    dict(latency_s=-0.1), dict(retry_wait_s=-1), dict(attempts=0), dict(cost_usd=-0.01),
    dict(reported_cost_usd=-0.01), dict(latency_s=float("nan")), dict(cost_usd=float("inf")),
])
def test_step_range_checks(bad):
    with pytest.raises(ValueError):
        step(**bad)


@pytest.mark.parametrize("bad", [dict(wall_time_s=-1), dict(repeat=-1), dict(wall_time_s=float("nan"))])
def test_run_range_checks(bad):
    with pytest.raises(ValueError):
        run(**bad)


def test_tool_call_duration_cannot_be_negative_or_nan():
    for bad in (-1, float("nan")):
        with pytest.raises(ValueError):
            tool_call(duration_s=bad)


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf")])
def test_answer_cannot_be_a_bool_or_non_finite(bad):
    with pytest.raises(ValueError):
        run(answer=bad)


def test_required_fields_are_enforced():
    with pytest.raises(ValueError):
        StepRecord(index=1)  # latency_s missing
    with pytest.raises(ValueError):
        ToolCallRecord(id="x", name="y", arguments_raw="{}")  # arguments, result, is_error missing


def test_defaults():
    t = tool_call()
    assert (t.duration_s, t.executed) == (0.0, True)
    s = StepRecord(index=1, latency_s=1.0)
    assert (s.tool_calls, s.prompt_tokens, s.cost_usd, s.attempts, s.retry_wait_s) == ([], 0, None, 1, 0.0)
    assert run(steps=[]).error_message is None


def test_default_lists_are_not_shared_between_instances():
    a, b = StepRecord(index=1, latency_s=1.0), StepRecord(index=2, latency_s=1.0)
    a.tool_calls.append(tool_call())
    assert b.tool_calls == []


# ---------- totals ----------

def test_totals_over_three_steps():
    r = run(steps=[
        step(index=1, prompt_tokens=100, completion_tokens=10, cached_tokens=0, cost_usd=0.001, latency_s=1.0),
        step(index=2, prompt_tokens=200, completion_tokens=20, cached_tokens=150, cost_usd=0.002, latency_s=2.0,
             attempts=2, retry_wait_s=1.0),
        step(index=3, prompt_tokens=300, completion_tokens=30, cached_tokens=250, cost_usd=0.004, latency_s=3.0,
             attempts=3, retry_wait_s=3.0),
    ])
    assert (r.total_prompt_tokens, r.total_completion_tokens, r.total_cached_tokens) == (600, 60, 400)
    assert r.total_cost_usd == pytest.approx(0.007)
    assert r.total_model_latency_s == pytest.approx(6.0)
    assert r.total_retry_wait_s == pytest.approx(4.0)
    assert (r.total_attempts, r.n_steps) == (6, 3)


def test_totals_over_two_steps():
    r = run(steps=[step(index=1, prompt_tokens=10, cost_usd=0.5), step(index=2, prompt_tokens=5, cost_usd=0.25)])
    assert (r.total_prompt_tokens, r.n_steps, r.total_cost_usd) == (15, 2, pytest.approx(0.75))


def test_a_run_with_no_steps_totals_zero():
    r = run(steps=[])
    assert (r.total_prompt_tokens, r.total_cost_usd, r.n_steps, r.n_tool_calls, r.total_attempts) == (0, 0, 0, 0, 0)


# ---------- cost ----------

def test_reported_cost_wins_over_computed():
    s = step(cost_usd=0.001, reported_cost_usd=0.0042)
    assert s.effective_cost_usd == 0.0042
    assert run(steps=[s]).total_cost_usd == 0.0042


def test_computed_cost_is_used_when_nothing_was_reported():
    assert step(cost_usd=0.001, reported_cost_usd=None).effective_cost_usd == 0.001


def test_no_cost_at_all_is_zero():
    s = step(cost_usd=None, reported_cost_usd=None)
    assert s.effective_cost_usd == 0.0 and s.cost_source == "none"


@pytest.mark.parametrize("cost,reported,source", [
    (0.001, 0.002, "reported"), (None, 0.002, "reported"), (0.001, None, "computed"), (None, None, "none"),
    (0.001, 0.0, "reported"),  # a reported cost of exactly zero is still a reported cost
    (0.0, None, "computed"),   # and a computed zero is still computed
])
def test_cost_source(cost, reported, source):
    assert step(cost_usd=cost, reported_cost_usd=reported).cost_source == source


def test_reported_zero_beats_a_nonzero_computed_cost():
    s = step(cost_usd=0.5, reported_cost_usd=0.0)
    assert s.effective_cost_usd == 0.0


def test_total_cost_mixes_reported_and_computed_steps():
    r = run(steps=[step(index=1, cost_usd=0.001, reported_cost_usd=0.01), step(index=2, cost_usd=0.002),
                   step(index=3, cost_usd=None)])
    assert r.total_cost_usd == pytest.approx(0.012)


# ---------- tool call counts ----------

def test_non_executed_calls_are_not_counted_as_calls():
    r = run(steps=[step(tool_calls=[tool_call(id="a"), tool_call(id="b", result=None, executed=False)])])
    assert r.n_tool_calls == 1


def test_n_tool_errors_counts_only_executed_errors():
    r = run(steps=[
        step(index=1, tool_calls=[
            tool_call(id="ok"),
            tool_call(id="bad", result={"error": "x"}, is_error=True),
            tool_call(id="skipped", result=None, is_error=True, executed=False),  # never ran: not counted
        ]),
        step(index=2, tool_calls=[tool_call(id="bad2", arguments=None, result={"error": "y"}, is_error=True)]),
    ])
    assert (r.n_tool_calls, r.n_tool_errors) == (3, 2)


def test_tool_counts_survive_a_round_trip():
    back = RunRecord.from_json_line(full_run().to_json_line())
    assert (back.n_tool_calls, back.n_tool_errors) == (2, 1)


# ---------- infra_error run ----------

def test_infra_error_record_without_a_final_answer_is_valid():
    r = RunRecord(run_id="r9", agent_name="a", model_id="m", prompt_name="p", question_id="q1", repeat=2,
                  status="infra_error", started_at="2026-01-01T00:00:00+00:00", wall_time_s=91.0,
                  final_text=None, parse_ok=False, answer=None, answerable=None, clean_format=False,
                  parse_problem="no final text", error_message="LLMUnavailable: gave up after 4 attempts",
                  config_snapshot={})
    assert r.steps == [] and r.total_cost_usd == 0 and r.n_steps == 0
    assert RunRecord.from_json_line(r.to_json_line()) == r
