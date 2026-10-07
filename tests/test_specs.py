import inspect
import json

import pytest

from agent_eval.tools.specs import TOOL_SPECS
from agent_eval.tools.toolbox import Toolbox

tb = Toolbox()
NAMES = ["list_tables", "describe_table", "sample_rows", "run_sql", "lookup_metric_definition", "calculate"]
FUNCS = {s["function"]["name"]: s["function"] for s in TOOL_SPECS}


def test_six_specs_with_expected_names():
    assert [s["function"]["name"] for s in TOOL_SPECS] == NAMES


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=NAMES)
def test_spec_structure(spec):
    assert spec["type"] == "function" and set(spec) == {"type", "function"}
    fn = spec["function"]
    assert fn["description"].strip()
    params = fn["parameters"]
    assert params["type"] == "object" and params["additionalProperties"] is False
    assert set(params["required"]) <= set(params["properties"])
    json.dumps(spec)  # must be JSON-serializable


@pytest.mark.parametrize("name", NAMES)
def test_spec_matches_method_signature(name):
    sig = inspect.signature(getattr(Toolbox, name))
    actual = [p for p in sig.parameters.values() if p.name != "self"]
    params = FUNCS[name]["parameters"]
    assert set(params["properties"]) == {p.name for p in actual}
    assert set(params["required"]) == {p.name for p in actual if p.default is inspect.Parameter.empty}


def test_sample_rows_n_spec():
    params = FUNCS["sample_rows"]["parameters"]
    assert params["properties"]["n"]["type"] == "integer"
    assert params["properties"]["n"]["minimum"] == 1 and params["properties"]["n"]["maximum"] == 10
    assert "n" not in params["required"]


def test_run_sql_and_calculate_descriptions_state_limits():
    sql = FUNCS["run_sql"]["description"]
    assert all(w in sql for w in ("SQLite", "read-only", "one statement", "50"))
    calc = FUNCS["calculate"]["description"]
    assert all(w in calc for w in ("round", "abs", "min", "max", "**"))


def test_call_accepts_dict_json_string_and_none():
    assert "customers" in tb.call("list_tables", None)["tables"]
    assert "customers" in tb.call("list_tables", "")["tables"]
    assert "customers" in tb.call("list_tables", {})["tables"]
    assert tb.call("calculate", {"expression": "2+3"})["result"] == 5
    assert tb.call("calculate", '{"expression": "2+3"}')["result"] == 5
    assert tb.call("sample_rows", '{"table": "plans", "n": 2}')["rows"][0]["plan_id"] == 1


def test_call_unknown_tool_lists_real_names():
    err = tb.call("drop_everything", {})["error"]
    assert all(n in err for n in NAMES)
    for bad in ("_connect", "call", "__init__", None, 5):
        assert "Available tools" in tb.call(bad, {})["error"]


def test_call_wrong_arguments_name_expected_parameters():
    missing = tb.call("describe_table", {})["error"]
    assert "table (required)" in missing
    extra = tb.call("sample_rows", {"table": "plans", "limit": 2})["error"]
    assert "table (required)" in extra and "n (optional)" in extra
    assert "none" in tb.call("list_tables", {"x": 1})["error"]


@pytest.mark.parametrize("args", ["{not json", "[1, 2]", "5", '"text"', "null", [1], 5])
def test_call_invalid_or_non_object_arguments(args):
    assert "error" in tb.call("calculate", args)


def test_call_always_returns_dict():
    for name, args in [("run_sql", {"query": "SELEC"}), ("calculate", {"expression": "1/0"}), (None, None)]:
        assert isinstance(tb.call(name, args), dict)
