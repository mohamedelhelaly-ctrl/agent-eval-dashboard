import sqlite3

import pytest

from agent_eval.tools.toolbox import Toolbox

tb = Toolbox()


def test_list_tables_includes_core_tables():
    assert {"customers", "subscriptions"} <= set(tb.list_tables()["tables"])


def test_describe_table_returns_columns():
    cols = tb.describe_table("customers")["columns"]
    assert {"name": "customer_id", "type": "INTEGER", "primary_key": True, "nullable": True} in cols


def test_describe_table_flags():
    cols = {c["name"]: c for c in tb.describe_table("customers")["columns"]}
    assert cols["customer_id"]["primary_key"] and not cols["company_name"]["nullable"]
    assert cols["industry"]["nullable"] and not cols["industry"]["primary_key"]


def test_describe_table_foreign_keys():
    fks = tb.describe_table("subscriptions")["foreign_keys"]
    assert {"column": "customer_id", "references_table": "customers", "references_column": "customer_id"} in fks
    assert "foreign_keys" not in tb.describe_table("plans")


def test_unknown_table_errors_list_available_tables():
    err = tb.describe_table("nope")["error"]
    assert "customers" in err and "subscriptions" in err
    assert "error" in tb.describe_table('customers"); DROP TABLE customers; --')
    assert "error" in tb.sample_rows("nope")


def test_missing_database_returns_error_not_exception(tmp_path):
    bad = Toolbox(tmp_path / "missing.db")
    assert "error" in bad.list_tables() and "error" in bad.describe_table("customers")
    assert "error" in bad.sample_rows("customers")


def test_connection_is_read_only():
    with pytest.raises(sqlite3.OperationalError):
        tb._connect().execute("DELETE FROM plans")


def test_sample_rows_default_and_order():
    out = tb.sample_rows("customers")
    assert out["table"] == "customers" and len(out["rows"]) == 3
    assert [r["customer_id"] for r in out["rows"]] == [1, 2, 3]
    assert "company_name" in out["rows"][0]


def test_sample_rows_clamps_and_coerces_n():
    assert len(tb.sample_rows("customers", 0)["rows"]) == 1
    assert len(tb.sample_rows("customers", 99)["rows"]) == 10
    assert len(tb.sample_rows("customers", "5")["rows"]) == 5
    assert "error" in tb.sample_rows("customers", "abc")
    assert "error" in tb.sample_rows("customers", None)


@pytest.mark.parametrize("term", ["MRR", "mrr", "Monthly Recurring Revenue", "  recurring revenue "])
def test_lookup_hits_by_name_alias_and_case(term):
    out = tb.lookup_metric_definition(term)
    assert out["term"] == "MRR" and out["definition"]
    assert len(out["conventions"]) >= 5  # full list on every hit


def test_lookup_miss_lists_known_terms():
    err = tb.lookup_metric_definition("vibes")["error"]
    assert "MRR" in err and "active customer" in err


def test_lookup_bad_glossary_returns_error(tmp_path):
    assert "error" in Toolbox(glossary_path=tmp_path / "missing.json").lookup_metric_definition("MRR")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert "error" in Toolbox(glossary_path=bad).lookup_metric_definition("MRR")
