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


def count(table):
    return tb.run_sql(f"SELECT COUNT(*) FROM {table}")["rows"][0][0]


def test_run_sql_basic_select():
    out = tb.run_sql("SELECT COUNT(*) AS n FROM customers")
    assert out == {"columns": ["n"], "rows": [[500]], "row_count": 1, "truncated": False}


def test_run_sql_cte_recursive_and_functions():
    assert tb.run_sql("WITH c AS (SELECT * FROM plans) SELECT COUNT(*) FROM c")["rows"] == [[4]]
    assert tb.run_sql("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r WHERE x<5) SELECT SUM(x) FROM r")["rows"] == [[15]]
    assert "error" not in tb.run_sql("SELECT c.country, COUNT(*) FROM customers c JOIN subscriptions s USING (customer_id) GROUP BY 1")


def test_run_sql_truncates_at_max_rows():
    big = tb.run_sql("SELECT * FROM invoices")
    assert big["row_count"] == 50 and len(big["rows"]) == 50 and big["truncated"] is True
    small = tb.run_sql("SELECT * FROM plans")
    assert small["row_count"] == 4 and small["truncated"] is False


@pytest.mark.parametrize("sql", [
    "INSERT INTO plans (plan_name, monthly_price, annual_price) VALUES ('x', 1, 1)",
    "UPDATE plans SET monthly_price = 0",
    "DELETE FROM plans",
    "DROP TABLE plans",
    "CREATE TABLE evil (a)",
])
def test_run_sql_denies_writes(sql):
    assert "error" in tb.run_sql(sql)
    assert count("plans") == 4 and "evil" not in tb.list_tables()["tables"]


@pytest.mark.parametrize("sql", [
    "PRAGMA table_info(customers)",
    "ATTACH DATABASE ':memory:' AS x",
    "SELECT load_extension('nothing')",
])
def test_run_sql_denies_escape_routes(sql):
    assert "error" in tb.run_sql(sql)


def test_run_sql_one_statement_only():
    assert "one SQL statement" in tb.run_sql("SELECT 1; SELECT 2")["error"]
    assert "error" in tb.run_sql("SELECT 1; DROP TABLE plans")
    assert count("plans") == 4


def test_run_sql_timeout(monkeypatch):
    monkeypatch.setattr("agent_eval.tools.toolbox.QUERY_TIMEOUT_S", 0.2)
    out = tb.run_sql("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) SELECT COUNT(*) FROM r")
    assert "timed out" in out["error"]


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_run_sql_rejects_bad_input(bad):
    assert "error" in tb.run_sql(bad)


def test_run_sql_passes_through_sql_errors():
    assert tb.run_sql("SELEC 1")["error"].startswith("SQL error:")
    assert tb.run_sql("SELECT * FROM nope")["error"].startswith("SQL error:")
