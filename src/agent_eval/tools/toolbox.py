import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from agent_eval.config import DB_PATH, GLOSSARY_PATH

MAX_ROWS = 50
QUERY_TIMEOUT_S = 5
ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


class Toolbox:
    def __init__(self, db_path=DB_PATH, glossary_path=GLOSSARY_PATH):
        self.db_path = db_path
        self.glossary_path = glossary_path

    def _connect(self) -> sqlite3.Connection:
        # as_uri() escapes odd characters; mode=ro makes SQLite itself refuse any write.
        return sqlite3.connect(Path(self.db_path).resolve().as_uri() + "?mode=ro", uri=True)

    def _table_names(self, con: sqlite3.Connection) -> list[str]:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [r[0] for r in rows]

    def _check_table(self, con: sqlite3.Connection, table: str) -> dict | None:
        # PRAGMA can't take ? parameters, so a name must match a real table before it enters SQL.
        names = self._table_names(con)
        if table not in names:
            return {"error": f"Unknown table {table!r}. Available tables: {', '.join(names)}"}
        return None

    # Tool 1: list the names of all user-created tables in the database.
    def list_tables(self) -> dict:
        try:
            with closing(self._connect()) as con:
                return {"tables": self._table_names(con)}
        except Exception as e:  # tools never raise; the agent gets the error as data
            return {"error": f"list_tables failed: {e}"}

    # Tool 2: describe a table's columns, including primary-key, nullable, and foreign-key details.
    def describe_table(self, table: str) -> dict:
        try:
            with closing(self._connect()) as con:
                if err := self._check_table(con, table):
                    return err
                # table_info rows: cid, name, type, notnull, default, pk
                cols = [
                    {"name": c[1], "type": c[2], "primary_key": bool(c[5]), "nullable": not c[3]}
                    for c in con.execute(f'PRAGMA table_info("{table}")')
                ]
                result = {"table": table, "columns": cols}
                # foreign_key_list rows: id, seq, ref_table, from, to, ...
                fks = [
                    {"column": f[3], "references_table": f[2], "references_column": f[4]}
                    for f in con.execute(f'PRAGMA foreign_key_list("{table}")')
                ]
                if fks:
                    result["foreign_keys"] = fks
                return result
        except Exception as e:
            return {"error": f"describe_table failed: {e}"}

    # Tool 3: return up to 10 example rows from a table so its data can be inspected.
    def sample_rows(self, table: str, n=3) -> dict:
        try:
            n = max(1, min(10, int(n)))
        except (TypeError, ValueError):
            return {"error": f"n must be an integer, got {n!r}"}
        try:
            with closing(self._connect()) as con:
                if err := self._check_table(con, table):
                    return err
                con.row_factory = sqlite3.Row  # rows convert to dicts keyed by column name
                rows = con.execute(f'SELECT * FROM "{table}" LIMIT ?', (n,)).fetchall()
                return {"table": table, "rows": [dict(r) for r in rows]}
        except Exception as e:
            return {"error": f"sample_rows failed: {e}"}

    # Tool 4: find a metric's glossary definition by its name or alias, with calculation conventions.
    def lookup_metric_definition(self, term: str) -> dict:
        try:
            glossary = json.loads(Path(self.glossary_path).read_text())
            wanted = str(term).strip().lower()
            for t in glossary["terms"]:
                if wanted in [t["term"].lower(), *(a.lower() for a in t.get("aliases", []))]:
                    # Full conventions on every hit: they change how every metric is computed.
                    return {"term": t["term"], "definition": t["definition"],
                            "conventions": glossary["conventions"]}
            known = ", ".join(t["term"] for t in glossary["terms"])
            return {"error": f"Unknown term {term!r}. Known terms: {known}"}
        except Exception as e:
            return {"error": f"lookup_metric_definition failed: {e}"}

    # Tool 5: run one safe, read-only SQL query and return a bounded result or an error.
    def run_sql(self, query: str) -> dict:

        # Reject missing or whitespace-only input before opening the database.
        if not isinstance(query, str) or not query.strip():
            return {"error": "query must be a non-empty string"}
        
        # Track whether SQLite stopped the query because it exceeded the time limit.
        timed_out = False

        # Use a monotonic clock so changes to the system clock do not affect the deadline.
        deadline = time.monotonic() + QUERY_TIMEOUT_S

        # SQLite calls this for database operations; only explicitly allowed read operations pass.
        def authorizer(action, *_):
            # Allowlist: anything not explicitly listed (writes, PRAGMA, ATTACH...) is denied.
            return sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


        # SQLite periodically calls this while executing a query to enforce the time limit.
        def progress():
            nonlocal timed_out
            # Record when the deadline has passed and request that SQLite abort the query.
            timed_out = time.monotonic() > deadline
            return 1 if timed_out else 0  # non-zero aborts the running query


        try:
            # Open a read-only connection and install the safety checks for this query.
            with closing(self._connect()) as con:
                con.set_authorizer(authorizer)
                con.set_progress_handler(progress, 1000)  # called every 1000 VM steps

                # Execute one SQL statement; SQLite rejects multiple statements in this call.
                cur = con.execute(query)

                # Capture output column names and fetch one extra row to detect truncation.
                columns = [d[0] for d in cur.description or []]
                rows = cur.fetchmany(MAX_ROWS + 1)  # one extra row reveals truncation

                # Return at most MAX_ROWS rows, with an explicit count and truncation flag.
                return {
                    "columns": columns,
                    "rows": [list(r) for r in rows[:MAX_ROWS]],
                    "row_count": min(len(rows), MAX_ROWS),
                    "truncated": len(rows) > MAX_ROWS,
                }
            
        except sqlite3.ProgrammingError:
            # Explain when the SQL API rejects multiple statements in one query.
            return {"error": "Only one SQL statement is allowed per query."}
        except sqlite3.Error as e:
            # Distinguish an interrupted over-time query from other SQLite errors.
            if timed_out:
                return {"error": f"Query timed out after {QUERY_TIMEOUT_S} seconds."}
            return {"error": f"SQL error: {e}"}
        except Exception as e:
            # Report unexpected failures in the same error-result format as other tools.
            return {"error": f"run_sql failed: {e}"}
