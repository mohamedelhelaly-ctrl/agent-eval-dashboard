import ast
import json
import math
import operator
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from agent_eval.config import DB_PATH, GLOSSARY_PATH
from agent_eval.tools.specs import TOOL_SPECS

MAX_ROWS = 50
QUERY_TIMEOUT_S = 5
MAX_EXPR_LEN = 200
MAX_EXPONENT = 100
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

    # Entry point for a model's tool call: look up the tool, parse its arguments, run it. Never raises.
    def call(self, name, arguments=None) -> dict:
        try:
            # Build a lookup of public tool names and their argument specifications.
            specs = {s["function"]["name"]: s["function"] for s in TOOL_SPECS}
            # Only names published in TOOL_SPECS are callable; private helpers like _connect are not.
            if not isinstance(name, str) or name not in specs:
                return {"error": f"Unknown tool {name!r}. Available tools: {', '.join(specs)}"}

            # Models may send a dict, a JSON string (possibly empty), or nothing at all.
            # Treat missing or empty arguments as an empty object.
            if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
                arguments = {}
            
            elif isinstance(arguments, str):
                # Convert JSON text into Python values before passing it to the tool.
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as e:
                    return {"error": f"arguments are not valid JSON: {e}"}
            
            # Tools expect named arguments, represented by a JSON object/dictionary.
            if not isinstance(arguments, dict):
                return {"error": "arguments must be a JSON object"}
            
            try:
                # Find the selected method by name and call it with the supplied arguments.
                return getattr(self, name)(**arguments)
            except TypeError:
                # If the call has wrong or missing parameters, report the expected parameter names.
                params = specs[name]["parameters"]
                expected = ", ".join(
                    f"{p} (required)" if p in params["required"] else f"{p} (optional)"
                    for p in params["properties"]
                ) or "none"
                return {"error": f"Invalid arguments for {name}. Expected parameters: {expected}"}
        except Exception as e:
            # Return unexpected failures as an error object rather than raising them to the caller.
            return {"error": f"call failed: {e}"}



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
                    {"name": c[1], "type": c[2], "primary_key": bool(c[5]), "nullable": not (c[3] or c[5])}  # a primary key can never hold NULL
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
        except (TypeError, ValueError, OverflowError):  # OverflowError: int(float('inf'))
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
                result = {
                    "columns": columns,
                    "rows": [list(r) for r in rows[:MAX_ROWS]],
                    "row_count": min(len(rows), MAX_ROWS),
                    "truncated": len(rows) > MAX_ROWS,
                }
                # Tell the caller the data is incomplete and how to narrow it.
                if result["truncated"]:
                    result["note"] = (f"Only the first {MAX_ROWS} rows are shown; "
                                      "add a WHERE, GROUP BY or LIMIT to narrow the result.")
                return result

        except sqlite3.Error as e:
            # Distinguish an interrupted over-time query from other SQLite errors.
            if timed_out:
                return {"error": f"Query timed out after {QUERY_TIMEOUT_S} seconds."}
            # Only the multiple-statement ProgrammingError gets the friendly message.
            if isinstance(e, sqlite3.ProgrammingError) and "one statement" in str(e):
                return {"error": "Only one SQL statement is allowed per query."}
            return {"error": f"SQL error: {e}"}
        except Exception as e:
            # Report unexpected failures in the same error-result format as other tools.
            return {"error": f"run_sql failed: {e}"}

    # Tool 6: safely evaluate an arithmetic expression without eval/exec.
    def calculate(self, expression: str) -> dict:
        if not isinstance(expression, str) or not expression.strip():
            return {"error": "expression must be a non-empty string"}
        
        if len(expression) > MAX_EXPR_LEN:
            return {"error": f"Expression too long (max {MAX_EXPR_LEN} characters)."}
        
        try:
            # Parse to a syntax tree (never executed), then walk it with our own allowlist.
            result = _eval_node(ast.parse(expression.strip(), mode="eval").body)
            if not math.isfinite(result):
                return {"error": "Result is not a finite number."}
            return {"expression": expression, "result": round(result, 10)}
        except ZeroDivisionError:
            return {"error": "Division by zero."}
        except RecursionError:
            return {"error": "Expression is nested too deeply."}
        except (ValueError, TypeError, OverflowError, SyntaxError) as e:
            return {"error": f"Invalid expression: {e}"}
        except Exception as e:
            return {"error": f"calculate failed: {e}"}


# Allowlists: the only operators and functions the calculator will ever run.
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max}


def _eval_node(node):
    """Evaluate one expression-tree node, allowing only explicitly supported syntax."""
    # If this node is a literal value, check that it is a permitted number.
    if isinstance(node, ast.Constant):
        # Require exactly int or float; using type() also excludes bool, which is an int subclass.
        if type(node.value) not in (int, float):  # type() check also rejects bool
            # Reject strings, booleans, and other literal types.
            raise ValueError("only int and float numbers are allowed")
        # Return the accepted numeric literal as-is.
        return node.value

    
    # If this is a binary operation, allow it only when its operator is on the approved list.
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        # Recursively evaluate the expression on the left and right of the operator.
        left, right = _eval_node(node.left), _eval_node(node.right)
        # Limit exponent size to avoid excessively expensive or enormous calculations.
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            # Reject powers whose exponent is outside the configured limit.
            raise ValueError(f"exponent magnitude must be at most {MAX_EXPONENT}")
        # Apply the approved operation to the evaluated left and right values.
        return _BINARY[type(node.op)](left, right)

    
    # If this is a unary operation, allow it only when its operator is approved.
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        # Apply unary plus or minus to the recursively evaluated operand.
        return _UNARY[type(node.op)](_eval_node(node.operand))

    
    # If this is a function call, verify that the function and its arguments are allowed.
    if isinstance(node, ast.Call):
        # Accept only direct calls to the approved function names.
        if not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ValueError("only round, abs, min and max may be called")
        # Disallow keyword arguments and argument unpacking such as *values.
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise ValueError("positional arguments only")
        # Require at least one argument for each supported function.
        if not node.args:
            raise ValueError(f"{node.func.id} needs at least one argument")
        # Safely evaluate each positional argument, then call the approved function.
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    
    # Reject names, attributes, comparisons, and any other unsupported syntax.
    raise ValueError(f"unsupported syntax: {type(node).__name__}")
