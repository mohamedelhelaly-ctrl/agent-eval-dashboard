def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SPECS = [
    _tool(
        "list_tables",
        "Returns the names of all tables in the SQLite database.",
        {},
        [],
    ),
    _tool(
        "describe_table",
        "Returns a table's columns (name, type, primary_key, nullable) and its foreign keys, if any.",
        {"table": {"type": "string", "description": "Exact name of a table in the database."}},
        ["table"],
    ),
    _tool(
        "sample_rows",
        "Returns the first n rows of a table, in table order, as a list of objects keyed by column name. "
        "n is clamped to 1-10.",
        {
            "table": {"type": "string", "description": "Exact name of a table in the database."},
            "n": {"type": "integer", "minimum": 1, "maximum": 10,
                  "description": "Number of rows to return (default 3)."},
        },
        ["table"],
    ),
    _tool(
        "run_sql",
        "Runs one read-only SQL SELECT statement (SQLite dialect; WITH/CTEs allowed) and returns "
        "columns and rows. Only one statement per call. Returns at most 50 rows and reports "
        "truncated=true when more exist. Queries running longer than 5 seconds are aborted. "
        "Writes and PRAGMA/ATTACH statements are rejected.",
        {"query": {"type": "string", "description": "A single SQLite SELECT statement."}},
        ["query"],
    ),
    _tool(
        "lookup_metric_definition",
        "Returns the glossary definition of a business term or metric, matched by name or alias "
        "(case-insensitive), together with the full list of calculation conventions.",
        {"term": {"type": "string", "description": "Metric or term name, or one of its aliases."}},
        ["term"],
    ),
    _tool(
        "calculate",
        "Evaluates an arithmetic expression. Allowed: integer and decimal numbers, + - * / // % **, "
        "unary + and -, parentheses, and round(), abs(), min(), max(). Maximum 200 characters; "
        "exponent magnitude at most 100; result rounded to 10 decimal places.",
        {"expression": {"type": "string", "description": "The arithmetic expression, e.g. '1200 * 1.07 ** 3'."}},
        ["expression"],
    ),
]
