# Repo Guide: agent-eval-dashboard

This repository is a **dashboard for comparing versions of one AI agent**. The agent is a "data analyst": it gets a business question in plain English ("What was the churn rate in Q1 2025?"), explores a database with a small set of read-only tools, and returns one answer that can be checked automatically. Agent versions differ only in the **model** and the **prompt**, so any change in results can be traced to one cause. They are scored on task success, cost and speed.

This guide walks the repo from the root down. It describes what exists **today**. Several parts of the plan (agent loop, eval questions, scoring, dashboard) are not built yet; the section "How the pieces fit together" says which is which.

## Contents

- [How the pieces fit together](#how-the-pieces-fit-together)
- [Setup and everyday commands](#setup-and-everyday-commands)
- [Root files](#root-files)
  - [`.env`](#env)
  - [`.env.example`](#envexample)
  - [`.gitignore`](#gitignore)
  - [`README.md`](#readmemd)
  - [`pyproject.toml`](#pyprojecttoml)
  - [`requirements.txt`](#requirementstxt)
- [`.claude/`](#claude)
  - [`.claude/skills/repo-guide/SKILL.md`](#claudeskillsrepo-guideskillmd)
- [`configs/`](#configs)
  - [`configs/models.yaml`](#configsmodelsyaml)
  - [`configs/agents/`](#configsagents)
  - [`configs/agents/gemini__minimal.yaml`](#configsagentsgemini__minimalyaml)
  - [`configs/agents/ling__minimal.yaml`](#configsagentsling__minimalyaml)
  - [`configs/agents/luna__minimal.yaml`](#configsagentsluna__minimalyaml)
  - [`configs/agents/solar__minimal.yaml`](#configsagentssolar__minimalyaml)
- [`data/`](#data)
  - [`data/db_meta.json`](#datadb_metajson)
  - [`data/generate_db.py`](#datagenerate_dbpy)
  - [`data/glossary.json`](#dataglossaryjson)
  - [`data/saas.db`](#datasaasdb)
- [`docs/`](#docs)
  - [`docs/project_plan.md`](#docsproject_planmd)
  - [`docs/REPO_GUIDE.md`](#docsrepo_guidemd)
- [`prompts/`](#prompts)
  - [`prompts/minimal.md`](#promptsminimalmd)
- [`results/`](#results)
- [`scripts/`](#scripts)
  - [`scripts/peek_db.py`](#scriptspeek_dbpy)
  - [`scripts/smoke_models.py`](#scriptssmoke_modelspy)
- [`src/agent_eval/`](#srcagent_eval)
  - [`src/agent_eval/__init__.py`](#srcagent_eval__init__py)
  - [`src/agent_eval/config.py`](#srcagent_evalconfigpy)
  - [`src/agent_eval/schemas.py`](#srcagent_evalschemaspy)
  - [`src/agent_eval/llm.py`](#srcagent_evalllmpy)
  - [`src/agent_eval/tools/__init__.py`](#srcagent_evaltools__init__py)
  - [`src/agent_eval/tools/specs.py`](#srcagent_evaltoolsspecspy)
  - [`src/agent_eval/tools/toolbox.py`](#srcagent_evaltoolstoolboxpy)
- [`tests/`](#tests)
  - [`tests/test_config.py`](#teststest_configpy)
  - [`tests/test_llm.py`](#teststest_llmpy)
  - [`tests/test_schemas.py`](#teststest_schemaspy)
  - [`tests/test_smoke_models.py`](#teststest_smoke_modelspy)
  - [`tests/test_specs.py`](#teststest_specspy)
  - [`tests/test_toolbox.py`](#teststest_toolboxpy)

## How the pieces fit together

**What is built today** (this is the chain the smoke script exercises):

```
data/saas.db  <--read-only--  tools/toolbox.py  <-- Toolbox.call(name, arguments) --+
data/glossary.json  <--------- (6 tools, never raise)                               |
                                                                                    |
configs/models.yaml --load_models--> ModelSpec (price, provider, ...)               |
configs/agents/*.yaml --load_agent--> AgentConfig (model + prompt + limits)         |
prompts/minimal.md --render_prompt (fills {today} from data/db_meta.json)--> system |
                                                                                    |
                 llm.chat(client, spec, messages, tools=TOOL_SPECS)  ---------------+
                   -> one call to OpenRouter, with retries, returning an LLMResult
                      (content, tool_calls, tokens, cost, latency, provider...)
```

1. **Config** (`config.py`) knows where every folder and file lives, loads secrets from `.env`, and fills the prompt's `{today}` with the database's own date (2026-01-01), not the real date.
2. **Schemas** (`schemas.py`) define and validate the YAML files (`ModelSpec`, `AgentConfig`) and the shape of one model call's result (`LLMResult`, `ToolCall`).
3. **Tools** (`tools/`) are the six things the agent may do. `specs.py` is the menu the model reads; `toolbox.py` is the code that runs each tool.
4. **LLM client** (`llm.py`) makes one chat call to OpenRouter, retries temporary failures, works out the cost, and returns an `LLMResult`.
5. **Smoke script** (`scripts/smoke_models.py`) wires all of the above together for a real two-turn check of each model.

**Planned in `docs/project_plan.md` but not built yet:** the agent loop that keeps going until an answer arrives, the 30-40 question eval set with expected answers, the scorers (task completion, tool accuracy, cost per success, p50/p95 latency, hallucination rate), the runner that writes full results, Langfuse tracing, and the Streamlit dashboard. Packages for those (`langfuse`, `pandas`, `streamlit`, `plotly`) are already in `requirements.txt` but no code uses them yet.

## Setup and everyday commands

- Install packages: `pip install -r requirements.txt` (inside `.venv`).
- Make the package importable: `pip install -e .`, or put `src` on the path with `PYTHONPATH=src` (the examples below use this form).
- Run the tests: `PYTHONPATH=src .venv/bin/python -m pytest`. Tests marked `live` are skipped by default.
- Look at the database: `.venv/bin/python scripts/peek_db.py`.
- Real model check (costs a fraction of a cent): `PYTHONPATH=src .venv/bin/python scripts/smoke_models.py`.

## Root files

### `.env`
**Purpose:** Your private settings file, loaded by `config.py` when the project starts. It holds secrets such as the OpenRouter API key. It is git-ignored, and this guide deliberately does not show its contents.

### `.env.example`
**Purpose:** A template that shows which settings the project expects, with no real values. Copy it to `.env` and fill it in.

**Variable names:** `OPENROUTER_API_KEY` (all model calls go through it; the only one used by code today), `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL` (reserved for Langfuse tracing, not used by code yet).

### `.gitignore`
**Purpose:** Tells git which files not to track. It is the standard Python template (caches, virtual environments, build folders) plus: `.env`, `.venv`, and `results/` (experiment output stays out of git).

### `README.md`
**Purpose:** A two-line project summary: a dashboard that compares agent versions on task success, cost and latency. Tech listed: Python, Langfuse, pytest.

### `pyproject.toml`
**Purpose:** Tells Python how to build and install the project, and configures pytest.

**Key settings:**
- `name = "agent-eval"`, `version = "0.1.0"`, `requires-python = ">=3.12"`.
- `[tool.setuptools.packages.find] where = ["src"]`: the importable package lives in `src/`.
- `[tool.pytest.ini_options]`: tests are in `tests/`; a `live` marker is declared for tests that call the real API; `addopts = "-m 'not live'"` skips them unless you run `pytest -m live`.

### `requirements.txt`
**Purpose:** The list of Python packages to install, with minimum versions.

**Groups:** model access and tracing (`openai`, `langfuse`), config and data models (`python-dotenv`, `pydantic`, `pyyaml`), metrics and dashboard (`pandas`, `streamlit`, `plotly`), and testing and linting (`pytest`, `ruff`). Only `openai`, `python-dotenv`, `pydantic`, `pyyaml` and `pytest` are used by code today.

## `.claude/`

A folder for Claude Code settings. It currently holds one project skill.

### `.claude/skills/repo-guide/SKILL.md`
**Purpose:** The instructions for the "repo-guide" skill, which is what generates this document (`docs/REPO_GUIDE.md`). It describes how to walk the tree, what to write per file, and rules such as never showing `.env`. It is not part of the application.

## `configs/`

Settings for the experiments, written as YAML. Nothing here is code; the loaders in `schemas.py` read and validate it.

### `configs/models.yaml`
**Purpose:** The catalogue of models the project can call, with prices and routing. Loaded by `load_models()`. The file has one top-level `models:` mapping; each key (`ling`, `solar`, `luna`, `gemini`) is the short name used everywhere else.

**What each field means:**
- `id`: the model name sent to OpenRouter (for example `google/gemini-3.8-flash`).
- `display_name`: label for tables and the dashboard.
- `tier`: `free`, `budget`, `mid` or `premium`, for grouping.
- `price_in_per_m`, `price_out_per_m`: dollars per million input and output tokens.
- `price_cached_in_per_m`: dollars per million cached input tokens (cheaper). Used in the cost calculation.
- `params_total_b`, `params_active_b`: model size in billions of parameters, and how many are used per token. `null` when the vendor does not say.
- `provider`: the upstream provider to pin the request to (`novita`, `upstage`, `openai`, `google-ai-studio`).
- `supports_temperature`: `false` for `luna`, whose hosts accept no temperature setting, so the request leaves it out.
- `notes`: free text.

| Key | Model id | Tier | Pinned provider | Temperature |
|---|---|---|---|---|
| `ling` | `inclusionai/ling-3.0-flash-sante` | budget | novita | yes |
| `solar` | `upstage/solar-mini4` | budget | upstage | yes |
| `luna` | `openai/gpt-6-luna` | mid | openai | no |
| `gemini` | `google/gemini-3.8-flash` | premium | google-ai-studio | yes |

### `configs/agents/`
Four agent definitions, one file per agent. Each file is a model plus a prompt plus limits. The file name must equal the `name` inside it (`load_agent` enforces this). Today all four use the same prompt and settings; only the model differs.

**Keys in every agent file:**
- `name`: must equal the file name without `.yaml`.
- `model`: a key from `models.yaml`.
- `prompt`: the name of a file in `prompts/` (without `.md`).
- `temperature` (0.0): sampling randomness, 0 to 2.
- `max_steps` (15): the most tool-calling turns allowed per question.
- `max_run_cost_usd` (0.25): the dollar cap for one run.
- `reasoning_effort` (`null`): optional `low`, `medium` or `high`.

### `configs/agents/gemini__minimal.yaml`
**Purpose:** The agent that uses model `gemini` with prompt `minimal` and the default limits above.

### `configs/agents/ling__minimal.yaml`
**Purpose:** The agent that uses model `ling` with prompt `minimal` and the default limits above.

### `configs/agents/luna__minimal.yaml`
**Purpose:** The agent that uses model `luna` with prompt `minimal` and the default limits above. Because `luna` ignores temperature, its `temperature: 0.0` has no effect.

### `configs/agents/solar__minimal.yaml`
**Purpose:** The agent that uses model `solar` with prompt `minimal` and the default limits above.

## `data/`

The synthetic company database the agent questions, plus everything needed to describe and regenerate it.

### `data/db_meta.json`
**Purpose:** A small fact sheet about the database, written by `generate_db.py`.

**Keys:** `seed` (42, which makes the data reproducible), `as_of_date` (`2026-01-01`, the "today" for all questions), `data_start` (2023-01-01), `data_end` (2025-12-31), `row_counts` per table, and `content_sha256` (a fingerprint of the whole database, so you can tell if it changed). `config.get_as_of_date()` reads `as_of_date` from here.

### `data/generate_db.py`
**Purpose:** Builds the synthetic SaaS database from a fixed random seed, so it is identical on every run. It also writes `db_meta.json`.

**How to run:** `python data/generate_db.py [output_path]`. With no argument it writes `data/saas.db` next to the script. **It overwrites the existing database and `db_meta.json`.**

**Main contents:**
- Constants at the top: `SEED = 42`, `AS_OF_DATE`, `DATA_START`, `DATA_END`, `N_CUSTOMERS = 500`, `N_TEST_ACCOUNTS = 15`, `N_TICKETS = 1500`, `N_PAUSED = 8`, the four `PLANS`, and the lists the random data is drawn from (names, countries, industries, channels, cancel reasons, ticket categories).
- `SCHEMA`: the SQL that creates the five tables.
- `build_customers(rng)`: creates 500 customers, 15 of them internal test accounts.
- `build_subscriptions(rng, customers)`: gives each customer a subscription history. It uses "stories" (steady, churned, upgrade, downgrade, resub, multi) through `build_segments` and `events_for`, which plant the edge cases: re-subscribers, plan changes, paused subscriptions, and dates exactly on quarter boundaries (`snap_to_boundary`).
- `build_invoices(rng, subs)` and `build_tickets(rng, customers, subs)`: billing history (including refunds and failed payments) and support tickets.
- `write_db(path, ...)`: writes the tables and `db_meta.json`.
- `main()`: runs the steps above.

**Notes:** the date rules baked into the data are repeated in the glossary: `end_date` is exclusive, and a plan change ends one subscription and starts another on the same day.

### `data/glossary.json`
**Purpose:** The business definitions the agent must look up instead of guessing. Served by the `lookup_metric_definition` tool.

**Structure:** `conventions` is a list of 5 rules that apply to every metric (today is 2026-01-01; test accounts are excluded; `end_date` is exclusive; how "active on a date" is decided; a plan change is not a cancellation). `terms` is a list of entries with `term`, `aliases` and `definition`.

**Terms:** active customer, MRR, churn rate, net revenue, ARPA, retention rate, resolution time (each with a few aliases such as "churn", "ARPU", "retention").

### `data/saas.db`
**Purpose:** The SQLite database itself (a binary file, about 544 KB). Produced by `generate_db.py` and opened read-only by the tools and `peek_db.py`.

**Tables and sizes:**

| Table | Rows | What it holds |
|---|---|---|
| `customers` | 500 (15 test accounts) | company, country, industry, signup date, channel, `is_test_account` |
| `plans` | 4 | Starter, Pro (formerly "Growth"), Business, Enterprise, with monthly and annual prices |
| `subscriptions` | 697 | each plan period for a customer: dates, billing cycle, status, cancel reason |
| `invoices` | 5,878 | amount, status (paid/failed/refunded), refunded amount |
| `support_tickets` | 1,500 | priority, category, created and resolved times, CSAT score |

## `docs/`

### `docs/project_plan.md`
**Purpose:** The project plan. It covers the task (a data-analyst agent), the database design and its planted edge cases, the six tools, how agent versions differ (model and prompt, one change at a time), the five eval metrics, and the 30-40 question eval set in six categories. Treat it as the target; this guide describes what is actually built.

### `docs/REPO_GUIDE.md`
**Purpose:** This file. It is generated by the repo-guide skill and refreshed whenever the repo changes.

## `prompts/`

System prompts, one Markdown file per prompt style. Agent files refer to them by name.

### `prompts/minimal.md`
**Purpose:** The baseline system prompt: you are a data analyst, use the tools (read-only access to the company database), and reply with only a JSON object.

**Placeholder:** `{today}` is replaced with `2026-01-01` by `render_prompt("minimal")`. The prompt also contains literal braces in its JSON example, which is why the code uses plain `str.replace` and not `.format`.

**Answer format it asks for:** `{"answer": <number or string or null>, "answerable": <true or false>}`, with plain numbers (no currency symbols or separators) and `answerable: false` plus `answer: null` when the data cannot answer the question.

## `results/`

Git-ignored output from runs. Today it only holds `results/smoke/`, with one JSON file per run of the smoke script, named `smoke_<UTC timestamp>.json` (for example `smoke_20261011T155442Z.json`). Each file has `started_at`, `question`, `models`, a flat `summary` row per model, and the full per-turn `records`. The folder is created by the script when needed.

## `scripts/`

Small command-line tools. They are not part of the importable package and pytest never runs them.

### `scripts/peek_db.py`
**Purpose:** A quick read-only look at the database, used to learn its shape.

**How to run:** `.venv/bin/python scripts/peek_db.py`

**What it prints:** each table's row count, column names and types, and three sample rows; then labeled counts that show why "active customer" depends on the definition: total customers, test accounts, subscriptions by status, customers active on 2026-01-01 decided by dates (excluding paused and test accounts), and customers with any subscription whose status is `active`. It runs top to bottom with no functions.

### `scripts/smoke_models.py`
**Purpose:** A manual check, not a test. It asks each model "How many customers does the database have?" with the real tools, for at most two turns, and shows what really comes back. It makes real, billed API calls.

**How to run:** `PYTHONPATH=src .venv/bin/python scripts/smoke_models.py [--models ling solar]`. With no `--models` it runs every model in `models.yaml`. An unknown key exits with an error that lists the valid keys, before any call.

**Main contents:**
- `run_model(client, spec, toolbox) -> dict`: turn 1 asks the question with `TOOL_SPECS`; if the model requests tools, each is run with `Toolbox.call`, the results are sent back, and turn 2 is one follow-up call. It never loops further. Returns the raw record of both turns.
- `summarize(key, record, pinned_provider=None) -> dict`: flattens a record into one table row: served model and provider, what each turn did, summed latency, tokens and costs, and two flags.
- `total(turns, field)`: sums a field across turns, or `None` if any turn lacks it.
- `normalize_provider(name)` and `pin_mismatch(pinned, served) -> bool`: compare provider names ignoring case and punctuation, so "Google AI Studio" matches `google-ai-studio`; flags when the pinned provider did not answer.
- `print_table(rows)`, `fmt(v, spec)`: print the aligned table. The flag column shows `COST DIFF >10%` (computed vs reported cost) and `PIN MISMATCH`.
- `scrub(text)`: removes the API key from any text before it is shown or saved.
- `main()`: runs each model (one failure never stops the others), prints the table, and saves the JSON to `results/smoke/`.

**Constants:** `QUESTION`, `COST_TOLERANCE = 0.10`.

## `src/agent_eval/`

The importable Python package.

### `src/agent_eval/__init__.py`
**Purpose:** An empty file that marks the folder as a Python package.

### `src/agent_eval/config.py`
**Purpose:** The single place that knows where things are, loads secrets, and prepares the prompt. Importing it also loads `.env`.

**Main contents:**
- `PROJECT_ROOT`: the repo root, worked out from the file's own location. Every path below is built from it, so nothing depends on where you run from.
- Path constants: `DATA_DIR`, `DB_PATH`, `GLOSSARY_PATH`, `DB_META_PATH`, `CONFIGS_DIR`, `MODELS_PATH`, `AGENTS_DIR`, `PROMPTS_DIR`, `RESULTS_DIR`.
- `IDENTIFIER_RE`: letters, digits, `_` and `-` only. Used to keep prompt names from pointing outside their folder.
- `require_env(*names) -> dict[str, str]`: returns the named environment variables, or raises `RuntimeError` naming every missing or empty one. It never prints values. Used by `llm.make_client`.
- `get_as_of_date() -> str`: the database's own "today" (`"2026-01-01"`) from `db_meta.json`. Raises `ValueError` if the key is missing.
- `render_prompt(name, prompts_dir=PROMPTS_DIR) -> str`: loads `prompts/<name>.md` and replaces `{today}`. Raises `ValueError` for a bad name, a missing file or a missing placeholder. Example: `render_prompt("minimal")`.

### `src/agent_eval/schemas.py`
**Purpose:** The data shapes of the project, validated with Pydantic, plus the loaders for the YAML configs. Every model rejects unknown fields (`extra="forbid"`), so a typo is an error instead of being silently ignored. Loaders **raise** `ValueError` on bad input (unlike tools), always naming the file or model.

**Main contents:**
- `ModelSpec`: one model. Fields: `id` (non-empty), `display_name`, `tier`, `price_in_per_m` and `price_out_per_m` (0 or more), `price_cached_in_per_m` (optional, 0 or more, no NaN or infinity), `params_total_b` and `params_active_b` (optional, above 0), `provider`, `notes`, `supports_temperature` (default `True`). A validator rejects active parameters greater than total.
- `load_models(path=MODELS_PATH) -> dict[str, ModelSpec]`: reads `models.yaml` (needs a top-level `models:` mapping) and validates every entry; the error names the bad model key.
- `AgentConfig`: one agent. Fields: `name`, `model`, `prompt`, `temperature` (default 0.0, 0 to 2, no NaN), `max_steps` (default 15, 1 to 50), `max_run_cost_usd` (default 0.25, above 0, no NaN), `reasoning_effort` (`low`, `medium`, `high` or none).
- `load_agent(path, models=None, prompts_dir=PROMPTS_DIR) -> AgentConfig`: reads one agent file, then checks that its `name` equals the file name, its `model` is a key in the models file (the error lists the valid keys), its prompt name is a plain identifier, and `<prompt>.md` exists.
- `load_all_agents(agents_dir=AGENTS_DIR) -> dict[str, AgentConfig]`: loads every `*.yaml` in the folder, keyed by name, reading `models.yaml` once. Raises if the folder has no agent files.
- `ToolCall`: one tool request from the model: `id`, `name`, `arguments` (the raw JSON string the model sent).
- `LLMResult`: everything about one model call: `content`, `tool_calls`, `assistant_message` (ready to append to the conversation), `finish_reason`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `cost_usd` (our estimate), `reported_cost_usd` (what OpenRouter says it charged), `latency_s`, `attempts`, `retry_wait_s`, `response_id`, `served_model`, `provider`, `started_at`.

**Notes:** `load_models` reads its file with the system's default text encoding, while `load_agent` forces UTF-8. Both work for the current plain-ASCII files.

### `src/agent_eval/llm.py`
**Purpose:** Makes one call to a model through OpenRouter and turns the response into an `LLMResult`. It is deliberately thin: it handles retries and cost, nothing else.

**Main contents:**
- `make_client() -> OpenAI`: an OpenAI-compatible client pointed at OpenRouter, with the key from `require_env("OPENROUTER_API_KEY")`. The SDK's own retries are turned off (`max_retries=0`) and the timeout is 90 seconds.
- `chat(client, spec, messages, tools=None, temperature=0.0, sleep=time.sleep, max_retries=3) -> LLMResult`: builds the request (model id, messages, usage reporting turned on), then:
  - adds `temperature` only if `spec.supports_temperature`;
  - adds a provider pin (`order`, `allow_fallbacks: False`, `require_parameters: True`) when `spec.provider` is set;
  - adds `tools` only when given;
  - retries temporary failures (rate limit, 5xx, timeout, connection error, and a response with no choices) up to 3 times, waiting 1, 2, 4 seconds or the server's `Retry-After` (capped at 60);
  - gives up with `LLMUnavailable`; any other error (bad request, bad key, `RuntimeError`...) propagates immediately and unchanged;
  - computes `cost_usd` as uncached input, cached input (at the cached price when the model has one) and output tokens; `None` unless both token counts exist.
  - `latency_s` and `started_at` describe only the successful attempt; `attempts` and `retry_wait_s` record the retrying.
- `LLMError`, `LLMUnavailable(LLMError)`: the two error types raised by this module.
- Helpers `_retry_wait` (how long to wait) and `_dig` (read a nested attribute, `None` if any part is missing).

**Constants:** `OPENROUTER_BASE_URL`, `REQUEST_TIMEOUT_S = 90`, `BACKOFF_BASE_S = 1`, `MAX_RETRY_AFTER_S = 60`, `RETRYABLE_ERRORS`.

**Example:** `result = chat(make_client(), models["ling"], messages, tools=TOOL_SPECS)`. Used by `scripts/smoke_models.py`.

### `src/agent_eval/tools/__init__.py`
**Purpose:** An empty file that marks `tools` as a Python package.

### `src/agent_eval/tools/specs.py`
**Purpose:** The "menu" the model reads: `TOOL_SPECS`, a list of six function definitions in the OpenAI format (name, description, and a parameters schema with `additionalProperties: False`). Descriptions say what each tool does and its limits, not when to use it. A test checks that every spec matches the real method's signature.

**Main contents:**
- `TOOL_SPECS`: `list_tables` (no arguments), `describe_table(table)`, `sample_rows(table, n)` (`n` is 1 to 10, optional), `run_sql(query)`, `lookup_metric_definition(term)`, `calculate(expression)`.
- `_tool(...)`: a small helper that builds one entry.

**Used by:** `Toolbox.call` (to know which names are allowed and what to say on bad arguments) and `scripts/smoke_models.py` (sent to the model).

### `src/agent_eval/tools/toolbox.py`
**Purpose:** The code behind the six tools. **No tool ever raises**: every failure comes back as `{"error": "..."}`, so the model can read it and react. The database connection is read-only.

**Main contents** (class `Toolbox(db_path=DB_PATH, glossary_path=GLOSSARY_PATH)`):
- `call(name, arguments=None) -> dict`: the entry point for a model's tool request. It accepts a dict, a JSON string or nothing; only runs names listed in `TOOL_SPECS`; returns an error that lists the real tool names for unknown tools, an error naming the expected parameters for wrong arguments, and an error for invalid JSON. Example: `Toolbox().call("run_sql", '{"query": "SELECT COUNT(*) FROM customers"}')`.
- `list_tables() -> dict`: `{"tables": [...]}`.
- `describe_table(table) -> dict`: columns with `name`, `type`, `primary_key`, `nullable`, plus `foreign_keys` when the table has any. The table name is checked against the real table list first; an unknown name returns an error listing the available tables.
- `sample_rows(table, n=3) -> dict`: up to 10 rows as dicts, in table order. `n` is converted to a whole number and limited to 1 to 10.
- `lookup_metric_definition(term) -> dict`: matches a glossary term or alias (ignoring case) and returns `term`, `definition` and the full list of `conventions`; a miss lists the known terms.
- `run_sql(query) -> dict`: runs **one** read-only SELECT (SQLite dialect). An allowlist (`ALLOWED_ACTIONS`) denies everything except SELECT, reads, functions and recursion; a progress handler aborts after 5 seconds. Returns `columns`, `rows` (lists), `row_count` and `truncated`, with at most 50 rows and a `note` when truncated. Multiple statements, writes, `PRAGMA` and `ATTACH` return errors.
- `calculate(expression) -> dict`: safe arithmetic with no `eval`. The text is parsed to a syntax tree and walked by `_eval_node`, which allows only int/float numbers (not booleans), `+ - * / // % **`, unary plus and minus, and positional calls to `round`, `abs`, `min`, `max`. Limits: 200 characters, exponent at most 100; division by zero returns an error; results are rounded to 10 decimals.
- Private helpers: `_connect` (opens the database with `mode=ro`), `_table_names`, `_check_table`.

**Constants:** `MAX_ROWS = 50`, `QUERY_TIMEOUT_S = 5`, `MAX_EXPR_LEN = 200`, `MAX_EXPONENT = 100`.

## `tests/`

The automated tests, run with `PYTHONPATH=src .venv/bin/python -m pytest`. There are **255 tests**; 254 run by default and one (`test_live_ling_round_trip`) is marked `live` and skipped because it calls the real API. All other tests use fakes or temporary files, never the network. Counts below are tests as pytest collects them (parameterised cases count separately).

### `tests/test_config.py`
**Purpose:** Covers `config.py`. **17 tests.** The data paths exist; `require_env` raises and names a missing variable; `get_as_of_date` reads the real date, a different date, and fails on a missing key; `render_prompt` fills `{today}` and keeps the JSON braces, rejects bad names (path tricks, empty, non-strings), fails on a missing file or placeholder, replaces every `{today}`, and reads UTF-8.

### `tests/test_llm.py`
**Purpose:** Covers `llm.py`. **47 tests** (46 run by default). A `FakeClient` returns canned responses or raises errors in a set order, and a fake `sleep` records waits, so nothing waits or calls the network. Areas: the request body (tools, temperature, usage reporting, provider pin); plain answers and tool calls; usage and cost, including cached tokens (the Ling turn-2 case), clamping and a free cache price; `provider`; retries (429, 5xx, timeout, connection error, no choices, `Retry-After`, the 4-call limit, latency counting only the successful attempt); errors that must not retry; and `make_client`. One live test calls `ling` for real.

### `tests/test_schemas.py`
**Purpose:** Covers `schemas.py`. **68 tests.** `ModelSpec` validation (bad tiers, prices, parameter counts, unknown fields, cached price, temperature flag); `load_models` with good and bad files; the real `models.yaml` (keys, prices, cached prices, providers, temperature flags, notes); `AgentConfig` ranges and boundaries; `load_agent` (name vs file name, unknown model lists the valid keys, missing prompt, unsafe prompt names, bad YAML, every error naming the file); and `load_all_agents` on the four real files and on empty or broken folders.

### `tests/test_smoke_models.py`
**Purpose:** Covers the helpers in `scripts/smoke_models.py`. **17 tests.** The script is loaded by file path (`scripts/` is not a package). Provider-name normalising, pin-mismatch rules (including that "Google" does not match `google-ai-studio`), the `PIN MISMATCH` flag in the printed table, and that every real model has a pinned provider.

### `tests/test_specs.py`
**Purpose:** Covers `tools/specs.py` and `Toolbox.call`. **26 tests.** Six specs with the right names and structure; each spec's parameter names and required list match the method's real signature; `sample_rows.n` limits; limits stated in descriptions; `call` with dict, JSON string and no arguments; unknown and private tool names; wrong arguments naming the expected parameters; invalid or non-object JSON; always returns a dict.

### `tests/test_toolbox.py`
**Purpose:** Covers the tools in `tools/toolbox.py`. **80 tests.** Table listing and description (primary key, nullable, foreign keys, unknown tables); the read-only connection; `sample_rows` clamping and coercion; glossary lookups by name, alias and case, with misses and a broken glossary; `run_sql` (basic, CTEs, the 50-row cap and its note, denied writes and escape routes such as `PRAGMA` and `ATTACH`, one statement only, timeout, bad input, SQL errors); and `calculate` (values, unsafe or invalid syntax, exponent and length caps, division by zero, bad input).

---

Last updated: 2026-10-12
