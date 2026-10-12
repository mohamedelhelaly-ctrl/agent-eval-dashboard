---
name: repo-guide
description: Create or update docs/REPO_GUIDE.md, a plain-English walkthrough of every directory and file in the repo, with each file's purpose, main functions/classes and how to use them.
---

# Repo Guide

Generate or refresh `docs/REPO_GUIDE.md`: a readable tour of the whole repository, starting at the root. The reader is someone who built this project step by step and wants to understand it again without opening every file. Write in plain English, short sentences, no jargon without a one-line explanation.

## Rules

- Read the real code of every file you describe. Never guess from the file name.
- Never read, print or quote `.env`. Mention that it exists and holds secrets. For `.env.example`, list variable names only.
- Skip entirely (do not list): `.git/`, `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `*.egg-info/`, `node_modules/`, `.DS_Store`.
- `results/` is git-ignored output: describe the folder in one paragraph and its file naming, but do not walk its contents.
- Binary or data files (for example `*.db`): say what they are, how they are produced, their size and main tables or fields (use read-only inspection such as listing tables). Never dump contents.
- Do not edit any file other than `docs/REPO_GUIDE.md`.

## Steps

1. **Map the tree.** From the repo root, list all files and directories (apply the skip list). Use `git ls-files` if available.
2. **Check for an existing guide.** If `docs/REPO_GUIDE.md` exists, read it and work in update mode (step 7). Otherwise create it.
3. **Write the header.** Title, one paragraph on what the project is (from `README.md` or `docs/project_plan.md` if present), then a table of contents.
4. **Write "How the pieces fit together".** A short end-to-end description of the data/control flow (for example: config loads agents, agent loop calls the LLM and tools, runner writes results, scorers compute metrics). Use a small text diagram if it helps. Write this last, after reading all files, but place it near the top.
5. **Walk the tree from the root.** Root files first (`pyproject.toml`, `README.md`, `.gitignore`, `.env.example`, and so on), then each directory in alphabetical order, recursing. For each directory write a heading, a short paragraph on what it contains and why it exists, and then its files.
6. **For each file write:**
   - **Purpose**: 1-3 plain-English sentences: what job it does and where it sits in the system.
   - **Main contents** (code files): each important class or function as a bullet with its signature, one line of what it does (inputs, outputs, notable behavior such as "never raises, returns an error dict"), and a one-line usage example or the place that calls it. Skip trivial private helpers unless they carry key logic; group them in one line instead.
   - **Key constants / fields** where they matter (config values, limits).
   - Config files (YAML/JSON/TOML): the role of the file and what each key means.
   - Prompts and markdown docs: what they are used for and any placeholders.
   - Tests: what area they cover, how many tests, and any non-obvious fixtures or markers (for example a `live` marker that is skipped by default).
   - Scripts: how to run them (exact command) and what they output.
7. **Update mode.** Keep the existing structure and wording of sections whose files have not changed. Add entries for new files, remove entries for deleted or renamed files, and rewrite entries whose code changed (compare signatures and behavior against the guide). Do not rewrite unchanged sections just for style.
8. **Finish.** Add a footer line `Last updated: <date>`. Then reply with a short summary: files added, removed, changed, and anything you could not describe confidently (and why).

## Entry format

```
### `path/to/file.py`
**Purpose:** ...

**Main contents:**
- `function_name(arg1, arg2=None) -> ReturnType`: what it does. Used by `other_file.py` / example: `function_name("x")`.

**Notes:** limits, gotchas, things that never raise, etc.
```

## Quality check before finishing

- Every non-skipped file in the tree appears exactly once.
- Every function named in the guide exists in the code with that signature.
- No secrets or `.env` values appear.
- The table of contents links match the headings.