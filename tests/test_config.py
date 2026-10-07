# Import pytest for its test helpers, including exception assertions.
import pytest

# Import the configured data paths and the helper that reads required environment variables.
from agent_eval import config
from agent_eval.config import DB_PATH, GLOSSARY_PATH, get_as_of_date, render_prompt, require_env


# Verify that both configured data-file paths point to existing files.
def test_data_paths_exist():
    # Confirm that the configured database path refers to a file.
    assert DB_PATH.is_file()
    # Confirm that the configured glossary path refers to a file.
    assert GLOSSARY_PATH.is_file()

# this checks that the helper fails loudly and tells you which variable is missing, which is the behavior you wanted.
def test_require_env_names_missing_variable(monkeypatch):
    # Remove the variable for this test, without failing if it was already absent.
    monkeypatch.delenv("AGENT_EVAL_NOT_SET", raising=False)
    # Expect require_env to raise RuntimeError whose message includes the variable name.
    with pytest.raises(RuntimeError, match="AGENT_EVAL_NOT_SET"):
        # Call the helper with the deliberately missing variable.
        require_env("AGENT_EVAL_NOT_SET")


# The database's "today" comes from db_meta.json, not the real clock.
def test_get_as_of_date_reads_real_meta():
    assert get_as_of_date() == "2026-01-01"


# A metadata file without as_of_date must fail loudly, and a different date must be picked up.
def test_get_as_of_date_missing_key_and_other_value(tmp_path, monkeypatch):
    meta = tmp_path / "db_meta.json"
    monkeypatch.setattr(config, "DB_META_PATH", meta)
    meta.write_text('{"seed": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="as_of_date"):
        get_as_of_date()
    meta.write_text('{"as_of_date": "2030-05-06"}', encoding="utf-8")
    assert get_as_of_date() == "2030-05-06"


# The real minimal prompt gets its date, and its literal JSON braces are left alone.
def test_render_minimal_prompt_fills_today_and_keeps_json_braces():
    out = render_prompt("minimal")
    assert "{today}" not in out and "Today's date is 2026-01-01." in out
    assert '{"answer": <number or string or null>, "answerable": <true or false>}' in out
    assert out.startswith("You are a data analyst for a SaaS company.")


# Only plain identifiers are allowed, so a name can't point outside the prompts folder.
@pytest.mark.parametrize("bad", ["../minimal", "a/b", "a.b", "", "x y", "minimal\n", None, 5])
def test_render_prompt_rejects_bad_names(bad, tmp_path):
    with pytest.raises(ValueError, match="Invalid prompt name"):
        render_prompt(bad, prompts_dir=tmp_path)


# Names with letters, digits, _ and - are accepted.
def test_render_prompt_accepts_identifier_names(tmp_path):
    (tmp_path / "my-prompt_2.md").write_text("Date: {today}", encoding="utf-8")
    assert render_prompt("my-prompt_2", prompts_dir=tmp_path) == "Date: 2026-01-01"


# A missing file and a file without the placeholder both raise ValueError.
def test_render_prompt_missing_file_and_missing_placeholder(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        render_prompt("nope", prompts_dir=tmp_path)
    (tmp_path / "bare.md").write_text("No date here.", encoding="utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        render_prompt("bare", prompts_dir=tmp_path)


# Only {today} is substituted (every occurrence); other braces are untouched.
def test_render_prompt_replaces_only_today(tmp_path):
    (tmp_path / "p.md").write_text('{today} then {today} {"k": {x}} {}', encoding="utf-8")
    assert render_prompt("p", prompts_dir=tmp_path) == '2026-01-01 then 2026-01-01 {"k": {x}} {}'


# Non-ASCII text is read as UTF-8.
def test_render_prompt_reads_utf8(tmp_path):
    (tmp_path / "u.md").write_text("Café — {today} ✓", encoding="utf-8")
    assert render_prompt("u", prompts_dir=tmp_path) == "Café — 2026-01-01 ✓"
