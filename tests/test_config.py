# Import pytest for its test helpers, including exception assertions.
import pytest

# Import the configured data paths and the helper that reads required environment variables.
from agent_eval.config import DB_PATH, GLOSSARY_PATH, require_env


# Verify that both configured data-file paths point to existing files.
def test_data_paths_exist():
    # Confirm that the configured database path refers to a file.
    assert DB_PATH.is_file()
    # Confirm that the configured glossary path refers to a file.
    assert GLOSSARY_PATH.is_file()


# Verify the error raised when a required environment variable is missing.
def test_require_env_names_missing_variable(monkeypatch):
    # Remove the variable for this test, without failing if it was already absent.
    monkeypatch.delenv("AGENT_EVAL_NOT_SET", raising=False)
    # Expect require_env to raise RuntimeError whose message includes the variable name.
    with pytest.raises(RuntimeError, match="AGENT_EVAL_NOT_SET"):
        # Call the helper with the deliberately missing variable.
        require_env("AGENT_EVAL_NOT_SET")
