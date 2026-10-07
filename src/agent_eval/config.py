import os
from pathlib import Path

from dotenv import load_dotenv

# This file is src/agent_eval/config.py; parents[2] climbs file -> agent_eval -> src -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Every other path in the project is built from PROJECT_ROOT, so nothing depends on where you run from.
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "saas.db"
GLOSSARY_PATH = DATA_DIR / "glossary.json"
DB_META_PATH = DATA_DIR / "db_meta.json"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_PATH = CONFIGS_DIR / "models.yaml"
AGENTS_DIR = CONFIGS_DIR / "agents"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

# Read PROJECT_ROOT/.env into os.environ (silently does nothing if the file is missing).
load_dotenv(PROJECT_ROOT / ".env")


def require_env(*names: str) -> dict[str, str]:
    """Return the named environment variables, or fail naming every missing/empty one."""
    values = {n: os.environ.get(n, "") for n in names}
    missing = [n for n, v in values.items() if not v.strip()]
    if missing:
        # Only the variable NAMES go into the message, never their values.
        raise RuntimeError(f"Missing or empty environment variables: {', '.join(missing)}")
    return values
