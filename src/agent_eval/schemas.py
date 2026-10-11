from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent_eval.config import AGENTS_DIR, IDENTIFIER_RE, MODELS_PATH, PROMPTS_DIR


class ModelSpec(BaseModel):
    # extra="forbid": a misspelled or unknown field is an error instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    display_name: str
    tier: Literal["free", "budget", "mid", "premium"]
    price_in_per_m: float = Field(ge=0)
    price_out_per_m: float = Field(ge=0)
    params_total_b: float | None = Field(default=None, gt=0)
    params_active_b: float | None = Field(default=None, gt=0)
    provider: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _active_not_above_total(self):
        # A model can't use more parameters per token than it has in total.
        if self.params_total_b is not None and self.params_active_b is not None:
            if self.params_active_b > self.params_total_b:
                raise ValueError("params_active_b must not exceed params_total_b")
        return self


# Load model definitions from YAML and validate each entry against the ModelSpec schema.
# Used when the application needs validated model configuration instead of raw YAML data.
def load_models(path=MODELS_PATH) -> dict[str, ModelSpec]:
    """Read and validate the models file (top-level `models:` mapping of key -> fields).

    Unlike tools, a loader raises: bad config is a bug to fix.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())  # safe_load never builds arbitrary Python objects
    except OSError as e:
        raise ValueError(f"Cannot read models file {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e
    # The entries live under a top-level `models:` key, leaving room for other sections later.
    entries = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path} must have a top-level 'models' mapping of model key -> fields")

    models = {}
    for key, fields in entries.items():
        try:
            models[key] = ModelSpec.model_validate(fields)
        except ValidationError as e:
            raise ValueError(f"Invalid model {key!r} in {path}: {e}") from e
    return models

# An agent is a model plus a prompt plus limits
class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    model: str  # checked against models.yaml by load_agent
    prompt: str  # checked against the prompts folder by load_agent
    # allow_inf_nan=False: NaN would slip past a range check, and inf would disable the cost cap.
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    max_steps: int = Field(default=15, ge=1, le=50)
    max_run_cost_usd: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    reasoning_effort: Literal["low", "medium", "high"] | None = None


def load_agent(path, models: dict[str, ModelSpec] | None = None, prompts_dir=PROMPTS_DIR) -> AgentConfig:
    """Read one agent YAML and check it against the models file and the prompts folder."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"Cannot read agent file {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping of agent settings")
    try:
        agent = AgentConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid agent config {path}: {e}") from e

    # The name must match the file, so files, results and tables all agree on identity.
    if agent.name != path.stem:
        raise ValueError(f"{path}: name {agent.name!r} must equal the file name {path.stem!r}")
    if models is None:
        models = load_models()
    if agent.model not in models:
        raise ValueError(f"{path}: model {agent.model!r} not found. Valid models: {', '.join(models)}")
    # Plain identifiers only, so the prompt name can't point outside prompts_dir.
    if not IDENTIFIER_RE.fullmatch(agent.prompt):
        raise ValueError(f"{path}: invalid prompt name {agent.prompt!r}: use only letters, digits, _ and -")
    if not (Path(prompts_dir) / f"{agent.prompt}.md").is_file():
        raise ValueError(f"{path}: prompt file {agent.prompt}.md not found in {prompts_dir}")
    return agent


def load_all_agents(agents_dir=AGENTS_DIR) -> dict[str, AgentConfig]:
    """Load every *.yaml in agents_dir, keyed by agent name. The models file is read once."""
    agents_dir = Path(agents_dir)
    files = sorted(agents_dir.glob("*.yaml"))
    if not files:
        raise ValueError(f"No agent files (*.yaml) found in {agents_dir}")
    models = load_models()
    return {a.name: a for a in (load_agent(f, models=models) for f in files)}


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: str  # the raw JSON string the model sent; Toolbox.call parses it


class LLMResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str | None = None  # None when the model only requested tools
    tool_calls: list[ToolCall] = Field(default_factory=list)
    assistant_message: dict  # OpenAI-format message, ready to append for the next turn
    finish_reason: str | None = None
    # Token counts and costs are None when the provider didn't report them.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None  # our estimate, from the ModelSpec prices
    reported_cost_usd: float | None = None  # what OpenRouter says it charged
    latency_s: float  # the successful attempt only, never the failed tries or the waits
    attempts: int = 1  # how many calls it took (1 = no retries)
    retry_wait_s: float = 0.0  # total seconds slept between attempts
    response_id: str | None = None
    served_model: str | None = None
    provider: str | None = None  # which upstream provider OpenRouter routed to
    started_at: str  # UTC, ISO 8601
