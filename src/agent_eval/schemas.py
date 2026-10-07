from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent_eval.config import MODELS_PATH


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
