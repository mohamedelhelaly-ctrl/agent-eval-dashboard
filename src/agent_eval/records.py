import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

# Stored in every record so old files stay recognizable if the shape ever changes.
SCHEMA_VERSION = 1
RunStatus = Literal["answered", "max_steps", "cost_cap", "infra_error"]

# Range checks catch a bug that writes a negative value; no NaN or infinity, since JSON has neither.
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Usd = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0)]


# One tool request from the model, and what happened when we ran it.
class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments_raw: str  # exactly what the model sent
    arguments: dict | None  # parsed; None if it was not valid JSON
    result: dict | str | None  # None when executed is False
    is_error: bool  # the result is {"error": ...}, or the arguments could not be parsed
    duration_s: Seconds = 0.0
    executed: bool = True  # False when the call was skipped (for example, the run stopped first)


# One model call (one turn) and the tool calls it asked for.
class StepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: Annotated[int, Field(ge=1)]  # starts at 1
    content: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    finish_reason: str | None = None
    # 0 here can mean "not reported"; cost_usd stays None when it could not be computed.
    prompt_tokens: Count = 0
    completion_tokens: Count = 0
    cached_tokens: Count = 0
    cost_usd: Usd | None = None  # our computed cost
    reported_cost_usd: Usd | None = None  # the provider-reported cost
    latency_s: Seconds  # the successful attempt only
    attempts: Annotated[int, Field(ge=1)] = 1
    retry_wait_s: Seconds = 0.0
    response_id: str | None = None
    served_model: str | None = None
    provider: str | None = None

    # The cost we trust: what the provider says it charged, else our own estimate, else 0.
    @computed_field
    @property
    def effective_cost_usd(self) -> float:
        if self.reported_cost_usd is not None:
            return self.reported_cost_usd
        return self.cost_usd or 0.0

    # Where effective_cost_usd came from. A reported cost of exactly 0.0 still counts as reported.
    @computed_field
    @property
    def cost_source(self) -> Literal["reported", "computed", "none"]:
        if self.reported_cost_usd is not None:
            return "reported"
        return "computed" if self.cost_usd is not None else "none"


# One agent answering one question once: the unit that gets stored and scored.
class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: str
    agent_name: str
    model_id: str
    prompt_name: str
    question_id: str
    repeat: Annotated[int, Field(ge=0)]  # which repeat of this question this is
    status: RunStatus
    started_at: str  # ISO 8601, UTC
    wall_time_s: Seconds
    final_text: str | None  # None when the run ended without a final message
    parse_ok: bool
    answer: int | float | str | None
    answerable: bool | None
    clean_format: bool
    parse_problem: str | None
    steps: list[StepRecord] = Field(default_factory=list)
    error_message: str | None = None
    # agent config, prompt sha256, temperature_requested, temperature_sent, as_of_date, db_sha256
    config_snapshot: dict

    # A bool would otherwise be quietly accepted as the number 1, and NaN is not valid JSON.
    @field_validator("answer", mode="before")
    @classmethod
    def _answer_must_be_a_real_value(cls, v):
        if isinstance(v, bool):
            raise ValueError("answer must not be a bool")
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("answer must be a finite number")
        return v

    # ---- totals and counts: calculated from the steps, never stored on their own ----

    @computed_field
    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @computed_field
    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @computed_field
    @property
    def total_cached_tokens(self) -> int:
        return sum(s.cached_tokens for s in self.steps)

    @computed_field
    @property
    def total_cost_usd(self) -> float:
        return sum(s.effective_cost_usd for s in self.steps)

    @computed_field
    @property
    def n_steps(self) -> int:
        return len(self.steps)

    # Only calls that were actually executed count; a skipped call never ran.
    @computed_field
    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps for tc in s.tool_calls if tc.executed)

    @computed_field
    @property
    def n_tool_errors(self) -> int:
        return sum(1 for s in self.steps for tc in s.tool_calls if tc.executed and tc.is_error)

    @computed_field
    @property
    def total_model_latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    @computed_field
    @property
    def total_retry_wait_s(self) -> float:
        return sum(s.retry_wait_s for s in self.steps)

    @computed_field
    @property
    def total_attempts(self) -> int:
        return sum(s.attempts for s in self.steps)

    # One line of pure ASCII with no newline inside, so splitting a file on lines can never cut a record
    # (raw characters such as U+2028 would be treated as line breaks by some readers).
    def to_json_line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":"), allow_nan=False)

    # Read one line back. The computed keys are dropped first (extra="forbid" would reject them);
    # they are recalculated from the steps. Raises ValueError on a bad line.
    @classmethod
    def from_json_line(cls, line: str) -> "RunRecord":
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError("a record line must be a JSON object")
        for key in cls.model_computed_fields:
            data.pop(key, None)
        steps = data.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    for key in StepRecord.model_computed_fields:
                        step.pop(key, None)
        return cls.model_validate(data)
