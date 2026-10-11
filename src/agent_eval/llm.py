import time
from datetime import datetime, timezone

import openai
from openai import OpenAI

from agent_eval.config import require_env
from agent_eval.schemas import LLMResult, ModelSpec, ToolCall

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_S = 90  # longest one request may take before it counts as timed out
BACKOFF_BASE_S = 1  # waits between attempts: 1, 2, 4 seconds
MAX_RETRY_AFTER_S = 60  # cap on a server-requested wait

# Temporary failures worth retrying. Everything else (bad request, bad key...) is a bug, not bad luck.
RETRYABLE_ERRORS = (openai.RateLimitError, openai.InternalServerError,
                    openai.APITimeoutError, openai.APIConnectionError)


class LLMError(Exception):
    """The API answered, but not with anything we can use."""


class LLMUnavailable(LLMError):
    """Every attempt failed with a temporary error; we gave up."""


# Create an OpenAI-compatible client configured to send requests to OpenRouter.
def make_client() -> OpenAI:
    # OpenRouter speaks the OpenAI protocol, so the OpenAI SDK works with a different base_url.
    key = require_env("OPENROUTER_API_KEY")["OPENROUTER_API_KEY"]
    # max_retries=0: the SDK's own hidden retries are off, so chat() is the only place that retries.
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=0, timeout=REQUEST_TIMEOUT_S)


# How long to wait before the next attempt: the server's Retry-After if it sent a usable one, else 1, 2, 4...
def _retry_wait(error: Exception, attempt: int) -> float:
    headers = getattr(getattr(error, "response", None), "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    try:
        # Retry-After can also be an HTTP date; anything that isn't plain seconds falls back to backoff.
        return min(max(float(value), 0.0), MAX_RETRY_AFTER_S)
    except (TypeError, ValueError):
        return BACKOFF_BASE_S * 2 ** (attempt - 1)


# Safely read a nested response attribute, returning None when a part is missing.
def _dig(obj, *path):
    """Follow attribute names through obj, returning None if any step is missing."""
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


# Send one chat request and convert the response, usage, and timing into an LLMResult.
def chat(client, spec: ModelSpec, messages: list[dict], tools: list[dict] | None = None,
         temperature: float = 0.0, sleep=time.sleep, max_retries: int = 3) -> LLMResult:
    """One model call. Temporary failures are retried; any other error propagates unchanged.

    sleep is a parameter so tests can pass a fake one that returns instantly.
    """
   
    # Build the payload for the OpenAI-compatible API call.
    kwargs = {
        # Ask OpenRouter to use this model ID for the request.
        "model": spec.id,
        # Send the conversation history the model should respond to.
        "messages": messages,
        # Control response randomness from 0.0 (deterministic) to higher values.
        "temperature": temperature,
        # Ask the provider to include usage statistics so we can compute real cost.
        "extra_body": {"usage": {"include": True}},  # ask OpenRouter to report the real cost
    }
    # Attach tool definitions only when they exist; some providers reject empty/null tool lists.
    if tools:
        kwargs["tools"] = tools  # some providers reject an empty or null tools field

    

    attempts, retry_wait_s = 0, 0.0
    while True:
        attempts += 1
        # Record when this attempt began; only the successful attempt's times end up in the result.
        started_at = datetime.now(timezone.utc).isoformat()
        # Capture the start time for latency measurement.
        t0 = time.perf_counter()
        try:
            # Make the actual API call and wait for the model response.
            response = client.chat.completions.create(**kwargs)
            # Measure how long the network + model call took.
            latency_s = time.perf_counter() - t0
            # A valid response must include at least one choice; an empty one is retried like an outage.
            if not getattr(response, "choices", None):
                raise LLMError(f"Response from {spec.id} contained no choices")
            break
        except (*RETRYABLE_ERRORS, LLMError) as e:
            if attempts > max_retries:
                raise LLMUnavailable(f"{spec.id} unavailable after {attempts} attempts; "
                                     f"last error: {type(e).__name__}: {e}") from e
            wait = _retry_wait(e, attempts)
            sleep(wait)
            retry_wait_s += wait


    # Pull the first response choice from the returned list.
    choice = response.choices[0]
    # Extract the message object from that choice.
    message = choice.message

    # Convert tool calls from the API format into the project's ToolCall schema.
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
        for tc in (getattr(message, "tool_calls", None) or [])
    ]

    # The assistant's plain text content may be None when it only emits tool calls.
    content = getattr(message, "content", None)
    
    # Package the assistant message into a JSON-like dict for later inspection.
    assistant_message = {"role": "assistant", "content": content}
    if tool_calls:
        # Include tool-call metadata in the message record when the model requested tools.
        assistant_message["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in tool_calls
        ]

    # Usage data may be absent, so read it defensively and keep missing values as None.
    usage = getattr(response, "usage", None)
    # Number of input tokens consumed by the prompt.
    prompt_tokens = _dig(usage, "prompt_tokens")
    # Number of tokens generated by the model completion.
    completion_tokens = _dig(usage, "completion_tokens")
    cost_usd = None
    # Only calculate cost when both token counts are present.
    if prompt_tokens is not None and completion_tokens is not None:
        cost_usd = (prompt_tokens * spec.price_in_per_m + completion_tokens * spec.price_out_per_m) / 1_000_000

    # Build the final structured result that callers can inspect or save.
    return LLMResult(
        # The raw assistant text content, if any.
        content=content,
        # Structured function calls requested by the model.
        tool_calls=tool_calls,
        # The assistant message in a format consistent with chat history.
        assistant_message=assistant_message,
        # Why the model stopped: stop, tool_calls, length, etc.
        finish_reason=getattr(choice, "finish_reason", None),
        # Usage counts from the API.
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        # Tokens served from cache if the provider includes them.
        cached_tokens=_dig(usage, "prompt_tokens_details", "cached_tokens"),
        # Computed price based on the model's token pricing.
        cost_usd=cost_usd,
        # Provider-reported cost, if available.
        reported_cost_usd=_dig(usage, "cost"),
        # End-to-end time from sending to receiving the answer.
        latency_s=latency_s,
        # How many calls it took and how long we slept between them.
        attempts=attempts,
        retry_wait_s=retry_wait_s,
        # Unique response identifier returned by the API.
        response_id=getattr(response, "id", None),
        # Model name the provider actually served, which may differ from the requested ID.
        served_model=getattr(response, "model", None),
        provider=getattr(response, "provider", None),
        # When the request started, in ISO 8601 UTC format.
        started_at=started_at,
    )
