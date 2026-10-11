from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from agent_eval import llm
from agent_eval.llm import REQUEST_TIMEOUT_S, LLMError, LLMUnavailable, chat, make_client
from agent_eval.schemas import LLMResult, ModelSpec, load_models

SPEC = ModelSpec(id="vendor/m1", display_name="M1", tier="budget", price_in_per_m=2.0, price_out_per_m=10.0)
MESSAGES = [{"role": "user", "content": "hi"}]


class FakeClient:
    """Mimics client.chat.completions.create: records the call and returns a canned response."""

    def __init__(self, response=None, error=None, script=None):
        # script: outcomes consumed one per call, in order; each is a response or an exception to raise.
        self.response, self.error, self.script, self.calls = response, error, list(script or []), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.script:
            outcome = self.script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if self.error:
            raise self.error
        return self.response


def tool_call(id="call_1", name="run_sql", arguments='{"query": "SELECT 1"}'):
    return NS(id=id, type="function", function=NS(name=name, arguments=arguments))


def response(content="hello", tool_calls=None, finish_reason="stop", usage="default", choices="default",
             provider=None):
    if usage == "default":
        usage = NS(prompt_tokens=1000, completion_tokens=200, cost=0.0042,
                   prompt_tokens_details=NS(cached_tokens=400))
    if choices == "default":
        choices = [NS(message=NS(content=content, tool_calls=tool_calls), finish_reason=finish_reason)]
    resp = NS(id="gen-123", model="vendor/m1-served", choices=choices, usage=usage)
    if provider is not None:  # OpenRouter adds this extra field; real responses may lack it
        resp.provider = provider
    return resp


def run(resp=None, **kwargs):
    return chat(FakeClient(resp or response()), SPEC, MESSAGES, **kwargs)


def test_request_arguments():
    client = FakeClient(response())
    chat(client, SPEC, MESSAGES, temperature=0.7)
    call = client.calls[0]
    assert call["model"] == "vendor/m1" and call["messages"] == MESSAGES and call["temperature"] == 0.7
    assert call["extra_body"] == {"usage": {"include": True}}
    assert "tools" not in call


def test_tools_sent_only_when_given():
    tools = [{"type": "function", "function": {"name": "x"}}]
    client = FakeClient(response())
    chat(client, SPEC, MESSAGES, tools=tools)
    assert client.calls[0]["tools"] == tools
    chat(client, SPEC, MESSAGES, tools=[])
    assert "tools" not in client.calls[1]


def test_plain_answer():
    r = run(response(content="42"))
    assert isinstance(r, LLMResult)
    assert r.content == "42" and r.finish_reason == "stop" and r.tool_calls == []
    assert r.assistant_message == {"role": "assistant", "content": "42"}
    assert r.response_id == "gen-123" and r.served_model == "vendor/m1-served" and r.started_at


def test_tool_calls_keep_raw_arguments_and_build_openai_message():
    r = run(response(content=None, tool_calls=[tool_call(), tool_call("call_2", "calculate", "{bad json")],
                     finish_reason="tool_calls"))
    assert [(t.id, t.name, t.arguments) for t in r.tool_calls] == [
        ("call_1", "run_sql", '{"query": "SELECT 1"}'), ("call_2", "calculate", "{bad json")]
    assert r.content is None and r.finish_reason == "tool_calls"
    assert r.assistant_message == {
        "role": "assistant", "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": '{"query": "SELECT 1"}'}},
            {"id": "call_2", "type": "function", "function": {"name": "calculate", "arguments": "{bad json"}},
        ],
    }


def test_provider_is_read_when_present():
    assert run(response(provider="DeepInfra")).provider == "DeepInfra"


def test_provider_is_none_when_absent():
    assert run(response()).provider is None


def test_usage_and_cost():
    r = run()
    assert (r.prompt_tokens, r.completion_tokens, r.cached_tokens) == (1000, 200, 400)
    assert r.reported_cost_usd == 0.0042
    assert r.cost_usd == pytest.approx((1000 * 2.0 + 200 * 10.0) / 1_000_000)  # 0.004


def test_missing_usage_gives_none_everywhere():
    r = run(response(usage=None))
    assert (r.prompt_tokens, r.completion_tokens, r.cached_tokens) == (None, None, None)
    assert r.cost_usd is None and r.reported_cost_usd is None


def test_cost_needs_both_token_counts():
    r = run(response(usage=NS(prompt_tokens=1000)))
    assert r.prompt_tokens == 1000 and r.completion_tokens is None and r.cost_usd is None


def test_missing_cached_and_reported_cost_are_none():
    r = run(response(usage=NS(prompt_tokens=10, completion_tokens=5)))
    assert r.cached_tokens is None and r.reported_cost_usd is None and r.cost_usd is not None


def test_latency_uses_perf_counter(monkeypatch):
    ticks = iter([100.0, 102.5])
    monkeypatch.setattr(llm, "time", NS(perf_counter=lambda: next(ticks)))
    assert run().latency_s == 2.5


@pytest.mark.parametrize("choices", [[], None])
def test_no_choices_raises_llm_unavailable(choices):
    with pytest.raises(LLMUnavailable, match="no choices") as e:
        run(response(choices=choices), sleep=lambda s: None)
    assert isinstance(e.value, LLMError)  # still catchable as the base class


def test_api_errors_propagate():
    with pytest.raises(RuntimeError, match="rate limited"):
        chat(FakeClient(error=RuntimeError("rate limited")), SPEC, MESSAGES)


def test_make_client_requires_key_and_never_leaks_it(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_client()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-secret")
    client = make_client()
    assert str(client.base_url).startswith("https://openrouter.ai/api/v1")
    assert client.max_retries == 0 and client.timeout == REQUEST_TIMEOUT_S == 90


# ---------- cache-aware cost ----------

LING = ModelSpec(id="inclusionai/ling-3.0-flash-sante", display_name="Ling", tier="budget", price_in_per_m=0.042,
                 price_out_per_m=0.1232, price_cached_in_per_m=0.0084)


def cost_for(spec, prompt, completion, cached="absent"):
    details = {} if cached == "absent" else {"prompt_tokens_details": NS(cached_tokens=cached)}
    usage = NS(prompt_tokens=prompt, completion_tokens=completion, **details)
    return run(response(usage=usage)) if spec is SPEC else chat(FakeClient(response(usage=usage)), spec, MESSAGES)


def test_ling_turn_two_cost_matches_what_openrouter_billed():
    r = cost_for(LING, 1103, 51, cached=1024)
    assert r.cost_usd == pytest.approx(1.82028e-05)
    assert r.cached_tokens == 1024


def test_no_cached_price_charges_cached_tokens_at_full_input_price():
    r = cost_for(SPEC, 1000, 200, cached=400)  # SPEC has no cached price
    assert r.cost_usd == pytest.approx((1000 * 2.0 + 200 * 10.0) / 1_000_000)


def test_missing_cached_tokens_behave_like_zero():
    full = (1103 * 0.042 + 51 * 0.1232) / 1_000_000
    assert cost_for(LING, 1103, 51, cached=None).cost_usd == pytest.approx(full)
    assert cost_for(LING, 1103, 51).cost_usd == pytest.approx(full)  # no details object at all
    assert cost_for(LING, 1103, 51, cached=0).cost_usd == pytest.approx(full)


def test_cached_tokens_above_prompt_tokens_are_clamped():
    r = cost_for(LING, 1000, 10, cached=5000)
    assert r.cost_usd == pytest.approx((1000 * 0.0084 + 10 * 0.1232) / 1_000_000)  # never negative
    assert r.cached_tokens == 5000  # the raw reported value is kept


def test_a_free_cached_price_of_zero_is_honored():
    free_cache = ModelSpec(id="v/m", display_name="M", tier="mid", price_in_per_m=1.0, price_out_per_m=1.0,
                           price_cached_in_per_m=0.0)
    assert cost_for(free_cache, 1000, 0, cached=1000).cost_usd == 0.0


def test_cost_still_needs_both_token_counts_with_cache_info():
    usage = NS(prompt_tokens=1000, prompt_tokens_details=NS(cached_tokens=500))
    assert run(response(usage=usage)).cost_usd is None


# ---------- temperature support ----------

def no_temp_spec():
    return ModelSpec(id="vendor/m1", display_name="M1", tier="mid", price_in_per_m=1.0, price_out_per_m=1.0,
                     supports_temperature=False)


def test_temperature_is_sent_when_supported():
    client = FakeClient(response())
    chat(client, SPEC, MESSAGES, temperature=0.3)
    assert client.calls[0]["temperature"] == 0.3


def test_temperature_is_left_out_when_not_supported():
    client = FakeClient(response())
    chat(client, no_temp_spec(), MESSAGES, temperature=0.3)
    call = client.calls[0]
    assert "temperature" not in call
    assert call["model"] == "vendor/m1" and call["messages"] == MESSAGES and call["extra_body"]["usage"]


def test_temperature_stays_out_on_retries_too():
    client = FakeClient(script=[rate_limit(), response()])
    chat(client, no_temp_spec(), MESSAGES, sleep=lambda s: None)
    assert all("temperature" not in c for c in client.calls) and len(client.calls) == 2


# ---------- provider pinning ----------

def pinned(tag="novita"):
    return ModelSpec(id="vendor/m1", display_name="M1", tier="budget", price_in_per_m=2.0, price_out_per_m=10.0,
                     provider=tag)


def test_no_provider_means_no_provider_key_but_usage_still_requested():
    client = FakeClient(response())
    chat(client, SPEC, MESSAGES)
    assert client.calls[0]["extra_body"] == {"usage": {"include": True}}
    assert "provider" not in client.calls[0]["extra_body"]


def test_provider_is_pinned_in_extra_body():
    client = FakeClient(response())
    chat(client, pinned("google-ai-studio"), MESSAGES, tools=[{"type": "function", "function": {"name": "x"}}])
    body = client.calls[0]["extra_body"]
    assert body["provider"] == {"order": ["google-ai-studio"], "allow_fallbacks": False, "require_parameters": True}
    assert body["usage"] == {"include": True}
    assert set(body) == {"usage", "provider"}


def test_pin_is_sent_again_on_every_retry():
    client = FakeClient(script=[rate_limit(), response()])
    chat(client, pinned(), MESSAGES, sleep=lambda s: None)
    assert [c["extra_body"]["provider"]["order"] for c in client.calls] == [["novita"], ["novita"]]
    assert client.calls[0]["extra_body"] == client.calls[1]["extra_body"]


# ---------- retries ----------

def req():
    return httpx.Request("POST", "https://x")


def status_error(cls, code, headers=None):
    return cls("msg", response=httpx.Response(code, request=req(), headers=headers), body=None)


def rate_limit(headers=None):
    return status_error(openai.RateLimitError, 429, headers)


class Sleeper:
    """Fake sleep: records what it was asked to wait and returns immediately."""

    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def run_script(script, **kwargs):
    client, sleeper = FakeClient(script=script), Sleeper()
    try:
        return chat(client, SPEC, MESSAGES, sleep=sleeper, **kwargs), client, sleeper
    except Exception as e:
        return e, client, sleeper


def test_first_try_success_has_no_retries():
    r, client, sleeper = run_script([response()])
    assert (r.attempts, r.retry_wait_s) == (1, 0.0) and sleeper.calls == [] and len(client.calls) == 1


def test_429_then_success():
    r, client, sleeper = run_script([rate_limit(), response()])
    assert (r.attempts, r.retry_wait_s) == (2, 1.0) and sleeper.calls == [1.0] and len(client.calls) == 2


def test_two_500s_then_success_doubles_the_wait():
    r, _, sleeper = run_script([status_error(openai.InternalServerError, 500)] * 2 + [response()])
    assert r.attempts == 3 and sleeper.calls == [1.0, 2.0] and r.retry_wait_s == 3.0


def test_always_503_gives_up_after_four_calls():
    err = status_error(openai.InternalServerError, 503)
    e, client, sleeper = run_script([err] * 10)
    assert isinstance(e, LLMUnavailable) and isinstance(e, LLMError)
    assert len(client.calls) == 4 and sleeper.calls == [1.0, 2.0, 4.0]
    assert "4 attempts" in str(e) and "InternalServerError" in str(e) and e.__cause__ is err


@pytest.mark.parametrize("cls,code", [
    (openai.BadRequestError, 400), (openai.AuthenticationError, 401),
    (openai.PermissionDeniedError, 403), (openai.NotFoundError, 404),
])
def test_client_errors_propagate_immediately(cls, code):
    err = status_error(cls, code)
    e, client, sleeper = run_script([err, response()])
    assert e is err and type(e) is cls and len(client.calls) == 1 and sleeper.calls == []


def test_runtime_error_propagates_immediately():
    e, client, sleeper = run_script([RuntimeError("boom"), response()])
    assert type(e) is RuntimeError and len(client.calls) == 1 and sleeper.calls == []


@pytest.mark.parametrize("error", [openai.APITimeoutError(request=req()),
                                   openai.APIConnectionError(message="down", request=req())])
def test_timeouts_and_connection_errors_are_retried(error):
    r, client, sleeper = run_script([error, response()])
    assert r.attempts == 2 and sleeper.calls == [1.0]


@pytest.mark.parametrize("header,expected", [("5", 5.0), ("9999", 60.0), ("0", 0.0),
                                             ("Wed, 21 Oct 2026 07:28:00 GMT", 1.0), ("-3", 0.0)])
def test_retry_after_header(header, expected):
    _, _, sleeper = run_script([rate_limit({"retry-after": header}), response()])
    assert sleeper.calls == [expected]


def test_empty_choices_then_good_response_is_retried():
    r, client, sleeper = run_script([response(choices=[]), response(content="ok")])
    assert r.content == "ok" and r.attempts == 2 and sleeper.calls == [1.0]


def test_latency_measures_only_the_successful_attempt(monkeypatch):
    # perf_counter is read at: failed attempt start, successful attempt start, successful attempt end.
    ticks = iter([0.0, 50.0, 52.0])
    monkeypatch.setattr(llm, "time", NS(perf_counter=lambda: next(ticks)))
    r, _, sleeper = run_script([rate_limit({"retry-after": "30"}), response()])
    assert r.latency_s == 2.0 and r.retry_wait_s == 30.0  # the 30s wait is not part of latency


def test_max_retries_is_configurable():
    e, client, _ = run_script([rate_limit()] * 5, max_retries=1)
    assert isinstance(e, LLMUnavailable) and len(client.calls) == 2


@pytest.mark.live
def test_live_ling_round_trip():
    r = chat(make_client(), load_models()["ling"], [{"role": "user", "content": "Reply with the single word: pong"}])
    assert r.content and r.latency_s > 0 and r.response_id
