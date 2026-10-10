from types import SimpleNamespace as NS

import pytest

from agent_eval import llm
from agent_eval.llm import LLMError, chat, make_client
from agent_eval.schemas import LLMResult, ModelSpec, load_models

SPEC = ModelSpec(id="vendor/m1", display_name="M1", tier="budget", price_in_per_m=2.0, price_out_per_m=10.0)
MESSAGES = [{"role": "user", "content": "hi"}]


class FakeClient:
    """Mimics client.chat.completions.create: records the call and returns a canned response."""

    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def tool_call(id="call_1", name="run_sql", arguments='{"query": "SELECT 1"}'):
    return NS(id=id, type="function", function=NS(name=name, arguments=arguments))


def response(content="hello", tool_calls=None, finish_reason="stop", usage="default", choices="default"):
    if usage == "default":
        usage = NS(prompt_tokens=1000, completion_tokens=200, cost=0.0042,
                   prompt_tokens_details=NS(cached_tokens=400))
    if choices == "default":
        choices = [NS(message=NS(content=content, tool_calls=tool_calls), finish_reason=finish_reason)]
    return NS(id="gen-123", model="vendor/m1-served", choices=choices, usage=usage)


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
def test_no_choices_raises_llm_error(choices):
    with pytest.raises(LLMError, match="no choices"):
        run(response(choices=choices))


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


@pytest.mark.live
def test_live_ling_round_trip():
    r = chat(make_client(), load_models()["ling"], [{"role": "user", "content": "Reply with the single word: pong"}])
    assert r.content and r.latency_s > 0 and r.response_id
