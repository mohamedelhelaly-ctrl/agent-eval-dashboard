import importlib.util

import pytest

from agent_eval.config import PROJECT_ROOT
from agent_eval.schemas import load_models

# scripts/ is not a package, so load the script by its file path. Importing it runs nothing: main() is guarded.
_spec = importlib.util.spec_from_file_location("smoke_models", PROJECT_ROOT / "scripts" / "smoke_models.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


@pytest.mark.parametrize("name", ["Google AI Studio", "google-ai-studio", "GOOGLE_AI_STUDIO", " google ai  studio "])
def test_normalize_makes_spellings_equal(name):
    assert smoke.normalize_provider(name) == "googleaistudio"


def test_normalize_handles_none_and_digits():
    assert smoke.normalize_provider(None) is None
    assert smoke.normalize_provider("Together.AI v2") == "togetheraiv2"


@pytest.mark.parametrize("pinned,served", [
    ("google-ai-studio", "Google AI Studio"), ("openai", "OpenAI"), ("novita", "Novita"), ("upstage", "Upstage"),
])
def test_matching_names_are_not_a_mismatch(pinned, served):
    assert smoke.pin_mismatch(pinned, served) is False


@pytest.mark.parametrize("pinned,served", [
    ("google-ai-studio", "Google"),  # a different provider name (or an unexpected spelling): must flag
    ("google-ai-studio", "Google Vertex"), ("openai", "Azure"), ("novita", None),
])
def test_different_or_missing_served_provider_is_a_mismatch(pinned, served):
    assert smoke.pin_mismatch(pinned, served) is True


@pytest.mark.parametrize("pinned", [None, ""])
def test_no_pin_never_flags(pinned):
    assert smoke.pin_mismatch(pinned, "Anything") is False and smoke.pin_mismatch(pinned, None) is False


def _record(provider):
    turn = {"served_model": "m", "provider": provider, "finish_reason": "stop", "tool_calls": [], "content": "x",
            "latency_s": 1.0, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.0, "reported_cost_usd": 0.0}
    return {"turn1": turn, "tool_results": [], "turn2": None}


def test_summarize_and_table_show_pin_mismatch(capsys):
    ok = smoke.summarize("a", _record("Novita"), "novita")
    bad = smoke.summarize("b", _record("Together"), "novita")
    assert ok["pin_mismatch"] is False and bad["pin_mismatch"] is True and bad["pinned_provider"] == "novita"
    smoke.print_table([ok, bad])
    lines = capsys.readouterr().out.splitlines()
    assert "PIN MISMATCH" not in lines[1] and "PIN MISMATCH" in lines[2]


def test_every_real_model_is_pinned():
    assert all(m.provider for m in load_models().values())
