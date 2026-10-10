import pytest

from agent_eval.config import AGENTS_DIR
from agent_eval.schemas import AgentConfig, ModelSpec, load_agent, load_all_agents, load_models

BASE = {"id": "vendor/m1", "display_name": "M1", "tier": "budget", "price_in_per_m": 0.1, "price_out_per_m": 0.4}


def make(**overrides):
    return ModelSpec(**{**BASE, **overrides})


def test_minimal_model_defaults_optional_fields_to_none():
    m = make()
    assert m.params_total_b is None and m.params_active_b is None and m.provider is None and m.notes is None


@pytest.mark.parametrize("bad", [
    {"tier": "luxury"}, {"price_in_per_m": -1}, {"price_out_per_m": -0.01}, {"id": ""},
    {"params_total_b": 0}, {"params_total_b": -5}, {"params_active_b": 0}, {"extra_field": 1},
])
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_free_tier_with_zero_price_is_allowed():
    assert make(tier="free", price_in_per_m=0, price_out_per_m=0).price_in_per_m == 0


def test_required_fields_are_enforced():
    with pytest.raises(ValueError):
        ModelSpec(id="x", display_name="X", tier="free", price_in_per_m=0)  # price_out_per_m missing


def test_active_params_must_not_exceed_total():
    with pytest.raises(ValueError, match="params_active_b"):
        make(params_total_b=10, params_active_b=11)
    assert make(params_total_b=10, params_active_b=10)
    assert make(params_active_b=11)  # only one set: nothing to compare
    assert make(params_total_b=10)


def write(tmp_path, text):
    p = tmp_path / "models.yaml"
    p.write_text(text)
    return p


def wrap(entries: str) -> str:
    """Indent entry text under the top-level `models:` key."""
    return "models:\n" + "".join("  " + line + "\n" for line in entries.splitlines())


def test_load_models_valid_file(tmp_path):
    p = write(tmp_path, wrap("alpha:\n  id: vendor/a\n  display_name: A\n  tier: mid\n  price_in_per_m: 1\n  price_out_per_m: 2"))
    models = load_models(p)
    assert list(models) == ["alpha"] and isinstance(models["alpha"], ModelSpec) and models["alpha"].price_out_per_m == 2


def test_load_models_error_names_the_bad_model_key(tmp_path):
    p = write(tmp_path, wrap("good:\n  id: a\n  display_name: A\n  tier: free\n  price_in_per_m: 0\n  price_out_per_m: 0\n"
                             "broken:\n  id: b\n  display_name: B\n  tier: nope\n  price_in_per_m: 0\n  price_out_per_m: 0"))
    with pytest.raises(ValueError, match="'broken'"):
        load_models(p)


@pytest.mark.parametrize("text", [
    "", "- a\n- b\n", "just a string\n", "key: [unclosed\n",
    "models:\n  key: not-a-mapping\n",  # entry is not a mapping
    "models:\n", "models: []\n", "models: {}\n",  # models is empty or not a mapping
    "alpha:\n  id: a\n",  # old layout: no top-level 'models' key
])
def test_load_models_rejects_bad_file_shapes(tmp_path, text):
    with pytest.raises(ValueError):
        load_models(write(tmp_path, text))


def test_load_models_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Cannot read"):
        load_models(tmp_path / "nope.yaml")


def test_real_models_file_loads():
    models = load_models()
    assert list(models) == ["ling", "solar", "luna", "gemini"]
    assert all(isinstance(m, ModelSpec) for m in models.values())
    ling = models["ling"]
    assert ling.tier == "free" and ling.price_in_per_m == 0 and ling.price_out_per_m == 0
    assert ling.params_total_b == 124 and ling.params_active_b == 5.1
    assert models["luna"].params_total_b is None and models["gemini"].params_active_b is None
    assert models["gemini"].price_out_per_m == 3.75


# ---------- AgentConfig ----------

AGENT = {"name": "a", "model": "ling", "prompt": "minimal"}


def agent(**overrides):
    return AgentConfig(**{**AGENT, **overrides})


def test_agent_defaults():
    a = agent()
    assert (a.temperature, a.max_steps, a.max_run_cost_usd, a.reasoning_effort) == (0.0, 15, 0.25, None)


@pytest.mark.parametrize("bad", [
    {"temperature": 3}, {"temperature": -0.1}, {"temperature": float("nan")}, {"temperature": float("inf")},
    {"max_steps": 0}, {"max_steps": 100}, {"max_steps": 51},
    {"max_run_cost_usd": -1}, {"max_run_cost_usd": 0}, {"max_run_cost_usd": float("inf")},
    {"max_run_cost_usd": float("nan")},
    {"reasoning_effort": "extreme"}, {"name": ""}, {"extra_field": 1},
])
def test_agent_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        agent(**bad)


@pytest.mark.parametrize("ok", [
    {"temperature": 0}, {"temperature": 2}, {"max_steps": 1}, {"max_steps": 50}, {"reasoning_effort": "high"},
])
def test_agent_boundary_values_accepted(ok):
    assert agent(**ok)


# ---------- load_agent ----------

MODELS = {"ling": None, "solar": None}  # load_agent only checks the keys


@pytest.fixture
def prompts(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "minimal.md").write_text("Date {today}", encoding="utf-8")
    return d


def agent_file(tmp_path, stem="a", **fields):
    values = {"name": stem, "model": "ling", "prompt": "minimal", **fields}
    p = tmp_path / f"{stem}.yaml"
    p.write_text("".join(f"{k}: {v}\n" for k, v in values.items()), encoding="utf-8")
    return p


def test_load_agent_valid(tmp_path, prompts):
    a = load_agent(agent_file(tmp_path, temperature=0.5), models=MODELS, prompts_dir=prompts)
    assert a.name == "a" and a.temperature == 0.5


def test_load_agent_name_must_match_file_stem(tmp_path, prompts):
    p = agent_file(tmp_path, stem="a", name="other")
    with pytest.raises(ValueError, match="file name") as e:
        load_agent(p, models=MODELS, prompts_dir=prompts)
    assert str(p) in str(e.value)


def test_load_agent_unknown_model_lists_valid_keys(tmp_path, prompts):
    p = agent_file(tmp_path, model="nope")
    with pytest.raises(ValueError, match="Valid models: ling, solar") as e:
        load_agent(p, models=MODELS, prompts_dir=prompts)
    assert str(p) in str(e.value)


def test_load_agent_missing_prompt_file(tmp_path, prompts):
    p = agent_file(tmp_path, prompt="ghost")
    with pytest.raises(ValueError, match="ghost.md") as e:
        load_agent(p, models=MODELS, prompts_dir=prompts)
    assert str(p) in str(e.value)


@pytest.mark.parametrize("prompt", ["../minimal", "a/b", "a.b"])
def test_load_agent_rejects_prompt_names_that_could_escape(tmp_path, prompts, prompt):
    p = agent_file(tmp_path, prompt=f"'{prompt}'")
    with pytest.raises(ValueError, match="invalid prompt name"):
        load_agent(p, models=MODELS, prompts_dir=prompts)


@pytest.mark.parametrize("field,value", [("temperature", 3), ("max_steps", 0), ("max_steps", 100),
                                         ("max_run_cost_usd", -1), ("reasoning_effort", "extreme"),
                                         ("surprise", 1)])
def test_load_agent_rejects_bad_settings_with_path(tmp_path, prompts, field, value):
    p = agent_file(tmp_path, **{field: value})
    with pytest.raises(ValueError) as e:
        load_agent(p, models=MODELS, prompts_dir=prompts)
    assert str(p) in str(e.value)


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "just text\n", "name: [unclosed\n"])
def test_load_agent_rejects_bad_files_with_path(tmp_path, prompts, text):
    p = tmp_path / "a.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as e:
        load_agent(p, models=MODELS, prompts_dir=prompts)
    assert str(p) in str(e.value)


def test_load_agent_missing_file(tmp_path, prompts):
    p = tmp_path / "nope.yaml"
    with pytest.raises(ValueError, match="Cannot read") as e:
        load_agent(p, models=MODELS, prompts_dir=prompts)
    assert str(p) in str(e.value)


# ---------- load_all_agents ----------

def test_load_all_agents_real_files():
    agents = load_all_agents()
    files = sorted(f.stem for f in AGENTS_DIR.glob("*.yaml"))
    assert len(files) == 4 and sorted(agents) == files
    models = load_models()
    for name, a in agents.items():
        assert a.name == name and a.model in models
        assert load_agent(AGENTS_DIR / f"{name}.yaml") == a  # the default arguments load them too


def test_load_all_agents_empty_or_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="No agent files"):
        load_all_agents(tmp_path)
    with pytest.raises(ValueError, match="No agent files"):
        load_all_agents(tmp_path / "missing")


def test_load_all_agents_bad_file_names_its_path(tmp_path):
    p = agent_file(tmp_path, model="nope")
    with pytest.raises(ValueError) as e:
        load_all_agents(tmp_path)
    assert str(p) in str(e.value) and "Valid models: ling, solar, luna, gemini" in str(e.value)
