import pytest

from agent_eval.schemas import ModelSpec, load_models

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
