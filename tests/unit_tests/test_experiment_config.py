import argparse
import importlib.util
from pathlib import Path

import pytest
import yaml


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "DSPy"
    / "dspy_avqa"
    / "experiment_config.py"
)
SPEC = importlib.util.spec_from_file_location("experiment_config_under_test", MODULE_PATH)
experiment_config = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment_config)

YAML_DIR = MODULE_PATH.parent / "yamls"
REASONER_ENV_KEYS = {
    "PLANNER_PROVIDER",
    "PLANNER_MODEL",
    "PLANNER_TEMPERATURE",
    "PLANNER_OUTPUT_SEQ_LEN",
    "PLANNER_REASONING_EFFORT",
    "PLANNER_SEED",
    "GEPA_REFLECTION_MODEL",
    "GEPA_REFLECTION_TEMPERATURE",
    "GEPA_REFLECTION_MAX_TOKENS",
    "GEPA_REFLECTION_REASONING_EFFORT",
    "GEPA_REFLECTION_SEED",
}


@pytest.fixture(autouse=True)
def restore_reasoner_environment():
    previous = {
        key: experiment_config.os.environ.get(key)
        for key in REASONER_ENV_KEYS
    }
    yield
    for key, value in previous.items():
        if value is None:
            experiment_config.os.environ.pop(key, None)
        else:
            experiment_config.os.environ[key] = value


@pytest.mark.parametrize(
    ("filename", "model", "effort"),
    (
        ("reasoner_elm_gpt4_1_none.yaml", "gpt-4.1", "none"),
        ("reasoner_elm_gpt5_4_none.yaml", "gpt-5.4", "none"),
        ("reasoner_elm_gpt5_4_medium.yaml", "gpt-5.4", "medium"),
    ),
)
def test_reasoner_yaml_applies_explicit_planner_values(monkeypatch, filename, model, effort):
    monkeypatch.setenv("PLANNER_MODEL", "stale-model")
    payload = experiment_config.load_reasoner_config_yaml(
        YAML_DIR / filename,
        role="planner",
    )

    assert payload["model"] == {"provider": "elm_gpt", "name": model}
    assert experiment_config.os.environ["PLANNER_PROVIDER"] == "elm_gpt"
    assert experiment_config.os.environ["PLANNER_MODEL"] == model
    assert experiment_config.os.environ["PLANNER_TEMPERATURE"] == "0.0"
    assert experiment_config.os.environ["PLANNER_OUTPUT_SEQ_LEN"] == "32768"
    assert experiment_config.os.environ["PLANNER_REASONING_EFFORT"] == effort
    assert experiment_config.os.environ["PLANNER_SEED"] == "1234"


def test_reasoner_yaml_can_configure_reflection_lm(monkeypatch):
    experiment_config.load_reasoner_config_yaml(
        YAML_DIR / "reasoner_elm_gpt5_4_medium.yaml",
        role="reflection",
    )

    assert experiment_config.os.environ["GEPA_REFLECTION_MODEL"] == "gpt-5.4"
    assert experiment_config.os.environ["GEPA_REFLECTION_TEMPERATURE"] == "0.0"
    assert experiment_config.os.environ["GEPA_REFLECTION_MAX_TOKENS"] == "32768"
    assert experiment_config.os.environ["GEPA_REFLECTION_REASONING_EFFORT"] == "medium"
    assert experiment_config.os.environ["GEPA_REFLECTION_SEED"] == "1234"


def test_reasoner_yaml_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  name: gpt-4.1\n  api_key: do-not-allow\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown reasoner config key"):
        experiment_config.load_reasoner_config_yaml(path, role="planner")


def test_resolved_config_is_serializable_redacted_and_optionally_printed(tmp_path, capsys):
    output_path = experiment_config.save_resolved_experiment_config(
        tmp_path,
        {
            "arguments": argparse.Namespace(output_dir=tmp_path),
            "planner": {"api_key": "live-secret", "model": "gpt-4.1"},
            "nested": {"access_token": "another-secret"},
        },
        print_config=True,
    )

    rendered = output_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(rendered)
    captured = capsys.readouterr().out
    assert loaded["planner"]["api_key"] == "<redacted:set>"
    assert loaded["nested"]["access_token"] == "<redacted:set>"
    assert "live-secret" not in rendered
    assert "another-secret" not in rendered
    assert rendered in captured


def test_g0_gepa_yaml_preserves_static_budget_and_tracking():
    payload = yaml.safe_load((YAML_DIR / "gepa_g0_planner.yaml").read_text(encoding="utf-8"))

    assert payload["max_full_evals"] == 16
    assert payload["reflection_minibatch_size"] == 16
    assert payload["candidate_selection_strategy"] == "pareto"
    assert payload["max_merge_invocations"] == 5
    assert payload["track_stats"] is True
    assert payload["track_best_outputs"] is True
    assert set(payload) == {
        "max_full_evals",
        "reflection_minibatch_size",
        "candidate_selection_strategy",
        "max_merge_invocations",
        "track_stats",
        "track_best_outputs",
        "run",
    }
    assert payload["run"]["algorithm"] == "gepa"
    assert "gepa_seed" not in payload["run"]
    assert "audio_caption_dir" not in payload["run"]
    assert payload["run"]["trainset_jsonl"].endswith("daily_omni_cuts_selectedTrain125.jsonl")
    assert payload["run"]["valset_jsonl"].endswith("daily_omni_cuts_selectedVal125.jsonl")
