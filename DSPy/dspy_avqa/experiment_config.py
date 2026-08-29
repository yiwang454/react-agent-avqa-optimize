"""Explicit reasoner configuration and reproducibility artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


_REASONER_TOP_LEVEL_KEYS = {"model", "sampling_params"}
_REASONER_MODEL_KEYS = {"provider", "name"}
_REASONER_SAMPLING_KEYS = {
    "temperature",
    "max_tokens",
    "reasoning_effort",
    "seed",
}
_SENSITIVE_KEY_PARTS = ("api_key", "token", "password", "secret", "credential")


def _mapping(value: Any, *, section: str, path: Path) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Reasoner config {section} must be a mapping: {path}")
    return value


def _reject_unknown_keys(
    payload: dict[str, Any], allowed: set[str], *, section: str, path: Path
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown reasoner config key(s) in {section}: {', '.join(unknown)} ({path})"
        )


def _validate_reasoner_config(config: dict[str, Any], path: Path) -> None:
    _reject_unknown_keys(config, _REASONER_TOP_LEVEL_KEYS, section="root", path=path)
    model = _mapping(config.get("model"), section="model", path=path)
    sampling = _mapping(config.get("sampling_params"), section="sampling_params", path=path)
    _reject_unknown_keys(model, _REASONER_MODEL_KEYS, section="model", path=path)
    _reject_unknown_keys(
        sampling,
        _REASONER_SAMPLING_KEYS,
        section="sampling_params",
        path=path,
    )
    if "provider" in model and not isinstance(model["provider"], str):
        raise ValueError(f"Reasoner model.provider must be a string: {path}")
    if "name" in model and not isinstance(model["name"], str):
        raise ValueError(f"Reasoner model.name must be a string: {path}")
    if "temperature" in sampling and (
        isinstance(sampling["temperature"], bool)
        or not isinstance(sampling["temperature"], (int, float))
    ):
        raise ValueError(f"Reasoner sampling_params.temperature must be numeric: {path}")
    for key in ("max_tokens", "seed"):
        if key in sampling and (
            isinstance(sampling[key], bool) or not isinstance(sampling[key], int)
        ):
            raise ValueError(f"Reasoner sampling_params.{key} must be an integer: {path}")
    if "reasoning_effort" in sampling and not isinstance(
        sampling["reasoning_effort"], str
    ):
        raise ValueError(
            f"Reasoner sampling_params.reasoning_effort must be a string: {path}"
        )


def load_reasoner_config_yaml(path: Path | None, *, role: str) -> dict[str, Any]:
    """Load a planner/reflection config and apply its explicit values to the environment."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Reasoner config must be a YAML mapping: {path}")
    _validate_reasoner_config(config, path)
    model = config.get("model") or {}
    sampling = config.get("sampling_params") or {}

    if role == "planner":
        env_mapping = {
            "provider": "PLANNER_PROVIDER",
            "name": "PLANNER_MODEL",
            "temperature": "PLANNER_TEMPERATURE",
            "max_tokens": "PLANNER_OUTPUT_SEQ_LEN",
            "reasoning_effort": "PLANNER_REASONING_EFFORT",
            "seed": "PLANNER_SEED",
        }
    elif role == "reflection":
        env_mapping = {
            "name": "GEPA_REFLECTION_MODEL",
            "temperature": "GEPA_REFLECTION_TEMPERATURE",
            "max_tokens": "GEPA_REFLECTION_MAX_TOKENS",
            "reasoning_effort": "GEPA_REFLECTION_REASONING_EFFORT",
            "seed": "GEPA_REFLECTION_SEED",
        }
    else:
        raise ValueError(f"Unsupported reasoner config role: {role!r}")

    for key, value in model.items():
        env_name = env_mapping.get(key)
        if env_name is not None:
            os.environ[env_name] = str(value)
    for key, value in sampling.items():
        env_name = env_mapping.get(key)
        if env_name is not None:
            os.environ[env_name] = str(value)
    return config


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return {key: _serializable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def redact_sensitive(value: Any) -> Any:
    """Recursively replace secret values while retaining whether they were set."""
    value = _serializable(value)
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = key.lower()
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                redacted[key] = "<redacted:set>" if item else "<redacted:unset>"
            else:
                redacted[key] = redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def planner_effective_config(context: Any) -> dict[str, Any]:
    """Return the effective reasoner settings without exposing credentials."""
    return {
        "provider": context.planner_provider,
        "model": context.planner_model,
        "api_key": "<redacted:set>"
        if context.planner_api_key and context.planner_api_key != "EMPTY"
        else "<redacted:unset>",
        "api_base": context.planner_api_base,
        "temperature": context.planner_temperature,
        "top_p": context.planner_top_p,
        "top_k": context.planner_top_k,
        "repetition_penalty": context.planner_repetition_penalty,
        "max_tokens": context.planner_max_tokens,
        "max_input_seq_len": context.planner_max_input_seq_len,
        "seed": context.planner_seed,
        "reasoning_effort": context.planner_reasoning_effort,
        "timeout": context.planner_timeout,
        "max_retries": context.planner_max_retries,
        "retry_delay_s": context.planner_retry_delay_s,
    }


def lm_effective_config(lm: Any) -> dict[str, Any]:
    """Return a redacted model/kwargs snapshot for a DSPy LM."""
    return redact_sensitive(
        {
            "model": getattr(lm, "model", None),
            "kwargs": getattr(lm, "kwargs", {}),
        }
    )


def save_resolved_experiment_config(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    print_config: bool,
) -> Path:
    """Save and optionally print the same redacted resolved experiment mapping."""
    resolved = redact_sensitive(payload)
    rendered = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "resolved_experiment_config.yaml"
    output_path.write_text(rendered, encoding="utf-8")
    if print_config:
        print("Resolved experiment config:")
        print(rendered, end="")
    return output_path
