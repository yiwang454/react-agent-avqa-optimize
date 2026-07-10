"""Prompt YAML loading for DSPy AVQA."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROMPT_YAML = Path(__file__).resolve().parent / "yamls" / "daily_qa_prompt_v8.yaml"

_PROMPT_YAML_PATH: Path | None = None
_PROMPTS: dict[str, Any] = {}
_PROMPT_OVERRIDES: ContextVar[dict[tuple[str, ...], str]] = ContextVar("DSPY_AVQA_PROMPT_OVERRIDES", default={})


def load_prompt_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load prompt configuration from YAML and make it active."""
    global _PROMPT_YAML_PATH, _PROMPTS

    yaml_path = Path(path) if path is not None else DEFAULT_PROMPT_YAML
    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Prompt config must be a YAML mapping: {yaml_path}")

    _PROMPT_YAML_PATH = yaml_path
    _PROMPTS = config
    return config


def active_prompt_yaml_path() -> Path:
    """Return the active prompt YAML path, loading the default if needed."""
    if _PROMPT_YAML_PATH is None:
        load_prompt_config()
    assert _PROMPT_YAML_PATH is not None
    return _PROMPT_YAML_PATH


def prompt_config() -> dict[str, Any]:
    """Return the active prompt mapping, loading the default if needed."""
    if not _PROMPTS:
        load_prompt_config()
    return _PROMPTS


@contextmanager
def prompt_overrides(overrides: dict[tuple[str, ...], str]):
    """Temporarily override selected prompt values for one program rollout."""
    current = dict(_PROMPT_OVERRIDES.get())
    current.update(overrides)
    token = _PROMPT_OVERRIDES.set(current)
    try:
        yield
    finally:
        _PROMPT_OVERRIDES.reset(token)


def prompt_value(*keys: str) -> str:
    """Return a string prompt value by nested YAML keys."""
    override_key = tuple(keys)
    overrides = _PROMPT_OVERRIDES.get()
    if override_key in overrides:
        return overrides[override_key]

    value: Any = prompt_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            joined = ".".join(keys)
            raise KeyError(f"Missing prompt config key: {joined}")
        value = value[key]
    if not isinstance(value, str):
        joined = ".".join(keys)
        raise ValueError(f"Prompt config value must be a string: {joined}")
    return value


def render_prompt(*keys: str, **kwargs: Any) -> str:
    """Render a configured prompt template with Python format variables."""
    return prompt_value(*keys).format(**kwargs).strip()
