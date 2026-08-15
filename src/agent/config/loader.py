import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agent.errors import ConfigError
from agent.config.schema import Settings

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "default.yaml"
USER_CONFIG_PATH = Path.home() / ".agent" / "config.yaml"

ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "AGENT_MODEL": ("provider", "model"),
    "AGENT_WORKDIR": ("agent", "workdir"),
    "AGENT_MAX_STEPS": ("agent", "max_steps"),
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ConfigError(f"Config file not found: {path}")
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return data


def _env_layer() -> dict[str, Any]:
    layer: dict[str, Any] = {}
    for variable, (section, field) in ENV_OVERRIDES.items():
        raw = os.environ.get(variable)
        if not raw:
            continue
        layer = deep_merge(layer, {section: {field: raw}})
    return layer


def load_settings(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    layers = [
        _read_yaml(DEFAULT_CONFIG_PATH),
        _read_yaml(USER_CONFIG_PATH),
        _read_yaml(config_path, required=True) if config_path else {},
        _env_layer(),
        overrides or {},
    ]

    merged: dict[str, Any] = {}
    for layer in layers:
        merged = deep_merge(merged, layer)

    try:
        return Settings(**merged)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc
