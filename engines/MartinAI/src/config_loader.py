"""Validated YAML configuration loading with recursive override support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"


def _merge(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar/list values."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def _validate_shape(
    defaults: dict[str, Any], configured: dict[str, Any], prefix: str = ""
) -> None:
    unknown = set(configured) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s) at {prefix or '<root>'}: "
            + ", ".join(sorted(unknown))
        )
    for key, value in configured.items():
        dotted = f"{prefix}.{key}" if prefix else key
        expected = defaults[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section '{dotted}' must be a mapping.")
            _validate_shape(expected, value, dotted)
        elif isinstance(expected, bool):
            if not isinstance(value, bool):
                raise ValueError(f"Configuration value '{dotted}' must be boolean.")
        elif isinstance(expected, int):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Configuration value '{dotted}' must be an integer.")
        elif isinstance(expected, float):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise ValueError(f"Configuration value '{dotted}' must be numeric.")
        elif expected is not None and not isinstance(value, type(expected)):
            raise ValueError(
                f"Configuration value '{dotted}' must be {type(expected).__name__}."
            )


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration YAML, reject unknown keys, and apply nested overrides."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def read_yaml(source: Path) -> dict[str, Any]:
        try:
            with source.open("r", encoding="utf-8") as config_file:
                document = yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML configuration: {source}") from error
        if not isinstance(document, dict):
            raise ValueError(f"Configuration root must be a YAML mapping: {source}")
        return document

    defaults = read_yaml(DEFAULT_CONFIG_PATH)
    try:
        loaded = read_yaml(config_path)
    except (OSError, ValueError):
        raise
    _validate_shape(defaults, loaded)
    config = deepcopy(defaults)
    _merge(config, loaded)
    if overrides:
        _merge(config, overrides)
    _validate_shape(defaults, config)
    return config


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten top-level sections into CLI-friendly values with explicit mapping."""
    sections = {
        "mcts": {
            "simulations": "simulations",
            "c_puct": "c_puct",
            "dirichlet_alpha": "dirichlet_alpha",
            "dirichlet_fraction": "dirichlet_fraction",
            "value_discount": "value_discount",
            "tactical_checks": "tactical_checks",
        },
        "self_play": {
            "games": "selfplay_games",
            "workers": "selfplay_workers",
            "inference_batch_size": "selfplay_batch_size",
            "max_moves": "max_moves",
            "temperature_switch_ply": "temperature_switch_ply",
            "greedy_after_ply": "greedy_after_ply",
            "early_temperature": "early_temperature",
            "late_temperature": "late_temperature",
        },
        "evaluation": {
            "simulations": "evaluation_simulations",
            "max_moves": "evaluation_max_moves",
        },
        "training": {
            "max_iterations": "num_iterations",
            "batch_size": "batch_size",
            "epochs": "epochs",
            "pretrain_epochs": "pretrain_epochs",
            "max_buffer_size": "max_buffer_size",
            "rl_learning_rate": "rl_learning_rate",
            "device": "device",
        },
        "loss": {
            "policy_weight": "policy_loss_weight",
            "value_weight": "value_loss_weight",
        },
        "arena": {
            "games": "arena_games",
            "early_stop": "arena_early_stop",
            "min_games": "arena_min_games",
            "opening_plies": "arena_opening_plies",
            "opening_temperature": "arena_opening_temperature",
            "promotion_threshold": "promotion_threshold",
        },
        "elo": {"games_per_anchor": "elo_games"},
    }
    flattened: dict[str, Any] = {}
    for section, keys in sections.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping.")
        for key, arg_name in keys.items():
            if key not in values:
                raise ValueError(
                    f"Configuration is missing required value '{section}.{key}'."
                )
            flattened[arg_name] = values[key]
    return flattened
