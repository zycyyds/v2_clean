"""Standalone config loader for v2 — reads v2/model_config.yaml (and optional
v2/model_config.local.yaml) without depending on the legacy configs/ package.
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import yaml

V2_DIR = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_config() -> dict:
    base = _load_yaml(V2_DIR / "model_config.yaml")
    local = (
        {}
        if os.environ.get("V2_SKIP_LOCAL_MODEL_CONFIG") == "1"
        else _load_yaml(V2_DIR / "model_config.local.yaml")
    )
    return _deep_merge(base, local)


_config: dict = _load_config()


def get_agent_config(agent_key: str = "react_planner") -> dict:
    cfg = deepcopy(_config.get(agent_key) or _config.get("react_planner") or {})
    # normalise aliases so downstream code can use either key
    if "model" in cfg and "model_name" not in cfg:
        cfg["model_name"] = cfg["model"]
    if "base_url" in cfg and "api_base" not in cfg:
        cfg["api_base"] = cfg["base_url"]
    return cfg
