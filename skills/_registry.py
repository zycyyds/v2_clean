"""Skill auto-discovery and registration.

Each skill lives under ``skills/<skill_name>/`` and exposes:

- ``SKILL``: a dict with metadata (name, step, description, when_to_use,
  inputs, outputs, prerequisites). Mirrors the SKILL.md frontmatter.
- ``register(toolkit)``: registers the underlying tool function(s) on the
  given Toolkit so the ReActAgent can call them. The function may simply
  forward to a tool registered by the legacy ``step*_tools.py`` module.

The registry walks ``v2/skills/`` once at startup, imports every
``<dir>/skill.py`` and asks each module to register itself. The same
module also contributes a one-line summary to a "skill manifest" that the
PlannerReActAgent receives in its system prompt, so the agent knows what
each skill is for without having to read full SKILL.md content unless it
asks for it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from agentscope.tool import Toolkit

from . import _common  # noqa: F401  -- side effect: legacy paths on sys.path

SKILLS_DIR = Path(__file__).resolve().parent

SKILL_LAYERS = ("explore", "process", "label", "util")
CORE_METADATA_FIELDS = (
    "name",
    "layer",
    "description",
    "when_to_use",
    "inputs",
    "outputs",
)
AGENT_METADATA_FIELDS = (
    *CORE_METADATA_FIELDS,
    "when_to_skip",
    "capability_types",
    "prerequisites",
    "adapter_policy",
    "preserves_raw_values",
    "lib_entrypoints",
)
DEFAULT_ADAPTER_POLICY = "direct_then_adapter_then_fork"


def _validate_text(skill_dir: str, metadata: Mapping[str, Any], field: str) -> str:
    if field not in metadata:
        raise RuntimeError(f"Skill '{skill_dir}' metadata is missing required field '{field}'.")
    value = metadata[field]
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Skill '{skill_dir}' metadata field '{field}' must be a non-empty string.")
    return value.strip()


def _validate_string_list(
    skill_dir: str,
    metadata: Mapping[str, Any],
    field: str,
    *,
    required: bool,
) -> list[str]:
    if field not in metadata:
        if required:
            raise RuntimeError(f"Skill '{skill_dir}' metadata is missing required field '{field}'.")
        return []
    value = metadata[field]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RuntimeError(f"Skill '{skill_dir}' metadata field '{field}' must be a list of strings.")
    result = list(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise RuntimeError(f"Skill '{skill_dir}' metadata field '{field}' must contain only non-empty strings.")
    return [item.strip() for item in result]


def normalize_skill_metadata(skill_dir: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate core metadata and add backward-compatible agent-facing defaults."""
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"Skill '{skill_dir}' must define SKILL as a mapping.")

    normalized = dict(metadata)
    for field in ("name", "layer", "description", "when_to_use"):
        normalized[field] = _validate_text(skill_dir, metadata, field)
    for field in ("inputs", "outputs"):
        normalized[field] = _validate_string_list(
            skill_dir,
            metadata,
            field,
            required=True,
        )

    if normalized["layer"] not in SKILL_LAYERS:
        allowed = ", ".join(SKILL_LAYERS)
        raise RuntimeError(
            f"Skill '{skill_dir}' metadata field 'layer' must be one of: {allowed}."
        )

    when_to_skip = metadata.get("when_to_skip")
    if when_to_skip is None:
        normalized["when_to_skip"] = "未满足使用条件，或该能力与当前任务无关时跳过。"
    elif not isinstance(when_to_skip, str) or not when_to_skip.strip():
        raise RuntimeError(
            f"Skill '{skill_dir}' metadata field 'when_to_skip' must be a non-empty string."
        )
    else:
        normalized["when_to_skip"] = when_to_skip.strip()

    capability_types = metadata.get("capability_types")
    if capability_types is None:
        normalized["capability_types"] = [normalized["name"]]
    else:
        normalized["capability_types"] = _validate_string_list(
            skill_dir,
            metadata,
            "capability_types",
            required=False,
        )
        if not normalized["capability_types"]:
            raise RuntimeError(
                f"Skill '{skill_dir}' metadata field 'capability_types' cannot be empty."
            )

    normalized["prerequisites"] = _validate_string_list(
        skill_dir,
        metadata,
        "prerequisites",
        required=False,
    )
    normalized["lib_entrypoints"] = _validate_string_list(
        skill_dir,
        metadata,
        "lib_entrypoints",
        required=False,
    )

    adapter_policy = metadata.get("adapter_policy", DEFAULT_ADAPTER_POLICY)
    if not isinstance(adapter_policy, str) or not adapter_policy.strip():
        raise RuntimeError(
            f"Skill '{skill_dir}' metadata field 'adapter_policy' must be a non-empty string."
        )
    normalized["adapter_policy"] = adapter_policy.strip()

    preserves_raw_values = metadata.get("preserves_raw_values", True)
    if not isinstance(preserves_raw_values, bool):
        raise RuntimeError(
            f"Skill '{skill_dir}' metadata field 'preserves_raw_values' must be a boolean."
        )
    normalized["preserves_raw_values"] = preserves_raw_values
    return normalized


def _iter_skill_modules(
    allowed_names: set[str] | None = None,
) -> Iterable[tuple[str, Any]]:
    for entry in sorted(SKILLS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        if allowed_names is not None and entry.name not in allowed_names:
            continue
        skill_py = entry / "skill.py"
        if not skill_py.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            f"v2_skills.{entry.name}", skill_py
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield entry.name, module


def load_skills(toolkit: Toolkit, allowed_names: set[str] | None = None) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    loaded_names: set[str] = set()
    for skill_dir, module in _iter_skill_modules(allowed_names):
        skill_meta = getattr(module, "SKILL", None)
        register_fn = getattr(module, "register", None)
        if not isinstance(skill_meta, dict) or register_fn is None:
            raise RuntimeError(
                f"Skill '{skill_dir}' must define both SKILL dict and register(toolkit)."
            )
        normalized_meta = normalize_skill_metadata(skill_dir, skill_meta)
        skill_name = normalized_meta["name"]
        if allowed_names is not None and skill_name not in allowed_names:
            continue
        if skill_name in loaded_names:
            raise RuntimeError(f"Duplicate skill name '{skill_name}' discovered in '{skill_dir}'.")
        loaded_names.add(skill_name)
        before_tools = set(toolkit.tools)
        register_fn(toolkit)
        registered_tools = sorted(set(toolkit.tools) - before_tools)
        manifest.append({"dir": skill_dir, **normalized_meta, "tool_names": registered_tools})
    return manifest


def load_all_skills(toolkit: Toolkit) -> list[dict[str, Any]]:
    return load_skills(toolkit)


def _render_values(values: Sequence[str]) -> str:
    return "、".join(f"`{value}`" for value in values) if values else "无"


def render_skill_manifest(manifest: list[dict[str, Any]]) -> str:
    lines = ["# Skill manifest（按能力层分组）\n"]
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for item in manifest:
        by_layer.setdefault(item.get("layer", "util"), []).append(item)
    layer_labels = {
        "explore": "## 探索层（先调用这些了解数据）",
        "process": "## 处理层（按需调用，无固定顺序）",
        "label":   "## 标签与输出层（根据任务目标选择）",
        "util":    "## 工具层（辅助）",
    }
    for layer_key in ("explore", "process", "label", "util"):
        items = by_layer.get(layer_key) or []
        if not items:
            continue
        lines.append(f"\n{layer_labels[layer_key]}")
        for item in items:
            lines.append(
                f"- `{item['name']}` — {item.get('description', '').strip()}"
            )
            lines.append(f"    使用条件：{item['when_to_use']}")
            lines.append(f"    跳过条件：{item['when_to_skip']}")
            lines.append(f"    能力：{_render_values(item['capability_types'])}")
            lines.append(f"    输入：{_render_values(item['inputs'])}")
            lines.append(f"    输出：{_render_values(item['outputs'])}")
            lines.append(f"    前置条件：{_render_values(item['prerequisites'])}")
            lines.append(f"    适配策略：`{item['adapter_policy']}`")
            preserves = "是" if item["preserves_raw_values"] else "否"
            lines.append(f"    保留原始值：{preserves}")
            lines.append(f"    可复用入口：{_render_values(item['lib_entrypoints'])}")
    return "\n".join(lines)
