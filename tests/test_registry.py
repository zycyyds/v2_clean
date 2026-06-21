"""Contract tests for skill discovery, metadata validation, and rendering."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

V2_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V2_DIR.parent
for path in (PROJECT_ROOT, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from skills import _registry  # noqa: E402


REQUIRED_AGENT_FIELDS = {
    "name",
    "layer",
    "description",
    "when_to_use",
    "when_to_skip",
    "capability_types",
    "inputs",
    "outputs",
    "prerequisites",
    "adapter_policy",
    "preserves_raw_values",
    "lib_entrypoints",
}


class FakeToolkit:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}


def _minimal_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "name": "inspect_rows",
        "layer": "explore",
        "description": "检查数据行和列结构。",
        "when_to_use": "需要确认输入表结构时使用。",
        "inputs": ["input_path"],
        "outputs": ["schema"],
    }
    metadata.update(overrides)
    return metadata


def _module(metadata: dict[str, object], tool_name: str) -> SimpleNamespace:
    def register(toolkit: FakeToolkit) -> None:
        toolkit.tools[tool_name] = object()

    return SimpleNamespace(SKILL=metadata, register=register)


@pytest.mark.parametrize("layer", ["explore", "process", "label", "util"])
def test_normalize_skill_metadata_accepts_four_layers(layer: str) -> None:
    normalized = _registry.normalize_skill_metadata(
        "inspect_rows",
        _minimal_metadata(layer=layer),
    )

    assert normalized["layer"] == layer
    assert REQUIRED_AGENT_FIELDS <= normalized.keys()


@pytest.mark.parametrize(
    "field",
    ["name", "layer", "description", "when_to_use", "inputs", "outputs"],
)
def test_normalize_skill_metadata_rejects_missing_core_field(field: str) -> None:
    metadata = _minimal_metadata()
    metadata.pop(field)

    with pytest.raises(RuntimeError, match=field):
        _registry.normalize_skill_metadata("inspect_rows", metadata)


def test_normalize_skill_metadata_rejects_unknown_layer() -> None:
    with pytest.raises(RuntimeError, match="layer"):
        _registry.normalize_skill_metadata(
            "inspect_rows",
            _minimal_metadata(layer="transform"),
        )


def test_normalize_skill_metadata_adds_compatible_defaults() -> None:
    normalized = _registry.normalize_skill_metadata(
        "inspect_rows",
        _minimal_metadata(),
    )

    assert normalized["when_to_skip"]
    assert normalized["capability_types"] == ["inspect_rows"]
    assert normalized["prerequisites"] == []
    assert normalized["adapter_policy"] == "direct_then_adapter_then_fork"
    assert normalized["preserves_raw_values"] is True
    assert normalized["lib_entrypoints"] == []


def test_load_skills_returns_normalized_metadata_and_rejects_duplicate_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = [
        ("first", _module(_minimal_metadata(), "first_tool")),
        ("second", _module(_minimal_metadata(), "second_tool")),
    ]
    monkeypatch.setattr(_registry, "_iter_skill_modules", lambda allowed_names=None: iter(modules))

    with pytest.raises(RuntimeError, match="Duplicate skill name"):
        _registry.load_skills(FakeToolkit())


def test_render_skill_manifest_exposes_agent_decision_contract() -> None:
    item = _registry.normalize_skill_metadata(
        "extract_rows",
        _minimal_metadata(
            name="extract_rows",
            layer="process",
            when_to_skip="输入只有自由文本时跳过。",
            capability_types=["structured_extract"],
            prerequisites=["schema_known"],
            adapter_policy="adapter_then_fork",
            preserves_raw_values=False,
            lib_entrypoints=["lib.extract_rows"],
        ),
    )

    rendered = _registry.render_skill_manifest([item])

    assert "`extract_rows`" in rendered
    assert "使用条件：需要确认输入表结构时使用。" in rendered
    assert "跳过条件：输入只有自由文本时跳过。" in rendered
    assert "输入：`input_path`" in rendered
    assert "输出：`schema`" in rendered
    assert "适配策略：`adapter_then_fork`" in rendered
    assert "能力：`structured_extract`" in rendered
    assert "前置条件：`schema_known`" in rendered


def _read_static_skill_metadata(skill_file: Path) -> dict[str, object]:
    tree = ast.parse(skill_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "SKILL" for target in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, dict)
            return value
    raise AssertionError(f"Missing SKILL metadata: {skill_file}")


def test_all_existing_skill_metadata_is_valid_and_named_consistently() -> None:
    skills_dir = V2_DIR / "skills"
    dirs = [
        directory
        for directory in sorted(skills_dir.iterdir())
        if directory.is_dir() and not directory.name.startswith("_")
    ]
    assert dirs, "No skill directories discovered"

    names: set[str] = set()
    for directory in dirs:
        skill_file = directory / "skill.py"
        assert skill_file.exists(), f"Missing file: {directory.name}/skill.py"
        normalized = _registry.normalize_skill_metadata(
            directory.name,
            _read_static_skill_metadata(skill_file),
        )
        assert normalized["name"] == directory.name
        assert normalized["name"] not in names
        assert REQUIRED_AGENT_FIELDS <= normalized.keys()
        names.add(str(normalized["name"]))


def test_all_existing_skills_explicitly_describe_agent_decision_metadata() -> None:
    skills_dir = V2_DIR / "skills"
    for skill_file in sorted(skills_dir.glob("*/skill.py")):
        metadata = _read_static_skill_metadata(skill_file)
        missing = REQUIRED_AGENT_FIELDS - metadata.keys()
        assert not missing, f"{skill_file.parent.name} missing explicit metadata: {sorted(missing)}"


def test_mimic_liver_skill_is_explicitly_dataset_and_task_specific() -> None:
    metadata = _read_static_skill_metadata(
        V2_DIR / "skills" / "build_mimic_liver_dataset" / "skill.py",
    )
    normalized = _registry.normalize_skill_metadata("build_mimic_liver_dataset", metadata)

    use_text = str(normalized["when_to_use"]).lower()
    skip_text = str(normalized["when_to_skip"]).lower()
    assert "mimic" in use_text and "肝" in use_text
    assert "非 mimic" in skip_text
    assert "gold" in skip_text
