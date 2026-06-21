from pathlib import Path

from agent_4.medical_column_selector import main as selector_main


def test_standalone_defaults_use_data_input_and_program_output():
    project_root = Path(selector_main.get_project_root())

    assert selector_main.get_default_input_dir() == str(project_root / "agent_4" / "data_input")
    assert selector_main.get_default_output_dir() == str(project_root / "program" / "output" / "step4_results")


def test_resolve_input_csv_prefers_input_csv(tmp_path):
    input_dir = tmp_path / "data_input"
    input_dir.mkdir()
    older = input_dir / "older.csv"
    preferred = input_dir / "input.csv"
    older.write_text("a\n1\n", encoding="utf-8")
    preferred.write_text("a\n2\n", encoding="utf-8")

    assert selector_main.resolve_input_csv(str(input_dir)) == str(preferred)
