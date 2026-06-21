from pathlib import Path

import agent_4.main as agent4_main


def test_agent4_standalone_defaults_use_task_clipping_paths():
    project_root = Path(agent4_main.get_project_root())

    assert agent4_main.get_default_input_dir() == str(project_root / "agent_4" / "data_input")
    assert agent4_main.get_default_output_dir() == str(project_root / "program" / "output" / "step4_results")
