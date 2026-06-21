from __future__ import annotations

import sys
from pathlib import Path

V2_DIR = Path(__file__).resolve().parents[1]
LIB_DIR = V2_DIR / "lib"


def ensure_paths_on_syspath() -> None:
    for p in (V2_DIR, LIB_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


ensure_paths_on_syspath()


def default_program_output(subdir: str) -> str:
    return str(V2_DIR / "output" / subdir)


def default_records_path() -> str:
    return str(V2_DIR / "output" / "reorganized_output" / "_meta" / "records.json")


def default_generated_reorganizer_path() -> str:
    return str(V2_DIR / "output" / "generated_reorganizer.py")


def default_input_path() -> str:
    return str(V2_DIR / "input")
