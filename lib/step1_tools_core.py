from __future__ import annotations

import sys
from pathlib import Path

STEP1_DIR = Path(__file__).resolve().parent
if str(STEP1_DIR) in sys.path:
    sys.path.remove(str(STEP1_DIR))
sys.path.insert(0, str(STEP1_DIR))

from step1_tools import *  # noqa: F401,F403
