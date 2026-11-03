"""Resolve repository/tool imports after moving test entrypoints one level down.

Supports direct scripts and unittest discovery without changing the working
directory, model configuration, RNG state or training/evaluation algorithms.
"""
from pathlib import Path
import sys


TEST_SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = TEST_SCRIPT_DIR.parent
REPO_ROOT = TOOLS_DIR.parent
for directory in (REPO_ROOT, TOOLS_DIR, TEST_SCRIPT_DIR):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)
