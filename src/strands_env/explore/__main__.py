from __future__ import annotations

import os
import sys

from .cli import explore_cmd

# Ensure the current working directory is importable so a user-provided --answer-fn resolves.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    explore_cmd()
