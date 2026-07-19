from __future__ import annotations

import sys
from pathlib import Path


EVALS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALS_ROOT.parent

for path in (EVALS_ROOT, REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
