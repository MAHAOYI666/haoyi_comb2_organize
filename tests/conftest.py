from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMB2_ROOT = REPO_ROOT / "vendor" / "comb2"
SIMBASE_ROOT = REPO_ROOT / "vendor" / "comb2-simbase"
PCMASTER_ROOT = REPO_ROOT / "vendor" / "comb2-pcmaster"

for path in (REPO_ROOT, COMB2_ROOT, SIMBASE_ROOT, PCMASTER_ROOT):
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
