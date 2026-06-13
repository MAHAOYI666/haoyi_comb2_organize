from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMBASE_ROOT = REPO_ROOT / "vendor" / "comb2-simbase"

if str(SIMBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBASE_ROOT))
