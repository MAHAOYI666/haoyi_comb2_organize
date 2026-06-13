from __future__ import annotations

import sys
from pathlib import Path

import pytest


COMB2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SIMBASE_ROOT = REPO_ROOT / "vendor" / "comb2-simbase"
sys.path.insert(0, str(COMB2_ROOT))
sys.path.insert(0, str(SIMBASE_ROOT))
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def cpu_float8_supported() -> bool:
    from src.codec import probe_cpu_float8_cast

    return probe_cpu_float8_cast()
