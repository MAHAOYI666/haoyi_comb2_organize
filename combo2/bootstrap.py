"""One compatibility bootstrap for running an uninstalled source checkout."""
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]

def bootstrap_source_tree() -> None:
    if not (SOURCE_ROOT / "vendor" / "comb2" / "comb2").is_dir():
        return
    for relative in ("vendor/comb2", "vendor/comb2-pcmaster", "vendor/comb2-simbase", "vendor/comb2-metrics", "evals"):
        path = str(SOURCE_ROOT / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
