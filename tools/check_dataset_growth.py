"""Real-data parity of a training Dataset grown in place against cold loads; run on a KF worker.

Builds the research loader and dataset of a runCombo config exactly as ComboBase does, constructs the first window
with ``capacity_days`` reserved, advances it with ``roll_forward`` through the given windows and compares every
step's full storage (instrument selection, X, Y, W) with a cold load of the same window.

    python tools/check_dataset_growth.py CONFIG.xml --steps 20191231:731,20200331:790 --capacity-days 900
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
for dependency in ("vendor/comb2-metrics", "vendor/comb2-simbase", "vendor/comb2-pcmaster", "vendor/comb2", "."):
    sys.path.insert(0, str(ROOT / dependency))

import torch

from combo2.config import load_config
from combo2.runtime import Node, load_combo_base_class


def rss_gib():
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1024**3


def peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def mismatches(left, right):
    """Number of differing elements (NaN equals NaN)."""
    if left.shape != right.shape:
        return -1
    same = left == right
    if left.is_floating_point():
        same |= torch.isnan(left) & torch.isnan(right)
    return int((~same).sum())


def build(combo, loader, end_ds, ndays, capacity_days=None):
    kwargs = dict(end_ds=end_ds, ndays=ndays, x_delay=combo.retDays, ts_days=combo.tsDays,
                  load_chunk_days=combo.load_chunk_days, codec=loader.codec)
    if capacity_days is not None:
        kwargs["capacity_days"] = capacity_days
    return combo.research_dataset_cls(loader, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--steps", required=True, help="END:NDAYS,...; the first is the initial build")
    parser.add_argument("--capacity-days", type=int, required=True)
    args = parser.parse_args()
    steps = [tuple(int(v) for v in step.split(":")) for step in args.steps.split(",")]

    config = load_config(args.config)
    scratch = Path(tempfile.mkdtemp(prefix="grow_parity_"))
    config["combo"]["paths"].update(output_dir=str(scratch), checkpoint_root=str(scratch / "checkpoints"))
    combo = load_combo_base_class(config["combo"])(Node(config))
    print(json.dumps({"phase": "setup", "dataset_cls": combo.research_dataset_cls.__module__,
                      "loader_cls": type(combo.loader).__module__, "rss_gib": round(rss_gib(), 3)}), flush=True)

    end_ds, ndays = steps[0]
    began = time.perf_counter()
    grown = build(combo, combo.loader, end_ds, ndays, args.capacity_days)
    print(json.dumps({"phase": "built", "end_ds": end_ds, "ndays": grown.ndays, "seconds": round(time.perf_counter() - began, 1),
                      "instruments": grown.numValidinsts, "storage_gib": round(grown.storage_nbytes() / 1024**3, 3),
                      "rss_gib": round(rss_gib(), 3)}), flush=True)
    ok = True
    for end_ds, ndays in steps[1:]:
        before = grown.validinsts.clone()
        began = time.perf_counter()
        rolled = grown.roll_forward(end_ds, ndays)
        roll_seconds = time.perf_counter() - began
        if not rolled:
            print(json.dumps({"phase": "roll", "end_ds": end_ds, "ndays": ndays, "rolled": False}), flush=True)
            ok = False
            break
        cold_loader = combo.research_loader_cls(combo.node.loader_config)
        began = time.perf_counter()
        cold = build(combo, cold_loader, end_ds, ndays)
        cold_seconds = time.perf_counter() - began
        report = {
            "phase": "compare", "end_ds": end_ds, "ndays": grown.ndays, "roll_seconds": round(roll_seconds, 1),
            "cold_seconds": round(cold_seconds, 1),
            "window": [grown.start_didx, grown.end_didx] == [cold.start_didx, cold.end_didx],
            "instruments": [grown.numValidinsts, cold.numValidinsts],
            "added": int((~torch.isin(grown.validinsts, before)).sum()),
            "removed": int((~torch.isin(before, grown.validinsts)).sum()),
            "validinsts_equal": torch.equal(grown.validinsts, cold.validinsts),
            "X_mismatch": {str(ti): mismatches(grown.X[ti], cold.X[ti]) for ti in cold.X},
            "Y_mismatch": mismatches(grown.Y, cold.Y),
            "W_mismatch": mismatches(grown.W, cold.W),
            "rss_gib": round(rss_gib(), 3), "peak_gib": round(peak_gib(), 3),
        }
        exact = (report["window"] and report["validinsts_equal"] and not report["Y_mismatch"]
                 and not report["W_mismatch"] and not any(report["X_mismatch"].values()))
        report["exact"] = exact
        ok &= exact
        print(json.dumps(report), flush=True)
        if not exact and report["validinsts_equal"]:
            # where the mismatches are: retained rows vs new rows, and added instruments vs the rest
            added = ~torch.isin(grown.validinsts, before)
            diff = (grown.W != cold.W).any(dim=1)
            print(json.dumps({"phase": "W_mismatch_where", "rows": torch.where(diff.any(dim=1))[0][:20].tolist(),
                              "added_columns": int(diff[:, added].sum()), "other_columns": int(diff[:, ~added].sum())}),
                  flush=True)
        del cold, cold_loader
        gc.collect()
    print(json.dumps({"phase": "done", "ok": ok, "peak_gib": round(peak_gib(), 3)}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
