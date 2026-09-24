"""Worker-only real-data prediction-window reuse versus cold window loading."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for dependency in ("vendor/comb2", "vendor/comb2-simbase", "vendor/comb2-pcmaster", "."):
    sys.path.insert(0, str(ROOT / dependency))

import torch
from comb2 import ComboBase, LoaderConfig


def make_combo(loader):
    combo = ComboBase.__new__(ComboBase)
    combo.loader = loader
    combo.tsDays = 10
    combo.buffer = {}
    combo.buffer_end_didx = {}
    combo._predict_feature_window = {}
    combo._predict_model_windows = {}
    combo.livetrading = False
    return combo


def main():
    torch.set_num_threads(8)
    model_path = Path("/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/Model.py")
    spec = importlib.util.spec_from_file_location("comb2_predict_benchmark_model", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    loader = module.ResearchLoader(LoaderConfig(
        cache_path="/home/data/CacheData",
        data_start_ds=20170101,
        dtype=torch.float16,
        compression="fp4",
        sample_times=(93000,),
        load_chunk_days=32,
    ))
    ti = 93000
    first_ds = loader.align_date(20200521)
    first_didx = loader.date2didx(first_ds)
    with loader.cache_scope():
        loader.set_current_ti(ti)
        loader.build_raw_feature(loader.didx2date(first_didx - 9))
    reused = make_combo(loader)
    began = time.perf_counter()
    reused.buffer_load(first_ds, ti)
    seed_seconds = time.perf_counter() - began
    results = []
    for offset in range(1, 11):
        ds = loader.didx2date(first_didx + offset)
        began = time.perf_counter()
        reused.buffer_load(ds, ti)
        reused_seconds = time.perf_counter() - began
        cold = make_combo(loader)
        began = time.perf_counter()
        cold.buffer_load(ds, ti)
        cold_seconds = time.perf_counter() - began
        torch.testing.assert_close(
            reused._predict_feature_window[ti],
            cold._predict_feature_window[ti],
            rtol=0, atol=0, equal_nan=True,
        )
        assert not any(loader.registry.working_cache.values())
        results.append({
            "ds": ds,
            "reused_seconds": round(reused_seconds, 4),
            "cold_seconds": round(cold_seconds, 4),
        })
    reused_total = sum(item["reused_seconds"] for item in results)
    cold_total = sum(item["cold_seconds"] for item in results)
    print(json.dumps({
        "seed_seconds": round(seed_seconds, 4),
        "steps": len(results),
        "reused_total_seconds": round(reused_total, 4),
        "cold_total_seconds": round(cold_total, 4),
        "speedup": round(cold_total / reused_total, 2),
        "exact_parity": True,
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        "results": results,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
