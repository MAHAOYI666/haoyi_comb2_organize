"""Worker-only framework Dataset roll and storage measurements."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
for dependency in ("vendor/comb2", "vendor/comb2-simbase", "vendor/comb2-pcmaster", "."):
    sys.path.insert(0, str(REPO_ROOT / dependency))

import torch

from comb2 import ComboTrainDataset, LoaderConfig
from comb2.codec import build_codec
from comb2_simbase import IndexMask


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run_roll():
    test_path = Path(__file__).resolve().parents[1] / "tests/test_cube_feature_groups.py"
    spec = importlib.util.spec_from_file_location("cube_feature_groups_benchmark", test_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    results = []
    with tempfile.TemporaryDirectory(prefix="comb2_roll_bench_") as temporary:
        root, dates, *_ = module.sources.__wrapped__(Path(temporary))
        for compression in ("none", "fp8", "fp4"):
            loader = module.ResearchLoader(LoaderConfig(
                cache_path=str(root), data_start_ds=dates[1],
                dtype=torch.float32, compression=compression,
                sample_times=(100000, 110000), load_chunk_days=3,
            ))
            validinsts = torch.arange(256)
            start = time.perf_counter()
            rolling = ComboTrainDataset(
                loader, dates[-4], 5, ts_days=3,
                validinsts=validinsts, codec=loader.codec,
            )
            initial_seconds = time.perf_counter() - start
            initial_bytes = rolling.storage_nbytes()
            for added_days in (1, 2):
                end_ds = dates[-4 + added_days]
                start = time.perf_counter()
                assert rolling.roll_forward(end_ds, 5)
                roll_seconds = time.perf_counter() - start
                start = time.perf_counter()
                cold = ComboTrainDataset(
                    loader, end_ds, 5, ts_days=3,
                    validinsts=validinsts, codec=loader.codec,
                )
                rebuild_seconds = time.perf_counter() - start
                assert rolling.storage_nbytes() == initial_bytes == cold.storage_nbytes()
                for idx in range(len(rolling)):
                    actual, expected = rolling[idx], cold[idx]
                    assert actual[:3] == expected[:3]
                    for left, right in zip(actual[3:], expected[3:]):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                assert not hasattr(loader.registry, "processed_cache")
                assert not any(loader.registry.working_cache.values())
                results.append({
                    "codec": compression,
                    "added_days": added_days,
                    "initial_seconds": round(initial_seconds, 4),
                    "roll_seconds": round(roll_seconds, 4),
                    "rebuild_seconds": round(rebuild_seconds, 4),
                    "speedup": round(rebuild_seconds / roll_seconds, 2),
                    "dataset_mib": round(initial_bytes / 1024**2, 2),
                    "peak_rss_gib": round(rss_gib(), 3),
                    "parity": True,
                })
                del cold
            rolling.release_storage()
            del rolling, loader
            gc.collect()
    print(json.dumps({"kind": "dataset_roll", "results": results}, indent=2), flush=True)


def run_storage(days, factors):
    codes = len(IndexMask().code)
    dtype = torch.float16
    codec = build_codec("fp4", dtype)
    start = time.perf_counter()
    x, meta = codec.allocate((days, codes, factors), "cpu", dtype)
    sample_days = days - 8 + 1
    y = torch.zeros((sample_days, 1, codes), dtype=dtype)
    w = torch.zeros_like(y, dtype=torch.bool)
    measured_bytes = x.numel() * x.element_size() + y.numel() * y.element_size() + w.numel()
    result = {
        "kind": "dataset_storage_only",
        "days": days,
        "codes": codes,
        "factors": factors,
        "codec": "fp4",
        "logical_dtype": str(dtype),
        "storage_shape": meta.storage_shape,
        "x_gib": round(x.numel() / 1024**3, 3),
        "dataset_gib": round(measured_bytes / 1024**3, 3),
        "peak_rss_gib": round(rss_gib(), 3),
        "seconds": round(time.perf_counter() - start, 3),
        "below_32_gib": rss_gib() < 32,
    }
    assert result["below_32_gib"]
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("roll", "storage"))
    parser.add_argument("--days", type=int, default=2000)
    parser.add_argument("--factors", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "roll":
        run_roll()
    else:
        run_storage(args.days, args.factors)

