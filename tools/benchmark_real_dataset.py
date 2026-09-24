"""Measure a real research Dataset on a KF worker; never run on the login host."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for dependency in ("vendor/comb2", "vendor/comb2-simbase", "vendor/comb2-pcmaster", "."):
    sys.path.insert(0, str(ROOT / dependency))

import torch
from comb2 import ComboTrainDataset, LoaderConfig


def rss_gib():
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1024**3


def peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def compare_samples(left, right, pair, index):
    report = []
    assert left[:3] == right[:3]
    for name, actual, expected in zip(("X", "Y", "W"), left[3:], right[3:]):
        if actual.dtype == torch.bool:
            differences = int((actual != expected).sum())
            max_abs = float(differences > 0)
            within_fp16_tolerance = differences == 0
        else:
            same = (actual == expected) | (
                torch.isnan(actual) & torch.isnan(expected)
            )
            differences = int((~same).sum())
            finite = torch.isfinite(actual) & torch.isfinite(expected)
            max_abs = float((actual[finite].float() - expected[finite].float()).abs().max()) if finite.any() else 0.0
            within_fp16_tolerance = bool(torch.isclose(
                actual, expected, rtol=0.005, atol=0.0002, equal_nan=True,
            ).all())
        report.append({
            "pair": pair, "sample_index": index, "tensor": name,
            "different_elements": differences, "max_abs": max_abs,
            "within_fp16_tolerance": within_fp16_tolerance,
        })
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--end-ds", type=int, default=20200519)
    parser.add_argument("--next-ds", type=int)
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    model_path = Path("/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/Model.py")
    spec = importlib.util.spec_from_file_location("comb2_real_benchmark_model", model_path)
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
    end_ds = loader.align_date(args.end_ds)
    next_ds = (
        loader.didx2date(loader.date2didx(end_ds) + 1)
        if args.next_ds is None else loader.align_date(args.next_ds)
    )
    ndays = args.days
    print(json.dumps({
        "phase": "start", "end_ds": end_ds, "next_ds": next_ds,
        "ndays": ndays, "sources": len(loader.input_names),
        "rss_gib": round(rss_gib(), 3),
    }), flush=True)
    began = time.perf_counter()
    dataset = ComboTrainDataset(
        loader, end_ds, ndays, ts_days=10, codec=loader.codec,
    )
    build_seconds = time.perf_counter() - began
    storage_gib = dataset.storage_nbytes() / 1024**3
    steady_gib = rss_gib()
    assert not any(loader.registry.working_cache.values())
    print(json.dumps({
        "phase": "built", "seconds": round(build_seconds, 2),
        "actual_days": dataset.ndays,
        "instruments": dataset.numValidinsts,
        "features": loader.num_features,
        "storage_gib": round(storage_gib, 3),
        "steady_rss_gib": round(steady_gib, 3),
        "peak_rss_gib": round(peak_gib(), 3),
    }), flush=True)
    assert steady_gib < 32, "Dataset steady RSS exceeded 32 GiB"
    began = time.perf_counter()
    rolled = dataset.roll_forward(next_ds, ndays)
    roll_seconds = time.perf_counter() - began
    if rolled:
        assert not any(loader.registry.working_cache.values())
    print(json.dumps({
        "phase": "roll", "rolled": rolled, "seconds": round(roll_seconds, 2),
        "instruments": dataset.numValidinsts,
        "storage_gib": round(dataset.storage_nbytes() / 1024**3, 3),
        "steady_rss_gib": round(rss_gib(), 3),
        "peak_rss_gib": round(peak_gib(), 3),
    }), flush=True)
    if not rolled:
        dataset.release_storage()
        del dataset
        gc.collect()
        before_rebuild_gib = rss_gib()
        began = time.perf_counter()
        replacement = ComboTrainDataset(
            loader, next_ds, ndays, ts_days=10, codec=loader.codec,
        )
        print(json.dumps({
            "phase": "fallback_rebuild",
            "seconds": round(time.perf_counter() - began, 2),
            "rss_after_release_gib": round(before_rebuild_gib, 3),
            "steady_rss_gib": round(rss_gib(), 3),
            "peak_rss_gib": round(peak_gib(), 3),
            "instruments": replacement.numValidinsts,
        }), flush=True)
        assert rss_gib() < 32
        replacement.release_storage()
        return
    if rolled and not args.no_rebuild:
        began = time.perf_counter()
        cold = ComboTrainDataset(
            loader, next_ds, ndays, ts_days=10, codec=loader.codec,
        )
        rebuild_seconds = time.perf_counter() - began
        assert torch.equal(dataset.validinsts, cold.validinsts)
        x_storage_equal = all(
            torch.equal(dataset.X[ti], cold.X[ti])
            for ti in loader.sample_times
        )
        w_storage_equal = torch.equal(dataset.W, cold.W)
        y_same = (dataset.Y == cold.Y) | (
            torch.isnan(dataset.Y) & torch.isnan(cold.Y)
        )
        y_different = int((~y_same).sum())
        y_within_tolerance = bool(torch.isclose(
            dataset.Y, cold.Y, rtol=0.005, atol=0.0002, equal_nan=True,
        ).all())
        print(json.dumps({
            "phase": "full_storage_parity",
            "x_encoded_exact": x_storage_equal,
            "w_exact": w_storage_equal,
            "y_exact_differences": y_different,
            "y_total_elements": dataset.Y.numel(),
            "y_within_fp16_tolerance": y_within_tolerance,
        }), flush=True)
        assert x_storage_equal and w_storage_equal and y_within_tolerance
        indices = (0, len(dataset) // 2, len(dataset) - 1)
        comparisons = []
        cold_samples = []
        for index in indices:
            actual, expected = dataset[index], cold[index]
            comparisons.extend(compare_samples(actual, expected, "roll_vs_cold", index))
            cold_samples.append(tuple(
                item.clone() if isinstance(item, torch.Tensor) else item
                for item in expected
            ))
        print(json.dumps({
            "phase": "roll_vs_cold", "rebuild_seconds": round(rebuild_seconds, 2),
            "speedup": round(rebuild_seconds / roll_seconds, 2),
            "peak_rss_gib": round(peak_gib(), 3),
            "comparisons": comparisons,
        }), flush=True)
        cold.release_storage()
        del cold
        gc.collect()
        began = time.perf_counter()
        cold_again = ComboTrainDataset(
            loader, next_ds, ndays, ts_days=10, codec=loader.codec,
        )
        cold_comparisons = []
        for index, original in zip(indices, cold_samples):
            cold_comparisons.extend(compare_samples(
                original, cold_again[index], "cold_vs_cold", index,
            ))
        print(json.dumps({
            "phase": "cold_reproducibility",
            "seconds": round(time.perf_counter() - began, 2),
            "peak_rss_gib": round(peak_gib(), 3),
            "comparisons": cold_comparisons,
        }), flush=True)
        cold_again.release_storage()
        del cold_again
    dataset.release_storage()
    del dataset
    gc.collect()
    print(json.dumps({"phase": "released", "rss_gib": round(rss_gib(), 3)}), flush=True)


if __name__ == "__main__":
    main()
