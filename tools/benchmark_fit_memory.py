"""Worker-only 500-day Dataset and representative two-batch GPU fit memory probe."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
for dependency in ("vendor/comb2", "vendor/comb2-simbase", "vendor/comb2-pcmaster", "."):
    sys.path.insert(0, str(ROOT / dependency))

import torch
from comb2 import ComboTrainDataset, LoaderConfig


def rss_gib():
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1024**3


class LimitedDataset:
    def __init__(self, source, size):
        self.source = source
        self.validinsts = source.validinsts
        self.size = min(size, len(source))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.source[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-epoch", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), "GPU worker required"
    model_path = Path("/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/Model.py")
    spec = importlib.util.spec_from_file_location("comb2_fit_memory_model", model_path)
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
    stop = threading.Event()
    samples = {"peak": rss_gib()}

    def poll():
        while not stop.wait(0.05):
            samples["peak"] = max(samples["peak"], rss_gib())

    monitor = threading.Thread(target=poll, daemon=True)
    monitor.start()
    try:
        start = time.perf_counter()
        dataset = ComboTrainDataset(
            loader, 20200521, 500, ts_days=10, codec=loader.codec,
        )
        print(json.dumps({
            "phase": "dataset",
            "seconds": round(time.perf_counter() - start, 2),
            "features": loader.num_features,
            "instruments": dataset.numValidinsts,
            "storage_gib": round(dataset.storage_nbytes() / 1024**3, 3),
            "steady_rss_gib": round(rss_gib(), 3),
            "sampled_peak_rss_gib": round(samples["peak"], 3),
        }), flush=True)
        assert not any(loader.registry.working_cache.values())
        model = module.ResearchModel({
            "dtype": torch.float16,
            "tsDays": 10,
            "num_features": loader.num_features,
            "sample_times": (93000,),
            "device": "cuda",
            "dropout": 0.3,
            "lr": 2e-6,
            "epochs": 1,
            "batch_size": 5,
            "seed": 42,
        })
        train_dataset = dataset if args.full_epoch else LimitedDataset(dataset, 10)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.fit(train_dataset)
        torch.cuda.synchronize()
        print(json.dumps({
            "phase": "fit_full_epoch" if args.full_epoch else "fit_two_batches",
            "seconds": round(time.perf_counter() - start, 2),
            "steady_rss_gib": round(rss_gib(), 3),
            "sampled_peak_rss_gib": round(samples["peak"], 3),
            "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "gpu_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "full_epoch": args.full_epoch,
        }), flush=True)
        assert rss_gib() < 32 and samples["peak"] < 32
        del train_dataset, model
        dataset.release_storage()
        del dataset
        gc.collect()
        print(json.dumps({"phase": "released", "rss_gib": round(rss_gib(), 3)}), flush=True)
    finally:
        stop.set()
        monitor.join(timeout=1)


if __name__ == "__main__":
    main()
