"""Replay one saved alpha through DailyBacktest from two source trees; run on a KF worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time


def use_tree(tree: Path):
    for dependency in ("vendor/comb2-pcmaster", "vendor/comb2-simbase", "vendor/comb2", "."):
        sys.path.insert(0, str(tree / dependency))


def load_organize_config(config_path: str, output_path: str | None = None):
    import config as organize_config_module

    organize = organize_config_module.load_config(config_path)
    if output_path is not None:
        organize["backtest"]["output_path"] = output_path
    return organize


def prepare(args):
    use_tree(Path(args.tree).resolve())
    import pandas as pd
    import torch
    from comb2 import LoaderConfig
    from comb2_simbase import IndexMask

    organize = load_organize_config(args.config)
    combo = organize["combo"]
    spec = importlib.util.spec_from_file_location("backtest_ab_model", combo["paths"]["research_loader_path"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    loader = module.ResearchLoader(LoaderConfig(
        cache_path=organize["constants"]["cache_path"],
        data_start_ds=int(combo["loader"]["data_start_ds"]),
        dtype=torch.float32,
        sample_times=tuple(combo["runtime"]["sample_times"]),
    ))
    source, field = organize["backtest"]["execution_price"].split(":")
    alpha = pd.read_parquet(args.alpha)
    codes = [str(code).zfill(6) for code in IndexMask().code]
    rows = []
    for date, ti in alpha.index:
        loader.set_current_ti(int(ti))
        rows.append(loader.source_field(source, field, int(date), int(date))[0].cpu().numpy().astype(float))
    prices = pd.DataFrame(rows, index=alpha.index, columns=codes)
    prices.to_parquet(args.prices)
    print(json.dumps({"phase": "prepared", "rows": len(prices), "prices": args.prices}), flush=True)


def run(args):
    use_tree(Path(args.tree).resolve())
    import pandas as pd
    from comb2_pcmaster import DailyBacktest
    from runCombo import build_backtest_node, build_strategy_file

    organize = load_organize_config(args.config, args.out)
    alpha = pd.read_parquet(args.alpha)
    prices = pd.read_parquet(args.prices)
    alpha.columns = [str(code) for code in alpha.columns]
    times = sorted({int(ti) for _, ti in alpha.index})

    began = time.perf_counter()
    backtest = DailyBacktest(build_backtest_node(build_strategy_file(organize), organize))
    init_seconds = time.perf_counter() - began
    began = time.perf_counter()
    for (date, ti), row in alpha.iterrows():
        backtest.step(
            int(date), row, ti=int(ti), prices=prices.loc[(date, ti)],
            last=int(ti) == times[-1],
        )
    step_seconds = time.perf_counter() - began
    began = time.perf_counter()
    backtest.finalize()
    finalize_seconds = time.perf_counter() - began
    print(json.dumps({
        "phase": "backtest", "tree": args.tree, "steps": len(alpha),
        "init_s": round(init_seconds, 2), "step_s": round(step_seconds, 2),
        "finalize_s": round(finalize_seconds, 2),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
    }), flush=True)


def compare(args):
    import numpy as np
    import pandas as pd

    left, right = Path(args.left), Path(args.right)
    names = sorted({p.relative_to(left).as_posix() for p in left.rglob("*") if p.is_file()}
                   | {p.relative_to(right).as_posix() for p in right.rglob("*") if p.is_file()})
    report = []
    for name in names:
        a, b = left / name, right / name
        if not a.exists() or not b.exists():
            report.append({"file": name, "status": "missing", "left": a.exists(), "right": b.exists()})
            continue
        if a.read_bytes() == b.read_bytes():
            report.append({"file": name, "status": "identical"})
            continue
        if name.endswith(".csv"):
            fa, fb = pd.read_csv(a), pd.read_csv(b)
            same_shape = fa.shape == fb.shape and list(fa.columns) == list(fb.columns)
            entry = {"file": name, "status": "csv_differs", "same_shape": same_shape}
            if same_shape:
                numeric = fa.select_dtypes("number").columns
                diff = (fa[numeric] - fb[numeric]).abs().to_numpy()
                entry["max_abs_diff"] = float(np.nanmax(diff)) if diff.size else 0.0
                entry["non_numeric_equal"] = fa.drop(columns=numeric).equals(fb.drop(columns=numeric))
            report.append(entry)
            continue
        report.append({
            "file": name, "status": "bytes_differ",
            "left_sha": hashlib.sha256(a.read_bytes()).hexdigest()[:12],
            "right_sha": hashlib.sha256(b.read_bytes()).hexdigest()[:12],
        })
    print(json.dumps({"phase": "compare", "files": report}, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tree", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--alpha", required=True)
    p.add_argument("--prices", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tree", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--alpha", required=True)
    p.add_argument("--prices", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
