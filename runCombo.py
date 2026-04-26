from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ORGANIZE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ORGANIZE_ROOT / "vendor"
for local_package_root in (VENDOR_ROOT / "comb2", VENDOR_ROOT / "comb2-pcmaster"):
    local_package_path = str(local_package_root)
    if local_package_path not in sys.path:
        sys.path.insert(0, local_package_path)

from comb2 import ComboBase, LoaderConfig
from comb2_pcmaster import BacktestNode, DailyBacktest
from factorsim import IndexMask, Memmaper2, fast, operator
from factorsim.config import NAN_DTYPE
from vendor.perf_monitor import PerfMonitor

organize_config_spec = importlib.util.spec_from_file_location("comb2_organize_config", ORGANIZE_ROOT / "config.py")
organize_config_module = importlib.util.module_from_spec(organize_config_spec)
assert organize_config_spec.loader is not None
organize_config_spec.loader.exec_module(organize_config_module)


class Node:
    def __init__(self, config: dict):
        instsz = len(IndexMask().code)
        self.alpha = torch.zeros(instsz, dtype=config["loader"]["dtype"])
        self.alpha_history: dict[int, torch.Tensor] = {}

        for section in ("paths", "runtime", "model", "output", "defaults"):
            for key, value in config[section].items():
                setattr(self, key, value)

        self.model_config = dict(config["model"])
        loader_fields = {field.name for field in fields(LoaderConfig)}
        self.loader_config = LoaderConfig(**{key: value for key, value in config["loader"].items() if key in loader_fields})


def print_daily_metrics(metrics: dict):

    columns = [
        ("date", str(metrics["date"]), 10),
        ("pnl", f"{metrics['pnl']:.2f}", 14),
        ("total_asset", f"{metrics['total_asset']:.2f}", 16),
        ("trade_cost", f"{metrics['trade_cost']:.2f}", 14),
        ("reserve_cash", f"{metrics['reserve_cash']:.2f}", 16),
        ("tvr", f"{metrics['tvr']:.4f}", 10),
        ("long_num", str(metrics["long_num"]), 10),
    ]
    header = " ".join(f"{name:<{width}}" for name, _, width in columns)
    row = " ".join(f"{value:<{width}}" for _, value, width in columns)
    print("[BACKTEST]")
    print(header)
    print(row)


def safe_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, (list, tuple)):
        return np.array(data)
    return data


def process_label(y, start_ds: int, end_ds: int, ashare_data_path: str):
    y = fast.purify(y)
    y = operator.baseUniMask(y, start_ds=start_ds, end_ds=end_ds, path=ashare_data_path, shift_n=-1)
    y = operator.trdMask(y, start_ds=start_ds, end_ds=end_ds, path=ashare_data_path, shift_n=-1)
    return y


def get_backtest_label(ashare_data_path: str, period: str, start_ds: int, end_ds: int):
    label = Memmaper2(f"{ashare_data_path}/1d_DailyLabel/DailyLabel.vwap30_label{period}").load(
        start_ds=start_ds,
        end_ds=end_ds,
        df_type=True,
    ).dloc[:]
    label[np.isnan(label)] = NAN_DTYPE
    return label


def calculate_alpha_ic(alpha: pd.DataFrame, ashare_data_path: str) -> pd.DataFrame:
    if alpha.index.nlevels > 1:
        alpha = alpha.reset_index("times", drop=True).sort_index()
    date_idx = alpha.index.astype(int)
    end_time = min(int(date_idx[-1]), 20240101)
    date_idx = date_idx[date_idx < end_time]
    start_time = int(date_idx[0])
    end_time = int(date_idx[-1])
    alpha = alpha.reindex(index=date_idx)
    x = alpha.values

    label_1d = get_backtest_label(ashare_data_path, "1d", start_time, end_time).reindex(index=date_idx).values
    label_5d = get_backtest_label(ashare_data_path, "5d", start_time, end_time).reindex(index=date_idx).values

    x_masked = process_label(torch.tensor(x), start_time, end_time, ashare_data_path).numpy()
    label_1d_masked = process_label(torch.tensor(label_1d), start_time, end_time, ashare_data_path).numpy()
    label_5d_masked = process_label(torch.tensor(label_5d), start_time, end_time, ashare_data_path).numpy()

    ic_1d = safe_to_numpy(fast.corr(x_masked, label_1d_masked, dim=-1, keepdims=True))
    ic_perc = safe_to_numpy(fast.corr(fast.perc_long(x_masked), fast.rank(label_1d_masked, dim=-1), dim=-1, keepdims=True))
    ic_rank = safe_to_numpy(fast.corr(fast.rank(x_masked, dim=-1), fast.rank(label_1d_masked, dim=-1), dim=-1, keepdims=True))
    ic_5d = safe_to_numpy(fast.corr(x_masked, label_5d_masked, dim=-1, keepdims=True))
    x_cov = (~np.isnan(x_masked) & ~np.isnan(label_1d_masked)).sum(axis=1).astype(float)
    label_cov = (~np.isnan(label_1d_masked)).sum(axis=1).astype(float)
    label_cov[label_cov == 0] = np.nan
    coverage = x_cov / label_cov
    daily_ic = np.concatenate([ic_1d, ic_5d, ic_rank, ic_perc, coverage[:, np.newaxis]], axis=-1)
    return pd.DataFrame(daily_ic, index=date_idx, columns=["ic", "5dic", "rankic", "percic", "coverage"])


def dump_alpha_analysis(node: Node, combo_config: dict):
    if not node.alpha_history:
        return
    output_dir = Path(combo_config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_history_path = Path(combo_config["output"]["alpha_history_path"])
    alpha_history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(node.alpha_history, alpha_history_path)

    codes = pd.Index([str(code).zfill(6) for code in IndexMask().code])
    alpha = pd.DataFrame(
        {date: value.detach().cpu().to(torch.float32).numpy() for date, value in node.alpha_history.items()},
        index=codes,
    ).T.sort_index()
    alpha.index = alpha.index.astype(int)
    alpha_path = output_dir / "alpha.parquet"
    alpha.to_parquet(alpha_path)

    ashare_data_path = combo_config["loader"]["ashare_data_path"]
    daily_ic = calculate_alpha_ic(alpha, ashare_data_path)
    daily_ic_path = output_dir / "daily_ic"
    daily_ic.to_csv(daily_ic_path, sep="\t", na_rep="NAN")
    print(f"[IC] alpha={alpha_path} daily_ic={daily_ic_path}")


def build_strategy_file(organize_config: dict) -> Path:
    return Path(organize_config["strategy"]["path"])


def build_backtest_node(strategy_path: Path, organize_config: dict) -> BacktestNode:
    strategy_config = organize_config["strategy"]
    backtest_config = organize_config["backtest"]
    output_path = Path(backtest_config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    return BacktestNode(
        start_ds=int(strategy_config["start_ds"]),
        end_ds=int(strategy_config["end_ds"]),
        output_path=str(output_path),
        strategy_path=str(strategy_path),
        strategy_class="AlphaStrategy",
        cash=float(backtest_config["cash"]),
        fee_rate=float(backtest_config["fee_rate"]),
        reserve_cash=float(backtest_config["reserve_cash"]),
        daily_metrics_file=backtest_config.get("daily_metrics_file", "daily_metrics.csv"),
        cache_path=organize_config["constants"]["cache_path"],
        verbose=bool(backtest_config["verbose"]),
        universe=backtest_config.get("universe", "base"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=None, help="Path to XML experiment config")
    parser.add_argument("--config", dest="config_flag", type=str, default=None, help="Path to XML experiment config")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config_flag or args.config
    organize_config = organize_config_module.load_config(config_path)
    monitor = PerfMonitor.from_config(organize_config)
    try:
        combo_config = organize_config["combo"]

        with monitor.section("setup"):
            node = Node(combo_config)
            node.monitor = monitor
            combo = ComboBase(node)
            codes = pd.Index([str(code).zfill(6) for code in IndexMask().code])

            strategy_path = build_strategy_file(organize_config)
            backtest_node = build_backtest_node(strategy_path, organize_config)
            backtest = DailyBacktest(backtest_node)

        for date in sorted(backtest.vwap_data.index):
            date_int = int(date)
            with monitor.section("combine", date=date_int):
                combo.Combine(date_int)
            with monitor.section("alpha_convert", date=date_int):
                alpha = node.alpha.detach().cpu().to(dtype=node.alpha.dtype).numpy()
            with monitor.section("backtest_step", date=date_int):
                metrics = backtest.step(date_int, pd.Series(alpha, index=codes))
            print_daily_metrics(metrics)

        with monitor.section("backtest_finalize"):
            backtest.finalize()
        with monitor.section("alpha_analysis"):
            dump_alpha_analysis(node, combo_config)
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
