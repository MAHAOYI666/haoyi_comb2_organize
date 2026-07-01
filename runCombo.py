from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any


DEFAULT_COMB_TORCH_THREADS = 64
DEFAULT_COMB_TORCH_INTEROP_THREADS = 1
_EXPLICIT_THREAD_ENV = {
    name: os.environ.get(name)
    for name in (
        "COMB_TORCH_THREADS",
        "TORCH_NUM_THREADS",
        "OMP_NUM_THREADS",
        "COMB_TORCH_INTEROP_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
    )
}


def _install_thread_env_defaults():
    requested_threads = (
        _EXPLICIT_THREAD_ENV["COMB_TORCH_THREADS"]
        or _EXPLICIT_THREAD_ENV["TORCH_NUM_THREADS"]
        or _EXPLICIT_THREAD_ENV["OMP_NUM_THREADS"]
    )
    if not requested_threads:
        requested_threads = str(min(DEFAULT_COMB_TORCH_THREADS, os.cpu_count() or DEFAULT_COMB_TORCH_THREADS))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, requested_threads)


_install_thread_env_defaults()

import numpy as np
import pandas as pd
import torch

ORGANIZE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ORGANIZE_ROOT / "vendor"
organize_root_path = str(ORGANIZE_ROOT)
if organize_root_path not in sys.path:
    sys.path.insert(0, organize_root_path)
for local_package_root in (VENDOR_ROOT / "comb2", VENDOR_ROOT / "comb2-pcmaster", VENDOR_ROOT / "comb2-simbase"):
    local_package_path = str(local_package_root)
    if local_package_path not in sys.path:
        sys.path.insert(0, local_package_path)

from comb2 import ComboBase, LoaderConfig
from src.DataLoader import ComboDataLoader, ComboTrainDataset
from comb2_simbase import IndexMask, Memmaper2, fast
from comb2_simbase.config import NAN_DTYPE
from vendor.perf_monitor import PerfMonitor, print_progress


def _load_organize_config_module():
    config_path = ORGANIZE_ROOT / "config.py"
    if config_path.exists():
        config_spec = importlib.util.spec_from_file_location("comb2_organize_config", config_path)
        config_module = importlib.util.module_from_spec(config_spec)
        assert config_spec.loader is not None
        config_spec.loader.exec_module(config_module)
        return config_module
    return importlib.import_module("config")


organize_config_module = _load_organize_config_module()


def load_combo_base_class(combo_config: dict) -> type[ComboBase]:
    combo_base_path = combo_config["paths"].get("combo_base_path")
    if not combo_base_path:
        return ComboBase
    custom_path = Path(combo_base_path).expanduser().resolve()
    if not custom_path.exists():
        raise FileNotFoundError(f"combo base file not found: {custom_path}")
    spec = importlib.util.spec_from_file_location(f"comb2_research_combo_base_{custom_path.stem}", custom_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "ComboBase"):
        raise AttributeError(f"{custom_path} must define ComboBase")
    custom_cls = getattr(module, "ComboBase")
    if not isinstance(custom_cls, type) or not issubclass(custom_cls, ComboBase):
        raise TypeError(f"ComboBase in {custom_path} must inherit from comb2.ComboBase")
    return custom_cls


def _parse_positive_int(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _explicit_env_int(*names: str) -> int | None:
    for name in names:
        value = _EXPLICIT_THREAD_ENV.get(name)
        if value is None or value == "":
            continue
        return _parse_positive_int(name, value)
    return None


def configure_torch_threads(organize_config: dict):
    runtime = organize_config["combo"]["runtime"]
    intra_threads = _explicit_env_int("COMB_TORCH_THREADS", "TORCH_NUM_THREADS", "OMP_NUM_THREADS")
    interop_threads = _explicit_env_int("COMB_TORCH_INTEROP_THREADS", "TORCH_NUM_INTEROP_THREADS")
    if intra_threads is None:
        intra_threads = _parse_positive_int("combo.runtime.torch_threads", runtime.get("torch_threads", DEFAULT_COMB_TORCH_THREADS))
    if interop_threads is None:
        interop_threads = _parse_positive_int(
            "combo.runtime.torch_interop_threads",
            runtime.get("torch_interop_threads", DEFAULT_COMB_TORCH_INTEROP_THREADS),
        )
    if intra_threads is not None:
        torch.set_num_threads(intra_threads)
    if interop_threads is not None:
        torch.set_num_interop_threads(interop_threads)
    print(
        f"[THREADS] torch_num_threads={torch.get_num_threads()} "
        f"torch_num_interop_threads={torch.get_num_interop_threads()}"
    )


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
        loader_config = {key: value for key, value in config["loader"].items() if key in loader_fields}
        loader_config["verbose"] = bool(getattr(self, "verbose", False))
        self.loader_config = LoaderConfig(**loader_config)


def _print_metric_table(title: str, columns: list[tuple[str, str, int]]) -> None:
    header = " ".join(f"{name:<{width}}" for name, _, width in columns)
    row = " ".join(f"{value:<{width}}" for _, value, width in columns)
    print(title)
    print(header)
    print(row)


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
    _print_metric_table("[BACKTEST]", columns)


def print_live_metrics(meta: dict):
    columns = [
        ("date", str(meta["trade_ds"]), 10),
        ("pred_ds", str(meta["pred_ds"]), 10),
        ("model_dt", str(meta["checkpoint_model_dt"]), 10),
        ("finite", str(meta["finite_count"]), 10),
        ("nonzero", str(meta["nonzero_count"]), 10),
        ("nan", str(meta["nan_count"]), 10),
        ("mean", f"{meta['mean']:.6f}" if meta["mean"] is not None else "NA", 12),
        ("std", f"{meta['std']:.6f}" if meta["std"] is not None else "NA", 12),
    ]
    _print_metric_table("[LIVE]", columns)


BASE_UNIVERSE_MASK_PATH = "1d_StockMask2/StockMask2.BaseUnivMask"
TRADING_MASK_PATH = "1d_StockMask2/StockMask2.LimitMask"


def load_shifted_mask(mask_path: str, start_ds: int, end_ds: int, shift_n: int = -1) -> np.ndarray:
    mask = Memmaper2(mask_path).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:].values
    if shift_n == 0:
        return mask
    if shift_n != -1:
        raise ValueError(f"unsupported mask shift_n={shift_n}")

    if mask.shape[0] == 1:
        raise ValueError("need at least one array to concatenate")
    shifted = np.full_like(mask, np.nan, dtype=np.float64)
    if mask.shape[0] > 2:
        shifted[1:-1] = mask[2:]
    return shifted


def apply_mask(y: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    mask_tensor = torch.as_tensor(mask, dtype=y.dtype, device=y.device)
    return y * mask_tensor


def process_label(y, start_ds: int, end_ds: int, ashare_data_path: str):
    y = fast.purify(y)
    base_mask = load_shifted_mask(f"{ashare_data_path}/{BASE_UNIVERSE_MASK_PATH}", start_ds, end_ds, shift_n=-1)
    trading_mask = load_shifted_mask(f"{ashare_data_path}/{TRADING_MASK_PATH}", start_ds, end_ds, shift_n=-1)
    y = apply_mask(y, base_mask)
    y = apply_mask(y, trading_mask)
    return y


def get_backtest_label(ashare_data_path: str, period: str, start_ds: int, end_ds: int):
    label = Memmaper2(f"{ashare_data_path}/1d_DailyLabel/DailyLabel.vwap30_label{period}").load(
        start_ds=start_ds,
        end_ds=end_ds,
        df_type=True,
    ).dloc[:]
    return label.mask(np.isnan(label), NAN_DTYPE)


def calculate_alpha_ic(alpha: pd.DataFrame, ashare_data_path: str) -> pd.DataFrame:
    if alpha.index.nlevels > 1:
        alpha = alpha.reset_index("times", drop=True).sort_index()
    date_idx = alpha.index.astype(int)
    start_time = int(date_idx[0])
    end_time = int(date_idx[-1])
    alpha = alpha.reindex(index=date_idx)
    x = alpha.values

    label_1d = get_backtest_label(ashare_data_path, "1d", start_time, end_time).reindex(index=date_idx).values
    label_5d = get_backtest_label(ashare_data_path, "5d", start_time, end_time).reindex(index=date_idx).values

    x_masked = process_label(torch.tensor(x), start_time, end_time, ashare_data_path).numpy()
    label_1d_masked = process_label(torch.tensor(label_1d), start_time, end_time, ashare_data_path).numpy()
    label_5d_masked = process_label(torch.tensor(label_5d), start_time, end_time, ashare_data_path).numpy()

    ic_1d = fast.corr(x_masked, label_1d_masked, dim=-1, keepdims=True)
    ic_perc = fast.corr(fast.perc_long(x_masked), fast.rank(label_1d_masked, dim=-1), dim=-1, keepdims=True)
    ic_rank = fast.corr(fast.rank(x_masked, dim=-1), fast.rank(label_1d_masked, dim=-1), dim=-1, keepdims=True)
    ic_5d = fast.corr(x_masked, label_5d_masked, dim=-1, keepdims=True)
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
    from comb2_pcmaster import BacktestNode

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
        execution_price=backtest_config.get("execution_price", "vwap30"),
        drawdown_stop=float(backtest_config.get("drawdown_stop", 0.0)),
        cooldown_days=int(backtest_config.get("cooldown_days", 0)),
    )


class ExperimentRunner:
    def __init__(self, organize_config: dict, monitor: PerfMonitor, config_path: str | None = None):
        self.organize_config = organize_config
        self.combo_config = organize_config["combo"]
        self.monitor = monitor
        self.config_path = str(Path(config_path).expanduser().resolve()) if config_path else None
        self.live_mode = bool(self.combo_config["runtime"].get("livetrading", False))
        self.combo_base_cls = load_combo_base_class(self.combo_config)
        self.node: Node | None = None
        self.combo: ComboBase | None = None
        self.codes: pd.Index | None = None
        self.backtest: Any | None = None
        self.live_output_dir = Path(self.combo_config["paths"]["output_dir"]) / "live"

    def setup(self):
        self.node = Node(self.combo_config)
        self.node.monitor = self.monitor
        self.combo = self.combo_base_cls(self.node)
        if self.monitor.enabled:
            install_research_model_decorators(self.monitor, self.combo.research_model_cls)
        self.codes = pd.Index([str(code).zfill(6) for code in IndexMask().code], name="code")

        if self.live_mode:
            self.live_output_dir.mkdir(parents=True, exist_ok=True)
            return

        strategy_path = build_strategy_file(self.organize_config)
        backtest_node = build_backtest_node(strategy_path, self.organize_config)
        from comb2_pcmaster import DailyBacktest

        self.backtest = DailyBacktest(backtest_node)

    def dates(self):
        if self.live_mode:
            start_ds = int(self.organize_config["strategy"]["start_ds"])
            end_ds = int(self.organize_config["strategy"]["end_ds"])
            dates = [
                int(ds)
                for ds in sorted(self.combo.loader.mask.date)
                if start_ds <= int(ds) <= end_ds
            ]
            if not dates:
                raise ValueError(f"no live trading dates in range {start_ds}-{end_ds}; live mode does not auto-align dates")
            return dates
        return sorted(self.backtest.vwap_data.index)

    def alpha_convert(self, date_int: int):
        return self.node.alpha.detach().cpu().to(dtype=self.node.alpha.dtype).numpy()

    def backtest_step(self, date_int: int, alpha):
        return self.backtest.step(date_int, pd.Series(alpha, index=self.codes))

    def backtest_finalize(self):
        self.backtest.finalize()

    def alpha_analysis(self):
        if not self.combo_config["output"].get("enable_alpha_analysis", True):
            print("[IC] alpha analysis disabled by config")
            return
        dump_alpha_analysis(self.node, self.combo_config)

    def live_step(self, date_int: int, alpha) -> dict:
        if self.combo.model is None or int(self.combo.model_dt) < 0:
            raise RuntimeError(f"live mode failed to load checkpoint for trade date {date_int}")
        pred_ds = self.combo._prev_date(date_int)
        alpha_array = np.asarray(alpha, dtype=float)
        finite_mask = np.isfinite(alpha_array)
        valid = alpha_array[finite_mask]
        meta = {
            "mode": "livetrading",
            "config": self.config_path,
            "trade_ds": int(date_int),
            "pred_ds": int(pred_ds),
            "snaptime": str(self.combo.snaptime),
            "checkpoint_model_dt": int(self.combo.model_dt),
            "checkpoint_dir": str(Path(self.combo.modelDir) / str(self.combo.model_dt)) if self.combo.modelDir else None,
            "model_path": str(self.combo.model_path),
            "git_commit": get_git_commit(),
            "finite_count": int(finite_mask.sum()),
            "nonzero_count": int(np.count_nonzero(valid)) if valid.size else 0,
            "nan_count": int((~finite_mask).sum()),
            "mean": float(valid.mean()) if valid.size else None,
            "std": float(valid.std()) if valid.size else None,
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
        }
        if meta["finite_count"] == 0:
            raise RuntimeError(f"live mode produced no finite alpha for trade date {date_int}")
        if meta["nonzero_count"] == 0:
            raise RuntimeError(f"live mode produced all-zero finite alpha for trade date {date_int}")

        alpha_path = self.live_output_dir / f"alpha_{date_int}.csv"
        meta_path = self.live_output_dir / f"meta_{date_int}.json"
        pd.Series(alpha_array, index=self.codes, name="alpha").to_csv(alpha_path)
        meta["alpha_path"] = str(alpha_path)
        meta["meta_path"] = str(meta_path)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return meta


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ORGANIZE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="runCombo",
        description="Run a comb2 experiment from an XML config.",
        epilog="example: runCombo config.xml",
    )
    parser.add_argument("config", nargs="?", default=None, help="Path to XML experiment config")
    parser.add_argument("--config", dest="config_flag", type=str, default=None, help="Path to XML experiment config")
    return parser.parse_args()


def resolve_config_arg(args: argparse.Namespace, prog: str = "runCombo") -> str | None:
    config_path = args.config_flag or args.config
    if not config_path:
        print(f"{prog}: missing config file; pass config.xml or --config config.xml", file=sys.stderr)
        return None
    resolved = Path(config_path).expanduser()
    if not resolved.is_file():
        print(f"{prog}: config file not found: {config_path}", file=sys.stderr)
        return None
    return str(resolved)


def install_perf_decorators(monitor: PerfMonitor):
    monitor.patch_method(ExperimentRunner, "setup", "setup")
    monitor.patch_method(ExperimentRunner, "alpha_convert", "alpha_convert", date_arg="date_int")
    monitor.patch_method(ExperimentRunner, "backtest_step", "backtest_step", date_arg="date_int")
    monitor.patch_method(ExperimentRunner, "backtest_finalize", "backtest_finalize")
    monitor.patch_method(ExperimentRunner, "alpha_analysis", "alpha_analysis")
    monitor.patch_method(ComboBase, "Combine", "combine", date_arg="di")
    monitor.patch_method(ComboBase, "LoadCheckpointModel", "combo_load_checkpoint", date_arg="dt")
    monitor.patch_method(ComboBase, "GenComboPos", "combo_gen_pos", date_arg="ds")
    monitor.patch_method(ComboBase, "Train", "combo_train", date_arg="ds")
    monitor.patch_method(ComboBase, "SaveCheckpointModel", "combo_save_checkpoint", date_arg="dt")
    monitor.patch_method(ComboTrainDataset, "__init__", "detail_dataset_init", date_arg="end_ds")
    monitor.patch_method(ComboTrainDataset, "_build_validinsts", "detail_build_validinsts")
    monitor.patch_method(ComboTrainDataset, "__getitem__", "detail_dataset_getitem", date_arg="idx")
    monitor.patch_method(ComboDataLoader, "gen_feature", "detail_gen_feature", date_arg="ds")
    monitor.patch_method(ComboDataLoader, "gen_label", "detail_gen_label", date_arg="ds")
    monitor.patch_method(ComboDataLoader, "gen_valid_mask", "detail_gen_valid_mask", date_arg="ds")
    monitor.patch_method(ComboDataLoader, "gen_base_universe_mask", "detail_gen_base_universe_mask", date_arg="ds")


def install_research_model_decorators(monitor: PerfMonitor, research_model_cls: type):
    monitor.patch_method(research_model_cls, "_next_batch", "detail_dataloader_next")
    monitor.patch_method(research_model_cls, "_batch_to_device", "detail_batch_to_device")
    monitor.patch_method(research_model_cls, "_zero_grad", "detail_zero_grad")
    monitor.patch_method(research_model_cls, "_forward_batch", "detail_forward")
    monitor.patch_method(research_model_cls, "_compute_loss", "detail_loss")
    monitor.patch_method(research_model_cls, "_backward_loss", "detail_backward")
    monitor.patch_method(research_model_cls, "_clip_grad", "detail_clip_grad")
    monitor.patch_method(research_model_cls, "_optimizer_step", "detail_optimizer_step")
    monitor.patch_method(research_model_cls, "_loss_to_float", "detail_loss_to_cpu")


def main() -> int:
    args = parse_args()
    config_path = resolve_config_arg(args)
    if config_path is None:
        return 2
    organize_config = organize_config_module.load_config(config_path)
    configure_torch_threads(organize_config)
    monitor = PerfMonitor.from_config(organize_config)
    if monitor.enabled:
        install_perf_decorators(monitor)
    try:
        runner = ExperimentRunner(organize_config, monitor, config_path=config_path)
        runner.setup()

        dates = runner.dates()
        loop_start = time.perf_counter()
        combine_time = 0.0
        alpha_time = 0.0
        backtest_time = 0.0
        live_output_time = 0.0
        verbose = bool(monitor.config.verbose)
        for update_idx, date in enumerate(dates, start=1):
            date_int = int(date)
            section_start = time.perf_counter()
            runner.combo.Combine(date_int)
            combine_time += time.perf_counter() - section_start
            section_start = time.perf_counter()
            alpha = runner.alpha_convert(date_int)
            alpha_time += time.perf_counter() - section_start
            if runner.live_mode:
                section_start = time.perf_counter()
                meta = runner.live_step(date_int, alpha)
                live_output_time += time.perf_counter() - section_start
                print_live_metrics(meta)
                if verbose:
                    print_progress(
                        "Stage:runComboLive",
                        update_idx,
                        len(dates),
                        loop_start,
                        f"combine {combine_time:.2f}, alpha {alpha_time:.2f}, output {live_output_time:.2f}",
                        final=update_idx == len(dates),
                    )
                continue
            section_start = time.perf_counter()
            metrics = runner.backtest_step(date_int, alpha)
            backtest_time += time.perf_counter() - section_start
            print_daily_metrics(metrics)
            if verbose:
                print_progress(
                    "Stage:runCombo",
                    update_idx,
                    len(dates),
                    loop_start,
                    f"combine {combine_time:.2f}, alpha {alpha_time:.2f}, backtest {backtest_time:.2f}",
                    final=update_idx == len(dates),
                )

        if not runner.live_mode:
            runner.backtest_finalize()
            runner.alpha_analysis()
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
