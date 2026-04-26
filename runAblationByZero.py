#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

ORGANIZE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ORGANIZE_ROOT / "vendor"
for local_package_root in (VENDOR_ROOT / "comb2", VENDOR_ROOT / "comb2-pcmaster"):
    local_package_path = str(local_package_root)
    if local_package_path not in sys.path:
        sys.path.insert(0, local_package_path)

from comb2 import ComboBase
from comb2_pcmaster import DailyBacktest
from factorsim import IndexMask
from runCombo import Node, build_backtest_node, build_strategy_file, dump_alpha_analysis, print_daily_metrics
from src.DataLoader import nan_to_num
from vendor.perf_monitor import PerfMonitor

organize_config_spec = importlib.util.spec_from_file_location("comb2_organize_config", ORGANIZE_ROOT / "config.py")
organize_config_module = importlib.util.module_from_spec(organize_config_spec)
assert organize_config_spec.loader is not None
organize_config_spec.loader.exec_module(organize_config_module)


class AblationCombo(ComboBase):
    def __init__(self, node, features: list[int] | None = None, train_enabled: bool = False, batch_size: int = 16):
        super().__init__(node)
        self.train_enabled = train_enabled
        self.ablation_batch_size = max(1, int(batch_size))
        feature_count = self.loader.num_features
        selected_features = list(range(feature_count)) if features is None else features
        for feature_idx in selected_features:
            if feature_idx < 0 or feature_idx >= feature_count:
                raise ValueError(f"feature index out of range: {feature_idx}, num_features={feature_count}")

        self.variant_features = {"baseline": None}
        for feature_idx in selected_features:
            self.variant_features[self._feature_variant_name(feature_idx)] = feature_idx
        self.variant_names = list(self.variant_features)
        self.variant_alphas: dict[str, torch.Tensor] = {}
        self.variant_histories: dict[str, dict[int, torch.Tensor]] = {name: {} for name in self.variant_names}

    def needTrain(self, ds: int) -> bool:
        return self.train_enabled and super().needTrain(ds)

    def _feature_variant_name(self, feature_idx: int) -> str:
        factor_paths = list(self.loader.config.factor_paths)
        if feature_idx < len(factor_paths):
            feature_name = Path(factor_paths[feature_idx]).name
        else:
            feature_name = f"feature_{feature_idx:03d}"
        feature_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature_name).strip("._-")
        if not feature_name:
            feature_name = f"feature_{feature_idx:03d}"
        return f"zero_{feature_idx:03d}_{feature_name}"

    def _empty_alpha(self) -> torch.Tensor:
        return torch.zeros_like(self.node.alpha.detach().cpu())

    def _store_variant_alpha(self, name: str, ds: int, alpha: torch.Tensor):
        stored = alpha.detach().cpu().clone()
        self.variant_alphas[name] = stored
        self.variant_histories[name][int(ds)] = stored.clone()

    def _store_empty_variants(self, ds: int):
        for name in self.variant_names:
            self._store_variant_alpha(name, ds, self._empty_alpha())
        self.variant_alphas["baseline"] = self.node.alpha.detach().cpu().clone()

    def _predict_alpha(self, feature_window: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        cur_pred = self._predict_with_refill(self.model, feature_window)
        if self.oldModel is not None:
            old_pred = self._predict_with_refill(self.oldModel, feature_window)
            cur_pred = cur_pred * self.model_smooth_rate + old_pred * (1 - self.model_smooth_rate)
        alpha = cur_pred.to(self.loader.dtype).clone()
        alpha[~valid_mask] = torch.nan
        return alpha

    def _can_predict_batch(self, model) -> bool:
        return bool(hasattr(model, "model") and getattr(model, "model") is not None and self._model_trainii(model) is not None)

    @torch.no_grad()
    def _predict_batch_with_refill(self, model, windows: torch.Tensor) -> torch.Tensor:
        if not self._can_predict_batch(model):
            return torch.stack([self._predict_with_refill(model, window) for window in windows], dim=0)
        trainii = self._model_trainii(model)
        assert trainii is not None
        model.model.eval()
        x = windows[:, :, trainii].to(model.device, dtype=torch.float32)
        pred = model.model(x).detach().cpu().to(dtype=self.loader.dtype)
        full_pred = torch.full((windows.shape[0], len(self.loader.mask.code)), torch.nan, dtype=self.loader.dtype)
        full_pred[:, trainii] = pred
        return full_pred

    def _store_feature_batch(self, feature_window: torch.Tensor, valid_mask: torch.Tensor, items: list[tuple[str, int | None]]):
        if not items:
            return
        windows = []
        for _, feature_idx in items:
            window = feature_window if feature_idx is None else feature_window.clone()
            if feature_idx is not None:
                window[:, :, feature_idx] = 0
            windows.append(window)
        batch = torch.stack(windows, dim=0)
        pred = self._predict_batch_with_refill(self.model, batch)
        if self.oldModel is not None:
            old_pred = self._predict_batch_with_refill(self.oldModel, batch)
            pred = pred * self.model_smooth_rate + old_pred * (1 - self.model_smooth_rate)
        pred = pred.to(self.loader.dtype)
        pred[:, ~valid_mask] = torch.nan
        for row, (name, _) in enumerate(items):
            self._store_variant_alpha(name, self.current_gen_ds, pred[row])

    def GenComboPos(self, ds: int):
        self.variant_alphas = {}
        if self.model is None:
            self._clear_alpha()
            self._record_alpha(ds)
            self._store_empty_variants(ds)
            self._log_alpha(ds, "no_model")
            return None
        if not self._buffer_ready(ds):
            self._clear_alpha()
            self._record_alpha(ds)
            self._store_empty_variants(ds)
            self._log_alpha(ds, "buffer_warmup")
            return None

        self.buffer_load(ds)
        end_didx = self.loader.date2didx(ds)
        didx_list = [end_didx - (self.tsDays - 1) + i for i in range(self.tsDays)]
        feature_window = nan_to_num(self.buffer.get(didx_list), 0.0).to(self.loader.dtype)
        valid_mask = self.loader.gen_valid_mask(ds)

        self.current_gen_ds = ds
        items = list(self.variant_features.items())
        for start in range(0, len(items), self.ablation_batch_size):
            self._store_feature_batch(feature_window, valid_mask, items[start : start + self.ablation_batch_size])

        baseline_alpha = self.variant_alphas["baseline"]
        self.node.alpha[:] = baseline_alpha
        self._record_alpha(ds)

        self._log_alpha(ds, "predict")
        return self.node.alpha


def parse_features(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Combo flow with one zero-ablation output per feature.")
    parser.add_argument("config", nargs="?", default=None, help="Path to XML experiment config")
    parser.add_argument("--config", dest="config_flag", type=str, default=None, help="Path to XML experiment config")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature indices to ablate")
    parser.add_argument("--output-suffix", default="ablation_by_zero", help="Subdirectory under combo output_dir for ablation outputs")
    parser.add_argument("--ablation-batch-size", type=int, default=4, help="Number of baseline/feature variants to predict per forward batch")
    parser.add_argument("--feature-group-size", type=int, default=20, help="Number of features to run per full Combo/backtest pass")
    parser.add_argument("--train", action="store_true", help="Allow training and checkpoint saving during the ablation run")
    return parser.parse_args()


def variant_config(base_config: dict, variant_dir: Path) -> dict:
    config = copy.deepcopy(base_config)
    config["combo"]["paths"]["output_dir"] = str(variant_dir)
    config["combo"]["output"]["alpha_history_path"] = str(variant_dir / "alpha_history.pt")
    config["backtest"]["output_path"] = str(variant_dir / "backtest")
    monitor_config = config.get("monitor")
    if monitor_config and monitor_config.get("output_path"):
        monitor_config["output_path"] = str(variant_dir / "perf_metrics.csv")
    return config


def feature_groups(features: list[int], group_size: int) -> list[list[int]]:
    group_size = max(1, int(group_size))
    return [features[start : start + group_size] for start in range(0, len(features), group_size)]


def run_group(
    organize_config: dict,
    features: list[int],
    output_suffix: str,
    ablation_batch_size: int,
    train_enabled: bool,
    monitor: PerfMonitor,
    include_baseline_output: bool,
):
    combo_config = organize_config["combo"]
    with monitor.section("setup"):
        node = Node(combo_config)
        node.monitor = monitor
        combo = AblationCombo(
            node,
            features=features,
            train_enabled=train_enabled,
            batch_size=ablation_batch_size,
        )
        codes = pd.Index([str(code).zfill(6) for code in IndexMask().code])

        base_output_dir = Path(combo_config["paths"]["output_dir"])
        ablation_root = base_output_dir / output_suffix
        variant_configs = {
            name: variant_config(organize_config, ablation_root / name)
            for name in combo.variant_names
            if include_baseline_output or name != "baseline"
        }

        strategy_path = build_strategy_file()
        backtests = {
            name: DailyBacktest(build_backtest_node(strategy_path, config))
            for name, config in variant_configs.items()
        }
        if include_baseline_output:
            loop_backtest = backtests["baseline"]
        else:
            loop_backtest = DailyBacktest(build_backtest_node(strategy_path, variant_config(organize_config, ablation_root / "baseline_probe")))

    for date in sorted(loop_backtest.vwap_data.index):
        date_int = int(date)
        with monitor.section("combine", date=date_int):
            combo.Combine(date_int)
        with monitor.section("backtest_step", date=date_int):
            baseline_metrics = None
            for name, backtest in backtests.items():
                alpha = combo.variant_alphas[name].to(dtype=node.alpha.dtype).numpy()
                metrics = backtest.step(date_int, pd.Series(alpha, index=codes))
                if name == "baseline":
                    baseline_metrics = metrics
        if baseline_metrics is not None:
            print_daily_metrics(baseline_metrics)

    with monitor.section("backtest_finalize"):
        for backtest in backtests.values():
            backtest.finalize()
    with monitor.section("alpha_analysis"):
        for name, config in variant_configs.items():
            dump_alpha_analysis(SimpleNamespace(alpha_history=combo.variant_histories[name]), config["combo"])


def main():
    args = parse_args()
    config_path = args.config_flag or args.config
    organize_config = organize_config_module.load_config(config_path)
    monitor = PerfMonitor.from_config(organize_config)

    try:
        probe_node = Node(organize_config["combo"])
        probe_combo = AblationCombo(probe_node, features=[], batch_size=1)
        selected_features = parse_features(args.features)
        if selected_features is None:
            selected_features = list(range(probe_combo.loader.num_features))
        for group_idx, group_features in enumerate(feature_groups(selected_features, args.feature_group_size), start=1):
            print(f"[ABLATION] group={group_idx} features={group_features[0]}-{group_features[-1]} count={len(group_features)}")
            run_group(
                organize_config=organize_config,
                features=group_features,
                output_suffix=args.output_suffix,
                ablation_batch_size=args.ablation_batch_size,
                train_enabled=args.train,
                monitor=monitor,
                include_baseline_output=group_idx == 1,
            )
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
