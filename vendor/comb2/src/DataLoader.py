# haoyi 2026/05/5

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from typing import Iterable, Protocol, Sequence

import numpy as np
import torch
from factorsim import IndexMask, Memmaper2
from torch.utils.data import Dataset

from .op_utils import cs_zscore, nan_to_num, nanmedian, nanstd, normalize_by_max_abs, to_bool_mask, truncate, winsorize_by_quantile

MASK = IndexMask()


def _maybe_section(monitor, event: str, ds: int | None = None, *, level: str = "full"):
    if monitor is None:
        return nullcontext()
    return monitor.section(event, date=ds)


class FeatureSource(Protocol):
    feature_dim: int

    def load_day(self, ds: int) -> torch.Tensor:
        ...


class MemmapFeatureSource:
    def __init__(self, paths: Sequence[str], dtype: torch.dtype):
        self.paths = list(paths)
        self.dtype = dtype
        self.feature_dim = len(self.paths)
        self._cache: dict[str, Memmaper2] = {}

    def _mmap(self, path: str) -> Memmaper2:
        if path not in self._cache:
            self._cache[path] = Memmaper2(path)
        return self._cache[path]

    def load_day(self, ds: int) -> torch.Tensor:
        monitor = getattr(self, "monitor", None)
        values = []
        with _maybe_section(monitor, "mmap_feature.load", int(ds), level="full"):
            for path in self.paths:
                data = self._mmap(path).load(start_ds=int(ds), end_ds=int(ds))[:]
                values.append(torch.as_tensor(np.asarray(data)[0], dtype=self.dtype))
        with _maybe_section(monitor, "mmap_feature.stack", int(ds), level="full"):
            return torch.stack(values, dim=-1)


class EmptyCubeSource:
    def __init__(self, feature_dim: int = 0, dtype: torch.dtype | None = None):
        self.feature_dim = feature_dim
        self.dtype = dtype or torch.float16

    def load_day(self, ds: int) -> torch.Tensor:
        return torch.zeros((len(MASK.code), self.feature_dim), dtype=self.dtype)


class MemmapLabelSource:
    def __init__(self, path: str, dtype: torch.dtype):
        self.path = path
        self.dtype = dtype
        self._mmap = Memmaper2(path)

    def load_day(self, ds: int) -> torch.Tensor:
        monitor = getattr(self, "monitor", None)
        with _maybe_section(monitor, "mmap_label.load", int(ds), level="full"):
            data = self._mmap.load(start_ds=int(ds), end_ds=int(ds))[:]
            label = torch.as_tensor(np.asarray(data)[0], dtype=self.dtype)
            return label


class MemmapMaskSource:
    def __init__(self, path: str | None):
        self.path = path
        self._mmap = None
        if path and os.path.exists(path):
            self._mmap = Memmaper2(path)

    def load_day(self, ds: int) -> torch.Tensor:
        if self._mmap is None:
            return torch.ones(len(MASK.code), dtype=torch.bool)
        monitor = getattr(self, "monitor", None)
        with _maybe_section(monitor, "mmap_mask.load", int(ds), level="full"):
            data = self._mmap.load(start_ds=int(ds), end_ds=int(ds))[:]
            mask = torch.as_tensor(np.asarray(data)[0])
            return to_bool_mask(mask)


@dataclass
class LoaderConfig:
    factor_paths: Sequence[str]
    label_path: str
    dtype: torch.dtype
    data_start_ds: int
    valid_path: str
    filtered_path: str
    base_universe_path: str | None = None


class ComboDataLoader:
    def __init__(self, config: LoaderConfig, cube_source: FeatureSource | None = None, feature_cache_size: int = 2500, label_cache_size: int = 2500):
        self.config = config
        self.dtype = self.config.dtype
        self.mask = MASK
        self.data_start_ds = int(self.config.data_start_ds)
        self.data_start_didx = int(self.mask.date2didx(self.data_start_ds))
        self.factor_source = MemmapFeatureSource(self.config.factor_paths, dtype=self.dtype)
        self.cube_source = cube_source or EmptyCubeSource(dtype=self.dtype)
        self.label_source = MemmapLabelSource(self.config.label_path, dtype=self.dtype)
        self.valid_source = MemmapMaskSource(self.config.valid_path)
        self.filtered_source = MemmapMaskSource(self.config.filtered_path)
        self.base_universe_source = MemmapMaskSource(self.config.base_universe_path)
        self.monitor = None
        self.num_features = self.factor_source.feature_dim + self.cube_source.feature_dim
        self._feature_cache: dict[int, torch.Tensor] = {}
        self._feature_cache_order: list[int] = []
        self._feature_cache_size = int(feature_cache_size)
        self._label_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._label_cache_order: list[tuple[int, int]] = []
        self._label_cache_size = int(label_cache_size)

    def _sync_monitor_refs(self):
        self.factor_source.monitor = self.monitor
        self.label_source.monitor = self.monitor
        self.valid_source.monitor = self.monitor
        self.filtered_source.monitor = self.monitor
        self.base_universe_source.monitor = self.monitor

    def date2didx(self, ds: int) -> int:
        didx = int(self.mask.date2didx(int(ds)))
        return max(didx, self.data_start_didx)

    def didx2date(self, didx: int) -> int:
        didx = max(int(didx), self.data_start_didx)
        return int(self.mask.date[didx])

    def align_date(self, ds: int) -> int:
        aligned = self.didx2date(self.date2didx(ds))
        return max(aligned, self.data_start_ds)


    def _cache_get(self, cache: dict, order: list, key):
        if key not in cache:
            return None
        order.remove(key)
        order.append(key)
        return cache[key]

    def _cache_set(self, cache: dict, order: list, key, value, max_size: int):
        if max_size <= 0:
            return
        if key in cache:
            order.remove(key)
        cache[key] = value
        order.append(key)
        while len(order) > max_size:
            stale_key = order.pop(0)
            cache.pop(stale_key, None)

    def gen_feature(self, ds: int) -> torch.Tensor:
        ds = self.align_date(ds)
        cached = self._cache_get(self._feature_cache, self._feature_cache_order, ds)
        if cached is not None:
            return cached

        self._sync_monitor_refs()
        with _maybe_section(self.monitor, "gen_feature.factor_load", ds, level="full"):
            factor = self.factor_source.load_day(ds).to(torch.float32)
        with _maybe_section(self.monitor, "gen_feature.cube_load", ds, level="full"):
            cube = self.cube_source.load_day(ds).to(torch.float32)
        if cube.shape[1] == 0:
            feature = factor
        else:
            feature = torch.cat([factor, cube], dim=-1)
        feature[torch.isinf(feature)] = torch.nan
        with _maybe_section(self.monitor, "gen_feature.cs_zscore", ds, level="full"):
            feature = cs_zscore(feature.transpose(0, 1)).transpose(0, 1)
        with _maybe_section(self.monitor, "gen_feature.truncate_nan_to_num", ds, level="full"):
            feature = truncate(feature, -4.0, 4.0)
            feature = nan_to_num(feature, 0.0)

        feature = feature.to(self.dtype)
        self._cache_set(self._feature_cache, self._feature_cache_order, ds, feature, self._feature_cache_size)
        return feature

    def gen_base_universe_mask(self, ds: int) -> torch.Tensor:
        ds = self.align_date(ds)
        return self.base_universe_source.load_day(ds)

    def gen_valid_mask(self, ds: int) -> torch.Tensor:
        ds = self.align_date(ds)
        valid = self.valid_source.load_day(ds)
        filtered = self.filtered_source.load_day(ds)
        return valid & filtered & self.gen_base_universe_mask(ds)

    def gen_label(self, ds: int, ret_days: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        ds = self.align_date(ds)
        cache_key = (ds, int(ret_days))
        cached = self._cache_get(self._label_cache, self._label_cache_order, cache_key)
        if cached is not None:
            return cached

        end_didx = self.date2didx(ds)
        start_didx = end_didx - ret_days + 1
        if start_didx < self.data_start_didx:
            raise ValueError(f"not enough label history for ds={ds}, ret_days={ret_days}")

        dates = [self.didx2date(didx) for didx in range(start_didx, end_didx + 1)]
        with _maybe_section(self.monitor, "gen_label.load_returns", ds, level="full"):
            returns = [self.label_source.load_day(cur_ds).to(torch.float32) for cur_ds in dates]
        with _maybe_section(self.monitor, "gen_label.aggregate", ds, level="full"):
            cret = torch.stack(returns, dim=0)
            cret = nan_to_num(cret, 0.0)
            decay_weights = torch.arange(ret_days, 0, -1, dtype=torch.float32, device=cret.device)
            cret = torch.tensordot(decay_weights, cret, dims=([0], [0]))
            cret[torch.isinf(cret)] = torch.nan

        with _maybe_section(self.monitor, "gen_label.valid_mask", ds, level="full"):
            valid_mask = self.gen_valid_mask(self.didx2date(start_didx)) & (~torch.isnan(cret))
        valid_values = cret[valid_mask]
        if valid_values.numel() > 0:
            valid_values = winsorize_by_quantile(valid_values, 0.01, 0.99)
            valid_values = valid_values - nanmedian(valid_values)
            valid_values = valid_values / (nanstd(valid_values) + 1e-8)
            valid_values = truncate(valid_values, -3.0, 3.0)
            valid_values = normalize_by_max_abs(valid_values)
            cret[valid_mask] = valid_values

        cret[~valid_mask] = 0.0
        cret = nan_to_num(cret, 0.0).to(self.dtype)
        label = (cret, to_bool_mask(valid_mask))
        self._cache_set(self._label_cache, self._label_cache_order, cache_key, label, self._label_cache_size)
        return label

    def _feature_available_mask(self, feature_window: torch.Tensor) -> torch.Tensor:
        per_day_available = torch.isnan(feature_window).sum(dim=-1) == 0
        return per_day_available.to(torch.float32).mean(dim=0) > 0

    def process_feature_window(self, feature_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        available_mask = self._feature_available_mask(feature_window)
        feature_window = nan_to_num(feature_window, 0.0).to(self.dtype)
        return feature_window[:, available_mask], available_mask

    def load_feature_window(self, end_ds: int, ts_days: int) -> tuple[torch.Tensor, torch.Tensor]:
        end_didx = self.date2didx(end_ds)
        start_didx = max(self.data_start_didx, end_didx - ts_days + 1)
        days = [self.didx2date(didx) for didx in range(start_didx, end_didx + 1)]
        features = [self.gen_feature(ds) for ds in days]
        if not features:
            raise ValueError("no feature data loaded")
        if len(features) < ts_days:
            pad = [torch.full_like(features[0], torch.nan) for _ in range(ts_days - len(features))]
            features = pad + features
        return self.process_feature_window(torch.stack(features, dim=0))


class ComboTrainDataset(Dataset):
    def __init__(self, loader: ComboDataLoader, end_ds: int, ndays: int, x_delay: int, step_size: int, validinsts: torch.Tensor | None = None):
        self.loader = loader
        self.end_ds = loader.align_date(end_ds)
        self.ndays = int(ndays)
        self.x_delay = int(x_delay)
        self.step_size = int(step_size)
        self.feat_size = loader.num_features
        self.end_didx = loader.date2didx(self.end_ds)
        self.start_didx = max(loader.data_start_didx + self.x_delay - 1, self.end_didx - self.ndays + 1)
        self.ndays = self.end_didx - self.start_didx + 1

        monitor = getattr(loader, "monitor", None)
        instsz = len(MASK.code)
        with _maybe_section(monitor, "dataset_init.build_validinsts", self.end_ds, level="full"):
            if validinsts is None:
                self.validinsts = self._build_validinsts()
            else:
                self.validinsts = validinsts.to(dtype=torch.long)
        self.numValidinsts = len(self.validinsts)
        if self.numValidinsts == 0:
            self.validinsts = torch.arange(instsz)
            self.numValidinsts = instsz
        with _maybe_section(monitor, "dataset_init.alloc_xyw", self.end_ds, level="full"):
            self.X = torch.zeros((self.ndays, self.numValidinsts, self.feat_size), dtype=loader.dtype)
            self.Y = torch.zeros((self.ndays, self.numValidinsts), dtype=loader.dtype)
            self.W = torch.zeros((self.ndays, self.numValidinsts), dtype=loader.dtype)

        with _maybe_section(monitor, "dataset_init.loop_total", self.end_ds, level="full"):
            for offset in range(self.ndays):
                label_didx = self.start_didx + offset
                feature_didx = label_didx - self.x_delay + 1
                label_ds = loader.didx2date(label_didx)
                feature_ds = loader.didx2date(feature_didx)
                x = loader.gen_feature(feature_ds)
                y, w = loader.gen_label(label_ds, ret_days=self.x_delay)
                self.X[offset] = torch.nan_to_num(x[self.validinsts], nan=0.0)
                self.Y[offset] = torch.nan_to_num(y[self.validinsts], nan=0.0)
                self.W[offset] = w[self.validinsts].to(loader.dtype)

    def _build_validinsts(self) -> torch.Tensor:
        masks = []
        for didx in range(self.start_didx, self.end_didx + 1):
            ds = self.loader.didx2date(didx - self.x_delay + 1)
            masks.append(self.loader.gen_base_universe_mask(ds))
        stacked = torch.stack(masks, dim=0)
        return torch.where(stacked.any(dim=0))[0]

    def __len__(self) -> int:
        return max(0, self.ndays - self.step_size + 1)

    def __getitem__(self, idx: int):
        feature_window = nan_to_num(self.X[idx:idx + self.step_size], 0.0).to(self.loader.dtype)
        y = self.Y[idx + self.step_size - 1]
        w = self.W[idx + self.step_size - 1]
        return idx, feature_window, y, w


class ComboBuffer:
    def __init__(self, feat_size: int, keepdays: int, instsz: int | None = None, dtype: torch.dtype | None = None):
        self.feat_size = feat_size
        self.keepdays = keepdays
        self.instsz = instsz or len(MASK.code)
        self.dtype = dtype or torch.float16
        self.buffer = torch.zeros((keepdays, self.instsz, feat_size), dtype=self.dtype)
        self.start_didx = -1

    def append(self, x: torch.Tensor, didx: int):
        if self.start_didx < 0:
            self.start_didx = didx
        pos = (didx - self.start_didx) % self.keepdays
        self.buffer[pos] = x.to(self.dtype)

    def get(self, didx_list: Iterable[int]) -> torch.Tensor:
        pos_list = [int((didx - self.start_didx) % self.keepdays) for didx in didx_list]
        return self.buffer[pos_list]