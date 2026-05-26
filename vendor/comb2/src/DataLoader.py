# Mengkang Li 2026/04/22

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import torch
from factorsim import IndexMask, Memmaper2
from torch.utils.data import Dataset

from .op_utils import cs_zscore, nan_to_num, nanmedian, nanstd, normalize_by_max_abs, to_bool_mask, truncate, winsorize_by_quantile

ORGANIZE_ROOT = Path(__file__).resolve().parents[3]
if str(ORGANIZE_ROOT) not in sys.path:
    sys.path.insert(0, str(ORGANIZE_ROOT))
from vendor.perf_monitor import print_progress

MASK = IndexMask()


class FeatureSource(Protocol):
    feature_dim: int

    def load_day(self, ds: int) -> torch.Tensor:
        ...

    def prefetch_days(self, days: Sequence[int]):
        ...


class MemmapFeatureSource:
    def __init__(self, paths: Sequence[str], dtype: torch.dtype):
        self.paths = list(paths)
        self.dtype = dtype
        self.feature_dim = len(self.paths)
        self.verbose = False
        self._cache: dict[str, Memmaper2] = {}
        self._day_cache: dict[int, torch.Tensor] = {}

    def _mmap(self, path: str) -> Memmaper2:
        if path not in self._cache:
            self._cache[path] = Memmaper2(path)
        return self._cache[path]

    def load_day(self, ds: int) -> torch.Tensor:
        ds = int(ds)
        cached = self._day_cache.pop(ds, None)
        if cached is not None:
            return cached
        values = []
        for path in self.paths:
            data = self._mmap(path).load(start_ds=ds, end_ds=ds)[:]
            values.append(torch.as_tensor(np.asarray(data)[0], dtype=self.dtype))
        return torch.stack(values, dim=-1)

    def prefetch_days(self, days: Sequence[int]):
        days = list(dict.fromkeys(int(ds) for ds in days))
        if not days:
            return
        needed_days = [ds for ds in days if ds not in self._day_cache]
        if not needed_days:
            return
        start_ds = min(needed_days)
        end_ds = max(needed_days)
        day_set = set(needed_days)
        trading_days = [int(ds) for ds in MASK.date if start_ds <= int(ds) <= end_ds]
        values_by_day = {ds: [] for ds in needed_days}
        for path in self.paths:
            data = self._mmap(path).load(start_ds=start_ds, end_ds=end_ds)[:]
            arr = np.asarray(data)
            for offset, ds in enumerate(trading_days[: len(arr)]):
                if ds in day_set:
                    values_by_day[ds].append(torch.as_tensor(arr[offset], dtype=self.dtype))
        for ds, values in values_by_day.items():
            if len(values) == len(self.paths):
                self._day_cache[ds] = torch.stack(values, dim=-1)


class EmptyCubeSource:
    def __init__(self, feature_dim: int = 0, dtype: torch.dtype | None = None):
        self.feature_dim = feature_dim
        self.dtype = dtype or torch.float16

    def load_day(self, ds: int) -> torch.Tensor:
        return torch.zeros((len(MASK.code), self.feature_dim), dtype=self.dtype)

    def prefetch_days(self, days: Sequence[int]):
        return None


class MemmapLabelSource:
    def __init__(self, path: str, dtype: torch.dtype):
        self.path = path
        self.dtype = dtype
        self._mmap = Memmaper2(path)
        self._day_cache: dict[int, torch.Tensor] = {}

    def load_day(self, ds: int) -> torch.Tensor:
        ds = int(ds)
        cached = self._day_cache.pop(ds, None)
        if cached is not None:
            return cached
        data = self._mmap.load(start_ds=ds, end_ds=ds)[:]
        label = torch.as_tensor(np.asarray(data)[0], dtype=self.dtype)
        return label

    def prefetch_days(self, days: Sequence[int]):
        days = list(dict.fromkeys(int(ds) for ds in days))
        if not days:
            return
        needed_days = [ds for ds in days if ds not in self._day_cache]
        if not needed_days:
            return
        start_ds = min(needed_days)
        end_ds = max(needed_days)
        day_set = set(needed_days)
        trading_days = [int(ds) for ds in MASK.date if start_ds <= int(ds) <= end_ds]
        data = self._mmap.load(start_ds=start_ds, end_ds=end_ds)[:]
        arr = np.asarray(data)
        for offset, ds in enumerate(trading_days[: len(arr)]):
            if ds in day_set:
                self._day_cache[ds] = torch.as_tensor(arr[offset], dtype=self.dtype)


class MemmapMaskSource:
    def __init__(self, path: str | None):
        self.path = path
        self._mmap = None
        if path and os.path.exists(path):
            self._mmap = Memmaper2(path)

    def load_day(self, ds: int) -> torch.Tensor:
        if self._mmap is None:
            return torch.ones(len(MASK.code), dtype=torch.bool)
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
    verbose: bool = False


class ComboDataLoader:
    def __init__(self, config: LoaderConfig, cube_source: FeatureSource | None = None):
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
        self.num_features = self.factor_source.feature_dim + self.cube_source.feature_dim
        self.verbose = bool(getattr(config, "verbose", False))
        self.factor_source.verbose = self.verbose
        self._processed_feature_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._processed_feature_cache_enabled = False
        self._processed_feature_cache_max_days = 0

    def date2didx(self, ds: int) -> int:
        didx = int(self.mask.date2didx(int(ds)))
        return max(didx, self.data_start_didx)

    def didx2date(self, didx: int) -> int:
        didx = max(int(didx), self.data_start_didx)
        return int(self.mask.date[didx])

    def align_date(self, ds: int) -> int:
        aligned = self.didx2date(self.date2didx(ds))
        return max(aligned, self.data_start_ds)

    def set_processed_feature_cache_enabled(self, enabled: bool):
        self._processed_feature_cache_enabled = bool(enabled)
        if not self._processed_feature_cache_enabled:
            self._processed_feature_cache_max_days = 0
            self._processed_feature_cache.clear()

    def set_processed_feature_cache_max_days(self, days: int):
        self._processed_feature_cache_enabled = True
        self._processed_feature_cache_max_days = max(0, int(days))
        if self._processed_feature_cache_max_days == 0:
            self._processed_feature_cache.clear()
            return
        while len(self._processed_feature_cache) > self._processed_feature_cache_max_days:
            self._processed_feature_cache.popitem(last=False)

    def _build_feature(self, ds: int) -> torch.Tensor:
        factor = self.factor_source.load_day(ds).to(torch.float32)
        cube = self.cube_source.load_day(ds).to(torch.float32)
        if cube.shape[1] == 0:
            feature = factor
        else:
            feature = torch.cat([factor, cube], dim=-1)
        feature[torch.isinf(feature)] = torch.nan
        feature = cs_zscore(feature.transpose(0, 1)).transpose(0, 1)
        feature = truncate(feature, -4.0, 4.0)
        feature = nan_to_num(feature, 0.0).to(self.dtype)
        return feature

    def _cache_processed_feature(self, ds: int, feature: torch.Tensor):
        if not self._processed_feature_cache_enabled or self._processed_feature_cache_max_days <= 0:
            return
        self._processed_feature_cache[ds] = feature
        while len(self._processed_feature_cache) > self._processed_feature_cache_max_days:
            self._processed_feature_cache.popitem(last=False)

    def gen_feature(self, ds: int) -> torch.Tensor:
        ds = self.align_date(ds)
        cached = self._processed_feature_cache.get(ds)
        if cached is not None:
            self._processed_feature_cache.move_to_end(ds)
            return cached
        feature = self._build_feature(ds)
        self._cache_processed_feature(ds, feature)
        return feature

    def prefetch_features(self, days: Sequence[int]):
        days = [self.align_date(ds) for ds in days]
        self.factor_source.prefetch_days(days)
        self.cube_source.prefetch_days(days)

    def prefetch_labels(self, days: Sequence[int], ret_days: int = 1):
        label_days: list[int] = []
        for ds in days:
            end_didx = self.date2didx(self.align_date(ds))
            start_didx = end_didx - int(ret_days) + 1
            if start_didx < self.data_start_didx:
                raise ValueError(f"not enough label history for ds={ds}, ret_days={ret_days}")
            label_days.extend(self.didx2date(didx) for didx in range(start_didx, end_didx + 1))
        self.label_source.prefetch_days(label_days)

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
        end_didx = self.date2didx(ds)
        start_didx = end_didx - ret_days + 1
        if start_didx < self.data_start_didx:
            raise ValueError(f"not enough label history for ds={ds}, ret_days={ret_days}")

        dates = [self.didx2date(didx) for didx in range(start_didx, end_didx + 1)]
        returns = [self.label_source.load_day(cur_ds).to(torch.float32) for cur_ds in dates]
        cret = torch.stack(returns, dim=0)
        cret = nan_to_num(cret, 0.0)
        decay_weights = torch.arange(ret_days, 0, -1, dtype=torch.float32, device=cret.device)
        cret = torch.tensordot(decay_weights, cret, dims=([0], [0]))
        cret[torch.isinf(cret)] = torch.nan

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
        return cret, to_bool_mask(valid_mask)

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
    def __init__(
        self,
        loader: ComboDataLoader,
        end_ds: int,
        ndays: int,
        x_delay: int,
        ts_days: int,
        validinsts: torch.Tensor | None = None,
        load_chunk_days: int | None = None,
        processed_feature_cache: bool = False,
    ):
        self.loader = loader
        self.end_ds = loader.align_date(end_ds)
        self.ndays = int(ndays)
        self.x_delay = int(x_delay)
        self.ts_days = int(ts_days)
        self.load_chunk_days = int(load_chunk_days or self.ts_days)
        self.feat_size = loader.num_features
        self.end_didx = loader.date2didx(self.end_ds)
        self.start_didx = max(loader.data_start_didx + self.x_delay - 1, self.end_didx - self.ndays + 1)
        self.ndays = self.end_didx - self.start_didx + 1
        if processed_feature_cache:
            loader.set_processed_feature_cache_max_days(self.ndays)
        else:
            loader.set_processed_feature_cache_enabled(False)

        instsz = len(MASK.code)
        if validinsts is None:
            self.validinsts = self._build_validinsts()
        else:
            self.validinsts = validinsts.to(dtype=torch.long)
        self.numValidinsts = len(self.validinsts)
        if self.numValidinsts == 0:
            self.validinsts = torch.arange(instsz)
            self.numValidinsts = instsz
        self.X = torch.zeros((self.ndays, self.numValidinsts, self.feat_size), dtype=loader.dtype)
        self.Y = torch.zeros((self.ndays, self.numValidinsts), dtype=loader.dtype)
        self.W = torch.zeros((self.ndays, self.numValidinsts), dtype=loader.dtype)

        progress_start = time.perf_counter()
        loaded_days = 0
        for window_start in range(0, self.ndays, self.load_chunk_days):
            window_end = min(window_start + self.load_chunk_days, self.ndays)
            feature_days = [loader.didx2date(self.start_didx + offset - self.x_delay + 1) for offset in range(window_start, window_end)]
            label_days = [loader.didx2date(self.start_didx + offset) for offset in range(window_start, window_end)]
            load_start = time.perf_counter()
            loader.prefetch_features(feature_days)
            loader.prefetch_labels(label_days, ret_days=self.x_delay)
            feature_load_time = time.perf_counter() - load_start
            for offset in range(window_start, window_end):
                label_didx = self.start_didx + offset
                feature_didx = label_didx - self.x_delay + 1
                label_ds = loader.didx2date(label_didx)
                feature_ds = loader.didx2date(feature_didx)
                x = loader.gen_feature(feature_ds)
                y, w = loader.gen_label(label_ds, ret_days=self.x_delay)
                self.X[offset] = torch.nan_to_num(x[self.validinsts], nan=0.0)
                self.Y[offset] = torch.nan_to_num(y[self.validinsts], nan=0.0)
                self.W[offset] = w[self.validinsts].to(loader.dtype)
                loaded_days += 1
            if loader.verbose:
                detail = f"days {feature_days[0]}-{feature_days[-1]}, load {feature_load_time:.2f}s"
                print_progress("Stage:load_train_days", loaded_days, self.ndays, progress_start, detail, final=loaded_days == self.ndays)

    def _build_validinsts(self) -> torch.Tensor:
        masks = []
        for didx in range(self.start_didx, self.end_didx + 1):
            ds = self.loader.didx2date(didx - self.x_delay + 1)
            masks.append(self.loader.gen_base_universe_mask(ds))
        stacked = torch.stack(masks, dim=0)
        return torch.where(stacked.any(dim=0))[0]

    def __len__(self) -> int:
        return max(0, self.ndays - self.ts_days + 1)

    def __getitem__(self, idx: int):
        x = self.X[idx:idx + self.ts_days].to(self.loader.dtype)
        y = self.Y[idx + self.ts_days - 1]
        w = self.W[idx + self.ts_days - 1]
        return idx, x, y, w


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
