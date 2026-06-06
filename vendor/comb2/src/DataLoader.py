# haoyi 2026/05/5

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .codec import Codec, PassthroughCodec, build_codec
from .DataRegistry import (
    DataItem as _DataItem,
    DataRegistry as _DataRegistry,
    FeatureSpec as _FeatureSpec,
    MASK,
    Universe,
    _coerce_data_item,
    _coerce_feature_spec,
    _maybe_section,
    _require_memmaper2_cls,
)
from .op_utils import cs_zscore, nan_to_num, nanmedian, nanstd, normalize_by_max_abs, to_bool_mask, truncate, winsorize_by_quantile

try:
    from factorsim import Memmaper2
except ModuleNotFoundError:
    Memmaper2 = None

from vendor.perf_monitor import print_progress


class FeatureSource(Protocol):
    feature_dim: int

    def load_day(self, ds: int) -> torch.Tensor:
        ...

    def prefetch_days(self, days: Sequence[int]):
        ...


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
        self._mmap = _require_memmaper2_cls()(path)
        self._day_cache: dict[int, torch.Tensor] = {}

    def load_day(self, ds: int) -> torch.Tensor:
        ds = int(ds)
        cached = self._day_cache.pop(ds, None)
        if cached is not None:
            return cached
        monitor = getattr(self, "monitor", None)
        with _maybe_section(monitor, "mmap_label.load", ds, level="full"):
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
            self._mmap = _require_memmaper2_cls()(path)

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
    factor_paths: Sequence[str] = ()
    label_path: str | None = None
    dtype: torch.dtype = torch.float16
    data_start_ds: int = 20160101
    valid_path: str | None = None
    filtered_path: str | None = None
    features: Sequence[_FeatureSpec] | None = None
    compression: str = "none"
    base_universe_path: str | None = None
    ashare_data_path: str | None = None
    data_items: Sequence[_DataItem | dict[str, Any]] = ()
    data_presets: Sequence[str] = ()
    factor_root: str | None = None
    config_path: str | None = None
    verbose: bool = False


def _legacy_feature_specs_to_data_items(features: Sequence[_FeatureSpec | dict[str, Any]]) -> tuple[_DataItem, ...]:
    items: list[_DataItem] = []
    for index, raw_spec in enumerate(features):
        if isinstance(raw_spec, dict) and str(raw_spec.get("kind", "")).lower() == "ref":
            ops = tuple(_coerce_op_spec(op) for op in raw_spec.get("ops", ()))
            data_name = str(raw_spec["data"])
            feature_name = str(raw_spec.get("name") or data_name)
            items.append(
                _DataItem(
                    name=feature_name,
                    module="builtin.ref",
                    path=data_name,
                    role="factor",
                    ops=ops,
                    params={"data": data_name},
                )
            )
            continue
        spec = _coerce_feature_spec(raw_spec)
        kind = spec.kind.strip().lower()
        if kind == "factor":
            module = "builtin.factor"
        elif kind == "alpha":
            module = "builtin.alpha_parquet"
        else:
            raise ValueError(f"unsupported legacy feature kind: {spec.kind}")
        items.append(
            _DataItem(
                name=spec.name or f"feature.{index:04d}",
                module=module,
                path=spec.path,
                role="factor",
                mode=spec.mode,
                config_path=spec.config_path,
                ops=spec.ops,
            )
        )
    return tuple(items)


def _legacy_factor_paths_to_data_items(factor_paths: Sequence[str]) -> tuple[_DataItem, ...]:
    return tuple(
        _DataItem(
            name=Path(path).name,
            module="builtin.factor",
            path=str(path),
            role="factor",
        )
        for path in factor_paths
    )


def _build_loader_data_items(config: LoaderConfig) -> tuple[_DataItem, ...]:
    items = [_coerce_data_item(item) for item in config.data_items]
    if not any(item.role == "factor" for item in items):
        if config.features:
            items.extend(_legacy_feature_specs_to_data_items(config.features))
        else:
            items.extend(_legacy_factor_paths_to_data_items(config.factor_paths))
    if config.label_path and not any(item.role == "label" for item in items):
        items.append(
            _DataItem(
                name="label.default",
                module="builtin.label",
                path=config.label_path,
                role="label",
            )
        )
    return tuple(items)


class ComboDataLoader:
    def __init__(self, config: LoaderConfig, cube_source: FeatureSource | None = None, feature_cache_size: int = 2500, label_cache_size: int = 2500):
        self.config = config
        self.dtype = self.config.dtype
        self.codec = build_codec(self.config.compression, self.dtype)
        self.mask = MASK
        self.universe = Universe.from_mask(self.mask, self.dtype)
        self.data_start_ds = int(self.config.data_start_ds)
        self.data_start_didx = self.universe.date2idx(self.data_start_ds)
        data_items = _build_loader_data_items(config)
        self.registry = _DataRegistry(
            data_items,
            universe=self.universe,
            data_start_ds=self.data_start_ds,
            ashare_data_path=self.config.ashare_data_path,
            factor_root=self.config.factor_root,
            config_path=self.config.config_path,
            presets=self.config.data_presets,
        )
        self.factor_names = tuple(item.name for item in data_items if item.role == "factor")
        label_names = tuple(item.name for item in data_items if item.role == "label")
        if len(label_names) > 1:
            raise ValueError(f"only one label data item is supported, got {label_names}")
        self.label_name = label_names[0] if label_names else None
        if not self.factor_names:
            raise ValueError("at least one role='factor' data item is required")
        self.cube_source = cube_source or EmptyCubeSource(dtype=self.dtype)
        self.valid_source = MemmapMaskSource(self.config.valid_path)
        self.filtered_source = MemmapMaskSource(self.config.filtered_path)
        self.base_universe_source = MemmapMaskSource(self.config.base_universe_path)
        self.monitor = None
        self.num_features = len(self.factor_names) + self.cube_source.feature_dim
        factor_display_names = tuple(str(self.registry.items[name].params.get("display_name", name)) for name in self.factor_names)
        self.feature_names = factor_display_names + tuple(
            f"cube_{idx:03d}" for idx in range(self.cube_source.feature_dim)
        )
        self.verbose = bool(getattr(config, "verbose", False))
        self._feature_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._feature_cache_size = int(feature_cache_size)
        self._label_cache: OrderedDict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._label_cache_size = int(label_cache_size)

    def _sync_monitor_refs(self):
        self.valid_source.monitor = self.monitor
        self.filtered_source.monitor = self.monitor
        self.base_universe_source.monitor = self.monitor

    def date2didx(self, ds: int) -> int:
        didx = self.universe.date2idx(int(ds))
        return max(didx, self.data_start_didx)

    def didx2date(self, didx: int) -> int:
        didx = max(int(didx), self.data_start_didx)
        return int(self.mask.date[didx])

    def align_date(self, ds: int) -> int:
        aligned = self.didx2date(self.date2didx(ds))
        return max(aligned, self.data_start_ds)

    def set_processed_feature_cache_enabled(self, enabled: bool):
        if not enabled:
            self._feature_cache_size = 0
            self._feature_cache.clear()

    def set_processed_feature_cache_max_days(self, days: int):
        self._feature_cache_size = max(0, int(days))
        if self._feature_cache_size == 0:
            self._feature_cache.clear()
            return
        while len(self._feature_cache) > self._feature_cache_size:
            self._feature_cache.popitem(last=False)

    def _cache_get(self, cache: OrderedDict, key):
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
        return cached

    def _cache_set(self, cache: OrderedDict, key, value, max_size: int):
        if max_size <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)

    def build_raw_feature(self, ds: int) -> torch.Tensor:
        self._sync_monitor_refs()
        with _maybe_section(self.monitor, "gen_feature.feature_load", ds, level="full"):
            self.registry._ensure_range(self.factor_names, ds, ds)
            idx = self.universe.date2idx(ds)
            factor = torch.stack(
                [self.registry.get_data(name)[idx].to(torch.float32) for name in self.factor_names],
                dim=-1,
            )
        with _maybe_section(self.monitor, "gen_feature.cube_load", ds, level="full"):
            cube = self.cube_source.load_day(ds).to(torch.float32)
        if cube.shape[1] == 0:
            feature = factor
        else:
            feature = torch.cat([factor, cube], dim=-1)
        feature[torch.isinf(feature)] = torch.nan
        return feature

    def preprocess_feature(self, feature: torch.Tensor, ds: int) -> torch.Tensor:
        with _maybe_section(self.monitor, "gen_feature.cs_zscore", ds, level="full"):
            feature = cs_zscore(feature.transpose(0, 1)).transpose(0, 1)
        with _maybe_section(self.monitor, "gen_feature.truncate_nan_to_num", ds, level="full"):
            feature = truncate(feature, -4.0, 4.0)
            feature = nan_to_num(feature, 0.0)
        return feature.to(self.dtype)

    def _build_feature(self, ds: int) -> torch.Tensor:
        return self.preprocess_feature(self.build_raw_feature(ds), ds)

    def gen_feature(self, ds: int) -> torch.Tensor:
        ds = self.align_date(ds)
        cached = self._cache_get(self._feature_cache, ds)
        if cached is not None:
            return cached
        feature = self._build_feature(ds)
        self._cache_set(self._feature_cache, ds, feature, self._feature_cache_size)
        return feature

    def prefetch_features(self, days: Sequence[int]):
        days = [self.align_date(ds) for ds in days]
        if days:
            self.registry._ensure_range(self.factor_names, min(days), max(days))
        self._sync_monitor_refs()
        self.cube_source.prefetch_days(days)

    def prefetch_labels(self, days: Sequence[int], ret_days: int = 1):
        if self.label_name is None:
            return
        label_days: list[int] = []
        for ds in days:
            end_didx = self.date2didx(self.align_date(ds))
            start_didx = end_didx - int(ret_days) + 1
            if start_didx < self.data_start_didx:
                raise ValueError(f"not enough label history for ds={ds}, ret_days={ret_days}")
            label_days.extend(self.didx2date(didx) for didx in range(start_didx, end_didx + 1))
        if label_days:
            self.registry._ensure_range((self.label_name,), min(label_days), max(label_days))

    def gen_base_universe_mask(self, ds: int) -> torch.Tensor:
        ds = self.align_date(ds)
        self._sync_monitor_refs()
        return self.base_universe_source.load_day(ds)

    def gen_valid_mask(self, ds: int) -> torch.Tensor:
        ds = self.align_date(ds)
        self._sync_monitor_refs()
        valid = self.valid_source.load_day(ds)
        filtered = self.filtered_source.load_day(ds)
        return valid & filtered & self.gen_base_universe_mask(ds)

    def preprocess_label(
        self,
        label_values: torch.Tensor,
        valid_mask: torch.Tensor,
        ds: int,
        ret_days: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_values = label_values[valid_mask]
        if valid_values.numel() > 0:
            valid_values = winsorize_by_quantile(valid_values, 0.01, 0.99)
            valid_values = valid_values - nanmedian(valid_values)
            valid_values = valid_values / (nanstd(valid_values) + 1e-8)
            valid_values = truncate(valid_values, -3.0, 3.0)
            valid_values = normalize_by_max_abs(valid_values)
            label_values[valid_mask] = valid_values

        label_values[~valid_mask] = 0.0
        label_values = nan_to_num(label_values, 0.0).to(self.dtype)
        return label_values, to_bool_mask(valid_mask)

    def gen_label(self, ds: int, ret_days: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        if self.label_name is None:
            raise ValueError("gen_label requires one role='label' data item")
        ds = self.align_date(ds)
        cache_key = (ds, int(ret_days))
        cached = self._cache_get(self._label_cache, cache_key)
        if cached is not None:
            return cached

        self._sync_monitor_refs()
        end_didx = self.date2didx(ds)
        start_didx = end_didx - ret_days + 1
        if start_didx < self.data_start_didx:
            raise ValueError(f"not enough label history for ds={ds}, ret_days={ret_days}")

        dates = [self.didx2date(didx) for didx in range(start_didx, end_didx + 1)]
        with _maybe_section(self.monitor, "gen_label.load_returns", ds, level="full"):
            self.registry._ensure_range((self.label_name,), dates[0], dates[-1])
            label_data = self.registry.get_data(self.label_name)
            returns = [label_data[self.universe.date2idx(cur_ds)].to(torch.float32) for cur_ds in dates]
        with _maybe_section(self.monitor, "gen_label.aggregate", ds, level="full"):
            cret = torch.stack(returns, dim=0)
            cret = nan_to_num(cret, 0.0)
            decay_weights = torch.arange(ret_days, 0, -1, dtype=torch.float32, device=cret.device)
            cret = torch.tensordot(decay_weights, cret, dims=([0], [0]))
            cret[torch.isinf(cret)] = torch.nan

        with _maybe_section(self.monitor, "gen_label.valid_mask", ds, level="full"):
            valid_mask = self.gen_valid_mask(self.didx2date(start_didx)) & (~torch.isnan(cret))
        label = self.preprocess_label(cret, valid_mask, ds, ret_days=ret_days)
        self._cache_set(self._label_cache, cache_key, label, self._label_cache_size)
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
        codec: Codec | None = None,
    ):
        self.loader = loader
        self.codec = codec or PassthroughCodec(loader.dtype)
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
            loader.set_processed_feature_cache_max_days(max(loader._feature_cache_size, self.ndays))

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
            self.X, self.X_meta = self.codec.allocate(
                (self.ndays, self.numValidinsts, self.feat_size),
                device="cpu",
                logical_dtype=loader.dtype,
            )
            self.Y = torch.zeros((self.ndays, self.numValidinsts), dtype=loader.dtype)
            self.W = torch.zeros((self.ndays, self.numValidinsts), dtype=loader.dtype)

        progress_start = time.perf_counter()
        loaded_days = 0
        with _maybe_section(monitor, "dataset_init.loop_total", self.end_ds, level="full"):
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
                    self.codec.encode_into(
                        self.X,
                        self.X_meta,
                        offset,
                        torch.nan_to_num(x[self.validinsts], nan=0.0),
                    )
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
        x = self.codec.decode(self.X, self.X_meta, slice(idx, idx + self.ts_days), out_dtype=self.loader.dtype)
        y = self.Y[idx + self.ts_days - 1]
        w = self.W[idx + self.ts_days - 1]
        return idx, x, y, w


class ComboBuffer:
    def __init__(
        self,
        feat_size: int,
        keepdays: int,
        instsz: int | None = None,
        dtype: torch.dtype | None = None,
        codec: Codec | None = None,
    ):
        self.feat_size = feat_size
        self.keepdays = keepdays
        self.instsz = instsz or len(MASK.code)
        self.dtype = dtype or torch.float16
        self.codec = codec or PassthroughCodec(self.dtype)
        self.buffer, self.meta = self.codec.allocate(
            (keepdays, self.instsz, feat_size),
            device="cpu",
            logical_dtype=self.dtype,
        )
        self.start_didx = -1

    def append(self, x: torch.Tensor, didx: int):
        if self.start_didx < 0:
            self.start_didx = didx
        pos = (didx - self.start_didx) % self.keepdays
        self.codec.encode_into(self.buffer, self.meta, pos, x)

    def get(self, didx_list: Iterable[int]) -> torch.Tensor:
        pos_list = [int((didx - self.start_didx) % self.keepdays) for didx in didx_list]
        return self.codec.decode(self.buffer, self.meta, pos_list, out_dtype=self.dtype)
