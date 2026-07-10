# haoyi 2026/05/5

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
import os
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .codec import Codec, PassthroughCodec, build_codec
from .DataRegistry import (
    BAR_COUNT_BY_FREQ,
    CANONICAL_BAR_TIMES,
    DataItem as _DataItem,
    DataRegistry as _DataRegistry,
    FREQ_ORDER,
    LoadStats,
    MASK,
    Universe,
    _coerce_data_item,
    _maybe_section,
    _require_memmaper2_cls,
    item_freq,
)
from .op_utils import cs_zscore, nan_to_num, nanmedian, nanstd, normalize_by_max_abs, to_bool_mask, truncate, winsorize_by_quantile

from vendor.perf_monitor import print_progress


@dataclass(frozen=True)
class FeatureGroups(MappingABC):
    data: dict[str, torch.Tensor]
    freqs: tuple[str, ...] | None = None

    def __post_init__(self):
        normalized = {str(freq): tensor for freq, tensor in self.data.items()}
        object.__setattr__(self, "data", normalized)
        if self.freqs is None:
            ordered = tuple(freq for freq in FREQ_ORDER if freq in normalized)
            object.__setattr__(self, "freqs", ordered)
        else:
            freqs = tuple(str(freq) for freq in self.freqs)
            missing = [freq for freq in freqs if freq not in normalized]
            if missing:
                raise KeyError(f"FeatureGroups freqs missing data: {missing}")
            object.__setattr__(self, "freqs", freqs)

    def __getitem__(self, freq: str) -> torch.Tensor:
        return self.data[str(freq)]

    def __contains__(self, freq: str) -> bool:
        return str(freq) in self.data

    def __iter__(self):
        return iter(self.freqs)

    def __len__(self) -> int:
        return len(self.freqs)

    def keys(self):
        return self.freqs

    def items(self):
        for freq in self.freqs:
            yield freq, self.data[freq]

    def values(self):
        for freq in self.freqs:
            yield self.data[freq]

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "FeatureGroups":
        converted: dict[str, torch.Tensor] = {}
        for freq, tensor in self.items():
            if device is None and dtype is None:
                converted[freq] = tensor
                continue
            kwargs: dict[str, Any] = {"non_blocking": non_blocking}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            converted[freq] = tensor.to(**kwargs)
        return FeatureGroups(converted, self.freqs)

    def select_stocks(self, indices: torch.Tensor | Sequence[int]) -> "FeatureGroups":
        selected: dict[str, torch.Tensor] = {}
        for freq, tensor in self.items():
            idx = indices
            if isinstance(idx, torch.Tensor) and idx.device != tensor.device:
                idx = idx.to(tensor.device)
            stock_dim = self._stock_dim(freq, tensor)
            selected[freq] = tensor.index_select(stock_dim, torch.as_tensor(idx, dtype=torch.long, device=tensor.device))
        return FeatureGroups(selected, self.freqs)

    def stock_count(self) -> int:
        if not self.freqs:
            return 0
        freq = self.freqs[0]
        tensor = self.data[freq]
        return int(tensor.shape[self._stock_dim(freq, tensor)])

    @staticmethod
    def _stock_dim(freq: str, tensor: torch.Tensor) -> int:
        if freq == "1d":
            if tensor.ndim == 2:
                return 0
            if tensor.ndim == 3:
                return 1
        else:
            if tensor.ndim == 3:
                return 0
            if tensor.ndim == 4:
                return 1
        raise ValueError(f"cannot infer stock dimension for freq={freq!r}, shape={tuple(tensor.shape)}")

    @classmethod
    def stack(cls, values: Sequence["FeatureGroups"], dim: int = 0) -> "FeatureGroups":
        if not values:
            raise ValueError("cannot stack empty FeatureGroups")
        freqs = values[0].freqs
        return cls(
            {freq: torch.stack([value[freq] for value in values], dim=dim) for freq in freqs},
            freqs,
        )


class GroupCodec:
    def __init__(self, codec: Codec, freqs: Sequence[str]):
        self.codec = codec
        self.freqs = tuple(freqs)

    def allocate(self, shapes: Mapping[str, tuple[int, ...]], *, device="cpu", logical_dtype=None):
        storage = {}
        meta = {}
        for freq in self.freqs:
            storage[freq], meta[freq] = self.codec.allocate(shapes[freq], device=device, logical_dtype=logical_dtype)
        return storage, meta

    def encode_into(self, storage: Mapping[str, torch.Tensor], meta: Mapping[str, Any], idx, value: FeatureGroups) -> None:
        for freq in self.freqs:
            self.codec.encode_into(storage[freq], meta[freq], idx, value[freq])

    def decode(self, storage: Mapping[str, torch.Tensor], meta: Mapping[str, Any], idx, out_dtype=None) -> FeatureGroups:
        return FeatureGroups(
            {freq: self.codec.decode(storage[freq], meta[freq], idx, out_dtype=out_dtype) for freq in self.freqs},
            self.freqs,
        )


class MemmapMaskSource:
    def __init__(self, path: str | None):
        self.path = path
        self._mmap = None
        if path is not None:
            if not os.path.exists(path):
                raise FileNotFoundError(f"mask data not found: {path}")
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
    dtype: torch.dtype = torch.float16
    data_start_ds: int = 20160101
    data_offset: int = 1024
    valid_path: str | None = None
    filtered_path: str | None = None
    compression: str = "none"
    base_universe_path: str | None = None
    ashare_data_path: str | None = None
    data_items: Sequence[_DataItem | dict[str, Any]] = ()
    data_presets: Sequence[str] = ()
    config_path: str | None = None
    verbose: bool = False


def _build_loader_data_items(config: LoaderConfig) -> tuple[_DataItem, ...]:
    items = tuple(_coerce_data_item(item) for item in config.data_items)
    if not any(item.role == "factor" for item in items):
        raise ValueError("LoaderConfig.data_items requires at least one role='factor' data item")
    return tuple(items)


class ComboDataLoader:
    def __init__(self, config: LoaderConfig, label_cache_size: int = 2500):
        self.config = config
        self.dtype = self.config.dtype
        self.codec = build_codec(self.config.compression, self.dtype)
        self.group_codec: GroupCodec | None = None
        self.mask = MASK
        self.data_offset = max(0, int(self.config.data_offset))
        self.universe = Universe.from_mask(self.mask, self.dtype, data_offset=self.data_offset)
        self.data_start_ds = int(self.config.data_start_ds)
        self.data_start_didx = self.universe.date2idx(self.data_start_ds)
        data_items = _build_loader_data_items(config)
        self.registry = _DataRegistry(
            data_items,
            universe=self.universe,
            data_start_ds=self.data_start_ds,
            ashare_data_path=self.config.ashare_data_path,
            config_path=self.config.config_path,
            presets=self.config.data_presets,
            verbose=bool(getattr(config, "verbose", False)),
        )
        self.factor_names = tuple(item.name for item in data_items if item.role == "factor")
        label_names = tuple(item.name for item in data_items if item.role == "label")
        if len(label_names) > 1:
            raise ValueError(f"only one label data item is supported, got {label_names}")
        self.label_name = label_names[0] if label_names else None
        if not self.factor_names:
            raise ValueError("at least one role='factor' data item is required")

        self.factor_names_by_freq = {
            freq: tuple(name for name in self.factor_names if item_freq(self.registry.items[name]) == freq)
            for freq in FREQ_ORDER
        }
        self.freqs = tuple(freq for freq in FREQ_ORDER if self.factor_names_by_freq[freq])
        self.num_features_by_freq = {freq: len(self.factor_names_by_freq[freq]) for freq in self.freqs}
        self.num_features = sum(self.num_features_by_freq.values())
        self.feature_names_by_freq = {
            freq: tuple(str(self.registry.items[name].params.get("display_name", name)) for name in self.factor_names_by_freq[freq])
            for freq in self.freqs
        }
        self.feature_names = tuple(name for freq in self.freqs for name in self.feature_names_by_freq[freq])
        self.valid_source = MemmapMaskSource(self.config.valid_path)
        self.filtered_source = MemmapMaskSource(self.config.filtered_path)
        self.base_universe_source = MemmapMaskSource(self.config.base_universe_path)
        self.monitor = None
        self.verbose = bool(getattr(config, "verbose", False))
        self.current_ti = 150000
        self._label_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._label_cache_size = int(label_cache_size)

    def _sync_monitor_refs(self):
        self.valid_source.monitor = self.monitor
        self.filtered_source.monitor = self.monitor
        self.base_universe_source.monitor = self.monitor
        self.registry.set_current_ti(self.current_ti)

    def set_current_ti(self, ti: int):
        self.current_ti = int(ti)

    def date2didx(self, ds: int) -> int:
        didx = self.universe.date2idx(int(ds))
        return max(didx, self.data_start_didx)

    def didx2date(self, didx: int) -> int:
        didx = max(int(didx), self.data_start_didx)
        return self.universe.idx2date(didx)

    def align_date(self, ds: int) -> int:
        aligned = self.didx2date(self.date2didx(ds))
        return max(aligned, self.data_start_ds)

    def _label_cache_get(self, key: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._label_cache.get(key)

    def _label_cache_set(self, key: tuple[int, int], value: tuple[torch.Tensor, torch.Tensor]) -> None:
        if self._label_cache_size <= 0:
            return
        self._label_cache[key] = value
        while len(self._label_cache) > self._label_cache_size:
            oldest = next(iter(self._label_cache))
            del self._label_cache[oldest]

    def feature_group_shapes(self, inst_count: int) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        for freq in self.freqs:
            feature_count = self.num_features_by_freq[freq]
            if freq == "1d":
                shapes[freq] = (int(inst_count), feature_count)
            else:
                shapes[freq] = (int(inst_count), BAR_COUNT_BY_FREQ[freq], feature_count)
        return shapes

    def _group_codec(self) -> GroupCodec:
        if self.group_codec is None:
            self.group_codec = GroupCodec(self.codec, self.freqs)
        return self.group_codec

    def build_raw_feature(self, ds: int) -> FeatureGroups:
        self._sync_monitor_refs()
        with _maybe_section(self.monitor, "gen_feature.feature_load", ds, level="full"):
            self.registry._ensure_range(self.factor_names, ds, ds)
            idx = self.universe.date2idx(ds)
            groups: dict[str, torch.Tensor] = {}
            for freq in self.freqs:
                tensors = [self.registry.get_data(name)[idx].to(torch.float32) for name in self.factor_names_by_freq[freq]]
                if freq == "1d":
                    group = torch.stack(tensors, dim=-1)
                else:
                    group = torch.stack([tensor.transpose(0, 1) for tensor in tensors], dim=-1)
                group[torch.isinf(group)] = torch.nan
                groups[freq] = group
        return FeatureGroups(groups, self.freqs)

    def preprocess_daily_features(self, feature: torch.Tensor, ds: int) -> torch.Tensor:
        with _maybe_section(self.monitor, "gen_feature.1d_cs_zscore", ds, level="full"):
            feature = cs_zscore(feature.transpose(0, 1)).transpose(0, 1)
        with _maybe_section(self.monitor, "gen_feature.1d_truncate_nan_to_num", ds, level="full"):
            feature = truncate(feature, -4.0, 4.0)
            feature = nan_to_num(feature, 0.0)
        return feature.to(self.dtype)

    def preprocess_feature_group(self, freq: str, feature: torch.Tensor, ds: int) -> torch.Tensor:
        if freq == "1d":
            return self.preprocess_daily_features(feature, ds)
        return feature.to(self.dtype)

    def preprocess_features(self, features: FeatureGroups, ds: int) -> FeatureGroups:
        return FeatureGroups(
            {freq: self.preprocess_feature_group(freq, feature, ds) for freq, feature in features.items()},
            features.freqs,
        )

    def _build_feature(self, ds: int) -> FeatureGroups:
        return self.preprocess_features(self.build_raw_feature(ds), ds)

    def gen_feature(self, ds: int) -> FeatureGroups:
        ds = self.align_date(ds)
        return self._build_feature(ds)

    def prefetch_features(self, days: Sequence[int]):
        days = [self.align_date(ds) for ds in days]
        self._sync_monitor_refs()
        stats = LoadStats(request_days=len(days))
        if days:
            stats = self.registry._ensure_range(self.factor_names, min(days), max(days))
        return stats

    def prefetch_labels(self, days: Sequence[int], ret_days: int = 1):
        if self.label_name is None:
            return LoadStats()
        label_days: list[int] = []
        for ds in days:
            end_didx = self.date2didx(self.align_date(ds))
            start_didx = end_didx - int(ret_days) + 1
            if start_didx < self.data_start_didx:
                raise ValueError(f"not enough label history for ds={ds}, ret_days={ret_days}")
            label_days.extend(self.didx2date(didx) for didx in range(start_didx, end_didx + 1))
        if label_days:
            return self.registry._ensure_range((self.label_name,), min(label_days), max(label_days))
        return LoadStats()

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
        cached = self._label_cache_get(cache_key)
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
        self._label_cache_set(cache_key, label)
        return label

    def _feature_available_mask(self, feature_window: FeatureGroups) -> torch.Tensor:
        return torch.ones(feature_window.stock_count(), dtype=torch.bool)

    def _mask_window_tail_by_ti(self, feature_window: FeatureGroups) -> FeatureGroups:
        if self.current_ti >= 150000:
            return feature_window
        masked: dict[str, torch.Tensor] = {}
        for freq, tensor in feature_window.items():
            if freq == "1d":
                masked[freq] = tensor
                continue
            future_mask = torch.as_tensor(
                np.asarray(CANONICAL_BAR_TIMES[freq], dtype=np.int64) > int(self.current_ti),
                dtype=torch.bool,
                device=tensor.device,
            )
            if not future_mask.any().item():
                masked[freq] = tensor
                continue
            cur = tensor.clone()
            cur[-1, :, future_mask, :] = torch.nan
            masked[freq] = cur
        return FeatureGroups(masked, feature_window.freqs)

    def transform_feature_window(self, feature_window: FeatureGroups, *, stage: str) -> FeatureGroups:
        feature_window = self._mask_window_tail_by_ti(feature_window)
        return feature_window.to(dtype=self.dtype)

    def process_feature_window(self, feature_window: FeatureGroups) -> tuple[FeatureGroups, torch.Tensor]:
        feature_window = self.transform_feature_window(feature_window, stage="predict")
        available_mask = self._feature_available_mask(feature_window)
        return feature_window.select_stocks(torch.where(available_mask)[0]), available_mask

    def load_feature_window(self, end_ds: int, ts_days: int) -> tuple[FeatureGroups, torch.Tensor]:
        end_didx = self.date2didx(end_ds)
        start_didx = max(self.data_start_didx, end_didx - ts_days + 1)
        days = [self.didx2date(didx) for didx in range(start_didx, end_didx + 1)]
        features = [self.gen_feature(ds) for ds in days]
        if not features:
            raise ValueError("no feature data loaded")
        if len(features) < ts_days:
            pad_data = {freq: torch.full_like(features[0][freq], torch.nan) for freq in features[0].freqs}
            pad = [FeatureGroups(pad_data, features[0].freqs) for _ in range(ts_days - len(features))]
            features = pad + features
        return self.process_feature_window(FeatureGroups.stack(features, dim=0))


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
        codec: Codec | None = None,
    ):
        self.loader = loader
        self.codec = codec or PassthroughCodec(loader.dtype)
        self.group_codec = GroupCodec(self.codec, loader.freqs)
        self.end_ds = loader.align_date(end_ds)
        self.ndays = int(ndays)
        self.x_delay = int(x_delay)
        self.ts_days = int(ts_days)
        self.load_chunk_days = int(load_chunk_days or self.ts_days)
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
            group_shapes = {
                freq: (self.ndays, *shape)
                for freq, shape in loader.feature_group_shapes(self.numValidinsts).items()
            }
            self.X, self.X_meta = self.group_codec.allocate(
                group_shapes,
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
                feature_days = [
                    loader.didx2date(self.start_didx + offset - self.x_delay + 1)
                    for offset in range(window_start, window_end)
                ]
                label_days = [loader.didx2date(self.start_didx + offset) for offset in range(window_start, window_end)]
                load_start = time.perf_counter()
                feature_stats = loader.prefetch_features(feature_days)
                label_stats = loader.prefetch_labels(label_days, ret_days=self.x_delay)
                load_time = time.perf_counter() - load_start
                if loader.verbose:
                    raw_time = feature_stats.raw_time + label_stats.raw_time
                    ops_time = feature_stats.ops_time + label_stats.ops_time
                    detail = (
                        f"days {feature_days[0]}-{feature_days[-1]}, chunk {window_end - window_start}, "
                        f"raw {raw_time:.2f}s, ops {ops_time:.2f}s, load {load_time:.2f}s"
                    )
                    print_progress(
                        "Stage:load_train_days",
                        window_end,
                        self.ndays,
                        progress_start,
                        detail,
                        final=window_end == self.ndays,
                    )
                for offset in range(window_start, window_end):
                    label_didx = self.start_didx + offset
                    feature_didx = label_didx - self.x_delay + 1
                    label_ds = loader.didx2date(label_didx)
                    feature_ds = loader.didx2date(feature_didx)
                    x = loader.gen_feature(feature_ds).select_stocks(self.validinsts)
                    y, w = loader.gen_label(label_ds, ret_days=self.x_delay)
                    self.group_codec.encode_into(self.X, self.X_meta, offset, x)
                    self.Y[offset] = torch.nan_to_num(y[self.validinsts], nan=0.0)
                    self.W[offset] = w[self.validinsts].to(loader.dtype)
                    loaded_days += 1

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
        x = self.group_codec.decode(self.X, self.X_meta, slice(idx, idx + self.ts_days), out_dtype=self.loader.dtype)
        x = self.loader.transform_feature_window(x, stage="train")
        y = self.Y[idx + self.ts_days - 1]
        w = self.W[idx + self.ts_days - 1]
        return idx, x, y, w


class ComboBuffer:
    def __init__(
        self,
        group_shapes: Mapping[str, tuple[int, ...]],
        keepdays: int,
        dtype: torch.dtype | None = None,
        codec: Codec | None = None,
        freqs: Sequence[str] | None = None,
    ):
        self.group_shapes = dict(group_shapes)
        self.keepdays = int(keepdays)
        self.dtype = dtype or torch.float16
        self.codec = codec or PassthroughCodec(self.dtype)
        self.freqs = tuple(freqs or self.group_shapes.keys())
        self.group_codec = GroupCodec(self.codec, self.freqs)
        storage_shapes = {freq: (self.keepdays, *self.group_shapes[freq]) for freq in self.freqs}
        self.buffer, self.meta = self.group_codec.allocate(
            storage_shapes,
            device="cpu",
            logical_dtype=self.dtype,
        )
        self.start_didx = -1

    def append(self, x: FeatureGroups, didx: int):
        if self.start_didx < 0:
            self.start_didx = didx
        pos = (didx - self.start_didx) % self.keepdays
        self.group_codec.encode_into(self.buffer, self.meta, pos, x)

    def get(self, didx_list: Iterable[int]) -> FeatureGroups:
        pos_list = [int((didx - self.start_didx) % self.keepdays) for didx in didx_list]
        return self.group_codec.decode(self.buffer, self.meta, pos_list, out_dtype=self.dtype)
