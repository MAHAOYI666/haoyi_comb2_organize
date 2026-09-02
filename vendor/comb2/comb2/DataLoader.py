# haoyi 2026/05/5

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
import os
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from comb2_simbase.cache_layout import (
    BASE_UNIVERSE_MASK_NAME,
    FILTERED_MASK_NAME,
    VALID_MASK_NAME,
    ashare_cache_path,
    stock_mask_path,
)
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
    available_bar_count,
    item_freq,
    target_bar_index,
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
    compression: str = "none"
    cache_path: str | None = None
    data_items: Sequence[_DataItem | dict[str, Any]] = ()
    data_presets: Sequence[str] = ()
    config_path: str | None = None
    registry_cache_days: int = 64
    freq: str = "1d"
    verbose: bool = False


def _build_loader_data_items(config: LoaderConfig) -> tuple[_DataItem, ...]:
    items = tuple(_coerce_data_item(item) for item in config.data_items)
    if not items:
        raise ValueError("LoaderConfig.data_items cannot be empty")
    if config.freq == "1d" and not any(item.role == "factor" for item in items):
        raise ValueError("daily mode requires at least one role='factor' data item")
    return tuple(items)


class ComboDataLoader:
    def __init__(self, config: LoaderConfig, label_cache_size: int = 2500):
        self.config = config
        self.execution_freq = str(config.freq)
        assert self.execution_freq in FREQ_ORDER
        self.dtype = self.config.dtype
        self.codec = build_codec(self.config.compression, self.dtype)
        self.group_codec: GroupCodec | None = None
        self.mask = MASK
        self.data_offset = max(0, int(self.config.data_offset))
        self.universe = Universe.from_mask(self.mask, self.dtype, data_offset=self.data_offset)
        self.data_start_ds = int(self.config.data_start_ds)
        self.data_start_didx = self.universe.date2idx(self.data_start_ds)
        data_items = _build_loader_data_items(config)
        ashare_path = ashare_cache_path(self.config.cache_path) if self.config.cache_path else None
        self.registry = _DataRegistry(
            data_items,
            universe=self.universe,
            data_start_ds=self.data_start_ds,
            ashare_cache_path=str(ashare_path) if ashare_path else None,
            config_path=self.config.config_path,
            presets=self.config.data_presets,
            verbose=bool(getattr(config, "verbose", False)),
            cache_days=int(self.config.registry_cache_days),
        )
        if self.execution_freq == "1d":
            self.input_names = tuple(item.name for item in data_items if item.role == "factor")
            label_names = tuple(item.name for item in data_items if item.role == "label")
            if len(label_names) > 1:
                raise ValueError(f"only one label data item is supported, got {label_names}")
            self.label_name = label_names[0] if label_names else None
            self.target_name = None
            self.target_freq = "1d"
            self.target_times = ()
            self.target_bar_ids = ()
        else:
            self.input_names, target_name = self.assign_data_items(data_items)
            self.target_name = target_name
            self.label_name = None
            self.target_freq = item_freq(self.registry.items[target_name])
            assert self.target_freq == self.execution_freq
            self.target_times = tuple(CANONICAL_BAR_TIMES[self.target_freq][1:])
            self.target_bar_ids = tuple(target_bar_index(self.target_freq, ti) for ti in self.target_times)

        self.input_names_by_freq = {
            freq: tuple(name for name in self.input_names if item_freq(self.registry.items[name]) == freq)
            for freq in FREQ_ORDER
        }
        self.freqs = tuple(freq for freq in FREQ_ORDER if self.input_names_by_freq[freq])
        self.num_features_by_freq = {freq: len(self.input_names_by_freq[freq]) for freq in self.freqs}
        self.num_features = sum(self.num_features_by_freq.values())
        self.feature_names_by_freq = {
            freq: tuple(str(self.registry.items[name].params.get("display_name", name)) for name in self.input_names_by_freq[freq])
            for freq in self.freqs
        }
        self.feature_names = tuple(name for freq in self.freqs for name in self.feature_names_by_freq[freq])
        self.factor_names = self.input_names
        self.factor_names_by_freq = self.input_names_by_freq
        self.valid_source = MemmapMaskSource(str(stock_mask_path(self.config.cache_path, VALID_MASK_NAME)) if self.config.cache_path else None)
        self.filtered_source = MemmapMaskSource(str(stock_mask_path(self.config.cache_path, FILTERED_MASK_NAME)) if self.config.cache_path else None)
        self.base_universe_source = MemmapMaskSource(
            str(stock_mask_path(self.config.cache_path, BASE_UNIVERSE_MASK_NAME)) if self.config.cache_path else None
        )
        self.monitor = None
        self.verbose = bool(getattr(config, "verbose", False))
        self.current_ti = 150000
        self._label_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._label_cache_size = int(label_cache_size)

    def assign_data_items(self, items: Sequence[_DataItem]) -> tuple[tuple[str, ...], str]:
        targets = tuple(item.name for item in items if item.role == "target")
        assert len(targets) == 1, f"expected exactly one target item, got {targets}"
        target_name = targets[0]
        input_names = tuple(item.name for item in items if item.name != target_name)
        assert input_names, "at least one input item is required"
        return input_names, target_name

    def _sync_monitor_refs(self):
        self.valid_source.monitor = self.monitor
        self.filtered_source.monitor = self.monitor
        self.base_universe_source.monitor = self.monitor
        self.registry.set_current_ti(self.current_ti)

    def set_current_ti(self, ti: int):
        self.current_ti = int(ti)

    def _label_cache_get(self, key: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._label_cache.get(key)

    def _label_cache_set(self, key: tuple[int, int], value: tuple[torch.Tensor, torch.Tensor]) -> None:
        if self._label_cache_size <= 0:
            return
        self._label_cache[key] = value
        while len(self._label_cache) > self._label_cache_size:
            oldest = next(iter(self._label_cache))
            del self._label_cache[oldest]

    def date2didx(self, ds: int) -> int:
        didx = self.universe.date2idx(int(ds))
        return max(didx, self.data_start_didx)

    def didx2date(self, didx: int) -> int:
        didx = max(int(didx), self.data_start_didx)
        return self.universe.idx2date(didx)

    def align_date(self, ds: int) -> int:
        aligned = self.didx2date(self.date2didx(ds))
        return max(aligned, self.data_start_ds)

    def previous_date(self, ds: int, offset: int = 1) -> int:
        didx = self.date2didx(ds) - int(offset)
        assert didx >= self.data_start_didx, f"date {ds} has no {offset}-day input history"
        return self.didx2date(didx)

    def source_date(self, ds: int, freq: str) -> int:
        if self.execution_freq != "1d" and freq == "1d":
            return self.previous_date(ds)
        return int(ds)

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
            groups: dict[str, torch.Tensor] = {}
            for freq in self.freqs:
                source_ds = self.source_date(ds, freq)
                tensors = [
                    self.registry.get_data(name, source_ds, source_ds)[0].to(torch.float32)
                    for name in self.input_names_by_freq[freq]
                ]
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
        cache_days = int(getattr(self.registry, "cache_days", self.config.registry_cache_days))
        assert len(days) <= cache_days, (
            f"feature prefetch chunk has {len(days)} days, cache_days={cache_days}"
        )
        self._sync_monitor_refs()
        stats = LoadStats()
        for freq in self.freqs:
            source_days = [self.source_date(ds, freq) for ds in days]
            if source_days:
                loaded_stats = self.registry._ensure_range(
                    self.input_names_by_freq[freq], min(source_days), max(source_days)
                )
                if loaded_stats is not None:
                    stats.merge(loaded_stats)
        return stats

    def prefetch_labels(self, days: Sequence[int], ret_days: int = 1) -> LoadStats:
        assert self.execution_freq == "1d"
        if self.label_name is None:
            return LoadStats()
        assert 0 < int(ret_days) <= self.registry.cache_days, (
            f"ret_days={ret_days} exceeds registry cache_days={self.registry.cache_days}"
        )
        stats = LoadStats()
        for ds in days:
            end_didx = self.date2didx(self.align_date(ds))
            start_didx = end_didx - int(ret_days) + 1
            if start_didx < self.data_start_didx:
                raise ValueError(f"not enough label history for ds={ds}, ret_days={ret_days}")
            stats.merge(
                self.registry._ensure_range(
                    (self.label_name,),
                    self.didx2date(start_didx),
                    self.didx2date(end_didx),
                )
            )
        return stats

    def prefetch_targets(self, days: Sequence[int]) -> LoadStats:
        assert self.execution_freq != "1d"
        days = [self.align_date(ds) for ds in days]
        assert len(days) <= self.registry.cache_days, (
            f"target prefetch chunk has {len(days)} days, cache_days={self.registry.cache_days}"
        )
        if not days:
            return LoadStats()
        assert self.target_name is not None
        return self.registry._ensure_range((self.target_name,), min(days), max(days))

    def gen_base_universe_mask(self, ds: int) -> torch.Tensor:
        ds = self.align_date(ds)
        self._sync_monitor_refs()
        return self.base_universe_source.load_day(ds)

    def gen_valid_mask(self, ds: int) -> torch.Tensor:
        ds = self.align_date(ds)
        if self.execution_freq != "1d":
            ds = self.previous_date(ds)
        self._sync_monitor_refs()
        valid = self.valid_source.load_day(ds)
        filtered = self.filtered_source.load_day(ds)
        return valid & filtered & self.gen_base_universe_mask(ds)

    def preprocess_target(
        self,
        target_values: torch.Tensor,
        valid_mask: torch.Tensor,
        ds: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_values = target_values.clone()
        valid_values = target_values[valid_mask]
        if valid_values.numel() > 0:
            valid_values = winsorize_by_quantile(valid_values, 0.01, 0.99)
            valid_values = valid_values - nanmedian(valid_values)
            valid_values = valid_values / (nanstd(valid_values) + 1e-8)
            valid_values = truncate(valid_values, -3.0, 3.0)
            valid_values = normalize_by_max_abs(valid_values)
            target_values[valid_mask] = valid_values

        target_values[~valid_mask] = 0.0
        target_values = nan_to_num(target_values, 0.0).to(self.dtype)
        return target_values, to_bool_mask(valid_mask)

    def preprocess_label(
        self,
        label_values: torch.Tensor,
        valid_mask: torch.Tensor,
        ds: int,
        ret_days: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess_target(label_values, valid_mask, ds)

    def gen_label(self, ds: int, ret_days: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.execution_freq == "1d"
        if self.label_name is None:
            raise ValueError("gen_label requires one role='label' data item")
        ds = self.align_date(ds)
        cache_key = (ds, int(ret_days))
        cached = self._label_cache_get(cache_key)
        if cached is not None:
            return cached
        end_didx = self.date2didx(ds)
        start_didx = end_didx - int(ret_days) + 1
        if start_didx < self.data_start_didx:
            raise ValueError(f"not enough label history for ds={ds}, ret_days={ret_days}")
        start_ds = self.didx2date(start_didx)
        with _maybe_section(self.monitor, "gen_label.load_returns", ds, level="full"):
            label_data = self.registry.get_data(self.label_name, start_ds, ds).to(torch.float32)
        with _maybe_section(self.monitor, "gen_label.aggregate", ds, level="full"):
            returns = nan_to_num(label_data, 0.0)
            decay_weights = torch.arange(
                int(ret_days), 0, -1, dtype=torch.float32, device=returns.device
            )
            cret = torch.tensordot(decay_weights, returns, dims=([0], [0]))
            cret[torch.isinf(cret)] = torch.nan
        with _maybe_section(self.monitor, "gen_label.valid_mask", ds, level="full"):
            valid_mask = self.gen_valid_mask(start_ds) & torch.isfinite(cret)
        label = self.preprocess_label(cret, valid_mask, ds, ret_days=int(ret_days))
        self._label_cache_set(cache_key, label)
        return label

    def gen_raw_target(self, ds: int, ti: int) -> torch.Tensor:
        assert self.execution_freq != "1d"
        ds = self.align_date(ds)
        self._sync_monitor_refs()
        bar_id = target_bar_index(self.target_freq, int(ti))
        with _maybe_section(self.monitor, "gen_target.load_returns", ds, level="full"):
            assert self.target_name is not None
            values = self.registry.get_data(self.target_name, ds, ds)[0]
        return values[bar_id].to(torch.float32)

    def gen_target(self, ds: int, ti: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.execution_freq != "1d"
        values = self.gen_raw_target(ds, ti)
        valid_mask = self.gen_valid_mask(ds) & torch.isfinite(values)
        return self.preprocess_target(values, valid_mask, ds)

    def _feature_available_mask(self, feature_window: FeatureGroups) -> torch.Tensor:
        return torch.ones(feature_window.stock_count(), dtype=torch.bool)

    def _mask_window_tail_by_ti(self, feature_window: FeatureGroups, target_ti: int) -> FeatureGroups:
        if self.execution_freq != "1d":
            target_bar_index(self.target_freq, int(target_ti))
        masked: dict[str, torch.Tensor] = {}
        for freq, tensor in feature_window.items():
            if freq == "1d":
                masked[freq] = tensor
                continue
            cur = tensor.clone()
            if self.execution_freq == "1d":
                future_mask = torch.as_tensor(
                    np.asarray(CANONICAL_BAR_TIMES[freq], dtype=np.int64) > int(target_ti),
                    dtype=torch.bool,
                    device=tensor.device,
                )
                if not future_mask.any().item():
                    masked[freq] = tensor
                    continue
                cur[-1, :, future_mask, :] = torch.nan
            else:
                available = available_bar_count(freq, self.target_freq, int(target_ti))
                if available >= tensor.shape[-2]:
                    masked[freq] = tensor
                    continue
                cur[-1, :, available:, :] = torch.nan
            masked[freq] = cur
        return FeatureGroups(masked, feature_window.freqs)

    def transform_feature_window(
        self, feature_window: FeatureGroups, *, target_ti: int | None = None, stage: str
    ) -> FeatureGroups:
        if self.execution_freq == "1d" and self.current_ti < 150000:
            feature_window = self._mask_window_tail_by_ti(feature_window, self.current_ti)
        elif self.execution_freq != "1d":
            assert target_ti is not None
            feature_window = self._mask_window_tail_by_ti(feature_window, target_ti)
        return feature_window.to(dtype=self.dtype)

    def process_feature_window(
        self, feature_window: FeatureGroups, *, target_ti: int | None = None
    ) -> tuple[FeatureGroups, torch.Tensor]:
        feature_window = self.transform_feature_window(
            feature_window, target_ti=target_ti, stage="predict"
        )
        available_mask = self._feature_available_mask(feature_window)
        return feature_window.select_stocks(torch.where(available_mask)[0]), available_mask

    def load_feature_window(
        self, end_ds: int, ts_days: int, target_ti: int | None = None
    ) -> tuple[FeatureGroups, torch.Tensor]:
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
        return self.process_feature_window(
            FeatureGroups.stack(features, dim=0), target_ti=target_ti
        )


class ComboTrainDataset(Dataset):
    def __init__(
        self,
        loader: ComboDataLoader,
        end_ds: int,
        ndays: int,
        x_delay: int | None = None,
        ts_days: int = 8,
        validinsts: torch.Tensor | None = None,
        load_chunk_days: int | None = None,
        codec: Codec | None = None,
    ):
        self.loader = loader
        self.codec = codec or PassthroughCodec(loader.dtype)
        self.group_codec = GroupCodec(self.codec, loader.freqs)
        self.end_ds = loader.align_date(end_ds)
        self.ndays = int(ndays)
        self.x_delay = int(x_delay or 1)
        self.ts_days = int(ts_days)
        self.load_chunk_days = min(
            int(load_chunk_days or loader.registry.cache_days), loader.registry.cache_days
        )
        self.end_didx = loader.date2didx(self.end_ds)
        if loader.execution_freq == "1d":
            self.start_didx = max(
                loader.data_start_didx + self.x_delay - 1,
                self.end_didx - self.ndays + 1,
            )
        else:
            self.start_didx = max(
                loader.data_start_didx + self.ts_days,
                self.end_didx - self.ndays + 1,
            )
        self.ndays = self.end_didx - self.start_didx + 1
        if loader.execution_freq == "1d":
            self.storage_start_didx = self.start_didx - self.x_delay + 1
            self.storage_days = self.ndays
        else:
            self.storage_start_didx = self.start_didx - self.ts_days + 1
            self.storage_days = self.end_didx - self.storage_start_didx + 1

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
                freq: (self.storage_days, *shape)
                for freq, shape in loader.feature_group_shapes(self.numValidinsts).items()
            }
            self.X, self.X_meta = self.group_codec.allocate(
                group_shapes,
                device="cpu",
                logical_dtype=loader.dtype,
            )
            if loader.execution_freq == "1d":
                self.Y = torch.zeros((self.ndays, self.numValidinsts), dtype=loader.dtype)
                self.W = torch.zeros((self.ndays, self.numValidinsts), dtype=loader.dtype)
            else:
                parts = len(loader.target_times)
                self.Y = torch.zeros((self.ndays, parts, self.numValidinsts), dtype=loader.dtype)
                self.W = torch.zeros((self.ndays, parts, self.numValidinsts), dtype=torch.bool)

        progress_start = time.perf_counter()
        loaded_days = 0
        with _maybe_section(monitor, "dataset_init.loop_total", self.end_ds, level="full"):
            if loader.execution_freq == "1d":
                for window_start in range(0, self.ndays, self.load_chunk_days):
                    window_end = min(window_start + self.load_chunk_days, self.ndays)
                    feature_days = [
                        loader.didx2date(self.storage_start_didx + offset)
                        for offset in range(window_start, window_end)
                    ]
                    label_days = [
                        loader.didx2date(self.start_didx + offset)
                        for offset in range(window_start, window_end)
                    ]
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
                        label_ds = loader.didx2date(self.start_didx + offset)
                        feature_ds = loader.didx2date(self.storage_start_didx + offset)
                        x = loader.gen_feature(feature_ds).select_stocks(self.validinsts)
                        y, w = loader.gen_label(label_ds, ret_days=self.x_delay)
                        self.group_codec.encode_into(self.X, self.X_meta, offset, x)
                        self.Y[offset] = torch.nan_to_num(y[self.validinsts], nan=0.0)
                        self.W[offset] = w[self.validinsts].to(loader.dtype)
                        loaded_days += 1
            else:
                for window_start in range(0, self.storage_days, self.load_chunk_days):
                    window_end = min(window_start + self.load_chunk_days, self.storage_days)
                    feature_days = [
                        loader.didx2date(self.storage_start_didx + offset)
                        for offset in range(window_start, window_end)
                    ]
                    load_start = time.perf_counter()
                    feature_stats = loader.prefetch_features(feature_days)
                    load_time = time.perf_counter() - load_start
                    if loader.verbose:
                        detail = (
                            f"days {feature_days[0]}-{feature_days[-1]}, chunk {window_end - window_start}, "
                            f"raw {feature_stats.raw_time:.2f}s, ops {feature_stats.ops_time:.2f}s, "
                            f"load {load_time:.2f}s"
                        )
                        print_progress(
                            "Stage:load_train_days",
                            window_end,
                            self.storage_days,
                            progress_start,
                            detail,
                            final=window_end == self.storage_days,
                        )
                    for offset in range(window_start, window_end):
                        feature_ds = loader.didx2date(self.storage_start_didx + offset)
                        x = loader.gen_feature(feature_ds).select_stocks(self.validinsts)
                        self.group_codec.encode_into(self.X, self.X_meta, offset, x)
                        loaded_days += 1

                for window_start in range(0, self.ndays, self.load_chunk_days):
                    window_end = min(window_start + self.load_chunk_days, self.ndays)
                    target_days = [
                        loader.didx2date(self.start_didx + offset)
                        for offset in range(window_start, window_end)
                    ]
                    loader.prefetch_targets(target_days)
                    for offset in range(window_start, window_end):
                        target_ds = loader.didx2date(self.start_didx + offset)
                        for part, ti in enumerate(loader.target_times):
                            y, w = loader.gen_target(target_ds, ti)
                            self.Y[offset, part] = y[self.validinsts]
                            self.W[offset, part] = w[self.validinsts]

        self._decoded_day: int | None = None
        self._decoded_window: FeatureGroups | None = None

    def _build_validinsts(self) -> torch.Tensor:
        masks = []
        for didx in range(self.start_didx, self.end_didx + 1):
            ds = self.loader.didx2date(didx)
            if self.loader.execution_freq == "1d":
                ds = self.loader.didx2date(didx - self.x_delay + 1)
            else:
                ds = self.loader.previous_date(ds)
            masks.append(self.loader.gen_base_universe_mask(ds))
        stacked = torch.stack(masks, dim=0)
        return torch.where(stacked.any(dim=0))[0]

    def __len__(self) -> int:
        if self.loader.execution_freq == "1d":
            return max(0, self.ndays - self.ts_days + 1)
        return max(0, self.ndays * len(self.loader.target_times))

    def sample_coordinates(self, idx: int) -> tuple[int, int, int, int]:
        assert self.loader.execution_freq != "1d"
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        day_offset, part = divmod(int(idx), len(self.loader.target_times))
        di = self.loader.didx2date(self.start_didx + day_offset)
        ti = int(self.loader.target_times[part])
        return day_offset, di, ti, int(self.loader.target_bar_ids[part])

    def __getitem__(self, idx: int):
        if self.loader.execution_freq == "1d":
            x = self.group_codec.decode(
                self.X,
                self.X_meta,
                slice(idx, idx + self.ts_days),
                out_dtype=self.loader.dtype,
            )
            x = self.loader.transform_feature_window(x, stage="train")
            y = self.Y[idx + self.ts_days - 1]
            w = self.W[idx + self.ts_days - 1]
            return idx, x, y, w
        day_offset, di, ti, _ = self.sample_coordinates(idx)
        if self._decoded_day != day_offset:
            self._decoded_window = self.group_codec.decode(
                self.X,
                self.X_meta,
                slice(day_offset, day_offset + self.ts_days),
                out_dtype=self.loader.dtype,
            )
            self._decoded_day = day_offset
        assert self._decoded_window is not None
        x = self.loader.transform_feature_window(
            self._decoded_window, target_ti=ti, stage="train"
        )
        y = self.Y[day_offset, self.loader.target_times.index(ti)]
        w = self.W[day_offset, self.loader.target_times.index(ti)]
        return idx, di, ti, x, y, w


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

    def clear(self):
        self.start_didx = -1

    def append(self, x: FeatureGroups, didx: int):
        if self.start_didx < 0:
            self.start_didx = didx
        pos = (didx - self.start_didx) % self.keepdays
        self.group_codec.encode_into(self.buffer, self.meta, pos, x)

    def get(self, didx_list: Iterable[int]) -> FeatureGroups:
        pos_list = [int((didx - self.start_didx) % self.keepdays) for didx in didx_list]
        return self.group_codec.decode(self.buffer, self.meta, pos_list, out_dtype=self.dtype)
