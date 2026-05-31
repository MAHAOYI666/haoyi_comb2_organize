# haoyi 2026/05/5

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd
import torch
from factorsim import IndexMask, Memmaper2
from torch.utils.data import Dataset

from .codec import Codec, PassthroughCodec, build_codec
from .op_utils import cs_zscore, nan_to_num, nanmedian, nanstd, normalize_by_max_abs, to_bool_mask, truncate, winsorize_by_quantile

ORGANIZE_ROOT = Path(__file__).resolve().parents[3]
if str(ORGANIZE_ROOT) not in sys.path:
    sys.path.insert(0, str(ORGANIZE_ROOT))
from vendor.perf_monitor import print_progress

MASK = IndexMask()


def _maybe_section(monitor, event: str, ds: int | None = None, *, level: str = "full"):
    if monitor is None:
        return nullcontext()
    return monitor.section(event, date=ds)


class FeatureSource(Protocol):
    feature_dim: int

    def load_day(self, ds: int) -> torch.Tensor:
        ...

    def prefetch_days(self, days: Sequence[int]):
        ...


@dataclass(frozen=True)
class OpSpec:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class FeatureSpec:
    kind: str
    name: str
    path: str | None = None
    mode: str = "read_dump"
    config_path: str | None = None
    ops: tuple[OpSpec, ...] = ()


def _coerce_op_spec(value: OpSpec | dict[str, Any]) -> OpSpec:
    if isinstance(value, OpSpec):
        return value
    return OpSpec(name=str(value["name"]), params=dict(value.get("params", {})))


def _coerce_feature_spec(value: FeatureSpec | dict[str, Any]) -> FeatureSpec:
    if isinstance(value, FeatureSpec):
        return value
    ops = tuple(_coerce_op_spec(op) for op in value.get("ops", ()))
    return FeatureSpec(
        kind=str(value["kind"]),
        name=str(value["name"]),
        path=value.get("path"),
        mode=str(value.get("mode", "read_dump")),
        config_path=value.get("config_path"),
        ops=ops,
    )


def _apply_feature_ops(x: torch.Tensor, ops: Sequence[OpSpec]) -> torch.Tensor:
    if not ops:
        return x
    out = x.to(torch.float32)
    for op in ops:
        name = op.name.strip().lower()
        params = op.params
        if name in {"cs_zscore", "zscore"}:
            out = cs_zscore(out.unsqueeze(0)).squeeze(0)
        elif name == "truncate":
            out = truncate(out, float(params.get("min", -4.0)), float(params.get("max", 4.0)))
        elif name in {"nan_to_num", "fillna"}:
            out = nan_to_num(out, float(params.get("value", 0.0)))
        elif name == "winsorize_by_quantile":
            out = winsorize_by_quantile(out, float(params.get("low", 0.01)), float(params.get("high", 0.99)))
        elif name == "normalize_by_max_abs":
            out = normalize_by_max_abs(out)
        else:
            raise ValueError(f"unsupported feature op: {op.name}")
    return out


class FeatureItemSource:
    feature_dim = 1

    def __init__(self, spec: FeatureSpec, dtype: torch.dtype):
        self.spec = spec
        self.name = spec.name
        self.dtype = dtype

    def _load_raw_day(self, ds: int) -> torch.Tensor:
        raise NotImplementedError

    def load_day(self, ds: int) -> torch.Tensor:
        x = self._load_raw_day(int(ds))
        x = _apply_feature_ops(x, self.spec.ops)
        x[torch.isinf(x)] = torch.nan
        return x.to(self.dtype)

    def prefetch_days(self, days: Sequence[int]):
        return None


class MemmapFactorItemSource(FeatureItemSource):
    def __init__(self, spec: FeatureSpec, dtype: torch.dtype):
        super().__init__(spec, dtype)
        if not spec.path:
            raise ValueError(f"factor feature {spec.name!r} requires path")
        self.path = spec.path
        self._mmap: Memmaper2 | None = None
        self._day_cache: dict[int, torch.Tensor] = {}

    def _get_mmap(self) -> Memmaper2:
        if self._mmap is None:
            self._mmap = Memmaper2(self.path)
        return self._mmap

    def _load_raw_day(self, ds: int) -> torch.Tensor:
        cached = self._day_cache.pop(int(ds), None)
        if cached is not None:
            return cached
        monitor = getattr(self, "monitor", None)
        with _maybe_section(monitor, "feature.factor_item_load", ds, level="full"):
            data = self._get_mmap().load(start_ds=ds, end_ds=ds)[:]
        return torch.as_tensor(np.asarray(data)[0], dtype=self.dtype)

    def prefetch_days(self, days: Sequence[int]):
        days = list(dict.fromkeys(int(ds) for ds in days))
        needed_days = [ds for ds in days if ds not in self._day_cache]
        if not needed_days:
            return
        start_ds = min(needed_days)
        end_ds = max(needed_days)
        day_set = set(needed_days)
        trading_days = [int(ds) for ds in MASK.date if start_ds <= int(ds) <= end_ds]
        data = self._get_mmap().load(start_ds=start_ds, end_ds=end_ds)[:]
        arr = np.asarray(data)
        for offset, ds in enumerate(trading_days[: len(arr)]):
            if ds in day_set:
                self._day_cache[ds] = torch.as_tensor(arr[offset], dtype=self.dtype)


class AlphaParquetItemSource(FeatureItemSource):
    def __init__(self, spec: FeatureSpec, dtype: torch.dtype, codes: Sequence[int | str]):
        super().__init__(spec, dtype)
        if spec.mode != "read_dump":
            raise NotImplementedError(f"alpha mode {spec.mode!r} is not implemented in the MVP")
        if not spec.path:
            raise ValueError(f"alpha feature {spec.name!r} requires path")
        self.path = spec.path
        self.codes = [str(code).zfill(6) for code in codes]
        self._frame: pd.DataFrame | None = None

    def _load_frame(self) -> pd.DataFrame:
        if self._frame is None:
            frame = pd.read_parquet(self.path)
            frame.index = frame.index.astype(int)
            frame.columns = frame.columns.astype(str).str.zfill(6)
            self._frame = frame.sort_index().reindex(columns=self.codes)
        return self._frame

    def _load_raw_day(self, ds: int) -> torch.Tensor:
        monitor = getattr(self, "monitor", None)
        with _maybe_section(monitor, "feature.alpha_parquet_load", ds, level="full"):
            frame = self._load_frame()
            if int(ds) not in frame.index:
                raise KeyError(f"alpha feature {self.name!r} missing date {ds} in {self.path}")
            row = frame.loc[int(ds)]
        return torch.as_tensor(row.to_numpy(dtype=np.float32), dtype=self.dtype)


class CompositeFeatureSource:
    def __init__(self, specs: Sequence[FeatureSpec], dtype: torch.dtype, codes: Sequence[int | str]):
        if not specs:
            raise ValueError("at least one feature is required")
        self.specs = tuple(specs)
        self.dtype = dtype
        self.sources = [self._build_source(spec, dtype, codes) for spec in self.specs]
        self.feature_dim = sum(source.feature_dim for source in self.sources)
        self.feature_names = tuple(source.name for source in self.sources)

    def _build_source(self, spec: FeatureSpec, dtype: torch.dtype, codes: Sequence[int | str]) -> FeatureItemSource:
        kind = spec.kind.strip().lower()
        if kind == "factor":
            return MemmapFactorItemSource(spec, dtype)
        if kind == "alpha":
            return AlphaParquetItemSource(spec, dtype, codes)
        raise ValueError(f"unsupported feature kind: {spec.kind}")

    def _sync_monitor_refs(self):
        monitor = getattr(self, "monitor", None)
        for source in self.sources:
            source.monitor = monitor

    def load_day(self, ds: int) -> torch.Tensor:
        self._sync_monitor_refs()
        parts = [source.load_day(ds) for source in self.sources]
        return torch.stack(parts, dim=-1)

    def prefetch_days(self, days: Sequence[int]):
        self._sync_monitor_refs()
        for source in self.sources:
            source.prefetch_days(days)


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
    features: Sequence[FeatureSpec] | None = None
    compression: str = "none"
    apply_global_ops: bool = True
    base_universe_path: str | None = None
    verbose: bool = False


class ComboDataLoader:
    def __init__(self, config: LoaderConfig, cube_source: FeatureSource | None = None, feature_cache_size: int = 2500, label_cache_size: int = 2500):
        self.config = config
        self.dtype = self.config.dtype
        self.codec = build_codec(self.config.compression, self.dtype)
        self.mask = MASK
        self.data_start_ds = int(self.config.data_start_ds)
        self.data_start_didx = int(self.mask.date2didx(self.data_start_ds))
        raw_feature_specs = self.config.features or tuple(
            FeatureSpec(kind="factor", name=Path(path).name, path=path) for path in self.config.factor_paths
        )
        feature_specs = tuple(_coerce_feature_spec(spec) for spec in raw_feature_specs)
        self.feature_source = CompositeFeatureSource(feature_specs, dtype=self.dtype, codes=self.mask.code)
        self.factor_source = self.feature_source
        self.cube_source = cube_source or EmptyCubeSource(dtype=self.dtype)
        self.label_source = MemmapLabelSource(self.config.label_path, dtype=self.dtype)
        self.valid_source = MemmapMaskSource(self.config.valid_path)
        self.filtered_source = MemmapMaskSource(self.config.filtered_path)
        self.base_universe_source = MemmapMaskSource(self.config.base_universe_path)
        self.monitor = None
        self.num_features = self.feature_source.feature_dim + self.cube_source.feature_dim
        self.feature_names = tuple(self.feature_source.feature_names) + tuple(
            f"cube_{idx:03d}" for idx in range(self.cube_source.feature_dim)
        )
        self.verbose = bool(getattr(config, "verbose", False))
        self._feature_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._feature_cache_size = int(feature_cache_size)
        self._label_cache: OrderedDict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._label_cache_size = int(label_cache_size)

    def _sync_monitor_refs(self):
        self.feature_source.monitor = self.monitor
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

    def _build_feature(self, ds: int) -> torch.Tensor:
        self._sync_monitor_refs()
        with _maybe_section(self.monitor, "gen_feature.feature_load", ds, level="full"):
            factor = self.feature_source.load_day(ds).to(torch.float32)
        with _maybe_section(self.monitor, "gen_feature.cube_load", ds, level="full"):
            cube = self.cube_source.load_day(ds).to(torch.float32)
        if cube.shape[1] == 0:
            feature = factor
        else:
            feature = torch.cat([factor, cube], dim=-1)
        feature[torch.isinf(feature)] = torch.nan
        if self.config.apply_global_ops:
            with _maybe_section(self.monitor, "gen_feature.cs_zscore", ds, level="full"):
                feature = cs_zscore(feature.transpose(0, 1)).transpose(0, 1)
            with _maybe_section(self.monitor, "gen_feature.truncate_nan_to_num", ds, level="full"):
                feature = truncate(feature, -4.0, 4.0)
                feature = nan_to_num(feature, 0.0)
        return feature.to(self.dtype)

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
        self._sync_monitor_refs()
        self.feature_source.prefetch_days(days)
        self.cube_source.prefetch_days(days)

    def prefetch_labels(self, days: Sequence[int], ret_days: int = 1):
        label_days: list[int] = []
        for ds in days:
            end_didx = self.date2didx(self.align_date(ds))
            start_didx = end_didx - int(ret_days) + 1
            if start_didx < self.data_start_didx:
                raise ValueError(f"not enough label history for ds={ds}, ret_days={ret_days}")
            label_days.extend(self.didx2date(didx) for didx in range(start_didx, end_didx + 1))
        self._sync_monitor_refs()
        self.label_source.prefetch_days(label_days)

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

    def gen_label(self, ds: int, ret_days: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
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
        x = self.codec.decode(
            self.X,
            self.X_meta,
            slice(idx, idx + self.ts_days),
            out_dtype=self.loader.dtype,
        )
        x = nan_to_num(x, 0.0).to(self.loader.dtype)
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
