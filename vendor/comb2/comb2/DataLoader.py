from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import inspect
import math
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from comb2_simbase.cache_layout import (
    ashare_cache_path, stock_mask_path, BASE_UNIVERSE_MASK_NAME,
    VALID_MASK_NAME, FILTERED_MASK_NAME,
)
from .DataRegistry import DataItem, DataRegistry, LoadedSource, LoadStats, MASK, Universe, _coerce_data_item
from .codec import build_codec, PassthroughCodec
from .op_utils import cs_zscore, nan_to_num, nanmedian, nanstd, normalize_by_max_abs, truncate, winsorize_by_quantile


@dataclass
class LoaderConfig:
    dtype: torch.dtype = torch.float16
    data_start_ds: int = 20160101
    data_offset: int = 1024
    compression: str = "none"
    cache_path: str | None = None
    sample_times: tuple[int, ...] = (100000,)
    verbose: bool = False
    load_chunk_days: int | None = None


class ComboDataLoader:
    """Research-owned sources; model inputs are plain [date, stock, feature] tensors."""

    def __init__(self, config: LoaderConfig):
        self.config = config
        self.dtype = config.dtype
        self.codec = build_codec(config.compression, self.dtype)
        self.load_chunk_days = 64 if config.load_chunk_days is None else int(config.load_chunk_days)
        assert self.load_chunk_days > 0
        self.mask = MASK
        self.universe = Universe.from_mask(self.mask, self.dtype, data_offset=config.data_offset)
        self.data_start_ds = int(config.data_start_ds)
        self.data_start_didx = self.universe.date2idx(self.data_start_ds)
        self.sample_times = tuple(config.sample_times)
        self.current_ti = self.sample_times[0]
        self.current_date = self.data_start_ds
        self.monitor = None
        self.verbose = config.verbose
        root = Path(inspect.getfile(type(self))).resolve().parent
        sources = []
        for declared in self.data_requirements():
            item = _coerce_data_item(declared)
            if item.path is not None:
                path = Path(item.path).expanduser()
                item = replace(item, path=str((path if path.is_absolute() else root / path).resolve()))
            sources.append(item)
        assert sources, "data_requirements must declare at least one source"
        self.validity = self.model_validity_source()
        if self.validity is None and config.cache_path:
            sources.extend(DataItem(
                f"mask.{name}", path=str(stock_mask_path(config.cache_path, field)), delay=1,
            ) for name, field in (
                ("base", BASE_UNIVERSE_MASK_NAME), ("valid", VALID_MASK_NAME),
                ("filtered", FILTERED_MASK_NAME),
            ))
        self.registry = DataRegistry(
            sources, universe=self.universe, data_start_ds=self.data_start_ds,
            ashare_cache_path=str(ashare_cache_path(config.cache_path)) if config.cache_path else None,
            config_path=str(root),
            load_chunk_days=self.load_chunk_days,
            process_source=self.process_source, verbose=config.verbose,
        )
        self.registry.set_current_ti(self.current_ti)
        self.input_names = tuple(self.model_input_sources())
        assert self.input_names and len(set(self.input_names)) == len(self.input_names)
        assert set(self.input_names) <= self.registry.items.keys()
        self.target = self.model_target()
        self._feature_names = ()

    def data_requirements(self) -> Sequence[DataItem]:
        raise NotImplementedError("declare Memmap sources in ResearchLoader.data_requirements")

    def process_source(self, source: DataItem, loaded: LoadedSource) -> LoadedSource:
        assert loaded.values.ndim == 3 and not loaded.times, (
            f"{source.name}: reduce raw bars in process_source before returning to the loader"
        )
        return loaded

    def model_input_sources(self) -> tuple[str, ...]:
        raise NotImplementedError

    def model_target(self) -> tuple[str, str] | None:
        return None

    def model_validity_source(self) -> tuple[str, tuple[str, ...], str] | None:
        return None

    def set_current_ti(self, ti):
        self.current_ti = int(ti)
        self.registry.set_current_ti(ti)

    def set_point(self, ds, ti, *, refresh=False):
        self.current_date = int(ds)
        self.set_current_ti(ti)
        self.registry.set_point(ds, ti, refresh=refresh)

    @contextmanager
    def cache_scope(self):
        with self.registry.cache_scope():
            yield self

    def release_working_cache(self):
        self.registry.release_working_cache()

    def date2didx(self, ds):
        return self.universe.date2idx(int(ds))

    def didx2date(self, idx):
        assert 0 <= int(idx) < len(self.universe.dates), "trading date index outside calendar"
        return self.universe.idx2date(int(idx))

    def previous_date(self, ds, offset=1):
        return self.didx2date(self.date2didx(ds) - int(offset))

    def align_date(self, ds):
        self.date2didx(ds)
        return int(ds)

    def source_history(self, name, start_ds, end_ds):
        return self.registry.get_data(name, start_ds, end_ds)

    def source_field(self, name, field, start_ds, end_ds):
        return self.registry.get_field(name, field, start_ds, end_ds)

    def source_field_day(self, name, field, ds):
        """Read one field for one day without allocating a leading-day stack."""
        name = self.registry._resolve_name(name)
        values = self.registry.get_day_many((name,), ds)[0]
        return values[..., self.registry.columns[name].index(field)]

    @property
    def feature_names(self):
        if not self._feature_names:
            self.gen_feature(self.current_date, self.current_ti)
        return self._feature_names

    @property
    def num_features(self):
        return len(self.feature_names)

    def build_raw_feature(self, ds):
        sources = self.registry.get_day_many(self.input_names, ds)
        names = tuple(f"{name}:{field}" for name in self.input_names for field in self.registry.columns[name])
        if self._feature_names:
            assert names == self._feature_names, "model feature order changed"
        self._feature_names = names
        values = torch.cat(sources, dim=-1)
        values[torch.isinf(values)] = torch.nan
        return values

    def preprocess_features(self, values, ds, ti):
        values = cs_zscore(values.transpose(0, 1)).transpose(0, 1)
        return nan_to_num(truncate(values, -4, 4), 0).to(self.dtype)

    def gen_feature(self, ds, ti=None):
        ds = self.align_date(ds)
        if ti is not None:
            self.set_current_ti(ti)
        value = self.preprocess_features(self.build_raw_feature(ds), ds, self.current_ti)
        assert value.shape == (len(self.mask.code), self.num_features)
        return value

    def prefetch_features(self, days, chunk_days=None):
        assert self.registry.in_cache_scope, "prefetch_features requires an active cache scope"
        stats = LoadStats()
        days = list(days)
        if not days:
            return stats
        width = self.load_chunk_days if chunk_days is None else int(chunk_days)
        assert width > 0
        for offset in range(0, len(days), width):
            chunk = days[offset:offset + width]
            stats.merge(self.registry._ensure_range(self.input_names, chunk[0], chunk[-1]))
        return stats

    def prefetch_targets(self, days, chunk_days=None):
        assert self.registry.in_cache_scope, "prefetch_targets requires an active cache scope"
        days = list(days)
        names = [self.target[0]] if self.target is not None else []
        if self.validity is not None:
            names.append(self.validity[0])
        elif self.config.cache_path:
            names.extend(("mask.base", "mask.valid", "mask.filtered"))
        stats = LoadStats()
        if not days:
            return stats
        width = self.load_chunk_days if chunk_days is None else int(chunk_days)
        assert width > 0
        for offset in range(0, len(days), width):
            chunk = days[offset:offset + width]
            stats.merge(self.registry._ensure_range(tuple(dict.fromkeys(names)), chunk[0], chunk[-1]))
        return stats

    def prefetch_base_universe(self, days, chunk_days=None):
        """Batch the source used by ``gen_base_universe_mask`` when one exists."""
        assert self.registry.in_cache_scope, (
            "prefetch_base_universe requires an active cache scope"
        )
        days = list(days)
        stats = LoadStats()
        if not days:
            return stats
        if self.validity is not None:
            names = (self.validity[0],)
        elif self.config.cache_path:
            names = ("mask.base",)
        else:
            return stats
        width = self.load_chunk_days if chunk_days is None else int(chunk_days)
        assert width > 0
        for offset in range(0, len(days), width):
            chunk = days[offset:offset + width]
            stats.merge(self.registry._ensure_range(names, chunk[0], chunk[-1]))
        return stats

    def gen_base_universe_mask(self, ds):
        if self.validity is None:
            if not self.config.cache_path:
                return torch.ones(len(self.mask.code), dtype=torch.bool)
            name = field = "mask.base"
        else:
            name, _, field = self.validity
        values = self.source_field_day(name, field, ds)
        return torch.isfinite(values) & (values != 0)

    def gen_valid_mask(self, ds, ti=None):
        if ti is not None:
            self.set_current_ti(ti)
        valid = self.gen_base_universe_mask(ds)
        if self.validity is not None:
            name, fields, _ = self.validity
            for field in fields:
                values = self.source_field_day(name, field, ds)
                valid &= torch.isfinite(values) & (values != 0)
        elif self.config.cache_path:
            for name in ("mask.valid", "mask.filtered"):
                values = self.source_field_day(name, name, ds)
                valid &= torch.isfinite(values) & (values != 0)
        return valid

    def gen_raw_target(self, ds, ti):
        """Default: the selected field on ds. Override to index the research target."""
        assert self.target is not None, "model_target must select a source and field"
        self.set_current_ti(ti)
        return self.source_field_day(*self.target, ds).to(torch.float32)

    def preprocess_target(self, values, valid_mask, ds, ti):
        values = values.clone()
        selected = values[valid_mask]
        if selected.numel():
            selected = winsorize_by_quantile(selected, .01, .99)
            selected = selected - nanmedian(selected)
            selected = selected / (nanstd(selected) + 1.e-8)
            selected = normalize_by_max_abs(truncate(selected, -3., 3.))
            values[valid_mask] = selected
        values[~valid_mask] = 0
        return values.to(self.dtype), valid_mask

    def gen_target(self, ds, ti, ret_days=1):
        assert int(ret_days) > 0
        start = self.date2didx(ds)
        components = torch.stack([
            self.gen_raw_target(self.didx2date(start + offset), ti)
            for offset in range(int(ret_days))
        ])
        valid = self.gen_valid_mask(ds, ti) & torch.isfinite(components).all(0)
        weights = torch.arange(ret_days, 0, -1, dtype=components.dtype)
        values = torch.tensordot(weights, torch.nan_to_num(components, nan=0.), dims=([0], [0]))
        return self.preprocess_target(values, valid, ds, ti)

    def transform_feature_window(self, values, *, target_ti, stage):
        assert values.ndim == 3
        return values.to(self.dtype)

    def load_feature_window(self, end_ds, ts_days, target_ti):
        end = self.date2didx(end_ds)
        start = end - int(ts_days) + 1
        assert start >= self.data_start_didx, "insufficient feature history"
        self.set_current_ti(target_ti)
        features = []
        with self.cache_scope():
            for first in range(start, end + 1, self.load_chunk_days):
                days = [self.didx2date(i) for i in range(first, min(first + self.load_chunk_days, end + 1))]
                self.prefetch_features(days)
                features.extend(self.gen_feature(ds, target_ti) for ds in days)
                self.release_working_cache()
        values = torch.stack(features)
        return self.transform_feature_window(values, target_ti=target_ti, stage="predict")


class ComboTrainDataset(Dataset):
    """Preloaded training snapshot with same-time windows across trading days."""

    # Spare instrument columns reserved so rolling can admit new instruments in place.
    column_slack = 0.10

    def __init__(self, loader, end_ds, ndays, x_delay=None, ts_days=8,
                 validinsts=None, load_chunk_days=None, codec=None):
        self.loader = loader
        assert not loader.registry.in_cache_scope, (
            "ComboTrainDataset must be constructed outside an active cache scope"
        )
        self.end_ds = loader.align_date(end_ds)
        self.ret_days = 1 if x_delay is None else int(x_delay)
        assert self.ret_days > 0
        self.ts_days = int(ts_days)
        self.load_chunk_days = loader.load_chunk_days if load_chunk_days is None else int(load_chunk_days)
        assert self.load_chunk_days > 0
        self.end_didx = loader.date2didx(self.end_ds)
        self.start_didx = max(loader.data_start_didx, self.end_didx - int(ndays) + 1)
        self.first_sample = self.start_didx + self.ts_days - 1
        self.last_sample = self.end_didx - self.ret_days + 1
        assert self.first_sample <= self.last_sample, "not enough training history"
        self.ndays = self.end_didx - self.start_didx + 1
        with loader.cache_scope():
            loader.build_raw_feature(loader.didx2date(self.first_sample))
            loader.release_working_cache()
            self._fixed_validinsts = validinsts is not None
            sample_days = self.last_sample - self.first_sample + 1
            self._base_validity = None
            if (
                not self._fixed_validinsts
                and type(self)._build_validinsts is ComboTrainDataset._build_validinsts
            ):
                self._base_validity = torch.empty(
                    (len(loader.sample_times), sample_days, len(loader.mask.code)),
                    dtype=torch.bool,
                )
            self.validinsts = self._build_validinsts() if validinsts is None else validinsts.to(torch.long)
            self.numValidinsts = len(self.validinsts)
            assert self.numValidinsts > 0, "no training instruments"
            self.codec = codec or PassthroughCodec(loader.dtype)
            capacity = self.numValidinsts
            if not self._fixed_validinsts:
                capacity += math.ceil(self.numValidinsts * self.column_slack)
            self._Y_storage = torch.empty((sample_days, len(loader.sample_times), capacity), dtype=loader.dtype)
            self._W_storage = torch.empty_like(self._Y_storage, dtype=torch.bool)
            storage_days = self.last_sample - self.start_didx + 1
            self._X_storage = {
                ti: self.codec.allocate(
                    (storage_days, capacity, loader.num_features), "cpu", loader.dtype,
                )
                for ti in loader.sample_times
            }
            self._set_columns(self.numValidinsts)
            for part, ti in enumerate(loader.sample_times):
                loader.set_current_ti(ti)
                for offset in range(0, storage_days, self.load_chunk_days):
                    stop = min(offset + self.load_chunk_days, storage_days)
                    days = [loader.didx2date(self.start_didx + i) for i in range(offset, stop)]
                    loader.prefetch_features(days, self.load_chunk_days)
                    for i, ds in enumerate(days, offset):
                        values = loader.gen_feature(ds, ti).index_select(0, self.validinsts)
                        self.codec.encode_into(self.X[ti], self.X_meta[ti], i, values)
                    loader.release_working_cache()
                for offset in range(0, sample_days, self.load_chunk_days):
                    stop = min(offset + self.load_chunk_days, sample_days)
                    loader.prefetch_targets([
                        loader.didx2date(self.first_sample + i)
                        for i in range(offset, stop + self.ret_days - 1)
                    ])
                    for day in range(offset, stop):
                        ds = loader.didx2date(self.first_sample + day)
                        y, w = loader.gen_target(ds, ti, ret_days=self.ret_days)
                        self.Y[day, part] = y[self.validinsts]
                        self.W[day, part] = w[self.validinsts]
                    loader.release_working_cache()
        self.load_stats = loader.registry.last_scope_stats

    @staticmethod
    def _shift_left_(values, offset):
        """Shift dim 0 without allocating an overlapping full-size temporary."""
        offset = int(offset)
        if offset <= 0:
            return
        keep = values.shape[0] - offset
        for start in range(0, keep, offset):
            width = min(offset, keep - start)
            values[start:start + width].copy_(
                values[start + offset:start + offset + width]
            )

    def _set_columns(self, count):
        """Expose the first ``count`` storage columns as X/Y/W."""
        self.X = {}
        self.X_meta = {}
        for ti, (storage, meta) in self._X_storage.items():
            self.X[ti] = storage[:, :count]
            self.X_meta[ti] = replace(
                meta,
                logical_shape=(meta.logical_shape[0], count, *meta.logical_shape[2:]),
                storage_shape=(meta.storage_shape[0], count, *meta.storage_shape[2:]),
            )
        self.Y = self._Y_storage[..., :count]
        self.W = self._W_storage[..., :count]

    def _remap_columns_(self, validinsts, offset):
        """Shift dim 0 left by ``offset`` and reorder columns to ``validinsts``.

        Instruments absent from the old selection were outside the base universe
        on every retained day. The data contract gives them zero features there,
        and the default target validity gives them zero targets and False weights.
        """
        kept = torch.isin(validinsts, self.validinsts)
        src = torch.searchsorted(self.validinsts, validinsts[kept])
        dst = torch.where(kept)[0]
        added = torch.where(~kept)[0]
        zero, zero_meta = self.codec.allocate((1, self.loader.num_features), "cpu", self.loader.dtype)
        self.codec.encode_into(
            zero, zero_meta, 0, torch.zeros(self.loader.num_features, dtype=self.loader.dtype),
        )
        storages = [(storage, 1, zero[0]) for storage, _ in self._X_storage.values()]
        storages += [(self._Y_storage, 2, 0), (self._W_storage, 2, False)]
        for storage, axis, fill in storages:
            added_index = (slice(None),) * axis + (added,)
            keep = storage.shape[0] - offset
            for start in range(0, keep, offset):
                width = min(offset, keep - start)
                rows = storage[start + offset:start + offset + width].index_select(axis, src)
                target = storage[start:start + width]
                target.index_copy_(axis, dst, rows)
                target[added_index] = fill
        self.validinsts = validinsts
        self.numValidinsts = len(validinsts)
        self._set_columns(self.numValidinsts)

    def storage_nbytes(self):
        tensors = [
            *(storage for storage, _ in self._X_storage.values()),
            self._Y_storage, self._W_storage, self._base_validity,
        ]
        return sum(
            value.numel() * value.element_size()
            for value in tensors
            if isinstance(value, torch.Tensor)
        )

    def release_storage(self):
        self.X.clear()
        self.X_meta.clear()
        self._X_storage.clear()
        self.Y = None
        self.W = None
        self._Y_storage = None
        self._W_storage = None
        self._base_validity = None

    def _roll_validity(self, first_sample, last_sample, offset):
        for part, ti in enumerate(self.loader.sample_times):
            self._shift_left_(self._base_validity[part], offset)
            self.loader.set_current_ti(ti)
            first = last_sample - offset + 1
            for begin in range(first, last_sample + 1, self.load_chunk_days):
                days = [
                    self.loader.didx2date(idx)
                    for idx in range(
                        begin,
                        min(begin + self.load_chunk_days, last_sample + 1),
                    )
                ]
                self.loader.prefetch_base_universe(days, self.load_chunk_days)
                for idx, ds in enumerate(days, begin):
                    self._base_validity[part, idx - first_sample] = (
                        self.loader.gen_base_universe_mask(ds)
                    )
                self.loader.release_working_cache()
        return torch.where(
            self._base_validity.reshape(-1, self._base_validity.shape[-1]).any(dim=0)
        )[0]

    def roll_forward(self, end_ds, ndays):
        """Update a fixed-size rolling snapshot in place; return False to rebuild."""
        assert self.Y is not None, "released Dataset cannot roll"
        assert int(ndays) == self.ndays
        end_ds = self.loader.align_date(end_ds)
        end_didx = self.loader.date2didx(end_ds)
        start_didx = end_didx - self.ndays + 1
        assert start_didx >= self.loader.data_start_didx
        first_sample = start_didx + self.ts_days - 1
        last_sample = end_didx - self.ret_days + 1
        storage_days = last_sample - start_didx + 1
        sample_days = last_sample - first_sample + 1
        assert sample_days == self.Y.shape[0]
        offset = start_didx - self.start_didx
        assert offset > 0
        if offset >= sample_days:
            return False

        remap = False
        if not self._fixed_validinsts:
            if self._base_validity is None:
                return False
            with self.loader.cache_scope():
                validinsts = self._roll_validity(first_sample, last_sample, offset)
            if not torch.equal(validinsts, self.validinsts):
                if len(validinsts) > self._Y_storage.shape[-1]:
                    return False
                remap = True

        if remap:
            self._remap_columns_(validinsts, offset)
        else:
            for storage in self.X.values():
                self._shift_left_(storage, offset)
            self._shift_left_(self.Y, offset)
            self._shift_left_(self.W, offset)

        self.end_ds = end_ds
        self.end_didx = end_didx
        self.start_didx = start_didx
        self.first_sample = first_sample
        self.last_sample = last_sample

        feature_first = storage_days - offset
        target_first = sample_days - offset
        with self.loader.cache_scope():
            for ti in self.loader.sample_times:
                self.loader.set_current_ti(ti)
                for begin in range(feature_first, storage_days, self.load_chunk_days):
                    stop = min(begin + self.load_chunk_days, storage_days)
                    days = [
                        self.loader.didx2date(self.start_didx + idx)
                        for idx in range(begin, stop)
                    ]
                    self.loader.prefetch_features(days, self.load_chunk_days)
                    for idx, ds in enumerate(days, begin):
                        values = self.loader.gen_feature(ds, ti).index_select(
                            0, self.validinsts
                        )
                        self.codec.encode_into(
                            self.X[ti], self.X_meta[ti], idx, values
                        )
                    self.loader.release_working_cache()

                part = self.loader.sample_times.index(ti)
                for begin in range(target_first, sample_days, self.load_chunk_days):
                    stop = min(begin + self.load_chunk_days, sample_days)
                    self.loader.prefetch_targets([
                        self.loader.didx2date(self.first_sample + idx)
                        for idx in range(begin, stop + self.ret_days - 1)
                    ])
                    for day in range(begin, stop):
                        ds = self.loader.didx2date(self.first_sample + day)
                        y, w = self.loader.gen_target(
                            ds, ti, ret_days=self.ret_days
                        )
                        self.Y[day, part] = y[self.validinsts]
                        self.W[day, part] = w[self.validinsts]
                    self.loader.release_working_cache()
        self.load_stats = self.loader.registry.last_scope_stats
        return True

    def _build_validinsts(self):
        valid = torch.zeros(len(self.loader.mask.code), dtype=torch.bool)
        for part, ti in enumerate(self.loader.sample_times):
            self.loader.set_current_ti(ti)
            for first in range(
                self.first_sample, self.last_sample + 1, self.load_chunk_days
            ):
                days = [
                    self.loader.didx2date(idx)
                    for idx in range(
                        first,
                        min(first + self.load_chunk_days, self.last_sample + 1),
                    )
                ]
                self.loader.prefetch_base_universe(days, self.load_chunk_days)
                for idx, ds in enumerate(days, first):
                    day_valid = self.loader.gen_base_universe_mask(ds)
                    valid |= day_valid
                    if self._base_validity is not None:
                        self._base_validity[part, idx - self.first_sample] = day_valid
                self.loader.release_working_cache()
        return torch.where(valid)[0]

    def __len__(self):
        return (self.last_sample - self.first_sample + 1) * len(self.loader.sample_times)

    def sample_coordinates(self, idx):
        assert 0 <= idx < len(self)
        day, part = divmod(int(idx), len(self.loader.sample_times))
        return self.loader.didx2date(self.first_sample + day), self.loader.sample_times[part]

    def __getitem__(self, idx):
        ds, ti = self.sample_coordinates(idx)
        day, part = divmod(int(idx), len(self.loader.sample_times))
        self.loader.set_current_ti(ti)
        values = self.codec.decode(
            self.X[ti], self.X_meta[ti], slice(day, day + self.ts_days),
            out_dtype=self.loader.dtype,
        )
        x = self.loader.transform_feature_window(values, target_ti=ti, stage="train")
        return idx, ds, ti, x, self.Y[day, part], self.W[day, part]


class ComboBuffer:
    def __init__(self, shape, keepdays, dtype=torch.float16, codec=None):
        self.keepdays = int(keepdays)
        self.dtype = dtype
        self.codec = codec or PassthroughCodec(dtype)
        self.buffer, self.meta = self.codec.allocate((keepdays, *shape), "cpu", dtype)
        self.start_didx = None

    def clear(self):
        self.start_didx = None

    def append(self, values, didx):
        if self.start_didx is None:
            self.start_didx = int(didx)
        self.codec.encode_into(self.buffer, self.meta, (didx - self.start_didx) % self.keepdays, values)

    def get(self, didx_list):
        positions = [(int(i) - self.start_didx) % self.keepdays for i in didx_list]
        return self.codec.decode(self.buffer, self.meta, positions, out_dtype=self.dtype)
