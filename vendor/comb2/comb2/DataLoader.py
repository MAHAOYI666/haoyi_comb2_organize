from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import inspect
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from comb2_simbase.cache_layout import ashare_cache_path
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
    registry_cache_days: int = 64
    sample_times: tuple[int, ...] = (100000,)
    verbose: bool = False


class ComboDataLoader:
    """Research-owned sources; model inputs are plain [date, stock, feature] tensors."""

    def __init__(self, config: LoaderConfig):
        self.config = config
        self.dtype = config.dtype
        self.codec = build_codec(config.compression, self.dtype)
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
        self.registry = DataRegistry(
            sources, universe=self.universe, data_start_ds=self.data_start_ds,
            ashare_cache_path=str(ashare_cache_path(config.cache_path)) if config.cache_path else None,
            config_path=str(root), cache_days=config.registry_cache_days,
            process_source=self.process_source, verbose=config.verbose,
        )
        self.input_names = tuple(self.model_input_sources())
        assert self.input_names and len(set(self.input_names)) == len(self.input_names)
        assert set(self.input_names) <= self.registry.items.keys()
        self.target = self.model_target()
        self.validity = self.model_validity_source()
        self._feature_names = ()
        self._features = OrderedDict()

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
        if refresh:
            for point in list(self._features):
                if point[0] >= int(ds):
                    del self._features[point]

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

    @property
    def feature_names(self):
        if not self._feature_names:
            self.gen_feature(self.current_date, self.current_ti)
        return self._feature_names

    @property
    def num_features(self):
        return len(self.feature_names)

    def build_raw_feature(self, ds):
        sources = [self.registry.get_data(name, ds, ds)[0] for name in self.input_names]
        names = tuple(f"{name}:{field}" for name in self.input_names for field in self.registry.columns[name])
        if self._feature_names:
            assert names == self._feature_names, "model feature order changed"
        self._feature_names = names
        return torch.cat(sources, dim=-1)

    def preprocess_features(self, values, ds, ti):
        values = cs_zscore(values.transpose(0, 1)).transpose(0, 1)
        return nan_to_num(truncate(values, -4, 4), 0).to(self.dtype)

    def gen_feature(self, ds, ti=None):
        ds = self.align_date(ds)
        if ti is not None:
            self.set_current_ti(ti)
        key = (ds, self.current_ti)
        if key not in self._features:
            value = self.preprocess_features(self.build_raw_feature(ds), ds, self.current_ti)
            assert value.shape == (len(self.mask.code), self.num_features)
            storage, meta = self.codec.allocate((1, *value.shape), "cpu", self.dtype)
            self.codec.encode_into(storage, meta, 0, value)
            self._features[key] = (storage, meta)
            while len(self._features) > self.config.registry_cache_days:
                self._features.popitem(last=False)
        self._features.move_to_end(key)
        storage, meta = self._features[key]
        return self.codec.decode(storage, meta, 0, out_dtype=self.dtype)

    def prefetch_features(self, days, chunk_days=None):
        stats = LoadStats()
        days = list(days)
        width = min(int(chunk_days or self.registry.cache_days), self.registry.cache_days)
        assert width > 0
        for offset in range(0, len(days), width):
            chunk = days[offset:offset + width]
            if all((ds, self.current_ti) in self._features for ds in chunk):
                continue
            stats.merge(self.registry._ensure_range(self.input_names, chunk[0], chunk[-1]))
            for ds in chunk:
                self.gen_feature(ds)
        return stats

    def gen_base_universe_mask(self, ds):
        if self.validity is None:
            return torch.ones(len(self.mask.code), dtype=torch.bool)
        name, _, field = self.validity
        values = self.source_field(name, field, ds, ds)[0]
        return torch.isfinite(values) & (values != 0)

    def gen_valid_mask(self, ds, ti=None):
        if ti is not None:
            self.set_current_ti(ti)
        valid = self.gen_base_universe_mask(ds)
        if self.validity is not None:
            name, fields, _ = self.validity
            for field in fields:
                values = self.source_field(name, field, ds, ds)[0]
                valid &= torch.isfinite(values) & (values != 0)
        return valid

    def gen_raw_target(self, ds, ti):
        """Default: the selected field on ds. Override to index the research target."""
        assert self.target is not None, "model_target must select a source and field"
        self.set_current_ti(ti)
        return self.source_field(*self.target, ds, ds)[0].to(torch.float32)

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
        self.prefetch_features([self.didx2date(i) for i in range(start, end + 1)])
        values = torch.stack([
            self.gen_feature(self.didx2date(idx), target_ti)
            for idx in range(start, end + 1)
        ])
        return self.transform_feature_window(values, target_ti=target_ti, stage="predict")


class ComboTrainDataset(Dataset):
    """Bounded feature cache; samples retain same-time snapshots across trading days."""

    def __init__(self, loader, end_ds, ndays, x_delay=None, ts_days=8,
                 validinsts=None, load_chunk_days=None, codec=None):
        self.loader = loader
        self.end_ds = loader.align_date(end_ds)
        self.ret_days = 1 if x_delay is None else int(x_delay)
        assert self.ret_days > 0
        self.ts_days = int(ts_days)
        self.load_chunk_days = min(int(load_chunk_days or loader.registry.cache_days), loader.registry.cache_days)
        self.end_didx = loader.date2didx(self.end_ds)
        self.start_didx = max(loader.data_start_didx, self.end_didx - int(ndays) + 1)
        self.first_sample = self.start_didx + self.ts_days - 1
        self.last_sample = self.end_didx - self.ret_days + 1
        assert self.first_sample <= self.last_sample, "not enough training history"
        loader.gen_feature(loader.didx2date(self.first_sample), loader.sample_times[0])
        self.ndays = self.end_didx - self.start_didx + 1
        self.validinsts = self._build_validinsts() if validinsts is None else validinsts.to(torch.long)
        self.numValidinsts = len(self.validinsts)
        assert self.numValidinsts > 0, "no training instruments"

    def _build_validinsts(self):
        valid = torch.zeros(len(self.loader.mask.code), dtype=torch.bool)
        for idx in range(self.first_sample, self.last_sample + 1):
            for ti in self.loader.sample_times:
                self.loader.set_current_ti(ti)
                valid |= self.loader.gen_base_universe_mask(self.loader.didx2date(idx))
        return torch.where(valid)[0]

    def __len__(self):
        return (self.last_sample - self.first_sample + 1) * len(self.loader.sample_times)

    def sample_coordinates(self, idx):
        assert 0 <= idx < len(self)
        day, part = divmod(int(idx), len(self.loader.sample_times))
        return self.loader.didx2date(self.first_sample + day), self.loader.sample_times[part]

    def __getitem__(self, idx):
        ds, ti = self.sample_coordinates(idx)
        end = self.loader.date2didx(ds)
        self.loader.set_current_ti(ti)
        self.loader.prefetch_features(
            [self.loader.didx2date(i) for i in range(end - self.ts_days + 1, end + 1)],
            self.load_chunk_days,
        )
        values = torch.stack([
            self.loader.gen_feature(self.loader.didx2date(i), ti).index_select(0, self.validinsts)
            for i in range(end - self.ts_days + 1, end + 1)
        ])
        x = self.loader.transform_feature_window(values, target_ti=ti, stage="train")
        y, w = self.loader.gen_target(ds, ti, ret_days=self.ret_days)
        return idx, ds, ti, x, y[self.validinsts], w[self.validinsts]


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
