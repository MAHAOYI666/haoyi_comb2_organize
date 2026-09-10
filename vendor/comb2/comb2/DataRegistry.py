# haoyi 2026/05/5

from __future__ import annotations

import ast
import bisect
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
import importlib
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .op_utils import (
    cs_zscore,
    nan_to_num,
    neut,
    normalize_by_max_abs,
    rank,
    rolling_mean,
    rolling_std,
    truncate,
    winsorize_by_quantile,
)

ORGANIZE_ROOT = Path(__file__).resolve().parents[3]
if str(ORGANIZE_ROOT) not in sys.path:
    sys.path.insert(0, str(ORGANIZE_ROOT))
SIMBASE_ROOT = ORGANIZE_ROOT / "vendor" / "comb2-simbase"
if str(SIMBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBASE_ROOT))

from comb2_simbase import IndexMask, Memmaper2, load_snap_vwap_labels
from comb2_simbase.cache_layout import (
    BARRA_STYLE_DIRNAME,
    BARRA_STYLE_PREFIX,
    DAILY_LABEL_DIRNAME,
    DAILY_LABEL_PREFIX,
)

BARRA_PRESET_FILES = (
    ("beta", "BETA"),
    ("btop", "BTOP"),
    ("earnyild", "EARNYILD"),
    ("growth", "GROWTH"),
    ("industry", "INDUSTRY"),
    ("leverage", "LEVERAGE"),
    ("liquidty", "LIQUIDTY"),
    ("momentum", "MOMENTUM"),
    ("resvol", "RESVOL"),
    ("size", "SIZE"),
    ("sizenl", "SIZENL"),
)

FACTORSIM_BASE_ITEMS = {
    "size": ("base.size", "1d_BarraCNE5/BarraCNE5.SIZE"),
    "btop": ("base.btop", "1d_BarraCNE5/BarraCNE5.BTOP"),
}
SUPPORTED_OPS = {
    "cs_zscore",
    "zscore",
    "rank",
    "truncate",
    "nan_to_num",
    "fillna",
    "winsorize_by_quantile",
    "normalize_by_max_abs",
    "rolling_mean",
    "rolling_std",
    "neut",
}
ROLLING_OPS = {
    "rolling_mean": rolling_mean,
    "rolling_std": rolling_std,
}
AXIS_NAME_TO_INDEX = {"date": 0, "code": -1}


def _require_index_mask_cls():
    if IndexMask is None:
        raise ModuleNotFoundError("comb2_simbase is required to build the comb2 trading universe")
    return IndexMask


def _require_memmaper2_cls():
    if Memmaper2 is None:
        raise ModuleNotFoundError("comb2_simbase is required to read Memmaper2 data")
    return Memmaper2


class _LazyIndexMask:
    def __init__(self):
        self._mask = None

    def get(self):
        if self._mask is None:
            self._mask = _require_index_mask_cls()()
        return self._mask

    def __getattr__(self, name: str):
        return getattr(self.get(), name)


MASK = _LazyIndexMask()


def _maybe_section(monitor, event: str, ds: int | None = None, *, level: str = "full"):
    if monitor is None:
        return nullcontext()
    return monitor.section(event, date=ds)


@dataclass(frozen=True)
class OpSpec:
    name: str
    params: dict[str, Any]


def _coerce_op_spec(value: OpSpec | dict[str, Any]) -> OpSpec:
    if isinstance(value, OpSpec):
        return value
    return OpSpec(name=str(value["name"]), params=dict(value.get("params", {})))


@dataclass
class Universe:
    dates: tuple[int, ...]
    codes: tuple[str, ...]
    dtype: torch.dtype
    _date_to_idx: dict[int, int] = field(init=False, repr=False)
    _code_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self._date_to_idx = {int(ds): idx for idx, ds in enumerate(self.dates)}
        self._code_to_idx = {str(code).zfill(6): idx for idx, code in enumerate(self.codes)}

    @classmethod
    def from_mask(cls, mask: Any, dtype: torch.dtype, data_offset: int = 0) -> "Universe":
        offset = max(0, int(data_offset))
        return cls(
            dates=tuple(int(ds) for ds in mask.date[offset:]),
            codes=tuple(str(code).zfill(6) for code in mask.code),
            dtype=dtype,
        )

    def date2idx(self, ds: int) -> int:
        ds = int(ds)
        exact = self._date_to_idx.get(ds)
        if exact is not None:
            return exact
        idx = bisect.bisect_left(self.dates, ds)
        if idx >= len(self.dates):
            raise KeyError(f"date {ds} is after universe end {self.dates[-1]}")
        return idx

    def idx2date(self, idx: int) -> int:
        return int(self.dates[int(idx)])

    def code2idx(self, code: str | int) -> int:
        normalized = str(code).zfill(6)
        if normalized not in self._code_to_idx:
            raise KeyError(f"code {code} is not in universe")
        return self._code_to_idx[normalized]


@dataclass(frozen=True)
class DataItem:
    name: str
    module: str = "builtin.factorsim"
    path: str | None = None
    delay: int = 0
    mode: str = "read_dump"
    config_path: str | None = None
    ops: tuple[OpSpec, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpRequirements:
    data_deps: tuple[str, ...] = ()
    lookback_days: int = 1


@dataclass(frozen=True)
class TensorSpec:
    axes: tuple[str, ...]
    source_name: str

    def normalize_axis(self, axis: Any) -> str:
        if isinstance(axis, str):
            normalized = axis.strip().lower()
            if normalized not in AXIS_NAME_TO_INDEX:
                raise ValueError(f"unsupported axis name {axis!r} for {self.source_name!r}")
            return normalized
        idx = int(axis)
        if idx < 0:
            idx += len(self.axes)
        if idx < 0 or idx >= len(self.axes):
            raise ValueError(f"item {self.source_name!r} has axes {self.axes}, cannot use axis={axis!r}")
        return self.axes[idx]

    def axis_index(self, axis: str) -> int:
        normalized = str(axis).strip().lower()
        if normalized not in self.axes:
            raise ValueError(f"item {self.source_name!r} has axes {self.axes}, cannot use axis={axis!r}")
        return self.axes.index(normalized)

    def require_axis(self, axis: str) -> int:
        return self.axis_index(axis)

    def without_axis(self, axis: str) -> "TensorSpec":
        idx = self.axis_index(axis)
        return TensorSpec(axes=self.axes[:idx] + self.axes[idx + 1 :], source_name=self.source_name)


@dataclass
class LoadStats:
    request_days: int = 0
    raw_chunks: int = 0
    raw_points: int = 0
    raw_time: float = 0.0
    ops_items: int = 0
    ops_points: int = 0
    ops_time: float = 0.0
    total_time: float = 0.0

    def merge(self, other: "LoadStats"):
        self.request_days += other.request_days
        self.raw_chunks += other.raw_chunks
        self.raw_points += other.raw_points
        self.raw_time += other.raw_time
        self.ops_items += other.ops_items
        self.ops_points += other.ops_points
        self.ops_time += other.ops_time
        self.total_time += other.total_time


def _coerce_data_item(value: DataItem | dict[str, Any]) -> DataItem:
    item = value if isinstance(value, DataItem) else DataItem(
        **{**value, "ops": tuple(_coerce_op_spec(op) for op in value.get("ops", ()))}
    )
    assert item.name and item.module
    assert isinstance(item.delay, int) and item.delay >= 0, "delay must be a nonnegative trading-day count"
    assert not ({"role", "freq", "nbar"} & set(item.params)), "source roles/frequencies are not supported"
    for op in item.ops:
        name, _ = _parse_op_call(op.name)
        assert name in SUPPORTED_OPS, f"unsupported data op: {op.name}"
    return item


def _parse_op_call(raw_name: str) -> tuple[str, tuple[str, ...]]:
    stripped = raw_name.strip()
    match = re.match(r"^([A-Za-z_]\w*)\((.*)\)$", stripped)
    if match is None:
        return stripped.lower(), ()
    name = match.group(1).strip().lower()
    raw_args = match.group(2).strip()
    if not raw_args:
        return name, ()
    return name, tuple(_split_op_args(raw_args))


def _split_op_args(raw_args: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for idx, char in enumerate(raw_args):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            value = raw_args[start:idx].strip()
            if value:
                args.append(value)
            start = idx + 1
    value = raw_args[start:].strip()
    if value:
        args.append(value)
    return args


def _strip_arg_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def _is_float_arg(value: str) -> bool:
    try:
        float(_strip_arg_quotes(value))
    except ValueError:
        return False
    return True


def _literal_list_arg(value: str) -> list[Any] | None:
    arg = _strip_arg_quotes(value)
    if not (arg.startswith("[") and arg.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(arg)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return None


def _float_list_arg(value: str) -> tuple[float, ...] | None:
    parsed = _literal_list_arg(value)
    if parsed is None:
        return None
    values: list[float] = []
    for item in parsed:
        try:
            values.append(float(str(item).strip()))
        except ValueError:
            return None
    return tuple(values)


def _coerce_neut_ratio(value: Any) -> float | tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    if isinstance(value, str):
        ratio_list = _float_list_arg(value)
        if ratio_list is not None:
            return ratio_list
        return float(_strip_arg_quotes(value))
    return float(value)


def _expand_neut_dep_arg(raw_arg: str) -> list[str]:
    arg = _strip_arg_quotes(raw_arg)
    if arg.startswith("[") and arg.endswith("]"):
        parsed = _literal_list_arg(arg)
        if parsed is not None:
            return [str(value).strip() for value in parsed if str(value).strip()]
        inner = arg[1:-1].strip()
        return [_strip_arg_quotes(value) for value in _split_op_args(inner)]
    return [arg] if arg else []


def _neut_deps_and_ratio(op: OpSpec, args: tuple[str, ...]) -> tuple[tuple[str, ...], float | tuple[float, ...]]:
    ratio = _coerce_neut_ratio(op.params.get("ratio", 1.0))
    dep_args = list(args)
    if dep_args:
        ratio_list = _float_list_arg(dep_args[-1])
        if ratio_list is not None:
            ratio = ratio_list
            dep_args.pop()
        elif _is_float_arg(dep_args[-1]):
            ratio = float(_strip_arg_quotes(dep_args.pop()))
    deps: list[str] = []
    for arg in dep_args:
        _merge_deps(deps, _expand_neut_dep_arg(arg))
    return tuple(deps), ratio


def _int_op_arg(op: OpSpec, args: tuple[str, ...], *param_names: str, default: int | None = None) -> int:
    for param_name in param_names:
        if param_name in op.params:
            return int(op.params[param_name])
    if args:
        return int(args[0])
    if default is not None:
        return int(default)
    raise ValueError(f"{op.name} requires an integer argument")


def _merge_deps(existing: list[str], values: Iterable[str]):
    for value in values:
        if value not in existing:
            existing.append(value)


def _op_requirements(ops: Sequence[OpSpec]) -> OpRequirements:
    lookback = 1
    deps: list[str] = []
    for op in ops:
        name, args = _parse_op_call(op.name)
        if name == "neut":
            neut_deps, _ = _neut_deps_and_ratio(op, args)
            _merge_deps(deps, neut_deps)
        elif name in ROLLING_OPS:
            axis = _op_axis_name(op, default="date")
            if axis == "date":
                lookback += max(0, _int_op_arg(op, args, "window", "n") - 1)
    return OpRequirements(data_deps=tuple(deps), lookback_days=max(1, lookback))


def _op_axis_name(op: OpSpec, *, default: str | None = None, spec: TensorSpec | None = None) -> str:
    axis = op.params.get("axis", default)
    if axis is None:
        raise ValueError(f"{op.name} requires axis")
    if spec is not None:
        return spec.normalize_axis(axis)
    if isinstance(axis, str):
        normalized = axis.strip().lower()
        if normalized in AXIS_NAME_TO_INDEX:
            return normalized
        raise ValueError(f"unsupported axis name {axis!r} for {op.name}")
    axis = int(axis)
    if axis == 0:
        return "date"
    if axis == 1:
        return "code"
    if axis == -1:
        return "code"
    raise ValueError(f"unsupported positional axis {axis!r} for {op.name}; use date/code")


@dataclass(frozen=True)
class LoadedSource:
    """Memmap data and its axes; processed values are [date, code, field]."""

    values: torch.Tensor
    dates: tuple[int, ...]
    codes: tuple[str, ...]
    columns: tuple[str, ...]
    times: tuple[int, ...] = ()


class FactorsimReader:
    def __init__(self, path: str):
        self.path = str(path)
        self.mmap = Memmaper2(path)
        self.codes = pd.Index(self.mmap._columns).astype(str).str.zfill(6)
        self.column_indices = None

    def load(self, item, registry, start_ds, end_ds):
        dates = tuple(registry.universe.idx2date(i) for i in range(
            registry.universe.date2idx(start_ds), registry.universe.date2idx(end_ds) + 1
        ))
        if int(self.mmap._meta[1]) == 1:
            source_dates = self.mmap._index
            lo = np.searchsorted(source_dates, start_ds)
            hi = np.searchsorted(source_dates, end_ds, side="right")
            assert tuple(map(int, source_dates[lo:hi])) == dates, f"{item.name}: incomplete date axis"
            values = self.mmap.load(start_ds=start_ds, end_ds=end_ds, df_type=False)[:]
            if self.column_indices is None:
                self.column_indices = (
                    slice(None) if self.codes.equals(pd.Index(registry.universe.codes))
                    else self.codes.get_indexer(registry.universe.codes)
                )
            if not isinstance(self.column_indices, slice):
                aligned = np.full((len(dates), len(registry.universe.codes)), np.nan, dtype=values.dtype)
                present = self.column_indices >= 0
                aligned[:, present] = values[:, self.column_indices[present]]
                values = aligned
            return LoadedSource(torch.as_tensor(values[..., None]), dates, registry.universe.codes, (item.name,))
        frame = self.mmap.load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]
        frame.columns = self.codes
        frame = frame.reindex(columns=registry.universe.codes)
        dates = tuple(registry.universe.idx2date(i) for i in range(
            registry.universe.date2idx(start_ds), registry.universe.date2idx(end_ds) + 1
        ))
        times = ()
        if isinstance(frame.index, pd.MultiIndex):
            times = tuple(int(t) for t in self.mmap._index[0, 1:] if np.isfinite(t))
            expected = pd.MultiIndex.from_product([dates, times], names=frame.index.names)
            assert set(dates) <= set(map(int, self.mmap._index[1:, 0])), f"{item.name}: missing source dates"
            assert frame.index.is_unique and frame.index.isin(expected).all()
            frame = frame.reindex(expected)
            values = frame.to_numpy(copy=True).reshape(len(dates), len(times), len(frame.columns))
            values = values.transpose(0, 2, 1)[..., None]
        else:
            assert tuple(map(int, frame.index)) == dates, f"{item.name}: incomplete date axis"
            values = frame.to_numpy(copy=True)[..., None]
        return LoadedSource(torch.as_tensor(values), dates, registry.universe.codes, (item.name,), times)


class DataRegistry:
    def __init__(
        self, items, *, universe, data_start_ds, ashare_cache_path=None,
        config_path=None, presets=(), verbose=False, cache_days=64,
        process_source=None,
    ):
        self.universe = universe
        self.data_start_ds = int(data_start_ds)
        self.data_start_idx = universe.date2idx(self.data_start_ds)
        self.ashare_cache_path = ashare_cache_path
        self.config_path = config_path
        self.verbose = verbose
        self.cache_days = int(cache_days)
        assert self.cache_days > 0
        self.process_source = process_source
        self.runtime_context = {"ti": 150000, "date": None}
        self.module_cache = {}
        self.modules = self._builtin_modules()
        self.items = {}
        for raw in (*items, *self._preset_items(presets)):
            item = _coerce_data_item(raw)
            assert item.name not in self.items, f"duplicate data source: {item.name}"
            self.items[item.name] = item
        self.processed_cache = {name: OrderedDict() for name in self.items}
        self.columns = {}
        self.aliases = self._build_aliases()

    def _builtin_modules(self):
        return {alias: fn for name, fn in (
            ("factorsim", _load_factorsim), ("alpha_parquet", _load_alpha_parquet),
            ("snap_label", _load_snap_label),
        ) for alias in (name, "builtin." + name)}

    def _preset_items(self, presets):
        items = []
        for preset in presets:
            assert preset == "barra" and self.ashare_cache_path
            for name, field in BARRA_PRESET_FILES:
                items.append(DataItem(
                    name=f"barra.{name}",
                    path=str(Path(self.ashare_cache_path) / BARRA_STYLE_DIRNAME / f"{BARRA_STYLE_PREFIX}{field}"),
                    delay=1,
                ))
        return items

    def _build_aliases(self):
        return {name.split(".", 1)[1]: name for name in self.items if name.startswith("barra.")}

    def set_current_ti(self, ti):
        self.runtime_context["ti"] = int(ti)

    def set_point(self, ds, ti, *, refresh=False):
        self.runtime_context.update(date=int(ds), ti=int(ti))
        if refresh:
            idx = self.universe.date2idx(int(ds))
            for cache in self.processed_cache.values():
                for key in list(cache):
                    if key[0] >= idx:
                        del cache[key]
            self.module_cache.clear()

    def _resolve_name(self, name):
        resolved = self.aliases.get(name, name)
        assert resolved in self.items, f"undeclared data source: {name}"
        return resolved

    def _module_for(self, item):
        if item.module in self.modules:
            return self.modules[item.module]
        module_name, function = item.module.split(":")
        if self.config_path and self.config_path not in sys.path:
            sys.path.insert(0, self.config_path)
        fn = getattr(importlib.import_module(module_name), function)
        self.modules[item.module] = fn
        return fn

    def get_data(self, name, start_ds=None, end_ds=None):
        name = self._resolve_name(name)
        assert start_ds is not None and end_ds is not None, "explicit logical dates are required"
        lo = self.universe.date2idx(int(start_ds))
        hi = self.universe.date2idx(int(end_ds))
        assert self.data_start_idx <= lo <= hi
        assert hi - lo + 1 <= self.cache_days, "request exceeds registry_cache_days; read in chunks"
        self._ensure_processed_range(name, start_ds, end_ds)
        ti = self.runtime_context["ti"]
        cache = self.processed_cache[name]
        for i in range(lo, hi + 1):
            cache.move_to_end((i, ti))
        return torch.stack([cache[(i, ti)] for i in range(lo, hi + 1)])

    def get_field(self, name, field, start_ds, end_ds):
        name = self._resolve_name(name)
        values = self.get_data(name, start_ds, end_ds)
        return values[..., self.columns[name].index(field)]

    def _cache_day(self, name, idx, value):
        cache = self.processed_cache[name]
        key = (int(idx), self.runtime_context["ti"])
        cache[key] = value.clone(memory_format=torch.contiguous_format)
        cache.move_to_end(key)
        while len(cache) > self.cache_days:
            cache.popitem(last=False)

    def _ensure_processed_range(self, name, start_ds, end_ds, stack=(), stats=None):
        name = self._resolve_name(name)
        assert name not in stack, f"cyclic source dependency: {(*stack, name)}"
        lo, hi = self.universe.date2idx(int(start_ds)), self.universe.date2idx(int(end_ds))
        assert self.data_start_idx <= lo <= hi
        item = self.items[name]
        ti = self.runtime_context["ti"]
        cache = self.processed_cache[name]
        missing = [i for i in range(lo, hi + 1) if (i, ti) not in cache]
        if not missing:
            return
        requirements = _op_requirements(item.ops)
        ranges = []
        for idx in missing:
            if ranges and idx == ranges[-1][1] + 1 and idx - ranges[-1][0] < self.cache_days:
                ranges[-1] = (ranges[-1][0], idx)
            else:
                ranges.append((idx, idx))
        for first, last in ranges:
            raw_lo = max(self.data_start_idx, first - requirements.lookback_days + 1)
            assert raw_lo - item.delay >= 0, f"{name}: insufficient delay history"
            logical_start = self.universe.idx2date(raw_lo)
            logical_end = self.universe.idx2date(last)
            for dependency in requirements.data_deps:
                self._ensure_processed_range(dependency, logical_start, logical_end, (*stack, name), stats)
            start = self.universe.idx2date(raw_lo - item.delay)
            end = self.universe.idx2date(last - item.delay)
            began = time.perf_counter()
            loaded = self._module_for(item)(item, self, start, end)
            assert isinstance(loaded, LoadedSource), f"{name}: reader must return LoadedSource"
            expected_dates = tuple(self.universe.idx2date(i - item.delay) for i in range(raw_lo, last + 1))
            assert loaded.dates == expected_dates and loaded.codes == self.universe.codes
            processed = self.process_source(item, loaded) if self.process_source else loaded
            assert isinstance(processed, LoadedSource)
            assert processed.dates == loaded.dates and processed.codes == loaded.codes
            values = torch.as_tensor(processed.values)
            assert values.ndim == 3 and values.shape[:2] == (last - raw_lo + 1, len(self.universe.codes)), (
                f"{name}: process_source must reduce to [date, code, field], got {tuple(values.shape)}"
            )
            assert not processed.times, f"{name}: reduced source must not retain a bar axis"
            assert len(processed.columns) == values.shape[-1] and len(set(processed.columns)) == len(processed.columns)
            if name in self.columns:
                assert self.columns[name] == processed.columns, f"{name}: feature columns changed"
            self.columns[name] = processed.columns
            del loaded, processed
            if stats:
                stats.raw_time += time.perf_counter() - began
                stats.raw_chunks += 1
                stats.raw_points += last - raw_lo + 1
            began = time.perf_counter()
            values = self._apply_ops(item, values, raw_lo, last)
            for idx in range(first, last + 1):
                self._cache_day(name, idx, values[idx - raw_lo])
            if stats:
                stats.ops_time += time.perf_counter() - began

    def _ensure_range(self, names, start_ds, end_ds):
        stats = LoadStats()
        for name in names:
            self._ensure_processed_range(name, start_ds, end_ds, stats=stats)
        return stats

    def _apply_ops(self, item, values, lo_idx, hi_idx):
        if not item.ops:
            return values
        columns = []
        for column in values.unbind(-1):
            out = column
            for op in item.ops:
                name, args = _parse_op_call(op.name)
                params = op.params
                axis = _op_axis_name(op, default="code", spec=TensorSpec(("date", "code"), item.name))
                dim = 0 if axis == "date" else 1
                if name in {"cs_zscore", "zscore"}:
                    out = cs_zscore(out, axis=dim)
                elif name == "rank":
                    out = rank(out, dim=None, axis=dim, pct=bool(params.get("pct", False)))
                elif name == "truncate":
                    out = truncate(out, float(params.get("min", -4)), float(params.get("max", 4)))
                elif name in {"fillna", "nan_to_num"}:
                    out = nan_to_num(out, float(params.get("value", 0)))
                elif name == "winsorize_by_quantile":
                    out = winsorize_by_quantile(out, float(params.get("low", .01)), float(params.get("high", .99)), axis=dim)
                elif name == "normalize_by_max_abs":
                    out = normalize_by_max_abs(out, axis=dim)
                elif name in ROLLING_OPS:
                    assert params.get("axis", "date") == "date"
                    out = ROLLING_OPS[name](out, _int_op_arg(op, args, "window", "n"), axis=0)
                elif name == "neut":
                    deps, ratio = _neut_deps_and_ratio(op, args)
                    xs = []
                    for dep in deps:
                        data = torch.cat([
                            self.get_data(dep, self.universe.idx2date(first), self.universe.idx2date(min(first + self.cache_days - 1, hi_idx)))
                            for first in range(lo_idx, hi_idx + 1, self.cache_days)
                        ])
                        xs.extend(data.unbind(-1))
                    out = neut(out, xs, ratio=ratio)
            columns.append(out)
        return torch.stack(columns, -1)


def _load_factorsim(item, registry, start_ds, end_ds):
    assert item.path, f"{item.name}: Memmap path is required"
    path = item.path.format(ti=registry.runtime_context["ti"])
    key = f"memmap:{path}"
    if key not in registry.module_cache:
        registry.module_cache[key] = FactorsimReader(path)
    return registry.module_cache[key].load(item, registry, start_ds, end_ds)


def _load_snap_label(item, registry, start_ds, end_ds):
    assert registry.ashare_cache_path
    labels = load_snap_vwap_labels(
        Path(registry.ashare_cache_path).parent,
        registry.runtime_context["ti"], start_ds, end_ds,
        periods=(1,),
    )[1]
    labels = labels.reindex(columns=registry.universe.codes)
    return LoadedSource(torch.as_tensor(labels.to_numpy(copy=True)[..., None]),
                        tuple(map(int, labels.index)), registry.universe.codes, (item.name,))


def _load_alpha_parquet(item, registry, start_ds, end_ds):
    assert item.path and item.mode == "read_dump"
    frame = pd.read_parquet(item.path)
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.xs(registry.runtime_context["ti"], level=1)
    frame.index = frame.index.astype(int)
    frame.columns = frame.columns.astype(str).str.zfill(6)
    dates = tuple(registry.universe.idx2date(i) for i in range(
        registry.universe.date2idx(start_ds), registry.universe.date2idx(end_ds) + 1
    ))
    frame = frame.reindex(index=dates, columns=registry.universe.codes)
    return LoadedSource(torch.as_tensor(frame.to_numpy(copy=True)[..., None]), dates,
                        registry.universe.codes, (item.name,))
