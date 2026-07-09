# haoyi 2026/05/5

from __future__ import annotations

import ast
import bisect
from contextlib import nullcontext
from dataclasses import dataclass, field
import importlib
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

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

from comb2_simbase import IndexMask, Memmaper2

BARRA_STYLE_DIRNAME = "1d_BarraCNE5"
BARRA_STYLE_PREFIX = "BarraCNE5."
BARRA_PRESET_STYLES = (
    "beta",
    "btop",
    "earnyild",
    "growth",
    "industry",
    "leverage",
    "liquidty",
    "momentum",
    "resvol",
    "size",
    "sizenl",
)

FACTORSIM_BASE_ITEMS = {
    "size": ("base.size", "1d_BarraCNE5/BarraCNE5.size"),
    "btop": ("base.btop", "1d_BarraCNE5/BarraCNE5.btop"),
}
FREQ_ORDER = ("1d", "5m", "1m")
SUPPORTED_FREQS = set(FREQ_ORDER)
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
AXIS_NAME_TO_INDEX = {"date": 0, "bar": 1, "code": -1}


def _hms_range(start_hhmm: int, end_hhmm: int, step_minutes: int) -> tuple[int, ...]:
    start_hour, start_minute = divmod(int(start_hhmm), 100)
    end_hour, end_minute = divmod(int(end_hhmm), 100)
    cur = start_hour * 60 + start_minute
    end = end_hour * 60 + end_minute
    values: list[int] = []
    while cur <= end:
        hour, minute = divmod(cur, 60)
        values.append(hour * 10000 + minute * 100)
        cur += int(step_minutes)
    return tuple(values)


CANONICAL_BAR_TIMES = {
    "1m": (
        *_hms_range(930, 1130, 1),
        *_hms_range(1301, 1457, 1),
        150000,
    ),
    "5m": (
        *_hms_range(930, 1130, 5),
        *_hms_range(1305, 1455, 5),
        150000,
    ),
}
BAR_COUNT_BY_FREQ = {freq: len(times) for freq, times in CANONICAL_BAR_TIMES.items()}


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
    module: str
    path: str | None = None
    role: str = "aux"
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
    if isinstance(value, DataItem):
        return _normalize_data_item(value)
    for legacy_key in ("dump_path", "source", "loader"):
        if legacy_key in value:
            raise ValueError(f"data item {value.get('name')!r} uses unsupported legacy key {legacy_key!r}; use path/module")
    module = value.get("module")
    if not module:
        raise ValueError(f"data item {value.get('name')!r} requires module")
    ops = tuple(_coerce_op_spec(op) for op in value.get("ops", ()))
    return _normalize_data_item(DataItem(
        name=str(value["name"]),
        module=str(module),
        path=value.get("path"),
        role=str(value.get("role", "aux") or "aux").lower(),
        mode=str(value.get("mode", "read_dump")),
        config_path=value.get("config_path"),
        ops=ops,
        params=dict(value.get("params", {})),
    ))


def _normalize_freq(value: Any, *, item_name: str) -> str:
    freq = "1d" if value is None or str(value).strip() == "" else str(value).strip().lower()
    if freq not in SUPPORTED_FREQS:
        supported = ", ".join(FREQ_ORDER)
        raise ValueError(f"item {item_name!r} has unsupported freq={value!r}; expected one of {supported}")
    return freq


def _normalize_data_item(item: DataItem) -> DataItem:
    params = dict(item.params)
    if "nbar" in params:
        raise ValueError(f"item {item.name!r} uses unsupported attribute nbar")
    freq = _normalize_freq(params.get("freq"), item_name=item.name)
    role = str(item.role or "aux").lower()
    if role == "label" and freq != "1d":
        raise ValueError(f"label item {item.name!r} only supports freq='1d'")
    params["freq"] = freq
    for op in item.ops:
        name, _ = _parse_op_call(op.name)
        if name not in SUPPORTED_OPS:
            raise ValueError(f"unsupported data op: {op.name}")
        if name in ROLLING_OPS and _op_axis_name(op, default="date") != "date":
            raise ValueError(f"{op.name} only supports axis='date'")
        if name == "neut" and freq != "1d":
            raise ValueError(f"neut only supports freq='1d', got item {item.name!r} freq={freq!r}")
    return DataItem(
        name=item.name,
        module=item.module,
        path=item.path,
        role=role,
        mode=item.mode,
        config_path=item.config_path,
        ops=item.ops,
        params=params,
    )


def item_freq(item: DataItem) -> str:
    return _normalize_freq(item.params.get("freq"), item_name=item.name)


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


def _missing_ranges(loaded: torch.Tensor, lo: int, hi: int) -> list[tuple[int, int]]:
    if hi < lo:
        return []
    missing = ~loaded[lo : hi + 1]
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for offset, is_missing in enumerate(missing.tolist()):
        idx = lo + offset
        if is_missing and start is None:
            start = idx
        elif (not is_missing) and start is not None:
            ranges.append((start, idx - 1))
            start = None
    if start is not None:
        ranges.append((start, hi))
    return ranges

def _as_ranked_tensor(value: Any, *, dtype: torch.dtype, allowed_ndims: tuple[int, ...], rank_label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=dtype)
    else:
        tensor = torch.as_tensor(np.asarray(value), dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim not in allowed_ndims:
        raise ValueError(f"data module must return {rank_label} data, got shape {tuple(tensor.shape)}")
    return tensor


def _as_2d_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return _as_ranked_tensor(value, dtype=dtype, allowed_ndims=(2,), rank_label="2-D")


def _as_data_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return _as_ranked_tensor(value, dtype=dtype, allowed_ndims=(2, 3), rank_label="2-D or 3-D")


DataLoadFn = Callable[[DataItem, "DataRegistry", int, int], torch.Tensor | np.ndarray]


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
        return "bar"
    if axis == -1:
        return "code"
    raise ValueError(f"unsupported positional axis {axis!r} for {op.name}; use date/bar/code")


def _expected_item_shape(item: DataItem, n_dates: int, n_codes: int) -> tuple[int, ...]:
    freq = item_freq(item)
    if freq == "1d":
        return (int(n_dates), int(n_codes))
    return (int(n_dates), BAR_COUNT_BY_FREQ[freq], int(n_codes))


def _item_axes(item: DataItem) -> tuple[str, ...]:
    return ("date", "code") if item_freq(item) == "1d" else ("date", "bar", "code")


class FactorsimReader:
    def __init__(self, path: str):
        self.path = str(path)
        self.mmap = _require_memmaper2_cls()(path)
        self.n_levels = int(self.mmap._meta[1])
        self.time_axis = self._load_time_axis()

    def _load_time_axis(self) -> np.ndarray:
        if self.n_levels <= 1:
            return np.array([], dtype=np.int64)
        axis = np.asarray(self.mmap._index[0, 1:])
        axis = axis[np.isfinite(axis)]
        return axis.astype(np.int64, copy=False)

    def _load_day_2d(self, ds: int, dtype: torch.dtype) -> torch.Tensor:
        data = self.mmap.load(start_ds=int(ds), end_ds=int(ds), df_type=False)[:]
        return _as_2d_tensor(data, dtype=dtype)

    def _load_day_3d(self, ds: int, dtype: torch.dtype, universe: Universe | None = None) -> torch.Tensor:
        frame = self.mmap.load(start_ds=int(ds), end_ds=int(ds), df_type=True).dloc[:]
        if universe is not None:
            frame.columns = frame.columns.astype(str).str.zfill(6)
            frame = frame.reindex(columns=universe.codes)
        if len(frame.index) == 0:
            width = len(universe.codes) if universe is not None else len(frame.columns)
            return torch.empty((0, width), dtype=dtype)
        return torch.as_tensor(frame.to_numpy(dtype=np.float32, copy=True), dtype=dtype)

    def load_2d(self, start_ds: int, end_ds: int, dtype: torch.dtype) -> torch.Tensor:
        if self.n_levels == 1:
            data = self.mmap.load(start_ds=int(start_ds), end_ds=int(end_ds), df_type=False)[:]
            return _as_2d_tensor(data, dtype=dtype)
        raise ValueError("load_2d is only valid for 2D sources")

    def validate_time_axis(self, freq: str) -> None:
        expected = np.asarray(CANONICAL_BAR_TIMES[freq], dtype=np.int64)
        if self.time_axis.shape != expected.shape or not np.array_equal(self.time_axis, expected):
            raise ValueError(
                f"factorsim source {self.path!r} time axis does not match canonical {freq}: "
                f"got len={len(self.time_axis)}, expected len={len(expected)}"
            )

    def load_intraday(
        self,
        start_ds: int,
        end_ds: int,
        *,
        dtype: torch.dtype,
        freq: str,
        universe: Universe,
    ) -> torch.Tensor:
        self.validate_time_axis(freq)
        dates = [universe.idx2date(idx) for idx in range(universe.date2idx(start_ds), universe.date2idx(end_ds) + 1)]
        rows: list[torch.Tensor] = []
        for ds in dates:
            day = self._load_day_3d(ds, dtype=torch.float32, universe=universe)
            if day.numel() == 0:
                rows.append(torch.full((BAR_COUNT_BY_FREQ[freq], len(universe.codes)), torch.nan, dtype=torch.float32))
                continue
            if day.shape[0] != BAR_COUNT_BY_FREQ[freq]:
                raise ValueError(
                    f"factorsim source {self.path!r} returned {day.shape[0]} bars for {ds}, "
                    f"expected {BAR_COUNT_BY_FREQ[freq]}"
                )
            rows.append(day)
        return torch.stack(rows, dim=0).to(dtype)


class DataRegistry:
    def __init__(
        self,
        items: Sequence[DataItem | dict[str, Any]],
        *,
        universe: Universe,
        data_start_ds: int,
        ashare_data_path: str | None,
        config_path: str | None,
        presets: Sequence[str] = (),
        verbose: bool = False,
    ):
        self.universe = universe
        self.data_start_ds = int(data_start_ds)
        self.data_start_idx = universe.date2idx(int(data_start_ds))
        self.ashare_data_path = ashare_data_path
        self.config_path = config_path
        self.verbose = bool(verbose)
        self.module_cache: dict[str, Any] = {}
        self.modules: dict[str, DataLoadFn] = self._builtin_modules()
        self.runtime_context: dict[str, Any] = {"ti": 150000}

        all_items = list(_coerce_data_item(item) for item in items)
        all_items.extend(self._preset_items(presets))
        if ashare_data_path:
            for short_name, (full_name, rel_path) in FACTORSIM_BASE_ITEMS.items():
                all_items.append(
                    DataItem(
                        name=full_name,
                        module="builtin.factorsim",
                        path=str(Path(ashare_data_path) / rel_path),
                        role="base",
                        params={"_builtin_short_name": short_name},
                    )
                )
        self.items: dict[str, DataItem] = {}
        for raw_item in all_items:
            item = _normalize_data_item(raw_item)
            if item.name in self.items:
                raise ValueError(f"duplicate data item name: {item.name}")
            self.items[item.name] = item

        self.processed_cache = {
            name: None
            for name in self.items
        }
        self.processed_loaded = {
            name: torch.zeros(len(universe.dates), dtype=torch.bool)
            for name in self.items
        }
        self.aliases = self._build_aliases()

    def _builtin_modules(self) -> dict[str, DataLoadFn]:
        loaders = {
            "factorsim": _load_factorsim,
            "alpha_parquet": _load_alpha_parquet,
        }
        return {
            alias: loader
            for name, loader in loaders.items()
            for alias in (name, f"builtin.{name}")
        }

    def _preset_items(self, presets: Sequence[str]) -> list[DataItem]:
        items: list[DataItem] = []
        for preset in presets:
            preset_name = str(preset).strip().lower()
            if preset_name != "barra":
                raise ValueError(f"unsupported data preset: {preset}")
            for style in BARRA_PRESET_STYLES:
                path = str(Path(self.ashare_data_path) / BARRA_STYLE_DIRNAME / f"{BARRA_STYLE_PREFIX}{style}") if self.ashare_data_path else style
                items.append(
                    DataItem(
                        name=f"barra.{style}",
                        module="builtin.factorsim",
                        path=path,
                        role="aux",
                    )
                )
        return items

    def _build_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for name in self.items:
            if name.startswith("barra."):
                short = name.split(".", 1)[1]
                if short not in self.items and short not in aliases:
                    aliases[short] = name
        return aliases

    def set_current_ti(self, ti: int) -> None:
        self.runtime_context["ti"] = int(ti)

    def _resolve_name(self, name: str) -> str:
        normalized = str(name).strip()
        if normalized in self.items:
            return normalized
        if normalized in self.aliases:
            return self.aliases[normalized]
        raise KeyError(f"unknown data name: {name}")

    def get_data(self, name: str, start_ds: int | None = None, end_ds: int | None = None) -> torch.Tensor:
        """Return processed data for a declared item.

        Passing a date range loads that range if needed and returns the aligned
        slice with date as the first dimension. Without dates, only already
        loaded cache is returned.
        """
        resolved = self._resolve_name(name)
        if start_ds is not None or end_ds is not None:
            if start_ds is None:
                start_ds = end_ds
            if end_ds is None:
                end_ds = start_ds
            self._ensure_range((resolved,), int(start_ds), int(end_ds))
            lo, hi = self._bounds_to_idx(int(start_ds), int(end_ds))
            if hi < lo:
                item = self.items[resolved]
                empty_shape = _expected_item_shape(item, 0, len(self.universe.codes))
                return torch.empty(empty_shape, dtype=self.universe.dtype)
            cache = self.processed_cache[resolved]
            if cache is None:
                raise ValueError(f"data item {resolved!r} has not been loaded")
            return cache[lo : hi + 1]
        cache = self.processed_cache[resolved]
        if cache is None:
            raise ValueError(f"data item {resolved!r} has not been loaded; pass start_ds/end_ds to load a range")
        return cache

    def _ensure_cache(self, name: str, item: DataItem, window_shape: tuple[int, ...]) -> torch.Tensor:
        cache = self.processed_cache[name]
        if cache is not None:
            return cache
        full_shape = (len(self.universe.dates),) + tuple(window_shape[1:])
        expected_full_shape = _expected_item_shape(item, len(self.universe.dates), len(self.universe.codes))
        if full_shape != expected_full_shape:
            raise ValueError(
                f"item {name!r} has shape {full_shape}, expected {expected_full_shape} for freq={item_freq(item)!r}"
            )
        cache = torch.full(full_shape, torch.nan, dtype=self.universe.dtype)
        self.processed_cache[name] = cache
        return cache

    def _module_for(self, item: DataItem) -> DataLoadFn:
        module_name = item.module.strip()
        if module_name in self.modules:
            return self.modules[module_name]
        if ":" in module_name:
            mod_name, func_name = module_name.split(":", 1)
        else:
            mod_name, func_name = module_name, "load_data"
        module = importlib.import_module(mod_name)
        fn = getattr(module, func_name)
        if not callable(fn):
            raise TypeError(f"data module {module_name!r} is not callable")
        self.modules[module_name] = fn
        return fn

    def _bounds_to_idx(self, start_ds: int, end_ds: int) -> tuple[int, int]:
        start_idx = max(self.data_start_idx, self.universe.date2idx(int(start_ds)))
        end_idx = self.universe.date2idx(int(end_ds))
        if end_idx < start_idx:
            return start_idx, start_idx - 1
        return start_idx, end_idx

    def _ensure_processed_range(self, name: str, start_ds: int, end_ds: int, stack: tuple[str, ...] = (), stats: LoadStats | None = None):
        name = self._resolve_name(name)
        if name in stack:
            chain = " -> ".join((*stack, name))
            raise ValueError(f"cyclic data dependency: {chain}")
        lo, hi = self._bounds_to_idx(start_ds, end_ds)
        if hi < lo:
            return
        if not _missing_ranges(self.processed_loaded[name], lo, hi):
            return

        item = self.items[name]
        requirements = _op_requirements(item.ops)
        for miss_lo, miss_hi in _missing_ranges(self.processed_loaded[name], lo, hi):
            raw_lo = max(self.data_start_idx, miss_lo - requirements.lookback_days + 1)
            raw_start_ds = self.universe.idx2date(raw_lo)
            raw_end_ds = self.universe.idx2date(miss_hi)
            for dep in requirements.data_deps:
                self._ensure_processed_range(self._resolve_neut_dep_name(dep), raw_start_ds, raw_end_ds, (*stack, name), stats)

            load_start = time.perf_counter()
            loaded = self._module_for(item)(item, self, raw_start_ds, raw_end_ds)
            tensor = _as_data_tensor(loaded, dtype=self.universe.dtype)
            expected_shape = _expected_item_shape(item, miss_hi - raw_lo + 1, len(self.universe.codes))
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"data module {item.module!r} for {name!r} returned shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            total_time = time.perf_counter() - load_start
            if stats is not None:
                stats.raw_chunks += 1
                stats.raw_points += miss_hi - raw_lo + 1
                stats.raw_time += total_time

            ops_start = time.perf_counter()
            processed_window = self._apply_ops(item, tensor.to(torch.float32), raw_lo, miss_hi)
            ops_time = time.perf_counter() - ops_start
            if tuple(processed_window.shape) != expected_shape:
                raise ValueError(
                    f"item {name!r} pipeline returned shape {tuple(processed_window.shape)}, "
                    f"expected shape {expected_shape}"
                )
            out_lo = miss_lo - raw_lo
            out_hi = miss_hi - raw_lo + 1
            cache = self._ensure_cache(name, item, tuple(processed_window.shape))
            cache[miss_lo : miss_hi + 1] = processed_window[out_lo:out_hi].to(self.universe.dtype)
            self.processed_loaded[name][miss_lo : miss_hi + 1] = True
            if item.ops and stats is not None:
                stats.ops_items += 1
                stats.ops_points += miss_hi - miss_lo + 1
                stats.ops_time += ops_time

    def _ensure_range(self, names: Sequence[str], start_ds: int, end_ds: int):
        ensure_start = time.perf_counter()
        stats = LoadStats()
        for name in names:
            self._ensure_processed_range(name, start_ds, end_ds, stats=stats)
        lo, hi = self._bounds_to_idx(start_ds, end_ds)
        stats.request_days = max(0, hi - lo + 1)
        stats.total_time = time.perf_counter() - ensure_start
        return stats

    def _apply_ops(self, item: DataItem, x: torch.Tensor, lo_idx: int, hi_idx: int) -> torch.Tensor:
        spec = TensorSpec(axes=_item_axes(item), source_name=item.name)
        out = x
        for op in item.ops:
            raw_name = op.name.strip()
            name, args = _parse_op_call(raw_name)
            params = op.params
            if name not in SUPPORTED_OPS:
                raise ValueError(f"unsupported data op: {op.name}")
            if name in {"cs_zscore", "zscore"}:
                out = cs_zscore(out, axis=spec.require_axis(_op_axis_name(op, default="code", spec=spec)))
            elif name == "rank":
                out = rank(
                    out,
                    dim=None,
                    axis=spec.require_axis(_op_axis_name(op, default="code", spec=spec)),
                    pct=bool(params.get("pct", False)),
                )
            elif name == "truncate":
                out = truncate(out, float(params.get("min", -4.0)), float(params.get("max", 4.0)))
            elif name in {"nan_to_num", "fillna"}:
                out = nan_to_num(out, float(params.get("value", 0.0)))
            elif name == "winsorize_by_quantile":
                out = winsorize_by_quantile(
                    out,
                    float(params.get("low", 0.01)),
                    float(params.get("high", 0.99)),
                    axis=spec.require_axis(_op_axis_name(op, default="code", spec=spec)),
                )
            elif name == "normalize_by_max_abs":
                out = normalize_by_max_abs(out, axis=spec.require_axis(_op_axis_name(op, default="code", spec=spec)))
            elif name == "neut":
                if item_freq(item) != "1d":
                    raise ValueError(f"neut only supports freq='1d', got item {item.name!r} freq={item_freq(item)!r}")
                deps, ratio = _neut_deps_and_ratio(op, args)
                if not deps:
                    raise ValueError(f"{op.name} requires at least one data dependency")
                _ = spec.require_axis("code")
                dep_names = [self._resolve_neut_dep_name(dep) for dep in deps]
                for dep_name in dep_names:
                    if item_freq(self.items[dep_name]) != "1d":
                        raise ValueError(f"neut dependency {dep_name!r} must have freq='1d'")
                xs = [self.get_data(dep_name)[lo_idx : hi_idx + 1].to(torch.float32) for dep_name in dep_names]
                out = neut(out, xs, ratio=ratio)
            elif name in ROLLING_OPS:
                axis_name = _op_axis_name(op, default="date", spec=spec)
                if axis_name != "date":
                    raise ValueError(f"{op.name} only supports axis='date'")
                out = ROLLING_OPS[name](
                    out,
                    _int_op_arg(op, args, "window", "n"),
                    axis=spec.require_axis(axis_name),
                )
            else:
                raise ValueError(f"unsupported data op: {op.name}")
        out[torch.isinf(out)] = torch.nan
        if tuple(out.shape) != tuple(x.shape):
            raise ValueError(
                f"item {item.name!r} op pipeline changed shape from {tuple(x.shape)} to {tuple(out.shape)}"
            )
        return out

    def _resolve_neut_dep_name(self, name: str) -> str:
        normalized = str(name).strip()
        if normalized in self.items:
            return normalized
        if normalized in self.aliases:
            return self.aliases[normalized]
        if normalized in FACTORSIM_BASE_ITEMS:
            return FACTORSIM_BASE_ITEMS[normalized][0]
        return self._resolve_name(normalized)


def _memmap_load_2d(registry: DataRegistry, path: str, start_ds: int, end_ds: int, dtype: torch.dtype) -> torch.Tensor:
    cache_key = f"memmap:{path}"
    mmap = registry.module_cache.get(cache_key)
    if mmap is None:
        mmap = _require_memmaper2_cls()(path)
        registry.module_cache[cache_key] = mmap
    data = mmap.load(start_ds=int(start_ds), end_ds=int(end_ds))[:]
    return _as_2d_tensor(data, dtype=dtype)


def _load_factorsim(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
    path = item.path
    if not path and item.role == "label":
        if not registry.ashare_data_path:
            raise ValueError(f"label data {item.name!r} requires path or registry.ashare_data_path")
        path = str(Path(registry.ashare_data_path) / "1d_DailyLabel" / "DailyLabel.vwap30_label1d")
    if not path:
        raise ValueError(f"factorsim data {item.name!r} requires path")
    cache_key = f"factorsim_reader:{path}"
    reader = registry.module_cache.get(cache_key)
    if reader is None:
        reader = FactorsimReader(path)
        registry.module_cache[cache_key] = reader
    freq = item_freq(item)
    if freq == "1d":
        if reader.n_levels != 1:
            raise ValueError(f"1d factorsim source {item.name!r} must be 2D")
        return reader.load_2d(start_ds, end_ds, registry.universe.dtype)

    if reader.n_levels <= 1:
        raise ValueError(f"{freq} factorsim source {item.name!r} must be 3D")
    return reader.load_intraday(
        start_ds,
        end_ds,
        dtype=registry.universe.dtype,
        freq=freq,
        universe=registry.universe,
    )


def _load_alpha_parquet(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
    if item.mode != "read_dump":
        raise NotImplementedError(f"alpha mode {item.mode!r} is not implemented")
    if not item.path:
        raise ValueError(f"alpha data {item.name!r} requires path")
    cache_key = f"alpha_parquet:{item.path}"
    frame = registry.module_cache.get(cache_key)
    if frame is None:
        frame = pd.read_parquet(item.path)
        frame.index = frame.index.astype(int)
        frame.columns = frame.columns.astype(str).str.zfill(6)
        frame = frame.sort_index().reindex(columns=registry.universe.codes)
        registry.module_cache[cache_key] = frame
    dates = [registry.universe.idx2date(idx) for idx in range(registry.universe.date2idx(start_ds), registry.universe.date2idx(end_ds) + 1)]
    aligned = frame.reindex(index=dates, columns=registry.universe.codes)
    values = aligned.to_numpy(dtype=np.float32, copy=True)
    return torch.as_tensor(values, dtype=registry.universe.dtype)
