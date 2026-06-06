# haoyi 2026/05/5

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import bisect
import importlib
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .op_utils import cs_zscore, nan_to_num, nanmean, neut, normalize_by_max_abs, truncate, winsorize_by_quantile

try:
    from factorsim import IndexMask, Memmaper2
except ModuleNotFoundError:
    IndexMask = None
    Memmaper2 = None

ORGANIZE_ROOT = Path(__file__).resolve().parents[3]
if str(ORGANIZE_ROOT) not in sys.path:
    sys.path.insert(0, str(ORGANIZE_ROOT))

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
NEUT_OP_PATTERN = re.compile(r"^neut\(([^()]+)\)$", re.IGNORECASE)


def _require_index_mask_cls():
    if IndexMask is None:
        raise ModuleNotFoundError("factorsim is required to build the comb2 trading universe")
    return IndexMask


def _require_memmaper2_cls():
    if Memmaper2 is None:
        raise ModuleNotFoundError("factorsim is required to read Memmaper2 data")
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


def _parse_barra_neut_name(name: str) -> tuple[str, ...] | None:
    match = NEUT_OP_PATTERN.match(name.strip())
    if match is None:
        return None
    raw_styles = match.group(1).strip()
    if not raw_styles:
        raise ValueError(f"invalid neutralization op: {name}")
    style_names = tuple(part.strip() for part in raw_styles.split(",") if part.strip())
    if not style_names:
        raise ValueError(f"invalid neutralization op: {name}")
    return style_names


def _normalize_barra_style_name(style_name: str) -> str:
    normalized = style_name.strip()
    prefix = BARRA_STYLE_PREFIX.lower()
    if normalized.lower().startswith(prefix):
        normalized = normalized[len(BARRA_STYLE_PREFIX) :]
    return normalized.strip().lower()


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
    def from_mask(cls, mask: Any, dtype: torch.dtype) -> "Universe":
        return cls(
            dates=tuple(int(ds) for ds in mask.date),
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


def _coerce_data_item(value: DataItem | dict[str, Any]) -> DataItem:
    if isinstance(value, DataItem):
        return value
    module = value.get("module", value.get("source", value.get("loader")))
    if not module:
        raise ValueError(f"data item {value.get('name')!r} requires module")
    ops = tuple(_coerce_op_spec(op) for op in value.get("ops", ()))
    return DataItem(
        name=str(value["name"]),
        module=str(module),
        path=value.get("path"),
        role=str(value.get("role", "aux") or "aux").lower(),
        mode=str(value.get("mode", "read_dump")),
        config_path=value.get("config_path"),
        ops=ops,
        params=dict(value.get("params", {})),
    )


def _parse_op_call(raw_name: str) -> tuple[str, tuple[str, ...]]:
    stripped = raw_name.strip()
    match = re.match(r"^([A-Za-z_]\w*)\((.*)\)$", stripped)
    if match is None:
        return stripped.lower(), ()
    name = match.group(1).strip().lower()
    raw_args = match.group(2).strip()
    if not raw_args:
        return name, ()
    return name, tuple(part.strip() for part in raw_args.split(",") if part.strip())


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
            _merge_deps(deps, args)
        elif name == "delay":
            lookback += max(0, _int_op_arg(op, args, "days", "periods", "n", default=1))
        elif name in {"ts_mean", "ts_avg"}:
            lookback += max(0, _int_op_arg(op, args, "window", "days", "n") - 1)
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


def _rolling_nanmean(x: torch.Tensor, window: int) -> torch.Tensor:
    window = int(window)
    if window <= 0:
        raise ValueError("ts_mean window must be positive")
    out = torch.full_like(x, torch.nan)
    for idx in range(x.shape[0]):
        lo = max(0, idx - window + 1)
        out[idx] = nanmean(x[lo : idx + 1], dim=0)
    return out


def _rowwise_winsorize(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if x.ndim == 1:
        return winsorize_by_quantile(x, low, high)
    return torch.stack([winsorize_by_quantile(row, low, high) for row in x], dim=0)


def _rowwise_normalize_by_max_abs(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return normalize_by_max_abs(x)
    return torch.stack([normalize_by_max_abs(row) for row in x], dim=0)


def _as_2d_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        if tensor.ndim != 2:
            raise ValueError(f"data module must return 2-D data, got shape {tuple(tensor.shape)}")
        return tensor
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"data module must return 2-D data, got shape {arr.shape}")
    return torch.as_tensor(arr, dtype=dtype)


DataLoadFn = Callable[[DataItem, "DataRegistry", int, int], torch.Tensor | np.ndarray]


class DataRegistry:
    def __init__(
        self,
        items: Sequence[DataItem | dict[str, Any]],
        *,
        universe: Universe,
        data_start_ds: int,
        ashare_data_path: str | None,
        factor_root: str | None,
        config_path: str | None,
        presets: Sequence[str] = (),
    ):
        self.universe = universe
        self.data_start_ds = int(data_start_ds)
        self.data_start_idx = universe.date2idx(int(data_start_ds))
        self.ashare_data_path = ashare_data_path
        self.factor_root = factor_root
        self.config_path = config_path
        self.module_cache: dict[str, Any] = {}
        self.modules: dict[str, DataLoadFn] = self._builtin_modules()

        all_items = list(_coerce_data_item(item) for item in items)
        all_items.extend(self._preset_items(presets))
        self.items: dict[str, DataItem] = {}
        for item in all_items:
            if item.name in self.items:
                raise ValueError(f"duplicate data item name: {item.name}")
            self.items[item.name] = item

        shape = (len(universe.dates), len(universe.codes))
        self.raw_cache = {
            name: torch.full(shape, torch.nan, dtype=universe.dtype)
            for name in self.items
        }
        self.processed_cache = {
            name: torch.full(shape, torch.nan, dtype=universe.dtype)
            for name in self.items
        }
        self.raw_loaded = {
            name: torch.zeros(shape[0], dtype=torch.bool)
            for name in self.items
        }
        self.processed_loaded = {
            name: torch.zeros(shape[0], dtype=torch.bool)
            for name in self.items
        }
        self.aliases = self._build_aliases()

    def _builtin_modules(self) -> dict[str, DataLoadFn]:
        return {
            "factor": _load_memmap_factor,
            "builtin.factor": _load_memmap_factor,
            "label": _load_label,
            "builtin.label": _load_label,
            "alpha_parquet": _load_alpha_parquet,
            "builtin.alpha_parquet": _load_alpha_parquet,
            "barra_style": _load_barra_style,
            "builtin.barra_style": _load_barra_style,
            "ref": _load_ref_data,
            "builtin.ref": _load_ref_data,
        }

    def _preset_items(self, presets: Sequence[str]) -> list[DataItem]:
        items: list[DataItem] = []
        for preset in presets:
            preset_name = str(preset).strip().lower()
            if preset_name != "barra":
                raise ValueError(f"unsupported data preset: {preset}")
            for style in BARRA_PRESET_STYLES:
                items.append(
                    DataItem(
                        name=f"barra.{style}",
                        module="builtin.barra_style",
                        path=style,
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

    def _resolve_name(self, name: str) -> str:
        normalized = str(name).strip()
        if normalized in self.items:
            return normalized
        if normalized in self.aliases:
            return self.aliases[normalized]
        raise KeyError(f"unknown data name: {name}")

    def get_data(self, name: str) -> torch.Tensor:
        return self.processed_cache[self._resolve_name(name)]

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

    def _ensure_raw_range(self, name: str, start_ds: int, end_ds: int):
        name = self._resolve_name(name)
        lo, hi = self._bounds_to_idx(start_ds, end_ds)
        if hi < lo:
            return
        item = self.items[name]
        for miss_lo, miss_hi in _missing_ranges(self.raw_loaded[name], lo, hi):
            chunk_start = self.universe.idx2date(miss_lo)
            chunk_end = self.universe.idx2date(miss_hi)
            loaded = self._module_for(item)(item, self, chunk_start, chunk_end)
            tensor = _as_2d_tensor(loaded, dtype=self.universe.dtype)
            expected_shape = (miss_hi - miss_lo + 1, len(self.universe.codes))
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"data module {item.module!r} for {name!r} returned shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            self.raw_cache[name][miss_lo : miss_hi + 1] = tensor
            self.raw_loaded[name][miss_lo : miss_hi + 1] = True

    def _ensure_processed_range(self, name: str, start_ds: int, end_ds: int, stack: tuple[str, ...] = ()):
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
        raw_lo = max(self.data_start_idx, lo - requirements.lookback_days + 1)
        raw_start_ds = self.universe.idx2date(raw_lo)
        for dep in requirements.data_deps:
            self._ensure_processed_range(dep, raw_start_ds, end_ds, (*stack, name))

        self._ensure_raw_range(name, raw_start_ds, end_ds)
        raw_window = self.raw_cache[name][raw_lo : hi + 1].to(torch.float32)
        processed_window = self._apply_ops(item, raw_window, raw_lo, hi)
        out_lo = lo - raw_lo
        out_hi = hi - raw_lo + 1
        self.processed_cache[name][lo : hi + 1] = processed_window[out_lo:out_hi].to(self.universe.dtype)
        self.processed_loaded[name][lo : hi + 1] = True

    def _ensure_range(self, names: Sequence[str], start_ds: int, end_ds: int):
        for name in names:
            self._ensure_processed_range(name, start_ds, end_ds)

    def _apply_ops(self, item: DataItem, x: torch.Tensor, lo_idx: int, hi_idx: int) -> torch.Tensor:
        out = x
        for op in item.ops:
            raw_name = op.name.strip()
            name, args = _parse_op_call(raw_name)
            params = op.params
            if name in {"cs_zscore", "zscore"}:
                out = cs_zscore(out)
            elif name == "truncate":
                out = truncate(out, float(params.get("min", -4.0)), float(params.get("max", 4.0)))
            elif name in {"nan_to_num", "fillna"}:
                out = nan_to_num(out, float(params.get("value", 0.0)))
            elif name == "winsorize_by_quantile":
                out = _rowwise_winsorize(out, float(params.get("low", 0.01)), float(params.get("high", 0.99)))
            elif name == "normalize_by_max_abs":
                out = _rowwise_normalize_by_max_abs(out)
            elif name == "neut":
                if not args:
                    raise ValueError(f"{op.name} requires at least one data dependency")
                xs = [self.get_data(dep)[lo_idx : hi_idx + 1].to(torch.float32) for dep in args]
                out = neut(out, xs)
            elif name == "delay":
                periods = max(0, _int_op_arg(op, args, "days", "periods", "n", default=1))
                shifted = torch.full_like(out, torch.nan)
                if periods == 0:
                    shifted = out
                elif periods < out.shape[0]:
                    shifted[periods:] = out[:-periods]
                out = shifted
            elif name in {"ts_mean", "ts_avg"}:
                out = _rolling_nanmean(out, _int_op_arg(op, args, "window", "days", "n"))
            else:
                raise ValueError(f"unsupported data op: {op.name}")
        out[torch.isinf(out)] = torch.nan
        return out


def _memmap_load_2d(registry: DataRegistry, path: str, start_ds: int, end_ds: int, dtype: torch.dtype) -> torch.Tensor:
    cache_key = f"memmap:{path}"
    mmap = registry.module_cache.get(cache_key)
    if mmap is None:
        mmap = _require_memmaper2_cls()(path)
        registry.module_cache[cache_key] = mmap
    data = mmap.load(start_ds=int(start_ds), end_ds=int(end_ds))[:]
    return _as_2d_tensor(data, dtype=dtype)


def _load_memmap_factor(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
    if not item.path:
        raise ValueError(f"factor data {item.name!r} requires path")
    return _memmap_load_2d(registry, item.path, start_ds, end_ds, registry.universe.dtype)


def _load_label(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
    path = item.path
    if not path or path == "label1d":
        if not registry.ashare_data_path:
            raise ValueError(f"label data {item.name!r} requires path or registry.ashare_data_path")
        path = str(Path(registry.ashare_data_path) / "1d_DailyLabel" / "DailyLabel.label1d")
    return _memmap_load_2d(registry, path, start_ds, end_ds, registry.universe.dtype)


def _load_barra_style(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
    style = item.path or item.params.get("style") or item.name.rsplit(".", 1)[-1]
    if not registry.ashare_data_path:
        raise ValueError(f"Barra data {item.name!r} requires registry.ashare_data_path")
    root = Path(registry.ashare_data_path) / BARRA_STYLE_DIRNAME
    cache_key = f"barra_paths:{root}"
    path_by_style = registry.module_cache.get(cache_key)
    if path_by_style is None:
        if not root.exists():
            raise FileNotFoundError(f"Barra style directory not found: {root}")
        path_by_style = {}
        for path in sorted(root.iterdir(), key=lambda item_path: item_path.name):
            if not path.name.lower().startswith(BARRA_STYLE_PREFIX.lower()):
                continue
            style_name = path.name[len(BARRA_STYLE_PREFIX) :]
            path_by_style[_normalize_barra_style_name(style_name)] = path
        if not path_by_style:
            raise FileNotFoundError(f"No Barra style files found under: {root}")
        registry.module_cache[cache_key] = path_by_style
    normalized = _normalize_barra_style_name(str(style))
    if normalized not in path_by_style:
        supported = ", ".join(sorted(path_by_style))
        raise KeyError(f"unsupported Barra style factor: {style}. Available: {supported}")
    path = path_by_style[normalized]
    return _memmap_load_2d(registry, str(path), start_ds, end_ds, registry.universe.dtype)


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
    return torch.as_tensor(aligned.to_numpy(dtype=np.float32), dtype=registry.universe.dtype)


def _load_ref_data(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
    ref_name = item.params.get("data") or item.params.get("ref") or item.path
    if not ref_name:
        raise ValueError(f"ref data {item.name!r} requires params.data")
    registry._ensure_processed_range(str(ref_name), start_ds, end_ds)
    lo = registry.universe.date2idx(start_ds)
    hi = registry.universe.date2idx(end_ds)
    return registry.get_data(str(ref_name))[lo : hi + 1]


class BarraStyleSource:
    def __init__(self, ashare_data_path: str, dtype: torch.dtype):
        self.ashare_data_path = ashare_data_path
        self.dtype = dtype
        self.root = Path(ashare_data_path) / BARRA_STYLE_DIRNAME
        self._path_by_style = self._discover_paths()
        self._mmap_by_style: dict[str, Memmaper2] = {}
        self._day_cache: dict[tuple[str, int], torch.Tensor] = {}

    def _discover_paths(self) -> dict[str, Path]:
        if not self.root.exists():
            raise FileNotFoundError(f"Barra style directory not found: {self.root}")
        path_by_style: dict[str, Path] = {}
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not path.name.startswith(BARRA_STYLE_PREFIX):
                continue
            style_name = path.name.removeprefix(BARRA_STYLE_PREFIX).strip()
            if style_name:
                path_by_style[style_name.lower()] = path
        if not path_by_style:
            raise FileNotFoundError(f"No Barra style files found under: {self.root}")
        return path_by_style

    def _normalize_style_name(self, style_name: str) -> str:
        normalized = _normalize_barra_style_name(style_name)
        if normalized not in self._path_by_style:
            supported = ", ".join(sorted(self._path_by_style))
            raise KeyError(f"unsupported Barra style factor: {style_name}. Available: {supported}")
        return normalized

    def _get_mmap(self, style_name: str) -> Memmaper2:
        normalized = self._normalize_style_name(style_name)
        mmap = self._mmap_by_style.get(normalized)
        if mmap is None:
            mmap = _require_memmaper2_cls()(str(self._path_by_style[normalized]))
            self._mmap_by_style[normalized] = mmap
        return mmap

    def load_day(self, style_name: str, ds: int) -> torch.Tensor:
        normalized = self._normalize_style_name(style_name)
        cache_key = (normalized, int(ds))
        cached = self._day_cache.pop(cache_key, None)
        if cached is not None:
            self._day_cache[cache_key] = cached
            return cached
        data = self._get_mmap(normalized).load(start_ds=int(ds), end_ds=int(ds))[:]
        exposure = torch.as_tensor(np.asarray(data)[0], dtype=self.dtype)
        self._day_cache[cache_key] = exposure
        return exposure


def _apply_feature_ops(
    x: torch.Tensor,
    ops: Sequence[OpSpec],
    *,
    ds: int | None = None,
    barra_source: BarraStyleSource | None = None,
) -> torch.Tensor:
    if not ops:
        return x
    out = x.to(torch.float32)
    for op in ops:
        raw_name = op.name.strip()
        name = raw_name.lower()
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
        elif (style_names := _parse_barra_neut_name(raw_name)) is not None:
            if ds is None:
                raise ValueError(f"{op.name} requires ds context")
            if barra_source is None:
                raise ValueError(f"{op.name} requires loader.ashare_data_path / BarraCNE5 support")
            out = neut(out, [barra_source.load_day(style_name, ds).to(torch.float32) for style_name in style_names])
        else:
            raise ValueError(f"unsupported feature op: {op.name}")
    return out


class FeatureItemSource:
    feature_dim = 1

    def __init__(self, spec: FeatureSpec, dtype: torch.dtype, barra_source: BarraStyleSource | None = None):
        self.spec = spec
        self.name = spec.name
        self.dtype = dtype
        self.barra_source = barra_source

    def _load_raw_day(self, ds: int) -> torch.Tensor:
        raise NotImplementedError

    def load_day(self, ds: int) -> torch.Tensor:
        x = self._load_raw_day(int(ds))
        x = _apply_feature_ops(x, self.spec.ops, ds=int(ds), barra_source=self.barra_source)
        x[torch.isinf(x)] = torch.nan
        return x.to(self.dtype)

    def prefetch_days(self, days: Sequence[int]):
        return None


class MemmapFactorItemSource(FeatureItemSource):
    def __init__(self, spec: FeatureSpec, dtype: torch.dtype, barra_source: BarraStyleSource | None = None):
        super().__init__(spec, dtype, barra_source=barra_source)
        if not spec.path:
            raise ValueError(f"factor feature {spec.name!r} requires path")
        self.path = spec.path
        self._mmap: Memmaper2 | None = None
        self._day_cache: dict[int, torch.Tensor] = {}

    def _get_mmap(self) -> Memmaper2:
        if self._mmap is None:
            self._mmap = _require_memmaper2_cls()(self.path)
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
    def __init__(self, spec: FeatureSpec, dtype: torch.dtype, codes: Sequence[int | str], barra_source: BarraStyleSource | None = None):
        super().__init__(spec, dtype, barra_source=barra_source)
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
    def __init__(self, specs: Sequence[FeatureSpec], dtype: torch.dtype, codes: Sequence[int | str], ashare_data_path: str | None = None):
        if not specs:
            raise ValueError("at least one feature is required")
        self.specs = tuple(specs)
        self.dtype = dtype
        self.barra_source = self._build_barra_source(ashare_data_path)
        self.sources = [self._build_source(spec, dtype, codes) for spec in self.specs]
        self.feature_dim = sum(source.feature_dim for source in self.sources)
        self.feature_names = tuple(source.name for source in self.sources)

    def _build_barra_source(self, ashare_data_path: str | None) -> BarraStyleSource | None:
        needs_barra = any(_parse_barra_neut_name(op.name) is not None for spec in self.specs for op in spec.ops)
        if not needs_barra:
            return None
        if not ashare_data_path:
            raise ValueError("Barra neutralization requires loader.ashare_data_path")
        return BarraStyleSource(ashare_data_path, dtype=self.dtype)

    def _build_source(self, spec: FeatureSpec, dtype: torch.dtype, codes: Sequence[int | str]) -> FeatureItemSource:
        kind = spec.kind.strip().lower()
        if kind == "factor":
            return MemmapFactorItemSource(spec, dtype, barra_source=self.barra_source)
        if kind == "alpha":
            return AlphaParquetItemSource(spec, dtype, codes, barra_source=self.barra_source)
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
