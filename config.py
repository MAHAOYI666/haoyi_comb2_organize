from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import torch

ORGANIZE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ORGANIZE_ROOT / "vendor"
COMB2_ROOT = VENDOR_ROOT / "comb2"
PCM_ROOT = VENDOR_ROOT / "comb2-pcmaster"


def _default_strategy_path() -> str:
    source_path = PCM_ROOT / "comb2_pcmaster" / "default_strategy.py"
    if source_path.is_file():
        return str(source_path)

    package_spec = importlib.util.find_spec("comb2_pcmaster")
    if package_spec is not None and package_spec.origin:
        package_strategy = Path(package_spec.origin).resolve().parent / "default_strategy.py"
        if package_strategy.is_file():
            return str(package_strategy)

    return str(PCM_ROOT / "examples" / "alpha_strategy.py")


def _builtin_factor_item(path: str) -> dict[str, Any]:
    return {
        "name": f"alpha.{path}",
        "module": "builtin.factorsim",
        "path": path,
        "role": "factor",
        "mode": "read_dump",
        "config_path": None,
        "ops": (),
        "params": {"display_name": path},
    }


def _builtin_label_item(path: str = "vwap30_label1d") -> dict[str, Any]:
    return {
        "name": "label.default",
        "module": "builtin.factorsim",
        "path": path,
        "role": "label",
        "mode": "read_dump",
        "config_path": None,
        "ops": (),
        "params": {},
    }


DEFAULT_FACTOR_PATHS = (
    "yz_20250219_02",
    "wjx_20240829_02",
    "guanxl_05",
    "alpha1_20251008_01",
    "alpha2_20251008_02",
    "alpha3_20251008_03",
    "alpha4_20251008_04",
    "alpha5_20251008_05",
)


DEFAULT_CONFIG = {
        "constants": {
            "cache_path": "data/Cache",
            "output_root": str(COMB2_ROOT / "output"),
            "checkpoint_root": None,
        },
    "strategy": {
        "start_ds": 20160111,
        "end_ds": 20200101,
        "path": _default_strategy_path(),
    },
    "combo": {
        "paths": {
            "base_dir": str(COMB2_ROOT),
            "output_dir": str(COMB2_ROOT / "output"),
            "model_path": str(COMB2_ROOT / "lgbm_model.py"),
            "combo_base_path": None,
            "research_loader_path": None,
            "research_dataset_path": None,
            "checkpoint_root": None,
        },
        "output": {
            "alpha_history_path": str(COMB2_ROOT / "output" / "alpha_history.pt"),
            "log_path": str(COMB2_ROOT / "output" / "train.log"),
            "enable_alpha_analysis": True,
        },
        "runtime": {
            "snaptime": "mlp_minimal",
            "snap_ti": None,
            "seed": None,
            "deterministic": False,
            "livetrading": False,
            "trainDelay": 0,
            "retDays": 1,
            "tsDays": 8,
            "load_chunk_days": None,
            "torch_threads": 64,
            "torch_interop_threads": 1,
            "model_smooth_rate": 0.7,
            "model_keep_num": 2,
            "select_days": 100,
            "max_train_days": 2000,
            "verbose": False,
        },
        "model": {},
        "data": {
            "imports": (),
            "items": (*(_builtin_factor_item(path) for path in DEFAULT_FACTOR_PATHS), _builtin_label_item()),
            "presets": (),
            "attrs": {},
        },
        "loader": {
            "ashare_data_path": None,
            "dtype": torch.float16,
            "compression": "none",
            "data_start_ds": 20160101,
            "data_offset": 1024,
            "valid_path": None,
            "filtered_path": None,
            "base_universe_path": None,
            "data_items": (),
            "data_presets": (),
            "config_path": None,
        },
        "defaults": {
            "selection_module": None,
        },
    },
    "backtest": {
        "output_path": str(COMB2_ROOT / "output" / "backtest"),
        "daily_metrics_file": "daily_pnl.csv",
        "cash": 10000000.0,
        "fee_rate": 0.0015,
        "reserve_cash": 0.95,
        "verbose": False,
        "universe": "base",
        "execution_price": "vwap30",
        "drawdown_stop": 0.0,
        "cooldown_days": 0,
    },
    "monitor": {
        "enabled": False,
        "output_path": None,
        "format": "csv",
        "print_summary": True,
        "collect_gpu": True,
        "sync_cuda": False,
        "verbose": False,
    },
}

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

PATH_FIELDS = {
    ("constants", "cache_path"),
    ("constants", "output_root"),
    ("constants", "checkpoint_root"),
    ("strategy", "path"),
    ("combo", "paths", "base_dir"),
    ("combo", "paths", "output_dir"),
    ("combo", "paths", "model_path"),
    ("combo", "paths", "combo_base_path"),
    ("combo", "paths", "research_loader_path"),
    ("combo", "paths", "research_dataset_path"),
    ("combo", "paths", "checkpoint_root"),
    ("combo", "output", "alpha_history_path"),
    ("combo", "output", "log_path"),
    ("combo", "loader", "ashare_data_path"),
    ("combo", "loader", "valid_path"),
    ("combo", "loader", "filtered_path"),
    ("combo", "loader", "base_universe_path"),
    ("backtest", "output_path"),
    ("monitor", "output_path"),
}

DATA_PATH_FIELDS = {"path", "config_path"}
BUILTIN_DATA_PRESETS = {"barra"}
DATA_FREQ_ORDER = ("1d", "5m", "1m")
SUPPORTED_DATA_FREQS = set(DATA_FREQ_ORDER)
SUPPORTED_DATA_OPS = {
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
DATA_ATTR_DEFAULTS = {
    key: DEFAULT_CONFIG["combo"]["loader"][key]
    for key in (
        "ashare_data_path",
        "dtype",
        "compression",
        "data_start_ds",
        "data_offset",
        "valid_path",
        "filtered_path",
        "base_universe_path",
    )
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _parse_dtype(value: str) -> torch.dtype:
    lowered = value.strip().lower()
    if lowered not in DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {value}")
    return DTYPE_MAP[lowered]


def _coerce_like(default_value, value: str):
    if value is None:
        return None
    if default_value is None:
        stripped = value.strip()
        return None if stripped == "" else _parse_scalar(stripped)
    if isinstance(default_value, bool):
        return _parse_bool(value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(value)
    if isinstance(default_value, float):
        return float(value)
    if isinstance(default_value, torch.dtype):
        return _parse_dtype(value)
    if isinstance(default_value, (tuple, list)):
        raise TypeError("sequence values must be parsed explicitly")
    return value


def _parse_scalar(value: str):
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"true", "false", "yes", "no", "y", "n", "on", "off"}:
        return _parse_bool(stripped)
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _parse_section_attributes(element: ET.Element | None, default_section: dict, allow_extra: bool = False) -> dict:
    if element is None:
        return {}
    parsed = {}
    for key, raw in element.attrib.items():
        if key not in default_section:
            if not allow_extra:
                raise ValueError(f"unsupported config key '{key}' in <{element.tag}>")
            parsed[key] = _parse_scalar(raw)
            continue
        parsed[key] = _coerce_like(default_section[key], raw)
    return parsed


def _parse_feature_ops(feature_element: ET.Element) -> tuple[dict[str, Any], ...]:
    ops = []
    for op_element in feature_element.findall("op"):
        name = op_element.attrib.get("name")
        if not name:
            raise ValueError(f"<op> under <{feature_element.tag}> requires name")
        op_name = name.split("(", 1)[0].strip().lower()
        if op_name not in SUPPORTED_DATA_OPS:
            raise ValueError(f"unsupported data op: {name}")
        params = {key: _parse_scalar(value) for key, value in op_element.attrib.items() if key != "name"}
        ops.append({"name": name, "params": params})
    return tuple(ops)


def _parse_data_item(item_element: ET.Element) -> dict[str, Any]:
    attrs = dict(item_element.attrib)
    name = attrs.pop("name", None)
    path = attrs.pop("path", None)
    config_path = attrs.pop("config_path", None)
    mode = attrs.pop("mode", "read_dump")
    role = attrs.pop("role", "aux")
    module = attrs.pop("module", "builtin.factorsim")
    legacy_keys = {"dump_path", "source", "loader"} & set(attrs)
    if legacy_keys:
        extra = ", ".join(sorted(legacy_keys))
        raise ValueError(f"<item> uses unsupported legacy attribute(s): {extra}; use path/module")
    if not name:
        raise ValueError("<item> requires name")
    if not module:
        module = "builtin.factorsim"
    if "nbar" in attrs:
        raise ValueError(f"<item name='{name}'> uses unsupported attribute nbar")
    role = str(role).lower()
    freq = str(attrs.pop("freq", "1d") or "1d").strip().lower()
    if freq not in SUPPORTED_DATA_FREQS:
        supported = ", ".join(DATA_FREQ_ORDER)
        raise ValueError(f"<item name='{name}'> has unsupported freq={freq!r}; expected one of {supported}")
    if role == "label" and freq != "1d":
        raise ValueError(f"<item name='{name}'> role='label' only supports freq='1d'")
    if role == "label" and not path:
        path = "vwap30_label1d"
    params = {key: _parse_scalar(value) for key, value in attrs.items()}
    params["freq"] = freq
    return {
        "name": str(name),
        "module": str(module),
        "path": path,
        "role": role,
        "mode": mode,
        "config_path": config_path,
        "ops": _parse_feature_ops(item_element),
        "params": params,
    }


def _parse_data_import(import_element: ET.Element) -> dict[str, Any]:
    attrs = dict(import_element.attrib)
    path = attrs.pop("path", None)
    preset = attrs.pop("preset", None)
    role = attrs.pop("role", None)
    roles = attrs.pop("roles", None)
    if bool(path) == bool(preset):
        raise ValueError("<import> requires exactly one of path=... or preset=...")
    if role and roles:
        raise ValueError("<import> supports only one of role=... or roles=...")
    if attrs:
        extra = ", ".join(sorted(attrs))
        raise ValueError(f"unsupported attribute(s) on <import>: {extra}")
    role_filter = role or roles
    parsed_roles = None
    if role_filter:
        parsed_roles = tuple(part.strip().lower() for part in role_filter.split(",") if part.strip())
        if not parsed_roles:
            raise ValueError("<import> role filter cannot be empty")
    return {"path": path, "preset": preset, "roles": parsed_roles}


def _parse_data_section(data_element: ET.Element | None) -> dict[str, Any] | None:
    if data_element is None:
        return None
    attrs = _parse_section_attributes(data_element, DATA_ATTR_DEFAULTS)
    imports = []
    items = []
    for child in list(data_element):
        if child.tag == "import":
            imports.append(_parse_data_import(child))
            continue
        if child.tag == "item":
            items.append(_parse_data_item(child))
            continue
        raise ValueError(f"unsupported tag <{child.tag}> under <{data_element.tag}>")
    parsed: dict[str, Any] = {"attrs": attrs}
    if imports or items:
        parsed["imports"] = tuple(imports)
        parsed["items"] = tuple(items)
    return parsed


def _resolve_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _resolve_data_item_path(
    value: str | None,
    *,
    module: str,
    role: str,
    field: str,
    ashare_data_path: str | None,
    base_dir: Path,
) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    normalized_module = module.strip().lower()
    if normalized_module.startswith("builtin."):
        normalized_module = normalized_module.removeprefix("builtin.")
    if path.is_absolute():
        return str(path.resolve())
    if field == "path" and normalized_module == "barra_style":
        return value
    if field == "path" and str(role).strip().lower() == "label":
        if ashare_data_path is None:
            raise ValueError("label items require combo.loader.ashare_data_path or constants.cache_path")
        return str((Path(ashare_data_path) / "1d_DailyLabel" / f"DailyLabel.{value}").resolve())
    return str((base_dir / path).resolve())


def _load_data_pack(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    if root.tag not in {"data-pack", "data"}:
        raise ValueError(f"data import root tag must be <data-pack> or <data>: {path}")
    parsed = _parse_data_section(root)
    return parsed or {"imports": (), "items": ()}


def _resolve_data_pack(
    parsed: dict[str, Any],
    *,
    base_dir: Path,
    ashare_data_path: str | None,
    seen_paths: set[tuple[str, tuple[str, ...] | None]],
    presets: list[str],
) -> list[dict[str, Any]]:
    resolved_items: list[dict[str, Any]] = []
    for import_spec in parsed.get("imports", ()):
        role_filter = import_spec.get("roles")
        preset = import_spec.get("preset")
        if preset is not None:
            if role_filter is not None and "aux" not in role_filter:
                continue
            preset_name = str(preset).strip().lower()
            if preset_name not in BUILTIN_DATA_PRESETS:
                raise ValueError(f"unsupported built-in data preset: {preset}")
            if preset_name not in presets:
                presets.append(preset_name)
            continue
        import_path = _resolve_path(import_spec.get("path"), base_dir)
        assert import_path is not None
        seen_key = (import_path, role_filter)
        if seen_key in seen_paths:
            continue
        seen_paths.add(seen_key)
        nested_path = Path(import_path)
        nested = _load_data_pack(nested_path)
        nested_presets: list[str] = []
        nested_items = _resolve_data_pack(
            nested,
            base_dir=nested_path.parent,
            ashare_data_path=ashare_data_path,
            seen_paths=seen_paths,
            presets=nested_presets,
        )
        if role_filter is not None:
            nested_items = [item for item in nested_items if str(item.get("role", "aux")).lower() in role_filter]
        resolved_items.extend(nested_items)
        for nested_preset in nested_presets:
            if nested_preset not in presets:
                presets.append(nested_preset)
    for item in parsed.get("items", ()):
        resolved_item = dict(item)
        module = str(resolved_item["module"])
        for field in DATA_PATH_FIELDS:
            resolved_item[field] = _resolve_data_item_path(
                resolved_item.get(field),
                module=module,
                role=str(resolved_item.get("role", "aux")),
                field=field,
                ashare_data_path=ashare_data_path,
                base_dir=base_dir,
            )
        resolved_items.append(resolved_item)
    return resolved_items


def _apply_constant_paths(config: dict) -> dict:
    updated = deepcopy(config)
    constants = updated["constants"]
    cache_path = Path(constants["cache_path"]) / "AshareCache"
    output_root = Path(constants["output_root"])

    updated["combo"]["paths"]["output_dir"] = str(output_root)
    updated["combo"]["paths"]["checkpoint_root"] = constants["checkpoint_root"]
    updated["combo"]["output"]["alpha_history_path"] = str(output_root / "alpha_history.pt")
    updated["combo"]["output"]["log_path"] = str(output_root / "train.log")
    loader_config = updated["combo"]["loader"]
    if loader_config.get("ashare_data_path") is None:
        loader_config["ashare_data_path"] = str(cache_path)
    if loader_config.get("valid_path") is None:
        loader_config["valid_path"] = str(cache_path / "Ashare")
    if loader_config.get("filtered_path") is None:
        loader_config["filtered_path"] = str(cache_path / "AshareFiltered")
    if loader_config.get("base_universe_path") is None:
        loader_config["base_universe_path"] = str(cache_path / "1d_StockMask2" / "StockMask2.BaseUnivMask")
    updated["backtest"]["output_path"] = str(output_root / "backtest")
    return updated


def _resolve_loaded_paths(config: dict, base_dir: Path) -> dict:
    normalized_config = deepcopy(config)
    data_attrs = normalized_config["combo"].get("data", {}).get("attrs", {})
    if data_attrs:
        normalized_config["combo"]["loader"].update(data_attrs)

    resolved = _apply_constant_paths(normalized_config)
    for path_key in PATH_FIELDS:
        section = resolved
        for key in path_key[:-1]:
            section = section[key]
        leaf_key = path_key[-1]
        if leaf_key in section:
            section[leaf_key] = _resolve_path(section[leaf_key], base_dir)

    presets: list[str] = list(resolved["combo"].get("data", {}).get("presets", ()))
    data_items = _resolve_data_pack(
        resolved["combo"].get("data", {"imports": (), "items": ()}),
        base_dir=base_dir,
        ashare_data_path=resolved["combo"]["loader"].get("ashare_data_path"),
        seen_paths=set(),
        presets=presets,
    )
    resolved["combo"]["data"] = {
        "attrs": dict(resolved["combo"].get("data", {}).get("attrs", {})),
        "items": tuple(data_items),
        "presets": tuple(presets),
    }
    resolved["combo"]["loader"]["data_items"] = tuple(data_items)
    resolved["combo"]["loader"]["data_presets"] = tuple(presets)
    resolved["combo"]["loader"]["config_path"] = str(base_dir.resolve())
    return resolved


def _load_xml_config(path: str) -> dict:
    root = ET.parse(path).getroot()
    if root.tag != "config":
        raise ValueError("xml config root tag must be <config>")

    constants = _parse_section_attributes(root.find("constants"), DEFAULT_CONFIG["constants"])
    strategy = _parse_section_attributes(root.find("strategy"), DEFAULT_CONFIG["strategy"])

    combo_element = root.find("combo")
    combo = {}
    if combo_element is not None:
        if combo_element.find("loader") is not None:
            raise ValueError("<combo><loader> is no longer supported; put loader/data attributes on <combo><data>")
        combo = {
            "paths": _parse_section_attributes(combo_element.find("paths"), DEFAULT_CONFIG["combo"]["paths"]),
            "output": _parse_section_attributes(combo_element.find("output"), DEFAULT_CONFIG["combo"]["output"]),
            "runtime": _parse_section_attributes(combo_element.find("runtime"), DEFAULT_CONFIG["combo"]["runtime"]),
            "model": _parse_section_attributes(combo_element.find("model"), DEFAULT_CONFIG["combo"]["model"], allow_extra=True),
            "defaults": _parse_section_attributes(combo_element.find("defaults"), DEFAULT_CONFIG["combo"]["defaults"]),
        }
        data_element = combo_element.find("data")
        if data_element is not None:
            combo["data"] = _parse_data_section(data_element) or {"attrs": {}, "imports": (), "items": ()}

    backtest = _parse_section_attributes(root.find("backtest"), DEFAULT_CONFIG["backtest"])
    monitor = _parse_section_attributes(root.find("monitor"), DEFAULT_CONFIG["monitor"])

    return {
        "constants": constants,
        "strategy": strategy,
        "combo": combo,
        "backtest": backtest,
        "monitor": monitor,
    }


def load_config(path: str | None = None) -> dict:
    if path is None:
        return _resolve_loaded_paths(deepcopy(DEFAULT_CONFIG), ORGANIZE_ROOT)
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() != ".xml":
        raise ValueError("config file must be an XML file")
    loaded = _load_xml_config(str(config_path))
    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    return _resolve_loaded_paths(merged, config_path.parent)
