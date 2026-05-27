from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

ORGANIZE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ORGANIZE_ROOT / "vendor"
COMB2_ROOT = VENDOR_ROOT / "comb2"
PCM_ROOT = VENDOR_ROOT / "comb2-pcmaster"

DEFAULT_CONFIG = {
    "constants": {
        "cache_path": "data/Cache",
        "factor_root": "data/Factor/FactorData",
        "output_root": str(COMB2_ROOT / "output"),
        "checkpoint_root": None,
    },
    "strategy": {
        "start_ds": 20160111,
        "end_ds": 20200101,
        "path": str(ORGANIZE_ROOT / "alpha_strategy.py"),
    },
    "combo": {
        "paths": {
            "base_dir": str(COMB2_ROOT),
            "output_dir": str(COMB2_ROOT / "output"),
            "model_path": str(COMB2_ROOT / "lgbm_model.py"),
            "checkpoint_root": None,
        },
        "output": {
            "alpha_history_path": str(COMB2_ROOT / "output" / "alpha_history.pt"),
            "log_path": str(COMB2_ROOT / "output" / "train.log"),
            "enable_alpha_analysis": True,
        },
        "runtime": {
            "snaptime": "mlp_minimal",
            "livetrading": False,
            "trainDelay": 1,
            "retDays": 1,
            "tsDays": 8,
            "model_smooth_rate": 0.7,
            "model_keep_num": 2,
            "select_days": 100,
            "max_train_days": 2000,
        },
        "model": {},
        "loader": {
            "factor_paths": (
                "yz_20250219_02",
                "wjx_20240829_02",
                "guanxl_05",
                "alpha1_20251008_01",
                "alpha2_20251008_02",
                "alpha3_20251008_03",
                "alpha4_20251008_04",
                "alpha5_20251008_05",
            ),
            "style_factor_paths": (),
            "alpha_factor_paths": (),
            "label_path": None,
            "ashare_data_path": None,
            "dtype": torch.float16,
            "compression": "none",
            "data_start_ds": 20160101,
            "valid_path": None,
            "filtered_path": None,
            "base_universe_path": None,
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
    },
    "monitor": {
        "enabled": False,
        "output_path": None,
        "format": "csv",
        "print_summary": True,
        "collect_gpu": True,
        "sync_cuda": False,
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
    ("constants", "factor_root"),
    ("constants", "output_root"),
    ("constants", "checkpoint_root"),
    ("strategy", "path"),
    ("combo", "paths", "base_dir"),
    ("combo", "paths", "output_dir"),
    ("combo", "paths", "model_path"),
    ("combo", "paths", "checkpoint_root"),
    ("combo", "output", "alpha_history_path"),
    ("combo", "output", "log_path"),
    ("combo", "loader", "label_path"),
    ("combo", "loader", "ashare_data_path"),
    ("combo", "loader", "valid_path"),
    ("combo", "loader", "filtered_path"),
    ("combo", "loader", "base_universe_path"),
    ("backtest", "output_path"),
    ("monitor", "output_path"),
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
        return None if stripped == "" else stripped
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


def _parse_path_list(parent_element: ET.Element | None, tag: str, default_paths=()) -> tuple[str, ...] | None:
    if parent_element is None:
        return None
    path_list_element = parent_element.find(tag)
    if path_list_element is None:
        return None
    paths = []
    for path_element in path_list_element.findall("path"):
        path_value = (path_element.text or "").strip()
        if path_value:
            paths.append(path_value)
    if not paths:
        return tuple(default_paths)
    return tuple(paths)


def _parse_factor_paths(loader_element: ET.Element | None, default_paths) -> tuple[str, ...] | None:
    return _parse_path_list(loader_element, "factor_paths", default_paths)


def _parse_style_factor_paths(loader_element: ET.Element | None) -> tuple[str, ...] | None:
    return _parse_path_list(loader_element, "style_factor_paths", ())


def _resolve_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _resolve_factor_path(factor_root: str, path: str) -> str:
    factor_path = Path(path).expanduser()
    if not factor_path.is_absolute():
        factor_path = Path(factor_root) / factor_path
    return str(factor_path.resolve())


def _apply_constant_paths(config: dict) -> dict:
    updated = deepcopy(config)
    constants = updated["constants"]
    cache_path = Path(constants["cache_path"]) / "AshareCache"
    output_root = Path(constants["output_root"])

    updated["combo"]["paths"]["output_dir"] = str(output_root)
    updated["combo"]["paths"]["checkpoint_root"] = constants["checkpoint_root"]
    updated["combo"]["output"]["alpha_history_path"] = str(output_root / "alpha_history.pt")
    updated["combo"]["output"]["log_path"] = str(output_root / "train.log")
    updated["combo"]["loader"]["label_path"] = str(cache_path / "1d_DailyLabel" / "DailyLabel.label1d")
    updated["combo"]["loader"]["ashare_data_path"] = str(cache_path)
    updated["combo"]["loader"]["valid_path"] = str(cache_path / "Ashare")
    updated["combo"]["loader"]["filtered_path"] = str(cache_path / "AshareFiltered")
    updated["combo"]["loader"]["base_universe_path"] = str(cache_path / "1d_StockMask2" / "StockMask2.BaseUnivMask")
    updated["backtest"]["output_path"] = str(output_root / "backtest")
    return updated


def _resolve_loaded_paths(config: dict, base_dir: Path) -> dict:
    resolved = _apply_constant_paths(config)
    for path_key in PATH_FIELDS:
        section = resolved
        for key in path_key[:-1]:
            section = section[key]
        leaf_key = path_key[-1]
        if leaf_key in section:
            section[leaf_key] = _resolve_path(section[leaf_key], base_dir)

    factor_root = resolved["constants"]["factor_root"]
    alpha_paths = resolved["combo"]["loader"].get("factor_paths") or ()
    style_paths = resolved["combo"]["loader"].get("style_factor_paths") or ()
    resolved_alpha_paths = tuple(_resolve_factor_path(factor_root, path) for path in alpha_paths)
    resolved_style_paths = tuple(_resolve_factor_path(factor_root, path) for path in style_paths)

    resolved["combo"]["loader"]["alpha_factor_paths"] = resolved_alpha_paths
    resolved["combo"]["loader"]["style_factor_paths"] = resolved_style_paths
    resolved["combo"]["loader"]["factor_paths"] = resolved_alpha_paths + resolved_style_paths

    resolved["combo"]["model"]["alpha_feature_count"] = len(resolved_alpha_paths)
    resolved["combo"]["model"]["style_feature_count"] = len(resolved_style_paths)
    resolved["combo"]["model"]["total_feature_count"] = len(resolved_alpha_paths) + len(resolved_style_paths)
    resolved["combo"]["model"]["use_style_gate"] = len(resolved_style_paths) > 0
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
        combo = {
            "paths": _parse_section_attributes(combo_element.find("paths"), DEFAULT_CONFIG["combo"]["paths"]),
            "output": _parse_section_attributes(combo_element.find("output"), DEFAULT_CONFIG["combo"]["output"]),
            "runtime": _parse_section_attributes(combo_element.find("runtime"), DEFAULT_CONFIG["combo"]["runtime"]),
            "model": _parse_section_attributes(combo_element.find("model"), DEFAULT_CONFIG["combo"]["model"], allow_extra=True),
            "loader": _parse_section_attributes(combo_element.find("loader"), DEFAULT_CONFIG["combo"]["loader"]),
            "defaults": _parse_section_attributes(combo_element.find("defaults"), DEFAULT_CONFIG["combo"]["defaults"]),
        }
        loader_element = combo_element.find("loader")
        factor_paths = _parse_factor_paths(loader_element, DEFAULT_CONFIG["combo"]["loader"]["factor_paths"])
        style_factor_paths = _parse_style_factor_paths(loader_element)
        if factor_paths is not None:
            combo["loader"]["factor_paths"] = factor_paths
        if style_factor_paths is not None:
            combo["loader"]["style_factor_paths"] = style_factor_paths

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
