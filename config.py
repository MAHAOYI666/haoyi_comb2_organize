from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import torch

ORGANIZE_ROOT = Path(__file__).resolve().parent
LOCAL_SIMBASE_ROOT = ORGANIZE_ROOT / "vendor" / "comb2-simbase"
if LOCAL_SIMBASE_ROOT.is_dir() and str(LOCAL_SIMBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SIMBASE_ROOT))

from comb2_simbase.cache_layout import daily_label_path

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


DEFAULT_OPTIMIZER_CONFIG = {
    "type": "opt1",
    "lambda0": 0.5,
    "shrinkage": 0.5,
    "ret_days": 60,
    "ret_delay": 1,
    "ret_method": 2,
    "benchmark": "000905.SH",
    "benchmark_delay": 1,
    "target_size": 1.0e8,
    "maxtvr": 0.2,
    "max_weight": 0.0075,
    "maxtrd": 0.0,
    "maxpos": 0.0,
    "liquidity_delay": 1,
    "lambda_slp": 0.0,
    "slippage_delay": 1,
    "min_participation_ratio": 0.07,
    "parti_penalty": 0.0,
    "trim_threshold": 1.0e-5,
    "min_valid_instruments": 200,
    "min_return_obs": 20,
    "soft_univ_penalty": 0.00025,
    "soft_risk_penalty": 0.00004,
    "soft_group_penalty": 0.0002,
    "num_mosek_threads": 1,
    "max_time": 30.0,
    "post_trim_renorm": False,
    "univ_list": (
        "ZZ500:0.18:0.70,1|"
        "AshareST:0.00:0.00,1|"
        "AshareSH:0.00:0.60,1|"
        "AshareSZ:0.00:0.60,1|"
        "NONETOP3000:0.00:0.17,1|"
        "AshareCYB:0.10:0.30,1"
    ),
    "soft_univ_list": (
        "ZZ1800:0.74:0.85:0.70,1|"
        "ZZ1800:0.79:0.85:0.45,1|"
        "ZZ1800:0.83:0.89:0.25,1|"
        "ZZ500:0.28:0.50:2.0,1|"
        "ZZ500:0.30:0.50:0.2,1|"
        "ZZ500:0.32:0.50:0.05,1|"
        "AshareSH:0.00:0.55:0.50,1|"
        "AshareCYB:0.10:0.25:1.00,1|"
        "AshareSZ:0.00:0.55:0.50,1|"
        "HS300:0.10:0.30:1.00,1|"
        "NONETOP3000:0.00:0.15:1.00,1"
    ),
    "risk_list": (
        "cap:-0.40:0.30,1,4|"
        "cap:-0.20:0.21,1,2|"
        "returns120:-0.14:0.14,1|"
        "vola_30:-0.30:0.30,1|"
        "vola_5:-0.30:0.30,1|"
        "close:-0.10:0.10,1|"
        "BarraCNE5.BETA:-0.20:0.30,1|"
        "BarraCNE5.GROWTH:-0.15:0.20,1|"
        "BarraCNE5.BTOP:-0.15:0.20,1|"
        "BarraCNE5.LEVERAGE:-0.30:0.30,1|"
        "BarraCNE5.RESVOL:-0.30:0.30,1"
    ),
    "soft_risk_list": (
        "cap:-0.02:0.07:2.0,1,4|"
        "returns120:-0.08:0.08:5.0,1|"
        "close:-0.05:0.05:1.0,1|"
        "BarraCNE5.BETA:0.00:0.04:1.7,1|"
        "BarraCNE5.GROWTH:-0.02:0.06:1.3,1|"
        "BarraCNE5.BTOP:-0.02:0.07:1.3,1|"
        "BarraCNE5.EARNYILD:-0.03:0.03:2.0,1"
    ),
    "group_list": "WindIndustry.sw1:-0.065:0.065,1",
    "soft_group_list": (
        "WindIndustry.sw1:-0.05:0.05,1|"
        "WindIndustry.sw3:-0.012:0.012,1"
    ),
}


DEFAULT_CONFIG = {
    "constants": {
        "cache_path": "data/Cache",
        "output_root": str(COMB2_ROOT / "output"),
    },
    "strategy": {
        "start_ds": 20160111,
        "end_ds": 20200101,
        "path": _default_strategy_path(),
        "optimizer": DEFAULT_OPTIMIZER_CONFIG,
    },
    "combo": {
        "paths": {
            "model_path": str(ORGANIZE_ROOT / "eg-torch" / "model.py"),
            "combo_base_path": None,
            "research_loader_path": str(ORGANIZE_ROOT / "eg-torch" / "loader.py"),
            "research_dataset_path": None,
        },
        "output": {
            "enable_alpha_analysis": True,
        },
        "runtime": {
            "snaptime": "mlp_minimal",
            "sample_times": "100000",
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
            "max_train_days": 2000,
            "verbose": False,
        },
        "model": {},
        "loader": {
            "dtype": torch.float16,
            "compression": "none",
            "data_start_ds": 20160101,
            "data_offset": 1024,
            "registry_cache_days": 64,
        },
    },
    "backtest": {
        "daily_metrics_file": "daily_pnl.csv",
        "cash": 10000000.0,
        "fee_rate": 0.00075,
        "reserve_cash": 0.95,
        "verbose": False,
        "universe": "base",
        "execution_price": "execution:execution",
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
    ("strategy", "path"),
    ("combo", "paths", "model_path"),
    ("combo", "paths", "combo_base_path"),
    ("combo", "paths", "research_loader_path"),
    ("combo", "paths", "research_dataset_path"),
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


def _resolve_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _apply_constant_paths(config: dict) -> dict:
    updated = deepcopy(config)
    constants = updated["constants"]
    cache_path = Path(constants["cache_path"])
    output_root = Path(constants["output_root"])

    updated["combo"]["paths"]["output_dir"] = str(output_root)
    updated["combo"]["paths"]["checkpoint_root"] = str(output_root / "checkpoints")
    updated["combo"]["output"]["alpha_history_path"] = str(output_root / "alpha_history.pt")
    updated["combo"]["output"]["log_path"] = str(output_root / "train.log")
    updated["backtest"]["output_path"] = str(output_root / "backtest")
    return updated


def _resolve_loaded_paths(config: dict, base_dir: Path) -> dict:
    resolved = deepcopy(config)
    for path_key in PATH_FIELDS:
        section = resolved
        for key in path_key[:-1]:
            section = section[key]
        leaf = path_key[-1]
        if leaf in section:
            section[leaf] = _resolve_path(section[leaf], base_dir)
    resolved = _apply_constant_paths(resolved)
    raw_times = resolved["combo"]["runtime"]["sample_times"]
    times = tuple(int(part.strip()) for part in raw_times.split(","))
    assert times and tuple(sorted(set(times))) == times, "sample_times must be unique and increasing"
    for ti in times:
        assert 0 <= ti // 10000 < 24 and 0 <= ti // 100 % 100 < 60 and 0 <= ti % 100 < 60
    resolved["combo"]["runtime"]["sample_times"] = times
    _validate_config(resolved)
    return resolved


def _validate_config(config: dict) -> None:
    runtime = config["combo"]["runtime"]
    nonnegative = ("trainDelay",)
    positive = ("retDays", "tsDays", "max_train_days", "torch_threads", "torch_interop_threads")
    for name in nonnegative:
        if int(runtime[name]) < 0:
            raise ValueError(f"combo.runtime.{name} must be nonnegative")
    for name in positive:
        if int(runtime[name]) <= 0:
            raise ValueError(f"combo.runtime.{name} must be positive")
    if runtime.get("load_chunk_days") is not None and int(runtime["load_chunk_days"]) <= 0:
        raise ValueError("combo.runtime.load_chunk_days must be positive when set")
    if int(runtime["max_train_days"]) < int(runtime["tsDays"]):
        raise ValueError("combo.runtime.max_train_days must be at least tsDays")
    smooth_rate = float(runtime["model_smooth_rate"])
    if not 0.0 <= smooth_rate <= 1.0:
        raise ValueError("combo.runtime.model_smooth_rate must be between 0 and 1")

    loader = config["combo"]["loader"]
    if int(loader["data_offset"]) < 0:
        raise ValueError("combo.data.data_offset must be nonnegative")
    if int(loader["registry_cache_days"]) <= 0:
        raise ValueError("combo.data.registry_cache_days must be positive")
    strategy = config["strategy"]
    if int(strategy["start_ds"]) > int(strategy["end_ds"]):
        raise ValueError("strategy.start_ds must not be after end_ds")
    optimizer = strategy["optimizer"]
    if len(runtime["sample_times"]) > 1:
        assert optimizer["type"] == "opt2", "multiple sample times require opt2"
    price = config["backtest"]["execution_price"].split(":")
    assert len(price) == 2 and all(price), "execution_price must be source:column"
    if optimizer["type"] not in {"opt1", "opt2"}:
        raise ValueError("strategy.optimizer.type must be opt1 or opt2")
    for name in ("ret_days", "min_valid_instruments", "min_return_obs", "num_mosek_threads"):
        if int(optimizer[name]) <= 0:
            raise ValueError(f"strategy.optimizer.{name} must be positive")
    for name in (
        "lambda0",
        "ret_delay",
        "benchmark_delay",
        "maxtvr",
        "maxtrd",
        "maxpos",
        "liquidity_delay",
        "lambda_slp",
        "slippage_delay",
        "trim_threshold",
        "soft_univ_penalty",
        "soft_risk_penalty",
        "soft_group_penalty",
        "max_time",
    ):
        if float(optimizer[name]) < 0:
            raise ValueError(f"strategy.optimizer.{name} must be nonnegative")
    if not 0.0 <= float(optimizer["shrinkage"]) <= 1.0:
        raise ValueError("strategy.optimizer.shrinkage must be between 0 and 1")
    if float(optimizer["max_weight"]) <= 0:
        raise ValueError("strategy.optimizer.max_weight must be positive")
    if float(optimizer["target_size"]) <= 0:
        raise ValueError("strategy.optimizer.target_size must be positive")
    if float(optimizer["parti_penalty"]) < 0:
        raise ValueError("strategy.optimizer.parti_penalty must be nonnegative")
    participation = float(optimizer["min_participation_ratio"])
    if not 0.0 <= participation <= 1.0:
        raise ValueError("strategy.optimizer.min_participation_ratio must be between 0 and 1")
    if int(optimizer["min_return_obs"]) > int(optimizer["ret_days"]):
        raise ValueError("strategy.optimizer.min_return_obs must not exceed ret_days")
    if int(optimizer["ret_method"]) not in {1, 2}:
        raise ValueError("strategy.optimizer.ret_method must be 1 or 2")
    if not isinstance(optimizer["benchmark"], str) or not optimizer["benchmark"]:
        raise ValueError("strategy.optimizer.benchmark must be a non-empty string")
    for name in (
        "univ_list",
        "soft_univ_list",
        "risk_list",
        "soft_risk_list",
        "group_list",
        "soft_group_list",
    ):
        if not isinstance(optimizer[name], str):
            raise ValueError(f"strategy.optimizer.{name} must be a string")

    backtest = config["backtest"]
    if float(backtest["fee_rate"]) < 0:
        raise ValueError("backtest.fee_rate must be nonnegative")
    if not 0.0 < float(backtest["reserve_cash"]) <= 1.0:
        raise ValueError("backtest.reserve_cash must be in (0, 1]")
    if float(backtest["drawdown_stop"]) < 0:
        raise ValueError("backtest.drawdown_stop must be nonnegative")
    if int(backtest["cooldown_days"]) < 0:
        raise ValueError("backtest.cooldown_days must be nonnegative")


def _load_xml_config(path: str) -> dict:
    root = ET.parse(path).getroot()
    if root.tag != "config":
        raise ValueError("xml config root tag must be <config>")

    constants = _parse_section_attributes(root.find("constants"), DEFAULT_CONFIG["constants"])
    strategy_element = root.find("strategy")
    strategy_defaults = {
        key: value for key, value in DEFAULT_CONFIG["strategy"].items() if key != "optimizer"
    }
    strategy = _parse_section_attributes(strategy_element, strategy_defaults)
    if strategy_element is not None:
        unsupported_children = [child.tag for child in strategy_element if child.tag != "optimizer"]
        if unsupported_children:
            raise ValueError(
                "unsupported <strategy> children: " + ", ".join(sorted(set(unsupported_children)))
            )
        optimizer_element = strategy_element.find("optimizer")
        if optimizer_element is not None:
            strategy["optimizer"] = _parse_section_attributes(
                optimizer_element,
                DEFAULT_OPTIMIZER_CONFIG,
            )

    combo_element = root.find("combo")
    combo = {}
    if combo_element is not None:
        if combo_element.find("data") is not None:
            raise ValueError("declare sources in ResearchLoader.data_requirements(); use <loader> for reading parameters")
        combo = {
            "paths": _parse_section_attributes(combo_element.find("paths"), DEFAULT_CONFIG["combo"]["paths"]),
            "loader": _parse_section_attributes(combo_element.find("loader"), DEFAULT_CONFIG["combo"]["loader"]),
            "output": _parse_section_attributes(combo_element.find("output"), DEFAULT_CONFIG["combo"]["output"]),
            "runtime": _parse_section_attributes(combo_element.find("runtime"), DEFAULT_CONFIG["combo"]["runtime"]),
            "model": _parse_section_attributes(combo_element.find("model"), DEFAULT_CONFIG["combo"]["model"], allow_extra=True),
        }
        if combo_element.find("defaults") is not None:
            raise ValueError("<combo><defaults> is no longer supported")

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
