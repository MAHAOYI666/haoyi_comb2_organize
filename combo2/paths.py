"""Experiment identity and output paths shared by producers and consumers."""
from pathlib import Path
import re

def validate_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[^\W_][\w.-]{0,127}", value, re.UNICODE):
        raise ValueError("config Name must start with a letter or digit and contain only letters, digits, _, . or - (max 128 characters)")
    if value.endswith('.'):
        raise ValueError("config Name must not end with a dot")
    return value

def artifact_filename(config: dict, legacy: str, suffix: str) -> str:
    name = config.get("Name")
    return f"{name}{suffix}" if name else legacy

def alpha_path(config: dict) -> Path:
    return Path(config["constants"]["output_root"]) / artifact_filename(config, "alpha.parquet", ".parquet")

def apply_output_paths(config: dict) -> None:
    name = validate_name(config.get("Name"))
    root = Path(config["constants"]["output_root"])
    if name:
        root = root / name
    config["constants"]["output_root"] = str(root)
    config["combo"]["paths"].update(output_dir=str(root), checkpoint_root=str(root / "checkpoints"))
    config["combo"]["output"].update(
        alpha_history_path=str(root / artifact_filename(config, "alpha_history.pt", ".alpha_history.pt")),
        log_path=str(root / artifact_filename(config, "train.log", ".train.log")),
    )
    config["backtest"]["output_path"] = str(root / "backtest")
    if not config["monitor"].get("output_path"):
        config["monitor"]["output_path"] = str(root / artifact_filename(config, "perf_metrics.csv", ".perf_metrics.csv"))
