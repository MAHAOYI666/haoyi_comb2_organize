"""Study constants for the eg-torch Optuna search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optuna_framework.paths import get_repo_root


STUDY_NAME = "study_eg_torch_v1"
ADAPTER_NAME = "eg_torch_v1"
BASELINE_CONFIG_PATH = get_repo_root() / "eg-torch" / "config.xml"

TUNING_RUN_WINDOW = (20200102, 20231229)
FULL_RUN_WINDOW = (20200102, 20240628)
SCORING_WINDOW = (20210104, 20231229)

N_TRIALS_DEFAULT = 60
FIXED_OVERRIDES = {"combo.output.enable_alpha_analysis": False}


@dataclass(frozen=True)
class StudyDescription:
    """Small descriptive record for scripts and reports."""

    name: str
    adapter_name: str
    baseline_config_path: Path
    tuning_run_window: tuple[int, int]
    full_run_window: tuple[int, int]
    scoring_window: tuple[int, int]
    n_trials: int
    fixed_overrides: dict[str, Any]


STUDY_DESCRIPTION = StudyDescription(
    name=STUDY_NAME,
    adapter_name=ADAPTER_NAME,
    baseline_config_path=BASELINE_CONFIG_PATH,
    tuning_run_window=TUNING_RUN_WINDOW,
    full_run_window=FULL_RUN_WINDOW,
    scoring_window=SCORING_WINDOW,
    n_trials=N_TRIALS_DEFAULT,
    fixed_overrides=FIXED_OVERRIDES,
)
