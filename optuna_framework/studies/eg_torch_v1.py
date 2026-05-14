"""Study specification for the eg-torch Optuna search."""

from __future__ import annotations

from optuna_framework.paths import get_repo_root
from optuna_framework.specs import SegmentSpec, StudySpec


TUNING_SEGMENTS = (
    SegmentSpec(
        "tuning_2020_2023",
        "tuning_period",
        20200102,
        20231229,
        score_start_ds=20210104,
        score_end_ds=20231229,
    ),
)

HOLDOUT_SEGMENTS = (
    SegmentSpec("holdout_2020", "holdout", 20200102, 20201231),
    SegmentSpec("holdout_2024h1", "holdout", 20240102, 20240628),
)

BASELINE_ONLY_SEGMENTS = (
    SegmentSpec("full_period", "baseline_only", 20200102, 20240628),
)

STUDY_SPEC = StudySpec(
    name="study_eg_torch_v1",
    baseline_config_path=get_repo_root() / "eg-torch" / "config.xml",
    adapter_name="eg_torch_v1",
    tuning_segments=TUNING_SEGMENTS,
    holdout_segments=HOLDOUT_SEGMENTS,
    baseline_only_segments=BASELINE_ONLY_SEGMENTS,
    n_trials=60,
    fixed_overrides={"combo.output.enable_alpha_analysis": False},
)

