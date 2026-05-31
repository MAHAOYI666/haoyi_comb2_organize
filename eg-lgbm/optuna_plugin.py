"""Model-owned Optuna plugin for the eg-lgbm example."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from optuna_framework.adapters.base import ModelAdapter
from optuna_framework.paths import get_repo_root
from optuna_framework.specs import SegmentSpec, StudySpec


TUNING_SEGMENTS = (
    SegmentSpec("seg01", "tuning", 20210104, 20211231),
    SegmentSpec("seg02", "tuning", 20220104, 20221230),
    SegmentSpec("seg03", "tuning", 20230103, 20231229),
)

HOLDOUT_SEGMENTS = (
    SegmentSpec("holdout_2020", "holdout", 20200102, 20201231),
    SegmentSpec("holdout_2024h1", "holdout", 20240102, 20240628),
)

BASELINE_ONLY_SEGMENTS = (
    SegmentSpec("full_period", "baseline_only", 20200102, 20240628),
)


class EgLgbmV1Adapter(ModelAdapter):
    """Adapter for the LightGBM research example model."""

    name = "eg_lgbm_v1"

    num_leaves_choices = (15, 31, 63, 127)
    bagging_freq_choices = (1, 3, 5)
    min_data_in_leaf_choices = (20, 50, 100, 200)

    def baseline_params(self) -> dict[str, Any]:
        """Return baseline-equivalent parameters in search-space coordinates."""

        return {
            "lr": 0.05,
            "epochs": 100,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 100,
        }

    def suggest_params(self, trial: Any) -> dict[str, Any]:
        """Suggest parameters from an Optuna-like trial object."""

        return {
            "lr": trial.suggest_float("lr", 1e-3, 2e-1, log=True),
            "epochs": trial.suggest_int("epochs", 50, 300),
            "num_leaves": trial.suggest_categorical("num_leaves", list(self.num_leaves_choices)),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_categorical("bagging_freq", list(self.bagging_freq_choices)),
            "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", list(self.min_data_in_leaf_choices)),
        }

    def materialize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Normalize LightGBM parameters without adding derived fields."""

        baseline = self.baseline_params()
        materialized: dict[str, Any] = {}
        for key, default in baseline.items():
            value = params.get(key, default)
            if isinstance(default, int) and not isinstance(default, bool):
                materialized[key] = int(value)
            elif isinstance(default, float):
                materialized[key] = float(value)
            else:
                materialized[key] = value
        return materialized

    def apply_params_to_xml(self, root: ET.Element, params: dict[str, Any]) -> dict[str, Any]:
        """Patch LightGBM hyperparameters into ``<combo><model>``."""

        materialized = self.materialize_params(params)
        model = root.find("./combo/model")
        if model is None:
            raise ValueError("baseline XML is missing <combo><model>")
        for key in (
            "lr",
            "epochs",
            "num_leaves",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "min_data_in_leaf",
        ):
            model.set(key, _format_xml_value(materialized[key]))
        return materialized


def create_adapter() -> ModelAdapter:
    """Return a fresh eg-lgbm adapter instance."""

    return EgLgbmV1Adapter()


ADAPTER = EgLgbmV1Adapter()

STUDY_SPEC = StudySpec(
    name="study_eg_lgbm_v1",
    baseline_config_path=get_repo_root() / "eg-lgbm" / "config.xml",
    adapter_name=EgLgbmV1Adapter.name,
    tuning_segments=TUNING_SEGMENTS,
    holdout_segments=HOLDOUT_SEGMENTS,
    baseline_only_segments=BASELINE_ONLY_SEGMENTS,
    n_trials=60,
    fixed_overrides={"combo.output.enable_alpha_analysis": False},
    plugin_path=Path(__file__).resolve(),
    phase_b_seeds=(42, 43, 44),
)


def _format_xml_value(value: Any) -> str:
    """Format Python values for XML attributes."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)
