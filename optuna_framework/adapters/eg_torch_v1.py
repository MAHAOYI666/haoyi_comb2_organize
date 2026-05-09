"""Search-space adapter for ``eg-torch/model.py``."""

from __future__ import annotations

from typing import Any
from xml.etree import ElementTree as ET

from optuna_framework.adapters.base import ModelAdapter


class EgTorchV1Adapter(ModelAdapter):
    """Adapter for the attention + MLP torch example model."""

    name = "eg_torch_v1"

    hidden_sizes = (256, 384, 512, 768)
    fc_sizes = (128, 256, 512)
    scheduler_step_ratios = (0.3, 0.5, 0.7, 1.0)

    def baseline_params(self) -> dict[str, Any]:
        """Return baseline-equivalent parameters in search-space coordinates."""

        return {
            "lr": 2e-6,
            "weight_decay": 1e-6,
            "dropout": 0.5,
            "hiddenSize": 512,
            "fcSize": 256,
            "epochs": 15,
            "scheduler_step_ratio": 0.7,
            "scheduler_gamma": 0.5,
        }

    def suggest_params(self, trial: Any) -> dict[str, Any]:
        """Suggest parameters from an Optuna-like trial object."""

        return {
            "lr": trial.suggest_float("lr", 1e-7, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "hiddenSize": trial.suggest_categorical("hiddenSize", list(self.hidden_sizes)),
            "fcSize": trial.suggest_categorical("fcSize", list(self.fc_sizes)),
            "epochs": trial.suggest_int("epochs", 5, 25),
            "scheduler_step_ratio": trial.suggest_categorical("scheduler_step_ratio", list(self.scheduler_step_ratios)),
            "scheduler_gamma": trial.suggest_float("scheduler_gamma", 0.3, 0.9),
        }

    def materialize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Add derived scheduler fields to the search-space parameters."""

        materialized = dict(params)
        epochs = int(materialized["epochs"])
        ratio = float(materialized["scheduler_step_ratio"])
        materialized["scheduler_step_size"] = max(1, int(epochs * ratio))
        return materialized

    def apply_params_to_xml(self, root: ET.Element, params: dict[str, Any]) -> dict[str, Any]:
        """Patch model hyperparameters into ``<combo><model>``."""

        materialized = self.materialize_params(params)
        model = root.find("./combo/model")
        if model is None:
            raise ValueError("baseline XML is missing <combo><model>")
        for key in (
            "lr",
            "weight_decay",
            "dropout",
            "hiddenSize",
            "fcSize",
            "epochs",
            "scheduler_step_size",
            "scheduler_gamma",
        ):
            model.set(key, _format_xml_value(materialized[key]))
        return materialized


def _format_xml_value(value: Any) -> str:
    """Format Python values for XML attributes."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)

