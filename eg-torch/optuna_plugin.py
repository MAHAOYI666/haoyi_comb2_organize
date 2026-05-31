"""Model-owned Optuna plugin for the eg-torch example."""

from __future__ import annotations

import textwrap
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

    def write_seeded_model(self, segment_dir: Path, baseline_config_path: Path, phase: str) -> Path | None:
        """Write a torch DataLoader seeding wrapper next to one Phase B/C run."""

        tree = ET.parse(baseline_config_path)
        root = tree.getroot()
        paths = root.find("./combo/paths")
        if paths is None or not paths.get("model_path"):
            raise ValueError("baseline XML is missing <combo><paths model_path=...>")
        model_path = Path(paths.get("model_path", ""))
        if not model_path.expanduser().is_absolute():
            model_path = baseline_config_path.parent / model_path
        model_path = model_path.expanduser().resolve()
        target = Path(segment_dir) / "seeded_model.py"
        target.write_text(
            SEEDED_MODEL_TEMPLATE.replace("__ORIGINAL_MODEL_PATH__", repr(str(model_path))),
            encoding="utf-8",
        )
        return target

    def seeded_overrides(self, seed: int, seeded_model_path: Path | None = None) -> dict[str, Any]:
        """Override seed and model path for a deterministic Phase B/C run."""

        overrides: dict[str, Any] = {"combo.model.seed": int(seed)}
        if seeded_model_path is not None:
            overrides["combo.paths.model_path"] = str(seeded_model_path)
        return overrides


def create_adapter() -> ModelAdapter:
    """Return a fresh eg-torch adapter instance."""

    return EgTorchV1Adapter()


ADAPTER = EgTorchV1Adapter()

STUDY_SPEC = StudySpec(
    name="study_eg_torch_v1",
    baseline_config_path=get_repo_root() / "eg-torch" / "config.xml",
    adapter_name=EgTorchV1Adapter.name,
    tuning_segments=TUNING_SEGMENTS,
    holdout_segments=HOLDOUT_SEGMENTS,
    baseline_only_segments=BASELINE_ONLY_SEGMENTS,
    n_trials=60,
    fixed_overrides={"combo.output.enable_alpha_analysis": False},
    plugin_path=Path(__file__).resolve(),
    phase_b_seeds=(42, 43, 44),
)


SEEDED_MODEL_TEMPLATE = textwrap.dedent(
    """
    from __future__ import annotations

    import importlib.util
    import random
    from pathlib import Path

    import numpy as np
    import torch


    _ORIGINAL_MODEL_PATH = Path(__ORIGINAL_MODEL_PATH__)
    _SPEC = importlib.util.spec_from_file_location("_seeded_base_model", _ORIGINAL_MODEL_PATH)
    _BASE_MODULE = importlib.util.module_from_spec(_SPEC)
    assert _SPEC.loader is not None
    _SPEC.loader.exec_module(_BASE_MODULE)


    def _apply_seed(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


    class ResearchModel(_BASE_MODULE.ResearchModel):
        def __init__(self, config):
            self.seed = int(config.get("seed", 42))
            _apply_seed(self.seed)
            super().__init__(config)

        def fit(self, dataset):
            _apply_seed(self.seed)
            original_dataloader = getattr(_BASE_MODULE, "DataLoader", None)
            if original_dataloader is None:
                return super().fit(dataset)

            def seeded_dataloader(*args, **kwargs):
                if kwargs.get("generator") is None:
                    generator = torch.Generator()
                    generator.manual_seed(self.seed)
                    kwargs["generator"] = generator
                return original_dataloader(*args, **kwargs)

            _BASE_MODULE.DataLoader = seeded_dataloader
            try:
                return super().fit(dataset)
            finally:
                _BASE_MODULE.DataLoader = original_dataloader
    """
).lstrip()


def _format_xml_value(value: Any) -> str:
    """Format Python values for XML attributes."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)
