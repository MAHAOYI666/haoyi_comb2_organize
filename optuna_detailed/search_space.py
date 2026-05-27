"""Config-driven Optuna search space and XML patching."""

from __future__ import annotations

import math
from typing import Any
from xml.etree import ElementTree as ET

from optuna_detailed.study_config import ParamSpec, StudyConfig, parse_scalar
from optuna_detailed.xml_patcher import get_section_attr, set_section_attr


class ConfigDrivenAdapter:
    """Adapter that derives search behavior from ``optuna_detailed/config.xml``."""

    name = "config_driven"

    def __init__(self, config: StudyConfig):
        self.config = config
        self._baseline_root = ET.parse(config.baseline_config_path).getroot()

    def baseline_params(self) -> dict[str, Any]:
        """Return baseline hyperparameters in configured search-space coordinates."""

        result: dict[str, Any] = {}
        for spec in self.config.params:
            result[spec.name] = self._baseline_value(spec)
        return result

    def suggest_params(self, trial: Any) -> dict[str, Any]:
        """Suggest one parameter set from an Optuna-like trial object."""

        suggested: dict[str, Any] = {}
        for spec in self.config.params:
            if spec.param_type == "float":
                suggested[spec.name] = trial.suggest_float(
                    spec.name,
                    float(spec.low),
                    float(spec.high),
                    log=bool(spec.log),
                )
            elif spec.param_type == "int":
                kwargs = {"log": bool(spec.log)}
                if spec.step is not None:
                    kwargs["step"] = int(spec.step)
                suggested[spec.name] = trial.suggest_int(
                    spec.name,
                    int(float(spec.low)),
                    int(float(spec.high)),
                    **kwargs,
                )
            elif spec.param_type == "categorical":
                suggested[spec.name] = trial.suggest_categorical(spec.name, self._typed_choices(spec))
            else:
                raise ValueError(f"unsupported param type for {spec.name}: {spec.param_type}")
        return suggested

    def materialize_params(self, params: dict[str, Any], xml_root: ET.Element | None = None) -> dict[str, Any]:
        """Return params plus derived fields that must be written to XML."""

        root = xml_root if xml_root is not None else self._baseline_root
        materialized = self.normalize_params(params)
        for derived in self.config.derived_params:
            if derived.kind != "int_product_xml_attr":
                raise ValueError(f"unsupported derived param kind: {derived.kind}")
            source_raw = get_section_attr(root, derived.source_section, derived.source_name)
            if source_raw is None:
                raise ValueError(f"derived source not found: combo.{derived.source_section}.{derived.source_name}")
            source_value = parse_scalar(source_raw)
            ratio = float(materialized[derived.param])
            derived_value = max(int(derived.min_value), int(float(source_value) * ratio))
            materialized[derived.name] = derived_value
            materialized.setdefault(derived.source_name, source_value)
        return materialized

    def apply_params_to_xml(self, root: ET.Element, params: dict[str, Any]) -> dict[str, Any]:
        """Patch configured model/runtime params into an XML tree."""

        materialized = self.materialize_params(params, root)
        for spec in self.config.params:
            if spec.is_virtual:
                continue
            if spec.section is None:
                raise ValueError(f"non-virtual param is missing section: {spec.name}")
            set_section_attr(root, spec.section, spec.name, materialized[spec.name])
        for derived in self.config.derived_params:
            set_section_attr(root, derived.section, derived.target, materialized[derived.name])
        return materialized

    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Coerce a partial or CSV-loaded params dict back to configured types."""

        baseline = self.baseline_params()
        normalized: dict[str, Any] = {}
        for spec in self.config.params:
            raw_value = params.get(spec.name, baseline[spec.name])
            if _is_missing(raw_value):
                raw_value = baseline[spec.name]
            normalized[spec.name] = self._coerce_param_value(spec, raw_value)
        return normalized

    def _baseline_value(self, spec: ParamSpec) -> Any:
        if spec.baseline is not None:
            return self._coerce_param_value(spec, spec.baseline)
        if spec.is_virtual:
            raise ValueError(f"virtual param needs explicit baseline: {spec.name}")
        if spec.section is None:
            raise ValueError(f"param needs section: {spec.name}")
        raw = get_section_attr(self._baseline_root, spec.section, spec.name)
        if raw is None:
            raise ValueError(f"baseline XML is missing combo.{spec.section}.{spec.name}")
        return self._coerce_param_value(spec, raw)

    def _coerce_param_value(self, spec: ParamSpec, value: Any) -> Any:
        if spec.param_type == "float":
            return float(value)
        if spec.param_type == "int":
            return int(float(value))
        if spec.param_type == "categorical":
            choices = self._typed_choices(spec)
            parsed = parse_scalar(str(value)) if isinstance(value, str) else value
            for choice in choices:
                if parsed == choice:
                    return choice
                if isinstance(choice, float):
                    try:
                        if abs(float(parsed) - choice) <= 1e-12:
                            return choice
                    except (TypeError, ValueError):
                        pass
            return parsed
        raise ValueError(f"unsupported param type for {spec.name}: {spec.param_type}")

    def _typed_choices(self, spec: ParamSpec) -> list[Any]:
        return [parse_scalar(choice) for choice in spec.choices]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False
