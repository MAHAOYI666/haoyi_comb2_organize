"""Validation and confirmation-plan generation for detailed Optuna configs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from optuna_framework.study_config import ParamSpec, StudyConfig, VALID_PARAM_TYPES, VALID_SECTIONS, parse_scalar
from optuna_framework.xml_patcher import find_section, get_section_attr, read_model_path, xml_attr_path


@dataclass(frozen=True)
class ValidationPlan:
    """Resolved config validation state."""

    config: StudyConfig
    status: str
    plan_hash: str
    recognized: tuple[dict[str, Any], ...]
    recognized_virtual: tuple[dict[str, Any], ...]
    derived: tuple[dict[str, Any], ...]
    unrecognized: tuple[dict[str, Any], ...]
    invalid: tuple[dict[str, Any], ...]
    baseline_model_path: str | None

    @property
    def is_ok(self) -> bool:
        return self.status == "OK"


def config_plan_path(config: StudyConfig) -> Path:
    """Return the default confirmation plan path for a study."""

    return config.study_root / "config_plan.txt"


def build_validation_plan(config: StudyConfig) -> ValidationPlan:
    """Validate configured params against the baseline XML."""

    baseline_root = ET.parse(config.baseline_config_path).getroot()
    baseline_model = read_model_path(baseline_root, config.baseline_config_path.parent)
    recognized: list[dict[str, Any]] = []
    recognized_virtual: list[dict[str, Any]] = []
    unrecognized: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    derived_users = {item.param for item in config.derived_params}
    for spec in config.params:
        errors = _param_errors(spec)
        if spec.is_virtual:
            if spec.baseline is None:
                errors.append("virtual param must define baseline")
            if spec.name not in derived_users and spec.baseline is None:
                errors.append("virtual param is not used by a derived param")
            if errors:
                invalid.append(_invalid_param(spec, "; ".join(errors)))
            else:
                recognized_virtual.append(_virtual_row(spec))
            continue

        if spec.section not in VALID_SECTIONS:
            errors.append("section must be model or runtime")
        if errors:
            invalid.append(_invalid_param(spec, "; ".join(errors)))
            continue

        element = find_section(baseline_root, spec.section)
        if element is None or spec.name not in element.attrib:
            unrecognized.append(_invalid_param(spec, f"baseline XML missing {xml_attr_path(spec.section, spec.name)}"))
            continue
        baseline_value = element.get(spec.name)
        range_error = _baseline_range_error(spec, baseline_value)
        if range_error:
            invalid.append(_invalid_param(spec, range_error))
            continue
        recognized.append(_recognized_row(spec, baseline_value))

    derived_rows = []
    for derived in config.derived_params:
        derived_errors = []
        if derived.kind != "int_product_xml_attr":
            derived_errors.append(f"unsupported derived kind: {derived.kind}")
        if derived.section not in VALID_SECTIONS:
            derived_errors.append("derived target section must be model or runtime")
        if derived.source_section not in VALID_SECTIONS:
            derived_errors.append("derived source section must be model or runtime")
        if derived.param not in {spec.name for spec in config.params}:
            derived_errors.append(f"derived source param not declared: {derived.param}")
        if derived.source_section in VALID_SECTIONS and get_section_attr(baseline_root, derived.source_section, derived.source_name) is None:
            derived_errors.append(f"baseline XML missing source {xml_attr_path(derived.source_section, derived.source_name)}")
        if derived.section in VALID_SECTIONS and get_section_attr(baseline_root, derived.section, derived.target) is None:
            derived_errors.append(f"baseline XML missing target {xml_attr_path(derived.section, derived.target)}")
        row = {
            "name": derived.name,
            "source": xml_attr_path(derived.source_section, derived.source_name),
            "target": xml_attr_path(derived.section, derived.target),
            "formula": f"max({derived.min_value}, int({derived.source_name} * {derived.param}))",
            "kind": derived.kind,
        }
        derived_rows.append(row)
        if derived_errors:
            invalid.append({"name": derived.name, "section": derived.section, "reason": "; ".join(derived_errors)})

    status = "BLOCKED" if unrecognized or invalid else "OK"
    payload = {
        "config_text": config.config_path.read_text(encoding="utf-8"),
        "baseline_text": config.baseline_config_path.read_text(encoding="utf-8"),
        "recognized": recognized,
        "recognized_virtual": recognized_virtual,
        "derived": derived_rows,
        "unrecognized": unrecognized,
        "invalid": invalid,
        "windows": {
            "tuning_run": config.tuning_run_window,
            "full_run": config.full_run_window,
            "scoring": config.scoring_window,
        },
        "seeds": config.phase_b.seeds,
        "fixed_overrides": config.fixed_overrides,
    }
    plan_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return ValidationPlan(
        config=config,
        status=status,
        plan_hash=plan_hash,
        recognized=tuple(recognized),
        recognized_virtual=tuple(recognized_virtual),
        derived=tuple(derived_rows),
        unrecognized=tuple(unrecognized),
        invalid=tuple(invalid),
        baseline_model_path=str(baseline_model) if baseline_model is not None else None,
    )


def write_config_plan(plan: ValidationPlan, path: str | Path | None = None) -> Path:
    """Write a human-readable confirmation plan."""

    target = Path(path) if path is not None else config_plan_path(plan.config)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# optuna_framework config plan",
        "",
        f"status: {plan.status}",
        f"plan_hash: {plan.plan_hash}",
        f"config_file: {plan.config.config_path}",
        f"baseline_config: {plan.config.baseline_config_path}",
        f"baseline_model_path: {plan.baseline_model_path}",
        f"study_name: {plan.config.study_name}",
        f"optuna_name: {plan.config.optuna_name}",
        f"study_root: {plan.config.study_root}",
        f"tuning_run_window: {plan.config.tuning_run_window[0]}-{plan.config.tuning_run_window[1]}",
        f"full_run_window: {plan.config.full_run_window[0]}-{plan.config.full_run_window[1]}",
        f"scoring_window: {plan.config.scoring_window[0]}-{plan.config.scoring_window[1]}",
        f"phase_b_seeds: {','.join(str(seed) for seed in plan.config.phase_b.seeds)}",
        "",
        "## fixed_overrides",
        *_format_dict_rows([{"path": key, "value": value} for key, value in plan.config.fixed_overrides.items()]),
        "",
        "## recognized tunable params",
        *_format_dict_rows(plan.recognized),
        "",
        "## recognized virtual params",
        *_format_dict_rows(plan.recognized_virtual),
        "",
        "## derived params",
        *_format_dict_rows(plan.derived),
        "",
        "## unrecognized params",
        *_format_dict_rows(plan.unrecognized),
        "",
        "## invalid params",
        *_format_dict_rows(plan.invalid),
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def ensure_config_plan(config: StudyConfig, write: bool = True) -> ValidationPlan:
    """Build and optionally write the current confirmation plan."""

    plan = build_validation_plan(config)
    if write:
        write_config_plan(plan)
    return plan


def assert_plan_ok(config: StudyConfig, skip_plan_check: bool = False) -> ValidationPlan:
    """Require an OK, current confirmation plan before a real run."""

    current = build_validation_plan(config)
    if skip_plan_check:
        if not current.is_ok:
            raise RuntimeError(f"config plan is BLOCKED; invalid={len(current.invalid)} unrecognized={len(current.unrecognized)}")
        return current

    path = config_plan_path(config)
    if not path.exists():
        raise FileNotFoundError(f"config plan is missing; run validate_config.py first: {path}")
    text = path.read_text(encoding="utf-8")
    saved_hash = _read_plan_field(text, "plan_hash")
    saved_status = _read_plan_field(text, "status")
    if saved_status != "OK":
        raise RuntimeError(f"config plan is not OK: status={saved_status} path={path}")
    if current.status != "OK":
        raise RuntimeError(f"current config plan is BLOCKED; rerun validate_config.py and review {path}")
    if saved_hash != current.plan_hash:
        raise RuntimeError(f"config plan hash mismatch; rerun validate_config.py and review {path}")
    return current


def _param_errors(spec: ParamSpec) -> list[str]:
    errors: list[str] = []
    if spec.param_type not in VALID_PARAM_TYPES:
        return [f"unsupported param type: {spec.param_type}"]
    if spec.param_type in {"float", "int"}:
        if spec.low is None or spec.high is None:
            errors.append("low/high are required")
        else:
            try:
                low = float(spec.low)
                high = float(spec.high)
            except ValueError:
                errors.append("low/high must be numeric")
            else:
                if low >= high:
                    errors.append("low must be < high")
                if spec.log and low <= 0:
                    errors.append("log search requires low > 0")
        if spec.param_type == "int" and spec.step is not None and int(float(spec.step)) <= 0:
            errors.append("int step must be positive")
    if spec.param_type == "categorical" and not spec.choices:
        errors.append("categorical choices are required")
    return errors


def _baseline_range_error(spec: ParamSpec, baseline_raw: str | None) -> str | None:
    if baseline_raw is None:
        return "baseline value missing"
    try:
        baseline = spec.baseline if spec.baseline is not None else baseline_raw
        if spec.param_type == "float":
            float(baseline)
        elif spec.param_type == "int":
            int(float(baseline))
        elif spec.param_type == "categorical":
            typed_choices = [parse_scalar(choice) for choice in spec.choices]
            parsed = parse_scalar(str(baseline))
            if parsed not in typed_choices:
                return f"baseline value {baseline!r} is not in choices"
    except ValueError as exc:
        return f"baseline value cannot be coerced: {exc}"
    return None


def _recognized_row(spec: ParamSpec, baseline_value: str | None) -> dict[str, Any]:
    inferred = parse_scalar(baseline_value or "")
    return {
        "section": spec.section,
        "name": spec.name,
        "xml_path": xml_attr_path(spec.section, spec.name),
        "baseline_value": baseline_value,
        "inferred_type": type(inferred).__name__,
        "optuna_type": spec.param_type,
        "range_or_choices": _range_display(spec),
    }


def _virtual_row(spec: ParamSpec) -> dict[str, Any]:
    return {
        "section": "virtual",
        "name": spec.name,
        "xml_path": "none",
        "baseline_value": spec.baseline,
        "inferred_type": type(parse_scalar(spec.baseline or "")).__name__,
        "optuna_type": spec.param_type,
        "range_or_choices": _range_display(spec),
    }


def _invalid_param(spec: ParamSpec, reason: str) -> dict[str, Any]:
    return {
        "section": spec.section,
        "name": spec.name,
        "type": spec.param_type,
        "target": spec.target,
        "reason": reason,
    }


def _range_display(spec: ParamSpec) -> str:
    if spec.param_type == "categorical":
        return ",".join(spec.choices)
    pieces = [f"low={spec.low}", f"high={spec.high}"]
    if spec.step is not None:
        pieces.append(f"step={spec.step}")
    if spec.log:
        pieces.append("log=true")
    return " ".join(pieces)


def _format_dict_rows(rows: Any) -> list[str]:
    rows = list(rows)
    if not rows:
        return ["(none)"]
    columns = sorted({key for row in rows for key in row})
    result = ["\t".join(columns)]
    for row in rows:
        result.append("\t".join(str(row.get(column, "")) for column in columns))
    return result


def _read_plan_field(text: str, field: str) -> str | None:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None
