"""XML-backed study configuration for the detailed Optuna runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from optuna_detailed.paths import get_repo_root, resolve_study_root


DEFAULT_CONFIG_PATH = get_repo_root() / "optuna_detailed" / "config.xml"
VALID_PARAM_TYPES = {"float", "int", "categorical"}
VALID_SECTIONS = {"model", "runtime"}


@dataclass(frozen=True)
class ParamSpec:
    """One user-declared tunable or virtual search-space parameter."""

    name: str
    param_type: str
    section: str | None = None
    low: str | None = None
    high: str | None = None
    step: str | None = None
    log: bool = False
    choices: tuple[str, ...] = ()
    baseline: str | None = None
    target: str | None = None

    @property
    def is_virtual(self) -> bool:
        return (self.target or "").strip().lower() == "none"


@dataclass(frozen=True)
class DerivedParamSpec:
    """One derived XML parameter rule."""

    name: str
    section: str
    target: str
    kind: str
    param: str
    source_section: str
    source_name: str
    min_value: int = 1


@dataclass(frozen=True)
class PhaseBConfig:
    seeds: tuple[int, ...]
    candidate_scan_top_n: int = 6
    candidate_limit: int = 3
    sharpe_margin: float = 0.15
    dd_multiplier: float = 1.15
    sharpe_std_max: float = 0.15


@dataclass(frozen=True)
class PhaseCConfig:
    holdout_years: tuple[str, ...]
    tuning_years: tuple[str, ...]
    holdout_sharpe_margin: float = 0.3
    full_dd_multiplier: float = 1.3
    min_tuning_years_better: int = 2
    full_sharpe_std_max: float = 0.25


@dataclass(frozen=True)
class StudyConfig:
    """Resolved configuration for one detailed Optuna workflow."""

    config_path: Path
    study_name: str
    optuna_name: str
    n_trials_default: int
    study_root: Path
    baseline_config_path: Path
    tuning_run_window: tuple[int, int]
    full_run_window: tuple[int, int]
    scoring_window: tuple[int, int]
    fixed_overrides: dict[str, Any]
    phase_b: PhaseBConfig
    phase_c: PhaseCConfig
    params: tuple[ParamSpec, ...]
    derived_params: tuple[DerivedParamSpec, ...]


def default_config_path() -> Path:
    """Return the default detailed Optuna XML config path."""

    return DEFAULT_CONFIG_PATH


def load_study_config(config_path: str | Path | None = None, study_root_override: str | Path | None = None) -> StudyConfig:
    """Load and resolve one ``optuna_detailed`` XML config."""

    resolved_config_path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser()
    if not resolved_config_path.is_absolute():
        resolved_config_path = get_repo_root() / resolved_config_path
    resolved_config_path = resolved_config_path.resolve()
    root = ET.parse(resolved_config_path).getroot()
    if root.tag != "optuna_study":
        raise ValueError("optuna detailed config root tag must be <optuna_study>")

    study_el = _required(root, "study")
    study_name = _required_attr(study_el, "name")
    optuna_name = study_el.get("optuna_name", study_name)
    n_trials_default = int(study_el.get("n_trials", "60"))
    study_root_raw = study_root_override or study_el.get("study_root")
    study_root = resolve_study_root(study_root_raw, study_name)

    baseline_el = _required(root, "baseline")
    baseline_config_path = _resolve_repo_path(_required_attr(baseline_el, "config_path"))

    windows = _required(root, "windows")
    tuning_run_window = (
        int(_required_attr(windows, "tuning_run_start_ds")),
        int(_required_attr(windows, "tuning_run_end_ds")),
    )
    full_run_window = (
        int(_required_attr(windows, "full_run_start_ds")),
        int(_required_attr(windows, "full_run_end_ds")),
    )
    scoring_window = (
        int(_required_attr(windows, "scoring_start_ds")),
        int(_required_attr(windows, "scoring_end_ds")),
    )

    fixed_overrides = _parse_fixed_overrides(root.find("fixed_overrides"))
    phase_b = _parse_phase_b(root.find("phase_b"))
    phase_c = _parse_phase_c(root.find("phase_c"))
    params = tuple(_parse_param(el) for el in _required(root, "search_space").findall("param"))
    derived_params = tuple(_parse_derived(el) for el in (root.find("derived_params") or ET.Element("derived_params")).findall("derived"))

    return StudyConfig(
        config_path=resolved_config_path,
        study_name=study_name,
        optuna_name=optuna_name,
        n_trials_default=n_trials_default,
        study_root=study_root,
        baseline_config_path=baseline_config_path,
        tuning_run_window=tuning_run_window,
        full_run_window=full_run_window,
        scoring_window=scoring_window,
        fixed_overrides=fixed_overrides,
        phase_b=phase_b,
        phase_c=phase_c,
        params=params,
        derived_params=derived_params,
    )


def parse_scalar(value: str) -> Any:
    """Parse config scalar strings into bool/int/float/string values."""

    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"true", "false", "yes", "no", "y", "n", "on", "off"}:
        return lowered in {"true", "yes", "y", "on"}
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def parse_csv(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated XML attribute."""

    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_param(element: ET.Element) -> ParamSpec:
    return ParamSpec(
        name=_required_attr(element, "name"),
        section=element.get("section"),
        param_type=_required_attr(element, "type").strip().lower(),
        low=element.get("low"),
        high=element.get("high"),
        step=element.get("step"),
        log=bool(parse_scalar(element.get("log", "false"))),
        choices=parse_csv(element.get("choices")),
        baseline=element.get("baseline"),
        target=element.get("target"),
    )


def _parse_derived(element: ET.Element) -> DerivedParamSpec:
    return DerivedParamSpec(
        name=_required_attr(element, "name"),
        section=_required_attr(element, "section"),
        target=_required_attr(element, "target"),
        kind=_required_attr(element, "kind"),
        param=_required_attr(element, "param"),
        source_section=_required_attr(element, "source_section"),
        source_name=_required_attr(element, "source_name"),
        min_value=int(element.get("min", "1")),
    )


def _parse_fixed_overrides(element: ET.Element | None) -> dict[str, Any]:
    if element is None:
        return {}
    overrides: dict[str, Any] = {}
    for override in element.findall("override"):
        overrides[_required_attr(override, "path")] = parse_scalar(_required_attr(override, "value"))
    return overrides


def _parse_phase_b(element: ET.Element | None) -> PhaseBConfig:
    if element is None:
        return PhaseBConfig(seeds=(42, 43, 44))
    return PhaseBConfig(
        seeds=tuple(int(item) for item in parse_csv(element.get("seeds", "42,43,44"))),
        candidate_scan_top_n=int(element.get("candidate_scan_top_n", "6")),
        candidate_limit=int(element.get("candidate_limit", "3")),
        sharpe_margin=float(element.get("sharpe_margin", "0.15")),
        dd_multiplier=float(element.get("dd_multiplier", "1.15")),
        sharpe_std_max=float(element.get("sharpe_std_max", "0.15")),
    )


def _parse_phase_c(element: ET.Element | None) -> PhaseCConfig:
    if element is None:
        return PhaseCConfig(holdout_years=("2020", "2024"), tuning_years=("2021", "2022", "2023"))
    return PhaseCConfig(
        holdout_years=parse_csv(element.get("holdout_years", "2020,2024")),
        tuning_years=parse_csv(element.get("tuning_years", "2021,2022,2023")),
        holdout_sharpe_margin=float(element.get("holdout_sharpe_margin", "0.3")),
        full_dd_multiplier=float(element.get("full_dd_multiplier", "1.3")),
        min_tuning_years_better=int(element.get("min_tuning_years_better", "2")),
        full_sharpe_std_max=float(element.get("full_sharpe_std_max", "0.25")),
    )


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = get_repo_root() / path
    return path.resolve()


def _required(root: ET.Element, tag: str) -> ET.Element:
    element = root.find(tag)
    if element is None:
        raise ValueError(f"optuna detailed config is missing <{tag}>")
    return element


def _required_attr(element: ET.Element, attr: str) -> str:
    value = element.get(attr)
    if value is None or value.strip() == "":
        raise ValueError(f"<{element.tag}> is missing required attribute: {attr}")
    return value
