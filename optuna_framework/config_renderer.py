"""Render isolated XML configs from the eg-torch baseline config."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from optuna_framework.adapters.base import ModelAdapter
from optuna_framework.specs import RunPaths


PATH_PATCH_KEYS = {
    "config.constants.@cache_path",
    "config.constants.@factor_root",
    "config.strategy.@start_ds",
    "config.strategy.@end_ds",
    "config.constants.@output_root",
    "config.constants.@checkpoint_root",
    "config.combo.paths.@model_path",
    "config.combo.paths.@research_loader_path",
    "config.combo.paths.@research_dataset_path",
    "config.combo.runtime.@snaptime",
}

_RELATIVE_PATH_SPECS = (
    ("./constants", "cache_path"),
    ("./constants", "factor_root"),
    ("./combo/paths", "model_path"),
    ("./combo/paths", "research_loader_path"),
    ("./combo/paths", "research_dataset_path"),
)

OPTUNA_RUNTIME_PATCH_KEYS = PATH_PATCH_KEYS | {
    "config.combo.output.@enable_alpha_analysis",
}

MODEL_PATCH_KEYS = {
    "config.combo.model.@lr",
    "config.combo.model.@weight_decay",
    "config.combo.model.@dropout",
    "config.combo.model.@hiddenSize",
    "config.combo.model.@fcSize",
    "config.combo.model.@epochs",
    "config.combo.model.@scheduler_step_size",
    "config.combo.model.@scheduler_gamma",
}


@dataclass(frozen=True)
class XmlDiff:
    """A structured XML field difference."""

    key: str
    before: Any
    after: Any


def render_config(
    baseline_config_path: str | Path,
    run_paths: RunPaths,
    adapter: ModelAdapter,
    params: dict[str, Any],
    fixed_overrides: dict[str, Any] | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Render ``config.xml`` plus metadata for one isolated run."""

    baseline_config_path = Path(baseline_config_path).expanduser().resolve()
    tree = ET.parse(baseline_config_path)
    root = tree.getroot()

    materialized = adapter.apply_params_to_xml(root, params)
    _set_attr(root, "./strategy", "start_ds", run_paths.segment.start_ds)
    _set_attr(root, "./strategy", "end_ds", run_paths.segment.end_ds)
    _set_attr(root, "./constants", "output_root", str(run_paths.output_root))
    _set_attr(root, "./constants", "checkpoint_root", str(run_paths.checkpoint_root))
    _set_attr(root, "./combo/runtime", "snaptime", run_paths.snaptime)

    for dotted_path, value in (fixed_overrides or {}).items():
        apply_fixed_override(root, dotted_path, value)

    _resolve_relative_paths(root, baseline_config_path.parent)

    run_paths.segment_dir.mkdir(parents=True, exist_ok=True)
    run_paths.output_root.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoint_root.mkdir(parents=True, exist_ok=True)

    _indent_xml(root)
    tree.write(run_paths.config_path, encoding="utf-8", xml_declaration=False)
    _write_json(run_paths.params_path, materialized)
    resolved_meta = {
        "trial_number": run_paths.trial_number,
        "segment": run_paths.segment.name,
        "start_ds": run_paths.segment.start_ds,
        "end_ds": run_paths.segment.end_ds,
        "output_root": str(run_paths.output_root),
        "checkpoint_root": str(run_paths.checkpoint_root),
        "snaptime": run_paths.snaptime,
        "seed": _read_xml_attr(root, "./combo/model", "seed"),
        "git_commit": git_commit if git_commit is not None else get_git_commit(baseline_config_path.parent),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "fixed_overrides": dict(fixed_overrides or {}),
        "kind": run_paths.kind,
    }
    _write_json(run_paths.resolved_meta_path, resolved_meta)
    return materialized


def _resolve_relative_paths(root: ET.Element, base_dir: Path) -> None:
    """Convert relative file paths to absolute paths anchored to the baseline config."""

    for xpath, attr in _RELATIVE_PATH_SPECS:
        element = root.find(xpath)
        if element is None:
            continue
        value = element.get(attr)
        if value and not Path(value).expanduser().is_absolute():
            element.set(attr, str((base_dir / value).resolve()))
    for item in root.findall("./combo/data/item"):
        module = (item.get("module") or "").strip().lower()
        for attr in ("path", "config_path"):
            value = item.get(attr)
            if not value or Path(value).expanduser().is_absolute():
                continue
            if attr == "path" and module in {"factor", "builtin.factor", "label", "builtin.label", "barra_style", "builtin.barra_style"}:
                continue
            item.set(attr, str((base_dir / value).resolve()))


def apply_fixed_override(root: ET.Element, dotted_path: str, value: Any) -> None:
    """Apply a dotted override like ``combo.model.device`` to an XML attribute."""

    parts = dotted_path.split(".")
    if len(parts) < 2:
        raise ValueError(f"invalid fixed override path: {dotted_path}")
    element_path = "./" + "/".join(parts[:-1])
    attr_name = parts[-1]
    element = root.find(element_path)
    if element is None and element_path == "./combo/output":
        combo = root.find("./combo")
        if combo is None:
            raise ValueError("XML is missing element: ./combo")
        element = ET.SubElement(combo, "output")
    if element is None:
        raise ValueError(f"XML is missing element: {element_path}")
    element.set(attr_name, _format_xml_value(value))


def structured_xml_diff(path_a: str | Path, path_b: str | Path) -> list[XmlDiff]:
    """Return semantic field-level differences between two XML files."""

    root_a = ET.parse(path_a).getroot()
    root_b = ET.parse(path_b).getroot()
    fields_a = flatten_xml(root_a)
    fields_b = flatten_xml(root_b)
    diffs: list[XmlDiff] = []
    for key in sorted(set(fields_a) | set(fields_b)):
        before = fields_a.get(key)
        after = fields_b.get(key)
        if not _semantic_equal(before, after):
            diffs.append(XmlDiff(key=key, before=before, after=after))
    return diffs


def assert_only_allowed_diffs(diffs: list[XmlDiff], allowed_keys: set[str]) -> None:
    """Raise if any diff is outside ``allowed_keys``."""

    unexpected = [diff for diff in diffs if diff.key not in allowed_keys]
    if unexpected:
        rendered = ", ".join(f"{diff.key}: {diff.before!r} -> {diff.after!r}" for diff in unexpected)
        raise AssertionError(f"unexpected XML differences: {rendered}")


def flatten_xml(root: ET.Element) -> dict[str, Any]:
    """Flatten XML attributes and non-empty text nodes into comparable fields."""

    fields: dict[str, Any] = {}

    def visit(element: ET.Element, path: str) -> None:
        for attr, raw_value in element.attrib.items():
            fields[f"{path}.@{attr}"] = _coerce_scalar(raw_value)
        text = (element.text or "").strip()
        if text:
            fields[f"{path}.#text"] = text
        totals: dict[str, int] = {}
        for child in list(element):
            totals[child.tag] = totals.get(child.tag, 0) + 1
        seen: dict[str, int] = {}
        for child in list(element):
            seen[child.tag] = seen.get(child.tag, 0) + 1
            child_name = child.tag if totals[child.tag] == 1 else f"{child.tag}[{seen[child.tag]}]"
            visit(child, f"{path}.{child_name}")

    visit(root, root.tag)
    return fields


def get_git_commit(cwd: str | Path) -> str | None:
    """Return the current git commit if available."""

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(cwd).resolve()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _set_attr(root: ET.Element, element_path: str, attr: str, value: Any) -> None:
    element = root.find(element_path)
    if element is None:
        raise ValueError(f"XML is missing element: {element_path}")
    element.set(attr, _format_xml_value(value))


def _read_xml_attr(root: ET.Element, element_path: str, attr: str) -> str | None:
    element = root.find(element_path)
    if element is None:
        return None
    return element.get(attr)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_xml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _coerce_scalar(raw_value: str) -> Any:
    stripped = raw_value.strip()
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return left == right


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent
