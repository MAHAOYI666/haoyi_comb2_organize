"""Render isolated XML configs from a detailed Optuna study config."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from optuna_framework.search_space import ConfigDrivenAdapter
from optuna_framework.specs import RunPaths
from optuna_framework.study_config import StudyConfig
from optuna_framework.xml_patcher import apply_fixed_override, coerce_scalar, format_xml_value, indent_xml, resolve_relative_paths


PATH_PATCH_KEYS = {
    "config.strategy.@start_ds",
    "config.strategy.@end_ds",
    "config.strategy.@path",
    "config.constants.@output_root",
    "config.combo.paths.@model_path",
    "config.combo.paths.@research_loader_path",
    "config.combo.paths.@research_dataset_path",
    "config.combo.paths.@combo_base_path",
    "config.constants.@cache_path",
    "config.combo.runtime.@snaptime",
}

OPTUNA_RUNTIME_PATCH_KEYS = PATH_PATCH_KEYS | {
    "config.combo.output.@enable_alpha_analysis",
}


@dataclass(frozen=True)
class XmlDiff:
    """A structured XML field difference."""

    key: str
    before: Any
    after: Any


def render_config(
    config: StudyConfig,
    run_paths: RunPaths,
    adapter: ConfigDrivenAdapter,
    params: dict[str, Any],
    extra_overrides: dict[str, Any] | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Render ``config.xml`` plus metadata for one isolated run."""

    tree = ET.parse(config.baseline_config_path)
    root = tree.getroot()

    materialized = adapter.apply_params_to_xml(root, params)
    _set_attr(root, "./strategy", "start_ds", run_paths.run_start_ds)
    _set_attr(root, "./strategy", "end_ds", run_paths.run_end_ds)
    _set_attr(root, "./constants", "output_root", str(run_paths.output_root))
    _set_attr(root, "./combo/runtime", "snaptime", run_paths.snaptime)

    fixed_overrides = {**config.fixed_overrides, **(extra_overrides or {})}
    for dotted_path, value in fixed_overrides.items():
        apply_fixed_override(root, dotted_path, value)

    resolve_relative_paths(root, config.baseline_config_path.parent)

    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.output_root.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoint_root.mkdir(parents=True, exist_ok=True)

    indent_xml(root)
    tree.write(run_paths.config_path, encoding="utf-8", xml_declaration=False)
    _write_json(run_paths.params_path, materialized)
    resolved_meta = {
        "study_name": config.study_name,
        "optuna_name": config.optuna_name,
        "config_file": str(config.config_path),
        "baseline_config": str(config.baseline_config_path),
        "trial_number": run_paths.trial_number,
        "run_window": {
            "start_ds": run_paths.run_start_ds,
            "end_ds": run_paths.run_end_ds,
        },
        "score_window": {
            "start_ds": run_paths.score_start_ds,
            "end_ds": run_paths.score_end_ds,
        },
        "output_root": str(run_paths.output_root),
        "checkpoint_root": str(run_paths.checkpoint_root),
        "snaptime": run_paths.snaptime,
        "seed": _read_xml_attr(root, "./combo/model", "seed"),
        "git_commit": git_commit if git_commit is not None else get_git_commit(config.baseline_config_path.parent),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "fixed_overrides": dict(fixed_overrides),
        "kind": run_paths.kind,
    }
    _write_json(run_paths.resolved_meta_path, resolved_meta)
    return materialized


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
            fields[f"{path}.@{attr}"] = coerce_scalar(raw_value)
        text = (element.text or "").strip()
        if text:
            fields[f"{path}.#text"] = text
        totals: dict[str, int] = {}
        for child_item in list(element):
            totals[child_item.tag] = totals.get(child_item.tag, 0) + 1
        seen: dict[str, int] = {}
        for child_item in list(element):
            seen[child_item.tag] = seen.get(child_item.tag, 0) + 1
            child_name = child_item.tag if totals[child_item.tag] == 1 else f"{child_item.tag}[{seen[child_item.tag]}]"
            visit(child_item, f"{path}.{child_name}")

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
    element.set(attr, format_xml_value(value))


def _read_xml_attr(root: ET.Element, element_path: str, attr: str) -> str | None:
    element = root.find(element_path)
    if element is None:
        return None
    return element.get(attr)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return left == right
