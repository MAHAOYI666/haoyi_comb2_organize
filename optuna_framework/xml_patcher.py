"""Small XML helpers for detailed Optuna config rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


SECTION_XPATH = {
    "model": "./combo/model",
    "runtime": "./combo/runtime",
}


def section_xpath(section: str) -> str:
    """Return the XML xpath for a supported user-tunable section."""

    if section not in SECTION_XPATH:
        raise ValueError(f"unsupported tunable section: {section}")
    return SECTION_XPATH[section]


def xml_attr_path(section: str, name: str) -> str:
    """Return a stable dotted display path for a supported attribute."""

    return f"combo.{section}.{name}"


def find_section(root: ET.Element, section: str) -> ET.Element | None:
    """Find ``combo.runtime`` or ``combo.model``."""

    return root.find(section_xpath(section))


def get_section_attr(root: ET.Element, section: str, name: str) -> str | None:
    """Return one supported XML section attribute."""

    element = find_section(root, section)
    if element is None:
        return None
    return element.get(name)


def set_section_attr(root: ET.Element, section: str, name: str, value: Any) -> None:
    """Set one supported XML section attribute."""

    element = find_section(root, section)
    if element is None:
        raise ValueError(f"baseline XML is missing section: combo.{section}")
    element.set(name, format_xml_value(value))


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
    element.set(attr_name, format_xml_value(value))


def resolve_relative_paths(root: ET.Element, base_dir: Path) -> None:
    """Convert runtime file paths to absolute paths anchored to the baseline config."""

    for xpath, attr in (("./combo/paths", "model_path"), ("./strategy", "path")):
        element = root.find(xpath)
        if element is None:
            continue
        value = element.get(attr)
        if value and not Path(value).expanduser().is_absolute():
            element.set(attr, str((base_dir / value).resolve()))


def read_model_path(root: ET.Element, baseline_dir: Path) -> Path | None:
    """Read and resolve ``combo.paths.@model_path`` from a baseline XML tree."""

    element = root.find("./combo/paths")
    if element is None:
        return None
    raw = element.get("model_path")
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = baseline_dir / path
    return path.resolve()


def format_xml_value(value: Any) -> str:
    """Format Python values for XML attributes."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def coerce_scalar(raw_value: str) -> Any:
    """Coerce an XML string into a simple comparable Python scalar."""

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


def indent_xml(element: ET.Element, level: int = 0) -> None:
    """Pretty-print an ElementTree in-place."""

    indent = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent
