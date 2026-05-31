"""Load model-owned Optuna plugins."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from optuna_framework.adapters.base import ModelAdapter
from optuna_framework.paths import get_repo_root
from optuna_framework.specs import StudySpec

DEFAULT_MODEL_DIR = "eg-torch"
PLUGIN_FILENAME = "optuna_plugin.py"


@dataclass(frozen=True)
class LoadedPlugin:
    """Resolved Optuna plugin module plus its public contract."""

    path: Path
    module: ModuleType
    study_spec: StudySpec
    adapter: ModelAdapter


def default_plugin_path() -> Path:
    """Return the default eg-torch plugin path."""

    return get_repo_root() / DEFAULT_MODEL_DIR / PLUGIN_FILENAME


def resolve_plugin_path(model_dir: str | Path | None = None, plugin: str | Path | None = None) -> Path:
    """Resolve a plugin path from ``--model-dir`` or ``--plugin`` style inputs."""

    if model_dir is not None and plugin is not None:
        raise ValueError("use either model_dir or plugin, not both")
    if plugin is not None:
        return _resolve_repo_relative(plugin)
    if model_dir is not None:
        return (_resolve_repo_relative(model_dir) / PLUGIN_FILENAME).resolve()
    return default_plugin_path().resolve()


def load_plugin(model_dir: str | Path | None = None, plugin: str | Path | None = None) -> LoadedPlugin:
    """Load and validate a model-owned Optuna plugin."""

    plugin_path = resolve_plugin_path(model_dir=model_dir, plugin=plugin)
    if not plugin_path.exists():
        raise FileNotFoundError(f"Optuna plugin not found: {plugin_path}")
    module = load_plugin_module(plugin_path)
    study_spec = _get_study_spec(module, plugin_path)
    adapter = _get_adapter(module, plugin_path)
    return LoadedPlugin(plugin_path, module, study_spec, adapter)


def load_plugin_module(plugin_path: str | Path) -> ModuleType:
    """Import a plugin module from an arbitrary file path."""

    path = Path(plugin_path).expanduser().resolve()
    module_name = f"_optuna_plugin_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import Optuna plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_study_spec(module: ModuleType, plugin_path: Path) -> StudySpec:
    study_spec = getattr(module, "STUDY_SPEC", None)
    if not isinstance(study_spec, StudySpec):
        raise TypeError(f"Optuna plugin must define STUDY_SPEC: {plugin_path}")
    return study_spec


def _get_adapter(module: ModuleType, plugin_path: Path) -> ModelAdapter:
    factory = getattr(module, "create_adapter", None)
    if callable(factory):
        adapter = factory()
    else:
        adapter = getattr(module, "ADAPTER", None)
    if not isinstance(adapter, ModelAdapter):
        raise TypeError(f"Optuna plugin must define create_adapter() or ADAPTER: {plugin_path}")
    return adapter


def _resolve_repo_relative(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = get_repo_root() / resolved
    return resolved.resolve()
