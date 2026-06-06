"""Adapter protocol for model-specific search spaces and XML patches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from xml.etree import ElementTree as ET


class ModelAdapter(ABC):
    """Abstract model adapter used by the generic study runner."""

    name: str

    @abstractmethod
    def baseline_params(self) -> dict[str, Any]:
        """Return the baseline hyperparameters in search-space coordinates."""

    @abstractmethod
    def suggest_params(self, trial: Any) -> dict[str, Any]:
        """Suggest one parameter set from an Optuna-like trial object."""

    @abstractmethod
    def materialize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return params plus derived fields that must be written to XML."""

    @abstractmethod
    def apply_params_to_xml(self, root: ET.Element, params: dict[str, Any]) -> dict[str, Any]:
        """Patch model parameters into an XML tree and return materialized params."""

