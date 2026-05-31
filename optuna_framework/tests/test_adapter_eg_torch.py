"""eg-torch adapter tests."""

from __future__ import annotations

from xml.etree import ElementTree as ET

from optuna_framework.paths import get_repo_root
from optuna_framework.plugin_loader import load_plugin


class FakeTrial:
    """Minimal Optuna-like trial for deterministic suggestions."""

    def suggest_float(self, name, low, high, log=False):
        assert low < high
        if name == "lr":
            assert log is True
            return 1e-6
        if name == "weight_decay":
            assert log is True
            return 1e-5
        if name == "dropout":
            return 0.3
        if name == "scheduler_gamma":
            return 0.4
        raise AssertionError(name)

    def suggest_categorical(self, name, choices):
        assert choices
        return choices[0]

    def suggest_int(self, name, low, high):
        assert name == "epochs"
        assert (low, high) == (5, 25)
        return 10


def test_suggest_params_ranges() -> None:
    adapter = load_plugin(model_dir="eg-torch").adapter
    params = adapter.suggest_params(FakeTrial())
    assert params["lr"] == 1e-6
    assert params["weight_decay"] == 1e-5
    assert params["dropout"] == 0.3
    assert params["hiddenSize"] in adapter.hidden_sizes
    assert params["fcSize"] in adapter.fc_sizes
    assert params["epochs"] == 10
    assert params["scheduler_step_ratio"] in adapter.scheduler_step_ratios


def test_baseline_materializes_to_original_model_fields() -> None:
    adapter = load_plugin(model_dir="eg-torch").adapter
    root = ET.parse(get_repo_root() / "eg-torch" / "config.xml").getroot()
    materialized = adapter.apply_params_to_xml(root, adapter.baseline_params())
    model = root.find("./combo/model")
    assert model is not None
    assert float(model.get("lr")) == 2e-6
    assert float(model.get("weight_decay")) == 1e-6
    assert float(model.get("dropout")) == 0.5
    assert int(model.get("hiddenSize")) == 512
    assert int(model.get("fcSize")) == 256
    assert int(model.get("epochs")) == 15
    assert int(model.get("scheduler_step_size")) == 10
    assert float(model.get("scheduler_gamma")) == 0.5
    assert materialized["scheduler_step_size"] == 10


def test_apply_params_to_xml_writes_model_attrs() -> None:
    adapter = load_plugin(model_dir="eg-torch").adapter
    root = ET.parse(get_repo_root() / "eg-torch" / "config.xml").getroot()
    params = {
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "dropout": 0.25,
        "hiddenSize": 384,
        "fcSize": 128,
        "epochs": 20,
        "scheduler_step_ratio": 0.5,
        "scheduler_gamma": 0.7,
    }
    adapter.apply_params_to_xml(root, params)
    model = root.find("./combo/model")
    assert model is not None
    assert float(model.get("lr")) == 1e-5
    assert int(model.get("hiddenSize")) == 384
    assert int(model.get("scheduler_step_size")) == 10

