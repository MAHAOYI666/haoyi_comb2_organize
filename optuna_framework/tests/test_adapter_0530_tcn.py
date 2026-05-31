"""0530.tcn adapter tests."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from optuna_framework.plugin_loader import load_plugin


PLUGIN_PATH = Path("/root/autodl/0530.tcn/optuna_plugin.py")
CONFIG_PATH = Path("/root/autodl/0530.tcn/config.xml")


class FakeTrial:
    """Minimal Optuna-like trial for deterministic suggestions."""

    def suggest_float(self, name, low, high, log=False):
        assert low < high
        if name == "lr":
            assert log is True
            return 1e-5
        if name == "weight_decay":
            assert log is True
            return 1e-5
        if name == "dropout":
            return 0.25
        if name == "scheduler_gamma":
            return 0.4
        raise AssertionError(name)

    def suggest_categorical(self, name, choices):
        assert choices
        return choices[0]

    def suggest_int(self, name, low, high):
        assert name == "epochs"
        assert (low, high) == (8, 20)
        return 10


def test_plugin_loads_from_absolute_path() -> None:
    loaded = load_plugin(plugin=PLUGIN_PATH)
    assert loaded.path == PLUGIN_PATH.resolve()
    assert loaded.study_spec.name == "study_0530_tcn_v1"
    assert loaded.adapter.name == "tcn_0530_v1"


def test_suggest_params_ranges() -> None:
    adapter = load_plugin(plugin=PLUGIN_PATH).adapter
    params = adapter.suggest_params(FakeTrial())
    assert params["lr"] == 1e-5
    assert params["weight_decay"] == 1e-5
    assert params["dropout"] == 0.25
    assert params["hiddenSize"] in adapter.hidden_sizes
    assert params["fcSize"] in adapter.fc_sizes
    assert params["tcnChannels"] in adapter.tcn_channels
    assert params["numBlocks"] in adapter.num_blocks
    assert params["epochs"] == 10
    assert params["scheduler_step_ratio"] in adapter.scheduler_step_ratios
    assert params["scheduler_gamma"] == 0.4


def test_baseline_materializes_to_original_model_fields() -> None:
    adapter = load_plugin(plugin=PLUGIN_PATH).adapter
    root = ET.parse(CONFIG_PATH).getroot()
    materialized = adapter.apply_params_to_xml(root, adapter.baseline_params())
    model = root.find("./combo/model")
    assert model is not None
    assert float(model.get("lr")) == 2e-5
    assert float(model.get("weight_decay")) == 1e-6
    assert float(model.get("dropout")) == 0.3
    assert int(model.get("hiddenSize")) == 256
    assert int(model.get("fcSize")) == 128
    assert int(model.get("tcnChannels")) == 64
    assert int(model.get("numBlocks")) == 2
    assert int(model.get("epochs")) == 15
    assert int(model.get("scheduler_step_size")) == 10
    assert float(model.get("scheduler_gamma")) == 0.5
    assert materialized["scheduler_step_size"] == 10


def test_seeded_wrapper_and_overrides() -> None:
    adapter = load_plugin(plugin=PLUGIN_PATH).adapter
    wrapper = adapter.write_seeded_model(Path("/tmp/tcn-seeded"), CONFIG_PATH, "phase_b")
    assert wrapper is not None
    source = wrapper.read_text(encoding="utf-8")
    assert "torch.manual_seed" in source
    assert "np.random.seed" in source
    assert "random.seed" in source
    assert "generator.manual_seed" in source
    overrides = adapter.seeded_overrides(43, wrapper)
    assert overrides["combo.model.seed"] == 43
    assert overrides["combo.paths.model_path"] == str(wrapper)
