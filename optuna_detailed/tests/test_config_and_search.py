from __future__ import annotations

import pytest

from optuna_detailed.config_validator import build_validation_plan
from optuna_detailed.search_space import ConfigDrivenAdapter
from optuna_detailed.study_config import load_study_config


def test_default_config_loads_and_baseline_params(tmp_path) -> None:
    config = load_study_config(study_root_override=tmp_path)
    adapter = ConfigDrivenAdapter(config)

    assert config.study_name == "study_hybrid_tcn_detailed"
    assert config.baseline_config_path.name == "config_hybrid_tcn.xml"
    assert config.tuning_run_window == (20200102, 20231229)
    assert config.full_run_window == (20200102, 20240628)
    assert config.scoring_window == (20210104, 20231229)
    assert adapter.baseline_params() == {
        "lr": 2e-6,
        "weight_decay": 1e-6,
        "dropout": 0.5,
        "hiddenSize": 512,
        "fcSize": 256,
        "scheduler_gamma": 0.5,
        "tcnChannels": 128,
        "tcnLayers": 1,
        "tcnKernelSize": 3,
        "tcnDropout": 0.5,
        "tcnDilationBase": 1,
    }


def test_validate_config_recognizes_runtime_and_model_params(tmp_path) -> None:
    config = load_study_config(study_root_override=tmp_path)
    plan = build_validation_plan(config)

    recognized = {row["xml_path"] for row in plan.recognized}
    assert plan.status == "OK"
    assert "combo.model.lr" in recognized
    assert "combo.model.hiddenSize" in recognized
    assert "combo.model.tcnChannels" in recognized
    assert "combo.model.tcnKernelSize" in recognized
    assert not plan.recognized_virtual
    assert not plan.unrecognized
    assert not plan.invalid


def test_unrecognized_param_blocks_plan(tmp_path) -> None:
    source = load_study_config().config_path.read_text(encoding="utf-8")
    bad_config = tmp_path / "bad_config.xml"
    bad_config.write_text(
        source.replace(
            "</search_space>",
            '<param section="model" name="does_not_exist" type="float" low="0.1" high="0.2" /></search_space>',
        ),
        encoding="utf-8",
    )
    config = load_study_config(bad_config, study_root_override=tmp_path / "study")

    plan = build_validation_plan(config)

    assert plan.status == "BLOCKED"
    assert any(row["name"] == "does_not_exist" for row in plan.unrecognized)


def test_no_scheduler_step_ratio_is_declared(tmp_path) -> None:
    config = load_study_config(study_root_override=tmp_path)
    adapter = ConfigDrivenAdapter(config)

    assert "scheduler_step_ratio" not in adapter.baseline_params()
    assert not config.derived_params


def test_fake_trial_suggests_configured_ranges(tmp_path) -> None:
    config = load_study_config(study_root_override=tmp_path)
    adapter = ConfigDrivenAdapter(config)

    class FakeTrial:
        def suggest_float(self, name, low, high, log=False):
            assert low < high
            if name in {"lr", "weight_decay"}:
                assert log is True
            return low

        def suggest_int(self, name, low, high, **kwargs):
            raise AssertionError(name)

        def suggest_categorical(self, name, choices):
            assert choices
            return choices[0]

    params = adapter.suggest_params(FakeTrial())

    assert params["lr"] == pytest.approx(1e-7)
    assert params["hiddenSize"] == 256
    assert params["tcnChannels"] == 64
    assert "scheduler_step_ratio" not in params
