"""Baseline runner tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from optuna_framework import baseline_runner
from optuna_framework.metrics_parser import SegmentMetrics
from optuna_framework.specs import SegmentSpec, StudySpec


class Adapter:
    def baseline_params(self) -> dict:
        return {"lr": 0.1}

    def apply_params_to_xml(self, root: ET.Element, params: dict) -> dict:
        root.find("./combo/model").set("lr", str(params["lr"]))
        return dict(params)


def _metric(sharpe: float, dd: float) -> SegmentMetrics:
    return SegmentMetrics(sharpe, dd, 0.1, 0.1, 1.0, 240, "row", ["row"])


def _study_spec(config_path: Path) -> StudySpec:
    return StudySpec(
        name="test",
        baseline_config_path=config_path,
        adapter_name="adapter",
        tuning_segments=(
            SegmentSpec("seg01", "tuning", 20210104, 20210630),
            SegmentSpec("seg02", "tuning", 20210701, 20211231),
        ),
        holdout_segments=(),
        baseline_only_segments=(SegmentSpec("full_period", "baseline_only", 20210104, 20211231),),
        fixed_overrides={"combo.output.enable_alpha_analysis": False},
    )


def test_worker_count_caps_to_devices() -> None:
    assert baseline_runner._worker_count(4, ("cuda:0", "cuda:1")) == 2
    assert baseline_runner._worker_count(4, ()) == 1


def test_parallel_baseline_assigns_devices_and_writes_thresholds(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.xml"
    config_path.write_text(
        """
<config>
  <constants output_root="output" checkpoint_root="checkpoints" />
  <strategy start_ds="20210104" end_ds="20211231" />
  <combo>
    <paths model_path="model.py" />
    <runtime snaptime="base" />
    <model device="cuda:9" lr="0.01" />
  </combo>
</config>
""".strip(),
        encoding="utf-8",
    )
    seen_devices: dict[str, str] = {}

    def fake_run_segment(run_paths):
        root = ET.parse(run_paths.config_path).getroot()
        seen_devices[run_paths.segment.name] = root.find("./combo/model").get("device")
        if run_paths.segment.name == "seg01":
            return _metric(1.0, 0.1)
        if run_paths.segment.name == "seg02":
            return _metric(1.2, 0.2)
        return _metric(1.1, 0.15)

    monkeypatch.setattr(baseline_runner, "run_segment", fake_run_segment)
    monkeypatch.setattr(baseline_runner, "parse_full_period", lambda _path: {"full": _metric(1.1, 0.15)})

    baseline_runner.run_baseline_evaluations(
        _study_spec(config_path),
        Adapter(),
        tmp_path / "study",
        n_jobs=2,
        devices=("cuda:0", "cuda:1"),
    )

    assert set(seen_devices.values()) <= {"cuda:0", "cuda:1"}
    assert seen_devices["seg01"] != "cuda:9"
    thresholds = json.loads((tmp_path / "study" / "baseline" / "baseline_thresholds.json").read_text(encoding="utf-8"))
    assert thresholds["tuning_min_sharpe"] == 1.0
    assert thresholds["tuning_max_dd"] == 0.2


def test_dry_run_prints_baseline_parallel_settings(tmp_path: Path, capsys) -> None:
    spec = SimpleNamespace(
        baseline_segments=lambda: (SegmentSpec("seg01", "tuning", 20210104, 20210630),),
    )

    baseline_runner.run_baseline_evaluations(
        spec,
        Adapter(),
        tmp_path / "study",
        dry_run=True,
        n_jobs=2,
        devices=("cuda:0", "cuda:1"),
    )

    output = capsys.readouterr().out
    assert "baseline n_jobs=2" in output
    assert "baseline devices=cuda:0,cuda:1" in output
