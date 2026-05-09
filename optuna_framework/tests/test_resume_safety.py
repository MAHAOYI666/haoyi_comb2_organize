"""Resume safety tests for baseline enqueue."""

from __future__ import annotations

import pytest

from optuna_framework.study_utils import maybe_enqueue_baseline


def test_maybe_enqueue_baseline_only_once() -> None:
    optuna = pytest.importorskip("optuna")

    class FakeStudy:
        def __init__(self):
            self.trials = []
            self.enqueued = []

        def enqueue_trial(self, params):
            self.enqueued.append(params)
            trial = type("FakeTrial", (), {})()
            trial.state = optuna.trial.TrialState.WAITING
            self.trials.append(trial)

    study = FakeStudy()
    assert maybe_enqueue_baseline(study, {"lr": 2e-6}) is True
    assert maybe_enqueue_baseline(study, {"lr": 2e-6}) is False
    assert len(study.enqueued) == 1

