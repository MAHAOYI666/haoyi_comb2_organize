"""eg-torch native seed handling tests."""

from __future__ import annotations

import importlib.util

import torch

from optuna_framework.paths import get_repo_root


class TinyDataset:
    validinsts = torch.tensor([0], dtype=torch.long)

    def __len__(self) -> int:
        return 0


def _load_eg_torch_model_module():
    model_path = get_repo_root() / "eg-torch" / "model.py"
    spec = importlib.util.spec_from_file_location("eg_torch_model_for_seed_test", model_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_research_model_uses_config_seed_for_dataloader_generator(monkeypatch) -> None:
    module = _load_eg_torch_model_module()
    captured = {}

    def fake_dataloader(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(module, "DataLoader", fake_dataloader)
    model = module.ResearchModel(
        {
            "seed": 123,
            "epochs": 0,
            "device": "cpu",
            "num_features": 1,
            "hidden_size": 2,
            "fc_size": 2,
        }
    )

    model.fit(TinyDataset())

    assert model.seed == 123
    assert captured["shuffle"] is True
    assert captured["generator"].initial_seed() == 123
