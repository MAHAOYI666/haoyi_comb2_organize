from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts" / "source_update_20260911"
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch
import config

torch.set_num_threads(2)
expected = {
    "README.md", "现在的负责点.txt", "combo框架整体架构.txt",
    "config.xml", "model.py", "loader.py", "dataset.py", "config.human",
}
assert {p.name for p in STAGE.iterdir() if p.is_file()} == expected
checks = []
for file in (p for p in STAGE.iterdir() if p.is_file()):
    content = file.read_text(encoding="utf-8-sig")
    assert "\ufffd" not in content, file
    if file.suffix == ".py":
        ast.parse(content, filename=str(file))
checks.append("8 files: UTF-8 readable; all 3 Python files parse")

for name in ("config.xml", "model.py", "loader.py", "dataset.py", "config.human"):
    original = ROOT / name if name == "config.human" else ROOT / "eg-torch" / name
    assert original.read_bytes() == (STAGE / name).read_bytes(), name
checks.append("5 copied files match current repository bytes exactly")

for link in re.findall(r"\]\(([^)]+)\)", (STAGE / "README.md").read_text(encoding="utf-8")):
    assert (STAGE / link).exists(), link
checks.append("README relative file links resolve")

cfg = config.load_config(str(STAGE / "config.xml"))
assert cfg["combo"]["runtime"]["sample_times"] == (100000,)
assert cfg["combo"]["runtime"]["trainDelay"] == 2
assert cfg["combo"]["runtime"]["retDays"] == 1
assert cfg["combo"]["runtime"]["tsDays"] == 8
assert cfg["strategy"]["optimizer"]["type"] == "opt1"
assert cfg["backtest"]["fee_rate"] == .0015
assert config.DEFAULT_CONFIG["backtest"]["fee_rate"] == .00075
assert cfg["backtest"]["execution_price"] == "execution:execution"
assert Path(cfg["constants"]["output_root"]) == STAGE / "output-torch"
for key in ("model_path", "research_loader_path", "research_dataset_path"):
    assert Path(cfg["combo"]["paths"][key]).is_file()
checks.append("current config parser: sample time, delay, optimizer, fees, execution field, paths")

spec = importlib.util.spec_from_file_location("source_export_model", STAGE / "model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model_cfg = dict(cfg["combo"]["model"])
model_cfg.update({
    "dtype": torch.float16, "tsDays": 8, "num_features": 8,
    "feature_names": tuple(f"factor_{i}" for i in range(8)),
    "sample_times": (100000,), "seed": 42,
    "hidden_size": 16, "fc_size": 8, "batch_size": 2, "epochs": 1,
})

class SyntheticDataset:
    validinsts = torch.arange(12, dtype=torch.long)

    def __init__(self):
        generator = torch.Generator().manual_seed(7)
        self.features = torch.randn(9, 12, 8, generator=generator)
        self.labels = torch.randn(2, 12, generator=generator)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        weights = torch.ones(12, dtype=torch.bool)
        weights[-1] = False
        return index, 20260115 + index, 100000, self.features[index:index + 8], self.labels[index], weights

dataset = SyntheticDataset()
model = module.ResearchModel(model_cfg).fit(dataset)
prediction = model.predict(dataset[0][3], di=20260115, ti=100000)
assert prediction.shape == (12,) and prediction.dtype == torch.float16
assert torch.isfinite(prediction).all()
assert torch.equal(model.trainii, dataset.validinsts)
buffer = io.BytesIO()
model.save(buffer)
buffer.seek(0)
loaded = module.ResearchModel(model_cfg).load(buffer)
reloaded = loaded.predict(dataset[0][3], di=20260115, ti=100000)
assert torch.equal(prediction, reloaded)
checks.append("synthetic six-field dataset: fit, tensor predict(di/ti), trainii, save/load roundtrip")

x = torch.tensor([[1., 2., 3., 100.]])
y = torch.tensor([[3., 1., 2., -100.]])
w = torch.tensor([[True, True, True, False]])
loss = module.ICLoss()(x, y, w)
x[0, -1] = -1e4
y[0, -1] = 1e4
assert torch.equal(loss, module.ICLoss()(x, y, w))
assert torch.isfinite(loss)
checks.append("IC loss excludes masked synthetic stock")

report = {
    "date": "2026-09-11", "scope": "local documentation and synthetic interface checks only",
    "python": sys.version.split()[0], "torch": torch.__version__, "checks": checks,
    "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in sorted(STAGE.iterdir()) if p.is_file()},
    "not_run": ["remote data verification", "factor-backed training", "MOSEK backtest", "deployment"],
}
(ROOT / "artifacts" / "source_update_20260911_validation.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({"status": "PASS", "checks": checks}, ensure_ascii=False, indent=2))
