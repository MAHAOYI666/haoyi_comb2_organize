"""Numerical and interface regression checks for the normalized hybrid."""
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_model(folder):
    spec = importlib.util.spec_from_file_location(folder, ROOT / "haoyi_models" / folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    torch.set_num_threads(2)
    return load_model("temporal_fullcs_protocol_20260920")


def test_ic_matches_valid_stock_pearson_and_affine_invariance(m):
    torch.manual_seed(17)
    p, y = torch.randn(3, 11), torch.randn(3, 11)
    w = torch.rand(3, 11) > .3
    expected = torch.stack([1 - torch.corrcoef(torch.stack([a[c], b[c]]))[0, 1]
                            for a, b, c in zip(p, y, w)]).mean()
    torch.testing.assert_close(m.ic_loss(p, y, w), expected)
    torch.testing.assert_close(m.ic_loss(p * 7 + 9, y * 3 - 4, w), expected)
    p[~w], y[~w] = float("nan"), float("inf")
    torch.testing.assert_close(m.ic_loss(p, y, w), expected)


@pytest.mark.parametrize("case", ["empty", "constant_pred", "constant_target", "single"])
def test_zero_norm_loss_and_gradients_are_finite(m, case):
    p = torch.randn(2, 7, 4)
    y = torch.randn(2, 7)
    w = torch.ones_like(y, dtype=torch.bool)
    if case == "empty":
        w[:] = False
    elif case == "constant_pred":
        p[:] = 0
    elif case == "constant_target":
        y[:] = 0
    else:
        w[:, 1:] = False
    p.requires_grad_()
    loss = m.MultiHeadICLoss(4)(p, y, w)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    torch.testing.assert_close(m.ic_loss(p[..., 0], y, w), torch.tensor(1.))


def test_masked_padding_does_not_change_loss_or_gradient(m):
    torch.manual_seed(23)
    p = torch.randn(2, 9, 4, requires_grad=True)
    y, w = torch.randn(2, 9), torch.ones(2, 9, dtype=torch.bool)
    loss = m.MultiHeadICLoss(4)(p, y, w)
    grad, = torch.autograd.grad(loss, p)
    padded = torch.cat([p.detach(), torch.full((2, 5, 4), 1000.)], 1).requires_grad_()
    padded_loss = m.MultiHeadICLoss(4)(padded, torch.cat([y, torch.full((2, 5), float("nan"))], 1),
                                         torch.cat([w, torch.zeros(2, 5, dtype=torch.bool)], 1))
    padded_loss.backward()
    torch.testing.assert_close(padded_loss, loss)
    torch.testing.assert_close(padded.grad[:, :9], grad)
    assert not padded.grad[:, 9:].count_nonzero()


def test_architecture_retained_and_full_stock_mask_is_input_only(m):
    old = load_model("temporal_fullcs_qsim_20260908")
    kwargs = dict(features=5, steps=3, width=8, temporal_width=8, heads=2,
                  layers=2, dropout=0., fc_size=8, output_heads=4)
    net, reference = m.Model(**kwargs).eval(), old.Model(**kwargs).eval()
    reference.load_state_dict(net.state_dict())
    x = torch.randn(2, 3, 9, 5).clamp(-4, 4)
    x[:, :, -1] = 0
    torch.testing.assert_close(net(x), reference(x))
    perm = torch.randperm(9)
    torch.testing.assert_close(net(x[:, :, perm]), net(x)[:, perm], atol=1e-6, rtol=1e-5)
    assert not net(x)[:, -1].count_nonzero()
    assert sum(p.numel() for p in m.Model(625).parameters()) == 3154967


def test_six_item_samples_fit_predict_and_checkpoint(m):
    samples = [(i, 20200102 + i, 100000, {"1d": torch.randn(3, 9, 5)},
                torch.randn(9), torch.ones(9, dtype=torch.bool)) for i in range(4)]
    class Dataset:
        numValidinsts = 9
        def __len__(self): return len(samples)
        def __getitem__(self, i): return samples[i]
    model = m.ResearchModel(dict(dtype="float16", device="cpu", num_features=5, tsDays=3,
                                hiddenSize=8, fcSize=8, temporal_width=8, numAttnHeads=2,
                                epochs=2, batchSize=3, dropout=0.))
    model.fit(Dataset())
    pred = model.predict(samples[0][3], di=1, ti=100000)
    assert pred.shape == (9,) and torch.isfinite(pred).all()
    buffer = io.BytesIO()
    model.save(buffer)
    buffer.seek(0)
    restored = m.ResearchModel(model.config).load(buffer)
    torch.testing.assert_close(pred, restored.predict(samples[0][3], di=1, ti=100000))


def test_split_data_interfaces_preserve_original_factor_order_and_label():
    folder = ROOT / "haoyi_models/temporal_fullcs_protocol_20260920"
    cfg = ET.parse(folder / "config.xml")
    classes = {}
    for key, cls in [("research_loader_path", "ResearchLoader"),
                     ("research_dataset_path", "ResearchDataset")]:
        path = folder / cfg.find("combo/paths").get(key)
        spec = importlib.util.spec_from_file_location(f"protocol_{cls}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes[cls] = getattr(module, cls)
    loader = object.__new__(classes["ResearchLoader"])
    loader.config = SimpleNamespace(cache_path="/cache")
    sources = loader.data_requirements()
    original = ET.parse(ROOT / "haoyi_models/temporal_fullcs_qsim_20260908/config.xml")
    factors = original.findall("combo/data/item")[:-1]
    assert [s.name for s in sources[:625]] == [f.get("name") for f in factors]
    assert [s.path for s in sources[:625]] == [f.get("path") for f in factors]
    assert all(s.delay == 1 for s in sources[:626])
    assert Path(sources[625].path).parts[-2:] == ("1d_DailyLabel", "DailyLabel.vwap30_label1d")
    assert sources[626].name == "execution" and sources[626].delay == 0
    assert loader.model_target() == ("label", "label")
    assert loader.model_input_sources() == tuple(s.name for s in sources[:625])
    dataset = object.__new__(classes["ResearchDataset"])
    dataset.loader = SimpleNamespace(mask=SimpleNamespace(code=list(range(5642))))
    torch.testing.assert_close(dataset._build_validinsts(), torch.arange(5642))
    assert cfg.find("constants").get("output_root") == "output"
    assert cfg.find("combo/runtime").get("retDays") == "5"
