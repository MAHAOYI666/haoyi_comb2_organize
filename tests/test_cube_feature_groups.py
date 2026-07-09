from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config import load_config
from src.DataLoader import ComboDataLoader, FeatureGroups, GroupCodec, LoaderConfig
from src.DataRegistry import DataItem, DataRegistry, OpSpec, Universe
from src.codec import build_codec


class FakeMask:
    date = [20200101, 20200102, 20200103, 20200106]
    code = ["000001", "000002", "000003"]


def _patch_mask(monkeypatch):
    import src.DataLoader as data_loader
    import src.DataRegistry as data_registry

    fake = FakeMask()
    monkeypatch.setattr(data_loader, "MASK", fake)
    monkeypatch.setattr(data_registry, "MASK", fake)
    return fake


def _daily_loader(item, registry, start_ds, end_ds):
    lo = registry.universe.date2idx(start_ds)
    hi = registry.universe.date2idx(end_ds)
    rows = hi - lo + 1
    return torch.arange(rows * len(registry.universe.codes), dtype=torch.float32).reshape(rows, len(registry.universe.codes))


def _intraday_loader(bars: int):
    def load(item, registry, start_ds, end_ds):
        lo = registry.universe.date2idx(start_ds)
        hi = registry.universe.date2idx(end_ds)
        rows = hi - lo + 1
        values = torch.arange(rows * bars * len(registry.universe.codes), dtype=torch.float32)
        return values.reshape(rows, bars, len(registry.universe.codes))

    return load


def test_config_normalizes_freq_and_rejects_removed_ops(tmp_path: Path):
    config_path = tmp_path / "config.xml"
    config_path.write_text(
        """
<config>
  <combo>
    <paths model_path="model.py" />
    <data>
      <item name="factor.daily" path="daily" role="factor" />
      <item name="factor.m5" module="builtin.factorsim" path="m5" role="factor" freq="5m" />
      <item name="label.default" module="builtin.factorsim" path="vwap30_label1d" role="label" />
    </data>
  </combo>
</config>
""",
        encoding="utf-8",
    )
    parsed = load_config(str(config_path))
    items = parsed["combo"]["loader"]["data_items"]
    assert items[0]["params"]["freq"] == "1d"
    assert items[1]["params"]["freq"] == "5m"
    assert items[0]["module"] == "builtin.factorsim"
    assert items[2]["module"] == "builtin.factorsim"

    bad_path = tmp_path / "bad.xml"
    bad_path.write_text(
        """
<config>
  <combo>
    <data>
      <item name="factor.bad" module="builtin.factorsim" path="bad" role="factor" nbar="10" />
      <item name="label.default" module="builtin.factorsim" path="vwap30_label1d" role="label" />
    </data>
  </combo>
</config>
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nbar"):
        load_config(str(bad_path))

    reducer_path = tmp_path / "reducer.xml"
    reducer_path.write_text(
        """
<config>
  <combo>
    <data>
      <item name="factor.bad" module="builtin.factorsim" path="bad" role="factor">
        <op name="mean" axis="bar" />
      </item>
      <item name="label.default" module="builtin.factorsim" path="vwap30_label1d" role="label" />
    </data>
  </combo>
</config>
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported data op"):
        load_config(str(reducer_path))


def test_registry_preserves_item_shape_by_freq():
    universe = Universe(dates=(20200101, 20200102), codes=("000001", "000002"), dtype=torch.float32)
    registry = DataRegistry(
        (
            DataItem(name="daily", module="test.daily", role="factor"),
            DataItem(name="m5", module="test.m5", role="factor", params={"freq": "5m"}),
            DataItem(name="m1", module="test.m1", role="aux", params={"freq": "1m"}),
        ),
        universe=universe,
        data_start_ds=20200101,
        ashare_data_path=None,
        config_path=None,
    )
    registry.modules["test.daily"] = _daily_loader
    registry.modules["test.m5"] = _intraday_loader(49)
    registry.modules["test.m1"] = _intraday_loader(239)

    with pytest.raises(ValueError, match="pass start_ds/end_ds"):
        registry.get_data("m1")

    assert registry.get_data("daily", 20200101, 20200102).shape == (2, 2)
    assert registry.get_data("m5", 20200101, 20200102).shape == (2, 49, 2)
    assert registry.get_data("m1", 20200101, 20200102).shape == (2, 239, 2)
    assert registry.get_data("m1", 20200102, 20200102).shape == (1, 239, 2)
    assert registry.get_data("m1").shape == (2, 239, 2)


def test_loader_groups_features_and_masks_only_window_tail(monkeypatch):
    _patch_mask(monkeypatch)
    loader = ComboDataLoader(
        LoaderConfig(
            dtype=torch.float32,
            data_start_ds=20200101,
            data_offset=0,
            data_items=(
                DataItem(name="daily", module="test.daily", role="factor"),
                DataItem(name="m5", module="test.m5", role="factor", params={"freq": "5m"}),
                DataItem(name="m1", module="test.m1", role="factor", params={"freq": "1m"}),
                DataItem(name="label", module="test.daily", role="label"),
            ),
        )
    )
    loader.registry.modules["test.daily"] = _daily_loader
    loader.registry.modules["test.m5"] = _intraday_loader(49)
    loader.registry.modules["test.m1"] = _intraday_loader(239)

    day = loader.gen_feature(20200101)
    assert day.freqs == ("1d", "5m", "1m")
    assert day["1d"].shape == (3, 1)
    assert day["5m"].shape == (3, 49, 1)
    assert day["1m"].shape == (3, 239, 1)

    loader.set_current_ti(100000)
    window = FeatureGroups.stack([day, day], dim=0)
    masked = loader.transform_feature_window(window, stage="train")
    assert torch.isfinite(masked["5m"][0]).all()
    assert torch.isnan(masked["5m"][-1, :, 7:, :]).all()
    assert torch.isfinite(masked["5m"][-1, :, :7, :]).all()
    assert torch.isfinite(masked["1d"]).all()


def test_group_codec_roundtrip_preserves_freqs_and_nan():
    groups = FeatureGroups(
        {
            "1d": torch.tensor([[1.0], [float("nan")]], dtype=torch.float32),
            "5m": torch.ones((2, 49, 1), dtype=torch.float32),
        }
    )
    codec = GroupCodec(build_codec("fp4", torch.float32), groups.freqs)
    storage, meta = codec.allocate({"1d": (3, 2, 1), "5m": (3, 2, 49, 1)}, logical_dtype=torch.float32)
    codec.encode_into(storage, meta, 0, groups)
    decoded = codec.decode(storage, meta, 0, out_dtype=torch.float32)
    assert decoded.freqs == ("1d", "5m")
    assert torch.isnan(decoded["1d"][1, 0])
    assert decoded["5m"].shape == (2, 49, 1)
