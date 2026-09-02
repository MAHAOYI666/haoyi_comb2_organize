from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from config import load_config
from comb2.DataLoader import ComboDataLoader, FeatureGroups, GroupCodec, LoaderConfig
from comb2.DataRegistry import (
    DataItem,
    DataRegistry,
    OpSpec,
    Universe,
    available_bar_count,
)
from comb2.codec import build_codec


class FakeMask:
    date = [20200101, 20200102, 20200103, 20200106]
    code = ["000001", "000002", "000003"]


def _patch_mask(monkeypatch):
    data_loader = importlib.import_module("comb2.DataLoader")
    data_registry = importlib.import_module("comb2.DataRegistry")

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
  <constants freq="5m" />
  <combo>
    <paths model_path="model.py" />
    <data>
      <item name="factor.daily" path="daily" />
      <item name="factor.m5" module="builtin.factorsim" path="m5" freq="5m" />
      <item name="returns.default" module="builtin.factorsim" path="5m_IntvReturns/IntvReturns.c2c" role="target" freq="5m" />
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
    assert items[2]["role"] == "target"
    assert items[2]["params"]["freq"] == "5m"

    bad_path = tmp_path / "bad.xml"
    bad_path.write_text(
        """
<config>
  <combo>
    <data>
      <item name="factor.bad" module="builtin.factorsim" path="bad" role="factor" nbar="10" />
      <item name="returns.default" module="builtin.factorsim" path="returns" role="target" freq="5m" />
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
      <item name="returns.default" module="builtin.factorsim" path="returns" role="target" freq="5m" />
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
        ashare_cache_path=None,
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
            freq="5m",
            data_items=(
                DataItem(name="daily", module="test.daily"),
                DataItem(name="m5", module="test.m5", params={"freq": "5m"}),
                DataItem(name="m1", module="test.m1", params={"freq": "1m"}),
                DataItem(name="returns", module="test.m5", role="target", params={"freq": "5m"}),
            ),
        )
    )
    loader.registry.modules["test.daily"] = _daily_loader
    loader.registry.modules["test.m5"] = _intraday_loader(49)
    loader.registry.modules["test.m1"] = _intraday_loader(239)

    day = loader.gen_feature(20200102)
    assert day.freqs == ("1d", "5m", "1m")
    assert day["1d"].shape == (3, 1)
    assert day["5m"].shape == (3, 49, 1)
    assert day["1m"].shape == (3, 239, 1)

    window = FeatureGroups.stack([day, day], dim=0)
    masked = loader.transform_feature_window(window, target_ti=100000, stage="train")
    assert torch.isfinite(masked["5m"][0]).all()
    assert torch.isnan(masked["5m"][-1, :, 6:, :]).all()
    assert torch.isfinite(masked["5m"][-1, :, :6, :]).all()
    assert torch.isnan(masked["1m"][-1, :, 26:, :]).all()
    assert torch.isfinite(masked["1m"][-1, :, :26, :]).all()
    assert torch.isfinite(masked["1d"]).all()


@pytest.mark.parametrize(
    ("target_ti", "expected_5m", "expected_1m"),
    (
        (93500, 1, 1),
        (94000, 2, 6),
        (100000, 6, 26),
        (113000, 24, 116),
        (130500, 25, 121),
        (150000, 48, 236),
    ),
)
def test_available_bar_count_uses_previous_target_bar_cutoff(
    target_ti, expected_5m, expected_1m
):
    assert available_bar_count("5m", "5m", target_ti) == expected_5m
    assert available_bar_count("1m", "5m", target_ti) == expected_1m


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
