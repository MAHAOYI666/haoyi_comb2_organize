from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from comb2 import ComboDataLoader, ComboTrainDataset, DataItem, LoadedSource, LoaderConfig
from comb2_simbase import IndexMask
from config import load_config


def write_source(path, values, dates, codes, times=()):
    path.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape(-1, len(codes))
    block = np.memmap(path / "0.ares", dtype=np.float64, mode="w+", shape=flat.shape)
    block[:] = flat
    block.flush()
    del block
    if times:
        index = np.full((len(dates) + 1, len(times) + 1), np.nan)
        index[0, 1:] = times
        index[1:, 0] = dates
        index[1:, 1:] = np.arange(len(dates) * len(times)).reshape(len(dates), len(times))
        levels = 2
    else:
        index = np.asarray(dates)
        levels = 1
    np.save(path / "index.npy", index)
    np.save(path / "columns.npy", np.asarray(codes, dtype=object))
    np.save(path / "meta.npy", np.array([
        np.float64, levels, *array.shape, len(dates), 1
    ], dtype=object))


class ResearchLoader(ComboDataLoader):
    def data_requirements(self):
        root = Path(self.config.cache_path)
        return (
            DataItem("daily", path=str(root / "daily"), delay=1),
            DataItem("bars", path=str(root / "bars")),
            DataItem("returns", path=str(root / "returns")),
        )

    def process_source(self, source, loaded):
        if source.name == "bars":
            before = torch.tensor(loaded.times) < self.current_ti
            assert before.any()
            return replace(loaded, values=loaded.values[:, :, before, :].mean(2), times=())
        return super().process_source(source, loaded)

    def model_input_sources(self):
        return "daily", "bars"

    def model_target(self):
        return "returns", "returns"

    def preprocess_features(self, values, ds, ti):
        return values.to(self.dtype)

    def preprocess_target(self, values, valid_mask, ds, ti):
        return values.to(self.dtype), valid_mask

    def gen_raw_target(self, ds, ti):
        self.set_current_ti(ti)
        endpoint = self.didx2date(self.date2didx(ds) + 1)
        return self.source_field(*self.model_target(), endpoint, endpoint)[0].float()


@pytest.fixture
def sources(tmp_path):
    mask = IndexMask()
    dates = tuple(map(int, mask.intv_trade_day(20240102, 20240112)))
    codes = tuple(str(c).zfill(6) for c in mask.code)
    times = (93000, 93500, 94000, 100000, 103000, 110000)
    stock = np.arange(len(codes), dtype=float) / 10000
    daily = np.arange(len(dates))[:, None] * 10 + stock
    bars = daily[:, None, :] + np.arange(len(times))[None, :, None]
    returns = daily / 1000
    write_source(tmp_path / "daily", daily, dates, codes)
    write_source(tmp_path / "bars", bars, dates, codes, times)
    write_source(tmp_path / "returns", returns, dates, codes)
    return tmp_path, dates, codes, times, daily, bars, returns


def test_memmap_delay_reduction_same_time_window_and_target(sources):
    root, dates, codes, times, daily, bars, returns = sources
    loader = ResearchLoader(LoaderConfig(
        cache_path=str(root), data_start_ds=dates[1], dtype=torch.float32,
        sample_times=(100000, 110000), registry_cache_days=4,
    ))
    dataset = ComboTrainDataset(loader, end_ds=dates[-2], ndays=6, ts_days=3)
    _, ds, ti, x, y, w = dataset[0]
    assert ti == 100000 and isinstance(x, torch.Tensor)
    row = dates.index(ds)
    assert x.shape == (3, len(codes), 2)
    np.testing.assert_allclose(x[:, :, 0], daily[row-3:row], atol=1.e-5)
    np.testing.assert_allclose(x[:, :, 1], bars[row-2:row+1, :3].mean(1), atol=1.e-5)
    np.testing.assert_allclose(y, returns[row+1], atol=1.e-7)
    assert w.all()
    _, ds2, ti2, x2, _, _ = dataset[1]
    assert ds2 == ds and ti2 == 110000
    np.testing.assert_allclose(x2[:, :, 1], bars[row-2:row+1, :5].mean(1), atol=1.e-5)
    torch.testing.assert_close(loader.load_feature_window(ds, 3, ti2), x2)
    assert len(loader._features) <= 4
    assert all(len(cache) <= 4 for cache in loader.registry.processed_cache.values())


def test_real_update_refreshes_only_new_snapshot_and_keeps_source_precision(sources):
    root, dates, codes, times, daily, bars, returns = sources
    loader = ResearchLoader(LoaderConfig(cache_path=str(root), data_start_ds=dates[1], dtype=torch.float32))
    ds = dates[3]
    before = loader.gen_feature(ds, 100000).clone()
    mmap = np.memmap(root / "bars" / "0.ares", dtype=np.float64, mode="r+", shape=bars.shape)
    mmap[3, 4:, :] += 1000
    mmap.flush()
    loader.set_point(ds, 100000, refresh=True)
    torch.testing.assert_close(loader.gen_feature(ds), before)
    mmap[3, :3, :] += 5
    mmap.flush()
    loader.set_point(ds, 100000, refresh=True)
    after = loader.gen_feature(ds)
    torch.testing.assert_close(after[:, 1], before[:, 1] + 5)
    raw = loader.source_field("returns", "returns", ds, ds)
    assert raw.dtype == torch.float64


def test_unreduced_source_is_rejected(sources):
    root, dates, *_ = sources
    class Unreduced(ResearchLoader):
        def process_source(self, source, loaded):
            return loaded
    loader = Unreduced(LoaderConfig(cache_path=str(root), data_start_ds=dates[1]))
    with pytest.raises(AssertionError, match="reduce"):
        loader.gen_feature(dates[2], 100000)


def test_generated_starter_is_array_based(tmp_path):
    import comboHelloWorld
    model_path = tmp_path / "Model.py"
    model_path.write_text(comboHelloWorld.MODEL_TEMPLATE)
    spec = importlib.util.spec_from_file_location("starter_array_test", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert issubclass(module.ResearchLoader, ComboDataLoader)
    assert "FeatureGroups" not in comboHelloWorld.MODEL_TEMPLATE
    config_path = tmp_path / "config.xml"
    config_path.write_text(comboHelloWorld.CONFIG_TEMPLATE)
    config = load_config(str(config_path))
    assert config["combo"]["runtime"]["sample_times"] == (100000,)
    assert config["combo"]["paths"]["research_loader_path"] == str(model_path)
    assert "data_items" not in config["combo"]["loader"]


def test_real_model_fit_predict_and_checkpoint_on_array_dataset(sources, tmp_path):
    root, dates, *_ = sources
    import comboHelloWorld
    module_name = "starter_fit_test"
    path = tmp_path / "research.py"
    path.write_text(comboHelloWorld.MODEL_TEMPLATE)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = ResearchLoader(LoaderConfig(
        cache_path=str(root), data_start_ds=dates[1], dtype=torch.float32,
        sample_times=(100000, 110000),
    ))
    data = ComboTrainDataset(
        loader, dates[-2], 5, ts_days=2,
        validinsts=torch.arange(20),
    )
    config = dict(dtype=torch.float32, tsDays=2, num_features=2, sample_times=(100000,110000),
                  device="cpu", epochs=1, batch_size=2, hidden_size=8, fc_size=8, dropout=0.)
    model = module.ResearchModel(config).fit(data)
    _, ds, ti, x, _, _ = data[0]
    prediction = model.predict(x, di=ds, ti=ti)
    assert prediction.shape == (20,) and torch.isfinite(prediction).all()
    checkpoint = tmp_path / "model"
    model.save(checkpoint)
    restored = module.ResearchModel(config).load(checkpoint)
    torch.testing.assert_close(restored.predict(x, di=ds, ti=ti), prediction)


def test_reduced_source_ops_and_dependency_delay_use_logical_dates(sources):
    from comb2 import OpSpec
    root, dates, codes, times, daily, bars, _ = sources
    class WithOps(ResearchLoader):
        def data_requirements(self):
            values = list(super().data_requirements())
            values[0] = replace(values[0], ops=(OpSpec("rolling_mean", {"window": 3}),))
            values[1] = replace(values[1], ops=(OpSpec("neut(daily)", {}),))
            return tuple(values)
    loader = WithOps(LoaderConfig(
        cache_path=str(root), data_start_ds=dates[1],
        registry_cache_days=8, dtype=torch.float32,
    ))
    ds = dates[6]
    values = loader.gen_feature(ds, 100000)
    np.testing.assert_allclose(values[:, 0], daily[3:6].mean(0), atol=1.e-5)
    assert float(values[:, 1].abs().max()) < 1.e-4


def test_lightgbm_uses_the_real_array_dataset(sources):
    root, dates, *_ = sources
    path = Path(__file__).resolve().parents[1] / "eg-lgbm" / "model.py"
    spec = importlib.util.spec_from_file_location("lgbm_array_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = ResearchLoader(LoaderConfig(
        cache_path=str(root), data_start_ds=dates[1], dtype=torch.float32,
        sample_times=(100000,110000),
    ))
    data = ComboTrainDataset(loader, dates[-2], 5, ts_days=2, validinsts=torch.arange(30))
    model = module.ResearchModel(dict(
        dtype=torch.float32, tsDays=2, num_features=2, epochs=2,
        min_data_in_leaf=2, num_threads=1,
    )).fit(data)
    _, ds, ti, x, _, _ = data[0]
    assert model.predict(x, di=ds, ti=ti).shape == (30,)


def test_feature_compression_does_not_quantize_target_sources(sources):
    root, dates, *_ = sources
    loader = ResearchLoader(LoaderConfig(
        cache_path=str(root), data_start_ds=dates[1], compression="fp4",
    ))
    feature = loader.gen_feature(dates[3], 100000)
    assert isinstance(feature, torch.Tensor) and feature.shape[-1] == 2
    storage, meta = loader._features[(dates[3],100000)]
    assert storage.dtype == torch.uint8
    assert loader.source_field("returns","returns", dates[3],dates[3]).dtype == torch.float64


def test_memmap_partial_current_day_does_not_require_future_bars(sources):
    root, dates, codes, times, daily, bars, _ = sources
    index_path = root / "bars" / "index.npy"
    index = np.load(index_path)
    index[-1, 4:] = np.nan
    np.save(index_path, index)
    loader = ResearchLoader(LoaderConfig(
        cache_path=str(root), data_start_ds=dates[1], dtype=torch.float32,
    ))
    values = loader.gen_feature(dates[-1], 100000)
    np.testing.assert_allclose(values[:, 1], bars[-1, :3].mean(0), atol=1.e-5)
