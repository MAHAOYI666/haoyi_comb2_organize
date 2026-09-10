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
from comb2_simbase.cache_layout import stock_mask_path, BASE_UNIVERSE_MASK_NAME, VALID_MASK_NAME, FILTERED_MASK_NAME
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
    for name in (BASE_UNIVERSE_MASK_NAME, VALID_MASK_NAME, FILTERED_MASK_NAME):
        write_source(stock_mask_path(tmp_path, name), np.ones_like(daily), dates, codes)
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
    import cProfile
    import pstats
    root, dates, *_ = sources
    loader = ResearchLoader(LoaderConfig(
        cache_path=str(root), data_start_ds=dates[1], compression="fp4",
    ))
    feature = loader.gen_feature(dates[3], 100000)
    assert isinstance(feature, torch.Tensor) and feature.shape[-1] == 2
    profile = cProfile.Profile()
    with profile:
        dataset = ComboTrainDataset(loader, dates[-2], 5, ts_days=2, codec=loader.codec)
    encodes = sum(v[1] for (_, _, name), v in pstats.Stats(profile).stats.items() if name == "_quantize")
    assert encodes == dataset.last_sample - dataset.start_didx + 1
    storage, meta = dataset.X[100000], dataset.X_meta[100000]
    assert storage.dtype == torch.uint8
    assert loader.source_field("returns","returns", dates[3],dates[3]).dtype == torch.float64


def test_default_masks_and_infinite_features_use_real_sources(sources):
    root, dates, codes, *_ = sources
    for stock, name in enumerate((BASE_UNIVERSE_MASK_NAME, VALID_MASK_NAME, FILTERED_MASK_NAME)):
        block = np.memmap(stock_mask_path(root, name) / "0.ares", dtype=np.float64,
                          mode="r+", shape=(len(dates), len(codes)))
        block[2, stock] = 0
        block.flush()
    loader = ResearchLoader(LoaderConfig(cache_path=str(root), data_start_ds=dates[1]))
    assert loader.gen_base_universe_mask(dates[3])[:4].tolist() == [False, True, True, True]
    assert loader.gen_valid_mask(dates[3])[:4].tolist() == [False, False, False, True]
    assert loader.gen_valid_mask(dates[2]).all()
    _, valid = loader.gen_target(dates[3], 100000)
    assert valid[:4].tolist() == [False, False, False, True]

    class StandardLoader(ResearchLoader):
        def preprocess_features(self, values, ds, ti):
            return ComboDataLoader.preprocess_features(self, values, ds, ti)

    block = np.memmap(root / "daily" / "0.ares", dtype=np.float64,
                      mode="r+", shape=(len(dates), len(codes)))
    block[2, 0] = np.inf
    block.flush()
    standard = StandardLoader(LoaderConfig(cache_path=str(root), data_start_ds=dates[1]))
    features = standard.gen_feature(dates[3], 100000)
    assert features[0, 0] == 0
    assert torch.isfinite(features).all()
    assert features[1:, 0].unique().numel() > 100


def test_registry_reads_only_missing_ranges_and_keeps_lru_days(sources):
    root, dates, *_ = sources
    loader = ResearchLoader(LoaderConfig(cache_path=str(root), data_start_ds=dates[1], registry_cache_days=8))
    registry = loader.registry
    registry.get_data("daily", dates[2], dates[7])
    stats = registry._ensure_range(("daily",), dates[1], dates[8])
    assert stats.raw_points == 2 and stats.raw_chunks == 2
    cache = registry.processed_cache["daily"]
    assert len(cache) == 8
    assert all(v.untyped_storage().nbytes() == v.numel() * v.element_size() for v in cache.values())
    assert all(v.dtype == torch.float64 for v in cache.values())
    cache.clear()
    registry.cache_days = 2
    for ds in (dates[1], dates[2], dates[1], dates[3]):
        registry.get_data("daily", ds, ds)
    assert list(cache) == [(loader.date2didx(ds), 100000) for ds in (dates[1], dates[3])]


def test_daily_array_reader_aligns_reordered_and_missing_stocks(tmp_path):
    from comb2.DataRegistry import DataRegistry, Universe
    dates = (20240102, 20240103)
    write_source(tmp_path / "daily", [[30.123456789, 10.], [31., 11.]], dates, ("000003", "000001"))
    registry = DataRegistry((DataItem("daily", path=str(tmp_path / "daily")),),
                            universe=Universe(dates, ("000001", "000002", "000003"), torch.float16),
                            data_start_ds=dates[0])
    values = registry.get_data("daily", *dates)[..., 0]
    torch.testing.assert_close(values, torch.tensor([[10., float("nan"), 30.123456789],
                                                   [11., float("nan"), 31.]], dtype=torch.float64), equal_nan=True)


@pytest.mark.parametrize("cache_days,compression", [(1, "none"), (4, "none"), (4, "fp4"), (4, "fp8")])
def test_real_prediction_reuses_windows_and_refreshes_live_source(sources, tmp_path, cache_days, compression):
    import cProfile
    import pstats
    import xml.etree.ElementTree as ET
    import comboHelloWorld
    import runCombo
    from comb2 import ComboBase

    root, dates, codes, times, *_ = sources
    rng = np.random.default_rng(42)
    write_source(root / "daily", rng.normal(size=(len(dates), len(codes))), dates, codes)
    write_source(root / "bars", rng.normal(size=(len(dates), len(times), len(codes))), dates, codes, times)
    write_source(root / "returns", rng.normal(size=(len(dates), len(codes))) / 100, dates, codes)
    path = tmp_path / "research.py"
    path.write_text(comboHelloWorld.MODEL_TEMPLATE.split("\nfrom pathlib import Path\n")[0] + '''
from pathlib import Path
from dataclasses import replace
class ResearchLoader(ComboDataLoader):
    def data_requirements(self):
        root = Path(self.config.cache_path)
        return (DataItem("daily", path=str(root / "daily"), delay=1),
                DataItem("bars", path=str(root / "bars")),
                DataItem("returns", path=str(root / "returns")))
    def model_input_sources(self):
        return ("daily", "bars")
    def model_target(self):
        return ("returns", "returns")
    def process_source(self, source, loaded):
        if source.name == "bars":
            before = torch.tensor(loaded.times) < self.current_ti
            return replace(loaded, values=loaded.values[:, :, before, :].mean(2), times=())
        return loaded
    def preprocess_features(self, values, ds, ti):
        return values.to(self.dtype)
    def transform_feature_window(self, values, *, target_ti, stage):
        return values - values.mean(dim=1, keepdim=True)
''')
    xml = ET.fromstring(comboHelloWorld.CONFIG_TEMPLATE)
    xml.find("constants").set("cache_path", str(root))
    xml.find("constants").set("output_root", str(tmp_path / "output"))
    xml.find("strategy").set("start_ds", str(dates[3]))
    xml.find("strategy").set("end_ds", str(dates[-1]))
    xml.find("strategy/optimizer").set("type", "opt2")
    for attr in ("model_path", "research_loader_path"):
        xml.find("combo/paths").set(attr, str(path))
    for attr, value in {"sample_times": "100000,110000", "tsDays": "2"}.items():
        xml.find("combo/runtime").set(attr, value)
    for attr, value in {"dtype": "float32", "data_start_ds": str(dates[1]),
                        "registry_cache_days": str(cache_days), "compression": compression}.items():
        xml.find("combo/loader").set(attr, value)
    for attr, value in {"device": "cpu", "epochs": "1", "hiddenSize": "8", "fcSize": "8", "batchSize": "2", "dropout": "0"}.items():
        xml.find("combo/model").set(attr, value)
    config_path = tmp_path / "config.xml"
    config_path.write_text(ET.tostring(xml, encoding="unicode"))
    combo = ComboBase(runCombo.Node(load_config(str(config_path))))
    dataset = ComboTrainDataset(combo.loader, dates[-2], 5, ts_days=2,
                               validinsts=torch.arange(20), codec=combo.loader.codec)
    combo.model = combo.research_model_cls(combo._model_config()).fit(dataset)
    for idx, expected_generations in ((0, 2), (0, 0), (1, 2), (2, 1), (3, 1)):
        _, ds, ti, x, _, _ = dataset[idx]
        expected = combo.model.predict(x, di=ds, ti=ti)
        with cProfile.Profile() as profile:
            prediction = combo.GenComboPos(ds, ti).clone()
        generations = sum(v[1] for (_, _, name), v in pstats.Stats(profile).stats.items() if name == "gen_feature")
        assert generations == expected_generations
        torch.testing.assert_close(prediction[:20], expected)
        assert torch.isnan(prediction[20:]).all()
    other = ComboTrainDataset(combo.loader, dates[-2], 5, ts_days=2,
                             validinsts=torch.arange(10, 30), codec=combo.loader.codec)
    combo.oldModel = combo.research_model_cls(combo._model_config()).fit(other)
    old_expected = combo.oldModel.predict(other[3][3], di=ds, ti=ti)
    mixed = combo.GenComboPos(ds, ti)
    torch.testing.assert_close(mixed[10:20], expected[10:20] * combo.model_smooth_rate
                               + old_expected[:10] * (1 - combo.model_smooth_rate))
    combo.oldModel = None
    before = combo._predict_feature_window[ti].clone()
    block = np.memmap(root / "bars" / "0.ares", dtype=np.float64, mode="r+",
                      shape=(len(dates), len(times), len(codes)))
    block[dates.index(ds), :3, 0] += 5
    block.flush()
    combo.livetrading = True
    combo.modelDir = None
    with cProfile.Profile() as profile:
        actual = combo.Combine(ds, ti).clone()
    generations = sum(v[1] for (_, _, name), v in pstats.Stats(profile).stats.items() if name == "gen_feature")
    assert generations == 1
    after = combo._predict_feature_window[ti]
    torch.testing.assert_close(before[:-1], after[:-1])
    assert not torch.equal(before[-1], after[-1])
    end = combo.loader.date2didx(ds)
    selected = torch.stack([combo.loader.gen_feature(combo.loader.didx2date(i), ti)[:20]
                            for i in range(end - 1, end + 1)])
    storage, meta = combo.loader.codec.allocate(tuple(selected.shape), "cpu", combo.loader.dtype)
    combo.loader.codec.encode_into(storage, meta, slice(None), selected)
    selected = combo.loader.codec.decode(storage, meta, slice(None))
    transformed = combo.loader.transform_feature_window(selected, target_ti=ti, stage="predict")
    torch.testing.assert_close(actual[:20], combo.model.predict(transformed, di=ds, ti=ti))


@pytest.mark.parametrize("compression", ["none", "fp4", "fp8"])
def test_preloaded_dataset_matches_windows_and_survives_source_removal(sources, compression):
    root, dates, *_ = sources

    class WindowLoader(ResearchLoader):
        def transform_feature_window(self, values, *, target_ti, stage):
            assert stage == "train" and self.current_ti == target_ti
            return values

    loader = WindowLoader(LoaderConfig(
        cache_path=str(root), data_start_ds=dates[1], dtype=torch.float32,
        sample_times=(100000, 110000), registry_cache_days=2, compression=compression,
    ))
    stocks = torch.tensor([7, 2, 19])
    data = ComboTrainDataset(
        loader, dates[-2], 6, ts_days=3, x_delay=2,
        validinsts=stocks, codec=loader.codec,
    )
    expected = []
    for idx in range(len(data)):
        ds, ti = data.sample_coordinates(idx)
        end = loader.date2didx(ds)
        values = torch.stack([
            loader.gen_feature(loader.didx2date(i), ti).index_select(0, stocks)
            for i in range(end - 2, end + 1)
        ])
        x = loader.transform_feature_window(values, target_ti=ti, stage="train")
        if compression == "fp4":
            x = torch.full_like(x, 5.)
        elif compression == "fp8":
            x = x.to(torch.float8_e4m3fn).to(loader.dtype)
        y, w = loader.gen_target(ds, ti, ret_days=2)
        expected.append((x, y[stocks], w[stocks]))
    for cache in loader.registry.processed_cache.values():
        cache.clear()
    loader.registry.module_cache.clear()
    for name in ("daily", "bars", "returns"):
        (root / name).rename(root / (name + "_unavailable"))
    for _ in range(2):
        for idx in torch.randperm(len(data)).tolist():
            sample = data[idx]
            assert sample[:3] == (idx, *data.sample_coordinates(idx))
            for actual, reference in zip(sample[3:], expected[idx]):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    sample = data[0]
    sample[3].fill_(-123)
    sample[4].zero_()
    sample[5].zero_()
    repeated = data[0]
    assert not repeated[4].any() and not repeated[5].any()
    if compression == "none":
        assert (repeated[3] == -123).all()
        assert (data[2][3][:-1] == -123).all()
    else:
        torch.testing.assert_close(repeated[3], expected[0][0], rtol=0, atol=0)
    for actual, reference in zip(data[1][3:], expected[1]):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


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
