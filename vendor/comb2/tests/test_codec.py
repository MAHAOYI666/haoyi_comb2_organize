from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import pandas as pd
import torch
import random

from comb2.codec import FP4Codec, FP4_VALUES, FP8Codec, PassthroughCodec, build_codec
from comb2.DataLoader import ComboDataLoader, FeatureGroups, LoaderConfig, MemmapMaskSource
from comb2.DataRegistry import CANONICAL_BAR_TIMES, DataItem, DataRegistry, OpSpec, Universe
from comb2.op_utils import cs_zscore, nan_to_num, nanmean, nanstd, normalize_by_max_abs, rank, truncate, winsorize_by_quantile
from comb2_simbase.cache_layout import (
    BASE_UNIVERSE_MASK_NAME,
    FILTERED_MASK_NAME,
    VALID_MASK_NAME,
    ashare_cache_path,
    stock_mask_path,
)


FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)


def test_nonempty_missing_mask_path_fails_fast(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="mask data not found"):
        MemmapMaskSource(str(tmp_path / "missing-mask"))


def test_train_delay_zero_does_not_add_framework_offset() -> None:
    from comb2.ComboBase import ComboBase

    combo = ComboBase.__new__(ComboBase)
    combo.trainDelay = 0
    combo._prev_date = lambda ds, offset=1: (ds, offset)

    assert combo._train_target_ds(20200102) == (20200102, 0)


def _write_memmaper2_fixture(base, data, index, columns, chunk_size=2):
    base.mkdir(parents=True, exist_ok=True)
    chunks = 0
    for chunks, start in enumerate(range(0, data.shape[0], chunk_size)):
        block = np.memmap(base / f"{chunks}.ares", dtype=np.float64, mode="w+", shape=data[start : start + chunk_size].shape)
        block[:] = data[start : start + chunk_size]
        block.flush()
        del block
    meta = np.array([np.float64, 1, data.shape[0], data.shape[1], chunk_size, chunks + 1], dtype=object)
    np.save(base / "meta.npy", meta)
    np.save(base / "index.npy", index.astype(np.int64))
    np.save(base / "columns.npy", columns)


def _write_memmaper2_3d_fixture(base, data, dates, times, columns, chunk_size=1):
    base.mkdir(parents=True, exist_ok=True)
    idx = np.arange(data.shape[0], dtype=float).reshape(len(dates), len(times))
    chunks = 0
    for chunks, start in enumerate(range(0, len(dates), chunk_size)):
        start_row = int(np.nanmin(idx[start]))
        end_row = int(np.nanmax(idx[min(start + chunk_size, len(dates)) - 1]))
        block = np.memmap(base / f"{chunks}.ares", dtype=np.float64, mode="w+", shape=data[start_row : end_row + 1].shape)
        block[:] = data[start_row : end_row + 1]
        block.flush()
        del block
    meta = np.array([np.float64, 2, data.shape[0], data.shape[1], chunk_size, chunks + 1], dtype=object)
    full_index = np.full((len(dates) + 1, len(times) + 1), np.nan, dtype=np.float64)
    full_index[0, 1:] = times.astype(float)
    full_index[1:, 0] = dates.astype(float)
    full_index[1:, 1:] = idx
    np.save(base / "meta.npy", meta)
    np.save(base / "index.npy", full_index)
    np.save(base / "columns.npy", columns)


def _finite_abs_max(x: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return torch.tensor(float("nan"), dtype=x.dtype)
    return torch.max(finite)


def _fp4_nearest(x: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(FP4_VALUES, dtype=torch.float32)
    distances = (x.to(torch.float32).unsqueeze(-1) - values).abs()
    return values[distances.argmin(dim=-1)]


def _encode_decode(codec, shape, logical_dtype, x, idx=slice(None), out_dtype=None):
    buf, meta = codec.allocate(shape, device="cpu", logical_dtype=logical_dtype)
    codec.encode_into(buf, meta, idx, x)
    return codec.decode(buf, meta, idx, out_dtype=out_dtype), buf, meta


def _fp8_expected(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).to(dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_passthrough_bit_identical(dtype: torch.dtype) -> None:
    codec = PassthroughCodec(dtype)
    x = torch.tensor(
        [-2.0, -1.5, -0.25, 0.0, 0.25, 1.5, 2.0, 3.0, -3.0, 4.0, -4.0, 0.5],
        dtype=dtype,
    ).reshape(2, 2, 3)

    decoded, buf, meta = _encode_decode(codec, x.shape, dtype, x)

    assert codec.storage_dtype == dtype
    assert meta.storage_dtype == dtype
    assert decoded.dtype == dtype
    assert torch.equal(decoded, x)
    assert torch.equal(buf, x)


@pytest.mark.parametrize("logical_dtype", (torch.float16, torch.bfloat16))
def test_fp4_accepts_float32_input_for_low_precision_logical_dtype(logical_dtype: torch.dtype) -> None:
    codec = FP4Codec()
    shape = (2, 3, 5)
    x = torch.linspace(-4.5, 4.5, math.prod(shape), dtype=torch.float32).reshape(shape)

    decoded, _, _ = _encode_decode(codec, shape, logical_dtype, x)

    expected = _fp4_nearest(x).to(logical_dtype)
    assert decoded.shape == x.shape
    assert decoded.dtype == logical_dtype
    assert torch.equal(decoded, expected)


def test_fp4_rejects_float64_input() -> None:
    codec = FP4Codec()
    shape = (1, 2, 3)
    buf, meta = codec.allocate(shape, device="cpu", logical_dtype=torch.float16)
    with pytest.raises(TypeError, match="float64"):
        codec.encode_into(buf, meta, slice(None), torch.zeros(shape, dtype=torch.float64))


def test_fp8_rejects_float64_input_when_supported(cpu_float8_supported: bool) -> None:
    if not cpu_float8_supported:
        pytest.skip("CPU float8 cast is not supported in this PyTorch environment")
    codec = FP8Codec()
    shape = (1, 2, 3)
    buf, meta = codec.allocate(shape, device="cpu", logical_dtype=torch.float16)
    with pytest.raises(TypeError, match="float64"):
        codec.encode_into(buf, meta, slice(None), torch.zeros(shape, dtype=torch.float64))


def test_fp4_roundtrip_matches_nearest_quantization() -> None:
    torch.manual_seed(0)
    codec = FP4Codec()
    shape = (4, 3, 9)
    x = (torch.rand(shape, dtype=torch.float32) * 12.0 - 6.0).to(torch.float32)

    decoded, _, _ = _encode_decode(codec, shape, torch.float32, x)

    assert torch.equal(decoded, _fp4_nearest(x))


def test_fp4_saturates_out_of_range_values() -> None:
    codec = FP4Codec()
    x = torch.tensor([[-10.0, -5.0, 5.0, 10.0]], dtype=torch.float32)

    decoded, _, _ = _encode_decode(codec, x.shape, torch.float32, x)

    assert torch.equal(decoded, torch.tensor([[-5.0, -5.0, 5.0, 5.0]], dtype=torch.float32))


def test_fp4_boundary_values_decode_exactly() -> None:
    codec = FP4Codec()
    x = torch.tensor(FP4_VALUES, dtype=torch.float32).reshape(1, len(FP4_VALUES))

    decoded, _, _ = _encode_decode(codec, x.shape, torch.float32, x)

    assert torch.equal(decoded, x)


def test_fp4_does_not_infer_even_last_dim_from_code_15() -> None:
    codec = FP4Codec()
    shape = (3, 2, 2)
    x = torch.zeros(shape, dtype=torch.float32)

    decoded, _, _ = _encode_decode(codec, shape, torch.float32, x)

    assert decoded.shape == shape
    assert torch.equal(decoded, x)


@pytest.mark.parametrize("last_dim", (1, 2, 3, 127))
def test_fp4_decode_restores_original_last_dim(last_dim: int) -> None:
    codec = FP4Codec()
    shape = (2, 1, last_dim)
    x = torch.zeros(shape, dtype=torch.float32)

    decoded, _, meta = _encode_decode(codec, shape, torch.float32, x)

    assert meta.orig_last_dim == last_dim
    assert decoded.shape == shape
    assert torch.equal(decoded, x)


def test_fp4_batch_size_one_boundary() -> None:
    codec = FP4Codec()
    shape = (1, 4, 3)
    x = torch.zeros(shape, dtype=torch.float32)

    decoded, _, _ = _encode_decode(codec, shape, torch.float32, x)

    assert decoded.shape == shape
    assert torch.equal(decoded, x)


def test_decode_int_index_returns_single_slice() -> None:
    codec = FP4Codec()
    values = torch.tensor(FP4_VALUES, dtype=torch.float32)
    x = values[torch.arange(5 * 2 * 3).remainder(len(values))].reshape(5, 2, 3)
    decoded, buf, meta = _encode_decode(codec, x.shape, torch.float32, x)

    one = codec.decode(buf, meta, 3, out_dtype=torch.float32)

    assert torch.equal(decoded, x)
    assert one.shape == (2, 3)
    assert torch.equal(one, x[3])


def test_decode_slice_index_shape() -> None:
    codec = FP4Codec()
    x = torch.zeros((5, 2, 3), dtype=torch.float32)
    _, buf, meta = _encode_decode(codec, x.shape, torch.float32, x)

    window = codec.decode(buf, meta, slice(1, 4), out_dtype=torch.float32)

    assert window.shape == (3, 2, 3)
    assert torch.equal(window, x[1:4])


def test_decode_list_index_preserves_order() -> None:
    codec = FP4Codec()
    values = torch.tensor(FP4_VALUES, dtype=torch.float32)
    x = values[torch.arange(5 * 2 * 3).remainder(len(values))].reshape(5, 2, 3)
    _, buf, meta = _encode_decode(codec, x.shape, torch.float32, x)

    out = codec.decode(buf, meta, [3, 0, 2], out_dtype=torch.float32)

    assert torch.equal(out, x[[3, 0, 2]])


def test_decode_tensor_index_allows_repeats() -> None:
    codec = FP4Codec()
    values = torch.tensor(FP4_VALUES, dtype=torch.float32)
    x = values[torch.arange(5 * 2 * 3).remainder(len(values))].reshape(5, 2, 3)
    _, buf, meta = _encode_decode(codec, x.shape, torch.float32, x)
    idx = torch.tensor([1, 1, 2], dtype=torch.long)

    out = codec.decode(buf, meta, idx, out_dtype=torch.float32)

    assert torch.equal(out, x[idx])


def test_fp8_roundtrip_when_cpu_float8_cast_supported(cpu_float8_supported: bool) -> None:
    if not cpu_float8_supported:
        pytest.skip("CPU float8 cast is not supported in this PyTorch environment")
    codec = FP8Codec()
    shape = (1, 2, 4)
    x = torch.tensor([0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 3.0, -3.0], dtype=torch.float32).reshape(shape)

    decoded, _, _ = _encode_decode(codec, shape, torch.float16, x)
    expected = _fp8_expected(x, torch.float16)

    assert decoded.dtype == torch.float16
    assert torch.equal(decoded, expected)


@pytest.mark.parametrize(
    ("idx", "out_dtype"),
    (
        (3, torch.float32),
        (slice(1, 4), torch.float16),
        ([3, 0, 2], torch.float32),
        (torch.tensor([1, 1, 2], dtype=torch.long), torch.float32),
    ),
)
def test_fp8_decode_index_semantics_when_cpu_float8_cast_supported(
    cpu_float8_supported: bool,
    idx,
    out_dtype: torch.dtype,
) -> None:
    if not cpu_float8_supported:
        pytest.skip("CPU float8 cast is not supported in this PyTorch environment")
    codec = FP8Codec()
    shape = (5, 2, 3)
    x = torch.linspace(-3.5, 3.5, math.prod(shape), dtype=torch.float32).reshape(shape)
    buf, meta = codec.allocate(shape, device="cpu", logical_dtype=torch.float16)
    codec.encode_into(buf, meta, slice(None), x)

    decoded = codec.decode(buf, meta, idx, out_dtype=out_dtype)
    expected = _fp8_expected(x[idx], out_dtype)

    assert decoded.dtype == out_dtype
    assert decoded.shape == expected.shape
    assert torch.equal(decoded, expected)


def test_fp8_decode_uses_logical_dtype_by_default_when_cpu_float8_cast_supported(cpu_float8_supported: bool) -> None:
    if not cpu_float8_supported:
        pytest.skip("CPU float8 cast is not supported in this PyTorch environment")
    codec = FP8Codec()
    shape = (5, 2, 3)
    x = torch.linspace(-3.5, 3.5, math.prod(shape), dtype=torch.float32).reshape(shape)
    buf, meta = codec.allocate(shape, device="cpu", logical_dtype=torch.bfloat16)
    codec.encode_into(buf, meta, slice(None), x)

    decoded = codec.decode(buf, meta, [3, 0, 2])

    assert decoded.dtype == torch.bfloat16
    assert torch.equal(decoded, _fp8_expected(x[[3, 0, 2]], torch.bfloat16))


def test_fp8_raises_when_cpu_float8_cast_unsupported(cpu_float8_supported: bool) -> None:
    if cpu_float8_supported:
        pytest.skip("CPU float8 cast is supported in this PyTorch environment")
    with pytest.raises(RuntimeError):
        FP8Codec()


@pytest.mark.parametrize(
    ("dtype", "bytes_per_value"),
    ((torch.float16, 2), (torch.float32, 4), (torch.float64, 8), (torch.bfloat16, 2)),
)
def test_passthrough_storage_bytes_formula(dtype: torch.dtype, bytes_per_value: int) -> None:
    shape = (10, 100, 64)
    buf, _ = PassthroughCodec(dtype).allocate(shape, device="cpu", logical_dtype=dtype)

    assert buf.element_size() * buf.numel() == bytes_per_value * math.prod(shape)


def test_fp4_storage_bytes_formula() -> None:
    shape = (10, 100, 65)
    buf, _ = FP4Codec().allocate(shape, device="cpu", logical_dtype=torch.float16)

    assert buf.element_size() * buf.numel() == shape[0] * shape[1] * math.ceil(shape[2] / 2)


def test_fp8_storage_bytes_formula_when_supported(cpu_float8_supported: bool) -> None:
    if not cpu_float8_supported:
        pytest.skip("CPU float8 cast is not supported in this PyTorch environment")
    shape = (2, 3, 5)
    codec = FP8Codec()
    buf, meta = codec.allocate(shape, device="cpu", logical_dtype=torch.float16)

    assert codec.storage_dtype == torch.uint8
    assert meta.storage_dtype == torch.uint8
    assert buf.dtype == torch.uint8
    assert buf.element_size() * buf.numel() == 1 * math.prod(shape)


def test_build_codec_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported compression"):
        build_codec("garbage", torch.float16)


def test_build_codec_rejects_float64_for_compressed_codecs() -> None:
    with pytest.raises(TypeError, match="float64"):
        build_codec("fp4", torch.float64)
    with pytest.raises(TypeError, match="float64"):
        build_codec("fp8", torch.float64)


def test_config_accepts_data_compression_attribute(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        '<config><combo><data compression="fp4" dtype="float16" /></combo></config>',
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))

    assert load_config()["combo"]["loader"]["compression"] == "none"
    assert loaded["combo"]["loader"]["compression"] == "fp4"
    assert loaded["combo"]["loader"]["data_offset"] == 1024


def test_config_accepts_data_items_and_rejects_loader_node(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    alpha_path = tmp_path / "alpha.parquet"
    xml_path.write_text(
        f"""
        <config>
          <constants cache_path="data/Cache" />
          <combo>
            <data dtype="float16">
              <item name="factor_a" module="builtin.factorsim" path="factor_a" role="factor">
                <op name="truncate" min="-2" max="2" />
              </item>
              <item name="base_lgbm" module="builtin.alpha_parquet" path="{alpha_path}" role="factor" mode="read_dump">
                <op name="nan_to_num" value="0" />
              </item>
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    data_items = loaded["combo"]["loader"]["data_items"]

    assert len(data_items) == 2
    assert data_items[0]["role"] == "factor"
    assert data_items[0]["module"] == "builtin.factorsim"
    assert data_items[0]["path"] == str((tmp_path / "factor_a").resolve())
    assert data_items[0]["ops"][0]["params"] == {"min": -2, "max": 2}
    assert data_items[1]["module"] == "builtin.alpha_parquet"
    assert data_items[1]["path"] == str(alpha_path.resolve())

    legacy_xml_path = tmp_path / "legacy.xml"
    legacy_xml_path.write_text(
        """
        <config>
          <combo>
            <loader>
              <factor_paths>
                <path>factor_a</path>
              </factor_paths>
            </loader>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="<combo><loader> is no longer supported"):
        load_config(str(legacy_xml_path))


def test_config_accepts_builtin_factor_until_registry_resolve(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <constants cache_path="data/Cache" />
          <combo>
            <data>
              <item name="alpha.legacy" module="builtin.factor" path="legacy_factor" role="factor" />
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    item = loaded["combo"]["loader"]["data_items"][0]

    assert item["module"] == "builtin.factor"
    assert item["path"] == str((tmp_path / "legacy_factor").resolve())

    registry = DataRegistry(
        [item],
        universe=Universe(dates=(20200101,), codes=("000001",), dtype=torch.float32),
        data_start_ds=20200101,
        ashare_cache_path=str(ashare_cache_path(loaded["constants"]["cache_path"])),
        config_path=None,
    )

    with pytest.raises(ModuleNotFoundError, match="No module named 'builtin'"):
        registry._ensure_range(("alpha.legacy",), 20200101, 20200101)


def test_config_rejects_loader_node(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        '<config><combo><loader dtype="float16" /></combo></config>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="<combo><loader> is no longer supported"):
        load_config(str(xml_path))


def test_config_rejects_legacy_data_item_aliases(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <combo>
            <data>
              <item name="alpha.bad" source="builtin.factorsim" path="factor_a" role="factor" />
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported legacy attribute"):
        load_config(str(xml_path))

    with pytest.raises(ValueError, match="unsupported legacy key"):
        DataRegistry(
            [{"name": "alpha.bad", "source": "builtin.factorsim", "path": "factor_a", "role": "factor"}],
            universe=Universe(dates=(20200101,), codes=("000001",), dtype=torch.float32),
            data_start_ds=20200101,
            ashare_cache_path=None,
            config_path=None,
        )


def test_config_accepts_research_loader_and_dataset_paths(tmp_path) -> None:
    from config import load_config

    loader_path = tmp_path / "loader.py"
    dataset_path = tmp_path / "dataset.py"
    loader_path.write_text("class ResearchLoader: pass\n", encoding="utf-8")
    dataset_path.write_text("class ResearchDataset: pass\n", encoding="utf-8")
    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <combo>
            <paths model_path="model.py" research_loader_path="loader.py" research_dataset_path="dataset.py" />
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    paths = loaded["combo"]["paths"]

    assert paths["research_loader_path"] == str(loader_path.resolve())
    assert paths["research_dataset_path"] == str(dataset_path.resolve())


def test_config_accepts_combo_base_path(tmp_path) -> None:
    from config import load_config

    combo_base_path = tmp_path / "combo_base.py"
    combo_base_path.write_text("class ComboBase: pass\n", encoding="utf-8")
    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <combo>
            <paths model_path="model.py" combo_base_path="combo_base.py" />
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    paths = loaded["combo"]["paths"]

    assert paths["combo_base_path"] == str(combo_base_path.resolve())


def test_config_accepts_data_section_roles_and_ops(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <constants cache_path="data/Cache" />
          <combo>
            <data dtype="float32" compression="fp4" data_start_ds="20200101" data_offset="7">
              <import preset="barra" />
              <item name="alpha.turn20" module="builtin.factorsim" path="turn20" role="factor">
                <op name="neut(barra.size, barra.btop)" />
                <op name="cs_zscore" />
              </item>
              <item name="label.ret1" module="builtin.factorsim" role="label" />
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    loader = loaded["combo"]["loader"]
    items = loader["data_items"]

    assert loader["dtype"] == torch.float32
    assert loader["compression"] == "fp4"
    assert loader["data_offset"] == 7
    assert loader["data_presets"] == ("barra",)
    assert items[0]["name"] == "alpha.turn20"
    assert items[0]["path"] == str((tmp_path / "turn20").resolve())
    assert items[0]["role"] == "factor"
    assert items[0]["ops"][0]["name"] == "neut(barra.size, barra.btop)"
    assert items[1]["role"] == "label"


def test_config_accepts_runtime_deterministic_flag(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <combo>
            <runtime deterministic="true" />
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    assert loaded["combo"]["runtime"]["deterministic"] is True


def test_config_accepts_runtime_seed(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <combo>
            <runtime seed="42" />
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    assert loaded["combo"]["runtime"]["seed"] == 42


def test_combo_data_loader_applies_default_feature_global_preprocess(monkeypatch) -> None:
    from comb2 import DataLoader as data_loader_module

    class FakeMask:
        date = (20200101,)
        code = tuple(f"{idx + 1:06d}" for idx in range(30))

    monkeypatch.setattr(data_loader_module, "MASK", FakeMask())
    factor_values = torch.zeros((1, 30), dtype=torch.float32)
    factor_values[0, 0] = torch.nan
    factor_values[0, -1] = 100.0
    loader = ComboDataLoader(
        LoaderConfig(
            dtype=torch.float32,
            data_start_ds=20200101,
            data_offset=0,
            data_items=(
                DataItem(
                    name="alpha.raw",
                    module="test.tensor",
                    role="factor",
                    params={"values": factor_values},
                ),
            ),
        )
    )

    def load_tensor(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
        return item.params["values"]

    loader.registry.modules["test.tensor"] = load_tensor
    feature = loader._build_feature(20200101)

    raw_feature = factor_values[0].unsqueeze(-1)
    expected = cs_zscore(raw_feature.transpose(0, 1)).transpose(0, 1)
    expected = nan_to_num(truncate(expected, -4.0, 4.0), 0.0)

    assert feature.freqs == ("1d",)
    assert torch.allclose(feature["1d"], expected, atol=1e-6)
    assert feature["1d"][0, 0].item() == 0.0
    assert feature["1d"][-1, 0].item() == 4.0


def test_combo_data_loader_preprocess_feature_hook_can_replace_default(monkeypatch) -> None:
    from comb2 import DataLoader as data_loader_module

    class FakeMask:
        date = (20200101,)
        code = ("000001", "000002", "000003")

    class RankLoader(ComboDataLoader):
        def preprocess_daily_features(self, feature: torch.Tensor, ds: int) -> torch.Tensor:
            finite = torch.isfinite(feature)
            filled = torch.nan_to_num(feature, nan=-float("inf"))
            rank = torch.argsort(torch.argsort(filled, dim=0), dim=0).to(torch.float32)
            return torch.where(finite, rank, torch.zeros_like(rank)).to(self.dtype)

    monkeypatch.setattr(data_loader_module, "MASK", FakeMask())
    factor_values = torch.tensor([[3.0, 1.0, 2.0]], dtype=torch.float32)
    loader = RankLoader(
        LoaderConfig(
            dtype=torch.float32,
            data_start_ds=20200101,
            data_offset=0,
            data_items=(
                DataItem(
                    name="alpha.raw",
                    module="test.tensor",
                    role="factor",
                    params={"values": factor_values},
                ),
            ),
        )
    )

    def load_tensor(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
        return item.params["values"]

    loader.registry.modules["test.tensor"] = load_tensor

    feature = loader.gen_feature(20200101)
    assert feature.freqs == ("1d",)
    assert torch.equal(feature["1d"], torch.tensor([[2.0], [0.0], [1.0]]))


def test_config_to_loader_3d_factorsim_ops_pipeline_end_to_end(tmp_path, monkeypatch) -> None:
    from config import load_config
    from comb2 import DataLoader as data_loader_module

    class FakeMask:
        date = (20200101, 20200102, 20200103)
        code = ("000001", "000002", "000003")

    class IdentityPreprocessLoader(ComboDataLoader):
        def preprocess_feature_group(self, freq: str, feature: torch.Tensor, ds: int) -> torch.Tensor:
            return feature.to(self.dtype)

    monkeypatch.setattr(data_loader_module, "MASK", FakeMask())

    cache_dir = tmp_path / "cache_root" / "AshareCache" / "1m_Grid1mBar" / "Grid1mBar.close"
    dates = np.array([20200101, 20200102, 20200103])
    times = np.array(CANONICAL_BAR_TIMES["1m"], dtype=np.int64)
    columns = np.array(["000002", "000001", "000003"], dtype=object)
    rows = []
    for date_idx in range(len(dates)):
        for bar_idx in range(len(times)):
            rows.append([10_000 * date_idx + bar_idx + 10.0, 10_000 * date_idx + bar_idx + 1.0, 10_000 * date_idx + bar_idx + 100.0])
    data = np.asarray(rows, dtype=np.float64)
    _write_memmaper2_3d_fixture(cache_dir, data, dates, times, columns, chunk_size=1)
    for mask_name in (VALID_MASK_NAME, FILTERED_MASK_NAME, BASE_UNIVERSE_MASK_NAME):
        _write_memmaper2_fixture(
            stock_mask_path(tmp_path / "cache_root", mask_name),
            np.ones((len(dates), len(columns)), dtype=np.float64),
            dates,
            columns,
            chunk_size=1,
        )

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <constants cache_path="cache_root" />
              <combo>
                <data dtype="float32" data_start_ds="20200101" data_offset="0">
                  <item
                    name="alpha.minute_cube"
                    module="builtin.factorsim"
                    path="cache_root/AshareCache/1m_Grid1mBar/Grid1mBar.close"
                    role="factor"
                    freq="1m"
                  >
                    <op name="rolling_mean" window="2" axis="date" />
                  </item>
                </data>
              </combo>
        </config>
        """,
        encoding="utf-8",
    )

    parsed = load_config(str(xml_path))
    loader_fields = LoaderConfig.__dataclass_fields__
    loader_values = {key: value for key, value in parsed["combo"]["loader"].items() if key in loader_fields}
    loader_values["cache_path"] = parsed["constants"]["cache_path"]
    loader_config = LoaderConfig(**loader_values)
    loader = IdentityPreprocessLoader(loader_config)

    loader.set_current_ti(93100)
    loader.prefetch_features([20200102])
    first = loader.gen_feature(20200102)

    loader.set_current_ti(93200)
    second = loader.gen_feature(20200102)

    assert first.freqs == ("1m",)
    assert first["1m"].shape == (3, len(times), 1)
    expected_first_bar = torch.tensor([(1.0 + 10001.0) / 2.0, (10.0 + 10010.0) / 2.0, (100.0 + 10100.0) / 2.0])
    assert torch.equal(first["1m"][:, 0, 0], expected_first_bar)
    assert torch.equal(first["1m"], second["1m"])


def test_date_rolling_reloads_raw_lookback_after_processed_cache_hit() -> None:
    universe = Universe(dates=(20200101, 20200102, 20200103), codes=("000001", "000002"), dtype=torch.float32)
    registry = DataRegistry(
        [
            DataItem(
                name="alpha.roll",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("rolling_mean", {"window": 2, "axis": "date"}),),
                params={"values": torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=torch.float32)},
            )
        ],
        universe=universe,
        data_start_ds=20200101,
        ashare_cache_path=None,
        config_path=None,
    )

    load_calls = []

    def load_tensor(item: DataItem, current_registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
        del current_registry
        load_calls.append((start_ds, end_ds))
        lo = universe.date2idx(start_ds)
        hi = universe.date2idx(end_ds)
        return item.params["values"][lo : hi + 1]

    registry.modules["test.tensor"] = load_tensor

    registry._ensure_range(("alpha.roll",), 20200102, 20200102)
    assert load_calls == [(20200101, 20200102)]
    assert tuple(registry.processed_cache["alpha.roll"]) == (1,)
    assert torch.equal(
        registry.get_data("alpha.roll", 20200102, 20200102)[0],
        torch.tensor([2.0, 20.0]),
    )

    registry._ensure_range(("alpha.roll",), 20200103, 20200103)
    assert load_calls == [(20200101, 20200102), (20200102, 20200103)]
    loaded = registry.get_data("alpha.roll", 20200102, 20200103)
    assert torch.equal(loaded[0], torch.tensor([2.0, 20.0]))
    assert torch.equal(loaded[1], torch.tensor([4.0, 40.0]))


def test_op_utils_axis_aware_transforms() -> None:
    x = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [100.0, 30.0, 300.0],
        ],
        dtype=torch.float32,
    )

    z_axis0 = cs_zscore(x, axis=0)
    expected_z_axis0 = torch.tensor(
        [
            [-0.7178, -1.2247, -1.2247],
            [-0.6963, 0.0000, 0.0000],
            [1.4142, 1.2247, 1.2247],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(z_axis0, expected_z_axis0, atol=1e-4)

    winsorized = winsorize_by_quantile(x, lower_q=0.5, upper_q=0.5, axis=0)
    expected_winsorized = torch.tensor(
        [
            [2.0, 20.0, 200.0],
            [2.0, 20.0, 200.0],
            [2.0, 20.0, 200.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(winsorized, expected_winsorized)

    normalized = normalize_by_max_abs(x, axis=0)
    expected_normalized = torch.tensor(
        [
            [0.01, 10.0 / 30.0, 100.0 / 300.0],
            [0.02, 20.0 / 30.0, 200.0 / 300.0],
            [1.00, 1.00, 1.00],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(normalized, expected_normalized, atol=1e-6)


def test_combo_data_loader_preprocess_label_hook_can_replace_default(monkeypatch) -> None:
    from comb2 import DataLoader as data_loader_module

    class FakeMask:
        date = (20200101,)
        code = ("000001", "000002", "000003")

    class RawLabelLoader(ComboDataLoader):
        def preprocess_label(self, label_values: torch.Tensor, valid_mask: torch.Tensor, ds: int, ret_days: int = 1):
            return torch.where(valid_mask, label_values, torch.zeros_like(label_values)).to(self.dtype), valid_mask

    monkeypatch.setattr(data_loader_module, "MASK", FakeMask())
    label_values = torch.tensor([[0.1, float("nan"), -0.2]], dtype=torch.float32)
    loader = RawLabelLoader(
        LoaderConfig(
            dtype=torch.float32,
            data_start_ds=20200101,
            data_offset=0,
            data_items=(
                DataItem(
                    name="alpha.raw",
                    module="test.tensor",
                    role="factor",
                    params={"values": torch.ones((1, 3), dtype=torch.float32)},
                ),
                DataItem(
                    name="label.raw",
                    module="test.tensor",
                    role="label",
                    params={"values": label_values},
                ),
            ),
        )
    )

    def load_tensor(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
        return item.params["values"]

    loader.registry.modules["test.tensor"] = load_tensor
    y, w = loader.gen_label(20200101)

    assert torch.equal(y, torch.tensor([0.1, 0.0, -0.2], dtype=torch.float32))
    assert torch.equal(w, torch.tensor([True, True, True]))


def test_universe_from_mask_applies_data_offset() -> None:
    class FakeMask:
        date = (20100104, 20100105, 20100106, 20100107)
        code = ("000001", "000002")

    universe = Universe.from_mask(FakeMask(), torch.float32, data_offset=2)

    assert universe.dates == (20100106, 20100107)
    assert universe.idx2date(0) == 20100106
    assert universe.date2idx(20100106) == 0


def test_barra_preset_registers_all_cne5_styles(tmp_path) -> None:
    registry, _ = _fake_registry([], presets=("barra",), ashare_cache_path=str(tmp_path))
    expected_styles = {
        "beta",
        "btop",
        "earnyild",
        "growth",
        "industry",
        "leverage",
        "liquidty",
        "momentum",
        "resvol",
        "size",
        "sizenl",
    }

    assert {f"barra.{style}" for style in expected_styles}.issubset(set(registry.items))
    assert {
        Path(registry.items[f"barra.{style}"].path).relative_to(tmp_path).as_posix()
        for style in expected_styles
    } == {f"1d_BarraCNE5/BarraCNE5.{style.upper()}" for style in expected_styles}
    assert Path(registry.items["base.size"].path).name == "BarraCNE5.SIZE"
    assert Path(registry.items["base.btop"].path).name == "BarraCNE5.BTOP"
    assert registry._resolve_name("size") == "barra.size"
    assert registry._resolve_name("sizenl") == "barra.sizenl"


def test_config_imports_data_pack(tmp_path) -> None:
    from config import load_config

    pack_path = tmp_path / "pack.xml"
    pack_path.write_text(
        """
        <data-pack>
          <item name="factor.pack_a" module="builtin.factorsim" path="pack_a" role="factor" />
        </data-pack>
        """,
        encoding="utf-8",
    )
    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        f"""
        <config>
          <constants cache_path="data/Cache" />
          <combo>
            <data>
              <import path="{pack_path.name}" />
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    items = loaded["combo"]["loader"]["data_items"]

    assert len(items) == 1
    assert items[0]["name"] == "factor.pack_a"
    assert items[0]["path"] == str((tmp_path / "pack_a").resolve())


def test_config_import_can_filter_roles_from_data_pack(tmp_path) -> None:
    from config import load_config

    pack_path = tmp_path / "source_pack.xml"
    pack_path.write_text(
        """
        <data-pack>
          <import preset="barra" />
          <item name="factor.pack_a" module="builtin.factorsim" path="pack_a" role="factor">
            <op name="neut(size)" />
          </item>
          <item name="label.ret1" module="builtin.factorsim" path="vwap30_label1d" role="label" />
        </data-pack>
        """,
        encoding="utf-8",
    )
    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        f"""
        <config>
          <constants cache_path="data/Cache" />
          <combo>
            <data>
              <import path="{pack_path.name}" role="factor" />
              <item name="label.local" module="builtin.factorsim" path="vwap30_label1d" role="label" />
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    loader = loaded["combo"]["loader"]
    items = loader["data_items"]

    assert loader["data_presets"] == ("barra",)
    assert [item["name"] for item in items] == ["factor.pack_a", "label.local"]
    assert items[0]["ops"][0]["name"] == "neut(size)"


def test_config_resolves_builtin_label_to_ashare_cache_daily_label(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <constants cache_path="cache_root" />
          <combo>
            <data>
              <item name="label.local" module="builtin.factorsim" path="my_custom_label" role="label" />
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    items = loaded["combo"]["loader"]["data_items"]

    assert [item["name"] for item in items] == ["label.local"]
    assert items[0]["path"] == str((tmp_path / "cache_root" / "AshareCache" / "1d_DailyLabel" / "DailyLabel.my_custom_label").resolve())


def test_config_resolves_label1d_with_same_builtin_label_rule(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <constants cache_path="cache_root" />
          <combo>
            <data>
              <item name="label.local" module="builtin.factorsim" path="label1d" role="label" />
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    items = loaded["combo"]["loader"]["data_items"]

    assert [item["name"] for item in items] == ["label.local"]
    assert items[0]["path"] == str((tmp_path / "cache_root" / "AshareCache" / "1d_DailyLabel" / "DailyLabel.label1d").resolve())


def test_config_import_rejects_full_config(tmp_path) -> None:
    from config import load_config

    source_path = tmp_path / "source_config.xml"
    source_path.write_text(
        """
        <config>
          <combo>
            <data>
          <item name="factor.pack_a" module="builtin.factorsim" path="pack_a" role="factor" />
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )
    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        f"""
        <config>
          <combo>
            <data>
              <import path="{source_path.name}" role="factor" />
            </data>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data import root tag"):
        load_config(str(xml_path))


def test_data_registry_builtin_alpha_parquet_loads_date_and_reindexes_codes(tmp_path) -> None:
    alpha_path = tmp_path / "alpha.parquet"
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=[20200101, 20200102],
        columns=["1", "000002"],
    ).to_parquet(alpha_path)
    registry, _ = _fake_registry(
        [
            DataItem(
                name="alpha.base",
                module="builtin.alpha_parquet",
                path=str(alpha_path),
                role="factor",
                ops=(OpSpec("nan_to_num", {"value": 0.0}),),
            )
        ]
    )

    loaded = registry.get_data("alpha.base", 20200102, 20200102)[0]

    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, torch.tensor([3.0, 4.0, 0.0]))


def test_data_registry_builtin_alpha_parquet_supports_neut_op(tmp_path) -> None:
    alpha_path = tmp_path / "alpha.parquet"
    pd.DataFrame(
        [[4.0, 7.0, 10.0]],
        index=[20200102],
        columns=["000001", "000002", "000003"],
    ).to_parquet(alpha_path)
    registry, _ = _fake_registry(
        [
            DataItem(
                name="alpha.base",
                module="builtin.alpha_parquet",
                path=str(alpha_path),
                role="factor",
                ops=(OpSpec("neut(size, btop)", {}),),
            ),
            DataItem(
                name="barra.size",
                module="test.tensor",
                role="aux",
                params={"values": torch.tensor([[1.0, 2.0, 3.0]]).repeat(4, 1)},
            ),
            DataItem(
                name="barra.btop",
                module="test.tensor",
                role="aux",
                params={"values": torch.tensor([[2.0, 3.0, 4.0]]).repeat(4, 1)},
            ),
        ]
    )

    loaded = registry.get_data("alpha.base", 20200102, 20200102)[0]

    assert loaded.dtype == torch.float32
    assert _finite_abs_max(loaded) < 1e-4


def test_data_registry_neut_supports_ratio() -> None:
    y = torch.tensor([[4.0, 7.0, 10.0], [5.0, 8.0, 11.0], [6.0, 9.0, 12.0], [7.0, 10.0, 13.0]])
    size = torch.tensor([[1.0, 2.0, 3.0]]).repeat(4, 1)
    registry, calls = _fake_registry(
        [
            DataItem(
                name="alpha.neut",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("neut(size, 0.5)", {}),),
                params={"values": y},
            ),
            DataItem(
                name="barra.size",
                module="test.tensor",
                role="aux",
                params={"values": size},
            ),
        ]
    )

    registry._ensure_range(("alpha.neut",), 20200101, 20200101)
    data = registry.get_data("alpha.neut")

    assert torch.allclose(data[0], y[0] * 0.5, atol=1e-4)
    assert calls == [("barra.size", 20200101, 20200101), ("alpha.neut", 20200101, 20200101)]


def test_data_registry_neut_supports_array_string_dependencies_and_ratio() -> None:
    y = torch.tensor([[4.0, 7.0, 10.0], [5.0, 8.0, 11.0], [6.0, 9.0, 12.0], [7.0, 10.0, 13.0]])
    registry, calls = _fake_registry(
        [
            DataItem(
                name="alpha.neut",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("neut(['size', 'btop'], 0.5)", {}),),
                params={"values": y},
            ),
            DataItem(
                name="barra.size",
                module="test.tensor",
                role="aux",
                params={"values": torch.tensor([[1.0, 2.0, 3.0]]).repeat(4, 1)},
            ),
            DataItem(
                name="barra.btop",
                module="test.tensor",
                role="aux",
                params={"values": torch.tensor([[2.0, 3.0, 4.0]]).repeat(4, 1)},
            ),
        ]
    )

    registry._ensure_range(("alpha.neut",), 20200101, 20200101)
    data = registry.get_data("alpha.neut")

    assert torch.allclose(data[0], y[0] * 0.5, atol=1e-4)
    assert calls == [
        ("barra.size", 20200101, 20200101),
        ("barra.btop", 20200101, 20200101),
        ("alpha.neut", 20200101, 20200101),
    ]


def test_data_registry_neut_supports_per_dependency_ratio_array() -> None:
    size = torch.tensor([[1.0, 0.0, -1.0]]).repeat(4, 1)
    btop = torch.tensor([[0.0, 1.0, -1.0]]).repeat(4, 1)
    y = 2.0 * size + 3.0 * btop
    registry, calls = _fake_registry(
        [
            DataItem(
                name="alpha.neut",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("neut(['size', 'btop'], [0.5, 1.0])", {}),),
                params={"values": y},
            ),
            DataItem(
                name="barra.size",
                module="test.tensor",
                role="aux",
                params={"values": size},
            ),
            DataItem(
                name="barra.btop",
                module="test.tensor",
                role="aux",
                params={"values": btop},
            ),
        ]
    )

    registry._ensure_range(("alpha.neut",), 20200101, 20200101)
    data = registry.get_data("alpha.neut")

    assert torch.allclose(data[0], torch.tensor([1.0, 0.0, -1.0]), atol=1e-4)
    assert calls == [
        ("barra.size", 20200101, 20200101),
        ("barra.btop", 20200101, 20200101),
        ("alpha.neut", 20200101, 20200101),
    ]


def _fake_registry(
    items: list[DataItem],
    presets: tuple[str, ...] = (),
    verbose: bool = False,
    ashare_cache_path: str | None = None,
) -> tuple[DataRegistry, list[tuple[str, int, int]]]:
    universe = Universe(
        dates=(20200101, 20200102, 20200103, 20200106),
        codes=("000001", "000002", "000003"),
        dtype=torch.float32,
    )
    calls: list[tuple[str, int, int]] = []
    registry = DataRegistry(
        items,
        universe=universe,
        data_start_ds=20200101,
        ashare_cache_path=ashare_cache_path,
        config_path=None,
        presets=presets,
        verbose=verbose,
    )

    def load_tensor(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
        calls.append((item.name, start_ds, end_ds))
        values = torch.as_tensor(item.params["values"], dtype=registry.universe.dtype)
        lo = registry.universe.date2idx(start_ds)
        hi = registry.universe.date2idx(end_ds)
        return values[lo : hi + 1]

    registry.modules["test.tensor"] = load_tensor
    return registry, calls


def test_data_registry_returns_load_timing_stats() -> None:
    registry, _ = _fake_registry(
        [
            DataItem(
                name="alpha.base",
                module="test.tensor",
                role="factor",
                params={"values": torch.ones((4, 3), dtype=torch.float32)},
            )
        ],
        verbose=True,
    )

    stats = registry._ensure_range(("alpha.base",), 20200101, 20200102)

    assert stats.request_days == 2
    assert stats.raw_points == 2
    assert stats.raw_chunks == 1
    assert stats.ops_points == 0
    assert stats.ops_items == 0
    assert stats.raw_time >= 0.0
    assert stats.total_time >= stats.raw_time


def test_data_registry_get_data_returns_requested_processed_matrix() -> None:
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]
    )
    registry, calls = _fake_registry(
        [
            DataItem(
                name="alpha.raw",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("rolling_mean", {"window": 2}),),
                params={"values": values},
            )
        ]
    )

    data = registry.get_data("alpha.raw", 20200102, 20200106)
    registry._ensure_range(("alpha.raw",), 20200102, 20200106)

    assert data.shape == (3, 3)
    assert torch.equal(data[0], torch.tensor([1.5, 3.0, 4.5]))
    assert torch.equal(data[1], torch.tensor([2.5, 5.0, 7.5]))
    assert torch.equal(data[2], torch.tensor([3.5, 7.0, 10.5]))
    assert calls == [("alpha.raw", 20200101, 20200106)]


def test_data_registry_rolling_mean_uses_history_window() -> None:
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 6.0, 9.0],
            [5.0, 10.0, 15.0],
            [7.0, 14.0, 21.0],
        ]
    )
    registry, _ = _fake_registry(
        [
            DataItem(
                name="alpha.mean",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("rolling_mean", {"window": 2}),),
                params={"values": values},
            )
        ]
    )

    data = registry.get_data("alpha.mean", 20200102, 20200106)

    assert torch.equal(data[0], torch.tensor([2.0, 4.0, 6.0]))
    assert torch.equal(data[1], torch.tensor([4.0, 8.0, 12.0]))
    assert torch.equal(data[2], torch.tensor([6.0, 12.0, 18.0]))


def test_data_registry_rolling_std_uses_history_window() -> None:
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 6.0, 9.0],
            [5.0, 10.0, 15.0],
            [7.0, 14.0, 21.0],
        ]
    )
    registry, _ = _fake_registry(
        [
            DataItem(
                name="alpha.std",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("rolling_std", {"window": 2}),),
                params={"values": values},
            )
        ]
    )

    data = registry.get_data("alpha.std", 20200102, 20200106)

    assert torch.equal(data[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(data[1], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(data[2], torch.tensor([1.0, 2.0, 3.0]))


def test_data_registry_axis_aware_ops_use_requested_axis() -> None:
    values = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [100.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=torch.float32,
    )
    registry, _ = _fake_registry(
        [
            DataItem(
                name="alpha.axis",
                module="test.tensor",
                role="factor",
                ops=(
                    OpSpec("winsorize_by_quantile", {"low": 0.5, "high": 0.5, "axis": 0}),
                    OpSpec("normalize_by_max_abs", {"axis": 0}),
                ),
                params={"values": values},
            )
        ]
    )

    registry._ensure_range(("alpha.axis",), 20200101, 20200106)
    data = registry.get_data("alpha.axis")
    expected = normalize_by_max_abs(winsorize_by_quantile(values, 0.5, 0.5, axis=0), axis=0)

    assert torch.allclose(data, expected, atol=1e-6)


def test_data_registry_positional_axis_respects_current_tensor_rank() -> None:
    values = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [4.0, 40.0, 400.0],
            [8.0, 80.0, 800.0],
        ],
        dtype=torch.float32,
    )
    registry, _ = _fake_registry(
        [
            DataItem(
                name="alpha.axis",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("rank", {"axis": 1}),),
                params={"values": values},
            )
        ]
    )

    registry._ensure_range(("alpha.axis",), 20200101, 20200106)

    assert torch.equal(registry.get_data("alpha.axis"), rank(values, dim=None, axis=1))


def test_data_registry_rank_is_whitelisted_and_reducers_are_rejected() -> None:
    values = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [100.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=torch.float32,
    )
    registry, _ = _fake_registry(
        [
            DataItem(
                name="alpha.rank",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("rank", {"axis": 0}),),
                params={"values": values},
            )
        ]
    )

    registry._ensure_range(("alpha.rank",), 20200101, 20200106)

    rank_data = registry.get_data("alpha.rank")

    expected_rank = rank(values, axis=0)

    assert torch.equal(rank_data, expected_rank)
    with pytest.raises(ValueError, match="unsupported data op: mean"):
        _fake_registry(
            [
                DataItem(
                    name="alpha.mean",
                    module="test.tensor",
                    role="factor",
                    ops=(OpSpec("mean", {"axis": "code"}),),
                    params={"values": values},
                )
            ]
        )
    with pytest.raises(ValueError, match="unsupported data op: std"):
        _fake_registry(
            [
                DataItem(
                    name="alpha.std",
                    module="test.tensor",
                    role="factor",
                    ops=(OpSpec("std", {"axis": "code"}),),
                    params={"values": values},
                )
            ]
        )


def test_data_registry_neut_resolves_data_dependencies_and_aliases() -> None:
    y = torch.tensor([[2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [4.0, 8.0, 12.0], [5.0, 10.0, 15.0]])
    size = torch.tensor([[1.0, 2.0, 3.0]]).repeat(4, 1)
    registry, calls = _fake_registry(
        [
            DataItem(
                name="alpha.neut",
                module="test.tensor",
                role="factor",
                ops=(OpSpec("neut(size)", {}),),
                params={"values": y},
            ),
            DataItem(
                name="barra.size",
                module="test.tensor",
                role="aux",
                params={"values": size},
            ),
        ]
    )

    registry._ensure_range(("alpha.neut",), 20200101, 20200101)
    data = registry.get_data("alpha.neut")

    assert _finite_abs_max(data[0]) < 1e-4
    assert calls == [("barra.size", 20200101, 20200101), ("alpha.neut", 20200101, 20200101)]


def test_data_registry_builtin_factorsim_2d_direct_load(tmp_path) -> None:
    cache_dir = tmp_path / "cache_root" / "AshareCache" / "1d_DailyKline" / "DailyKline.close"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = np.arange(12, dtype=np.float64).reshape(4, 3)
    index = np.array([20200101, 20200102, 20200103, 20200106])
    columns = np.array(["000001", "000002", "000003"], dtype=object)

    _write_memmaper2_fixture(cache_dir, data, index, columns, chunk_size=2)

    universe = Universe(dates=(20200101, 20200102, 20200103, 20200106), codes=("000001", "000002", "000003"), dtype=torch.float32)
    registry = DataRegistry(
        [
            DataItem(
                name="alpha.base",
                module="builtin.factorsim",
                path=str(cache_dir),
                role="factor",
            )
        ],
        universe=universe,
        data_start_ds=20200101,
        ashare_cache_path=str(tmp_path / "cache_root" / "AshareCache"),
        config_path=None,
    )

    loaded = registry.get_data("alpha.base", 20200102, 20200106)
    assert torch.equal(loaded[0], torch.tensor([3.0, 4.0, 5.0]))
    assert torch.equal(loaded[2], torch.tensor([9.0, 10.0, 11.0]))


def test_data_registry_builtin_factorsim_3d_preserves_cube_by_freq(tmp_path) -> None:
    cache_dir = tmp_path / "cache_root" / "AshareCache" / "1m_Grid1mBar" / "Grid1mBar.close"
    dates = np.array([20200102, 20200103])
    times = np.array(CANONICAL_BAR_TIMES["1m"], dtype=np.int64)
    columns = np.array(["000002", "000001", "000003"], dtype=object)
    rows = []
    for date_idx in range(len(dates)):
        for bar_idx in range(len(times)):
            rows.append([10_000 * date_idx + bar_idx + 10.0, 10_000 * date_idx + bar_idx + 1.0, 10_000 * date_idx + bar_idx + 100.0])
    data = np.asarray(rows, dtype=np.float64)
    _write_memmaper2_3d_fixture(cache_dir, data, dates, times, columns, chunk_size=1)

    universe = Universe(dates=(20200102, 20200103), codes=("000001", "000002", "000003"), dtype=torch.float32)
    registry = DataRegistry(
        [
            DataItem(
                name="alpha.base",
                module="builtin.factorsim",
                path=str(cache_dir),
                role="factor",
                params={"freq": "1m"},
            )
        ],
        universe=universe,
        data_start_ds=20200102,
        ashare_cache_path=str(tmp_path / "cache_root" / "AshareCache"),
        config_path=None,
    )

    registry._ensure_range(("alpha.base",), 20200102, 20200103)
    loaded = registry.get_data("alpha.base")

    assert loaded.shape == (2, len(times), 3)
    assert torch.equal(loaded[0, 0], torch.tensor([1.0, 10.0, 100.0]))
    assert torch.equal(loaded[1, -1], torch.tensor([10_000.0 + len(times), 10_000.0 + len(times) + 9.0, 10_000.0 + len(times) + 99.0]))


def test_data_registry_builtin_factorsim_3d_requires_canonical_time_axis(tmp_path) -> None:
    cache_dir = tmp_path / "cache_root" / "AshareCache" / "1m_Grid1mBar" / "Grid1mBar.close"
    data = np.arange(24, dtype=np.float64).reshape(8, 3)
    dates = np.array([20200102, 20200103])
    times = np.array([93000, 93100, 93200, 93300])
    columns = np.array(["000001", "000002", "000003"], dtype=object)
    _write_memmaper2_3d_fixture(cache_dir, data, dates, times, columns, chunk_size=1)

    universe = Universe(dates=(20200102, 20200103), codes=("000001", "000002", "000003"), dtype=torch.float32)
    registry = DataRegistry(
        [
            DataItem(
                name="alpha.base",
                module="builtin.factorsim",
                path=str(cache_dir),
                role="factor",
                params={"freq": "1m"},
            )
        ],
        universe=universe,
        data_start_ds=20200102,
        ashare_cache_path=str(tmp_path / "cache_root" / "AshareCache"),
        config_path=None,
    )

    with pytest.raises(ValueError, match="time axis does not match canonical 1m"):
        registry._ensure_range(("alpha.base",), 20200102, 20200103)


def test_factorsim_reader_3d_loads_full_canonical_cube(tmp_path) -> None:
    from comb2.DataRegistry import FactorsimReader

    cache_dir = tmp_path / "cache_root" / "AshareCache" / "5m_Intv5mBar" / "Intv5mBar.close"
    times = np.array(CANONICAL_BAR_TIMES["5m"], dtype=np.int64)
    data = np.repeat(np.arange(len(times), dtype=np.float64).reshape(-1, 1), 3, axis=1)
    dates = np.array([20200102])
    columns = np.array(["000001", "000002", "000003"], dtype=object)
    _write_memmaper2_3d_fixture(cache_dir, data, dates, times, columns, chunk_size=1)

    reader = FactorsimReader(str(cache_dir))
    universe = Universe(dates=(20200102,), codes=("000001", "000002", "000003"), dtype=torch.float32)

    cube = reader.load_intraday(20200102, 20200102, dtype=torch.float32, freq="5m", universe=universe)

    bar_idx = list(CANONICAL_BAR_TIMES["5m"]).index(100000)
    assert cube.shape == (1, len(times), 3)
    assert torch.equal(cube[0, bar_idx], torch.tensor([float(bar_idx)] * 3))


def test_factorsim_reader_3d_reindexes_columns_to_universe(tmp_path) -> None:
    from comb2.DataRegistry import FactorsimReader

    cache_dir = tmp_path / "cache_root" / "AshareCache" / "1m_Grid1mBar" / "Grid1mBar.close"
    dates = np.array([20200102])
    times = np.array(CANONICAL_BAR_TIMES["1m"], dtype=np.int64)
    data = np.asarray([[float(idx + 1), float(idx + 101), float(idx + 201)] for idx in range(len(times))], dtype=np.float64)
    columns = np.array(["000002", "000001", "000003"], dtype=object)
    _write_memmaper2_3d_fixture(cache_dir, data, dates, times, columns, chunk_size=1)

    reader = FactorsimReader(str(cache_dir))
    universe = Universe(dates=(20200102,), codes=("000001", "000002", "000003"), dtype=torch.float32)

    cube = reader.load_intraday(20200102, 20200102, dtype=torch.float32, freq="1m", universe=universe)

    assert torch.equal(cube[0, 0], torch.tensor([101.0, 1.0, 201.0]))


def test_data_registry_rejects_nbar_and_reducer_ops_for_3d_items(tmp_path) -> None:
    cache_dir = tmp_path / "cache_root" / "AshareCache" / "1m_Grid1mBar" / "Grid1mBar.close"
    universe = Universe(dates=(20200102, 20200103), codes=("000001", "000002", "000003"), dtype=torch.float32)
    common = {
        "universe": universe,
        "data_start_ds": 20200102,
        "ashare_cache_path": str(tmp_path / "cache_root" / "AshareCache"),
        "config_path": None,
    }
    with pytest.raises(ValueError, match="nbar"):
        DataRegistry(
            [
                DataItem(
                    name="alpha.bad_nbar",
                    module="builtin.factorsim",
                    path=str(cache_dir),
                    role="factor",
                    params={"freq": "1m", "nbar": 2},
                )
            ],
            **common,
        )
    with pytest.raises(ValueError, match="unsupported data op: mean"):
        DataRegistry(
            [
                DataItem(
                    name="alpha.bad_mean",
                    module="builtin.factorsim",
                    path=str(cache_dir),
                    role="factor",
                    ops=(OpSpec("mean", {"axis": "bar"}),),
                    params={"freq": "1m"},
                )
            ],
            **common,
        )


def test_data_registry_fails_when_3d_source_has_neither_nbar_nor_freq(tmp_path) -> None:
    cache_dir = tmp_path / "cache_root" / "AshareCache" / "1m_Grid1mBar" / "Grid1mBar.close"
    data = np.arange(24, dtype=np.float64).reshape(8, 3)
    dates = np.array([20200102, 20200103])
    times = np.array([93000, 93100, 93200, 93300])
    columns = np.array(["000001", "000002", "000003"], dtype=object)
    _write_memmaper2_3d_fixture(cache_dir, data, dates, times, columns, chunk_size=1)

    universe = Universe(dates=(20200102, 20200103), codes=("000001", "000002", "000003"), dtype=torch.float32)
    registry = DataRegistry(
        [
            DataItem(
                name="alpha.bad",
                module="builtin.factorsim",
                path=str(cache_dir),
                role="factor",
                params={},
            )
        ],
        universe=universe,
        data_start_ds=20200102,
        ashare_cache_path=str(tmp_path / "cache_root" / "AshareCache"),
        config_path=None,
    )

    with pytest.raises(ValueError, match="must be 2D"):
        registry._ensure_range(("alpha.bad",), 20200102, 20200103)


def test_factorsim_reader_3d_does_not_keep_source_cache(tmp_path) -> None:
    from comb2.DataRegistry import FactorsimReader

    cache_dir = tmp_path / "cache_root" / "AshareCache" / "1m_Grid1mBar" / "Grid1mBar.close"
    data = np.arange(24, dtype=np.float64).reshape(8, 3)
    dates = np.array([20200102, 20200103])
    times = np.array([93000, 93100, 93200, 93300])
    columns = np.array(["000001", "000002", "000003"], dtype=object)
    _write_memmaper2_3d_fixture(cache_dir, data, dates, times, columns, chunk_size=1)

    reader = FactorsimReader(str(cache_dir))
    day = reader._load_day_3d(20200102, dtype=torch.float32)

    assert day.shape == (4, 3)
    assert not hasattr(reader, "source_cache")


def test_combo_data_loader_current_ti_masks_only_window_tail(monkeypatch) -> None:
    from comb2 import DataLoader as data_loader_module

    class FakeMask:
        date = (20200101, 20200102)
        code = ("000001", "000002", "000003")

    monkeypatch.setattr(data_loader_module, "MASK", FakeMask())

    loader = ComboDataLoader(
        LoaderConfig(
            dtype=torch.float32,
            data_start_ds=20200101,
            data_offset=0,
            data_items=(
                DataItem(
                    name="alpha.daily",
                    module="test.daily",
                    role="factor",
                    params={},
                ),
                DataItem(
                    name="alpha.m5",
                    module="test.m5",
                    role="factor",
                    params={"freq": "5m"},
                ),
            ),
        )
    )

    def load_daily(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
        del item
        lo = registry.universe.date2idx(start_ds)
        hi = registry.universe.date2idx(end_ds)
        return torch.ones((hi - lo + 1, len(registry.universe.codes)), dtype=torch.float32)

    def load_m5(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
        del item
        lo = registry.universe.date2idx(start_ds)
        hi = registry.universe.date2idx(end_ds)
        rows = hi - lo + 1
        values = torch.arange(rows * len(CANONICAL_BAR_TIMES["5m"]) * len(registry.universe.codes), dtype=torch.float32)
        return values.reshape(rows, len(CANONICAL_BAR_TIMES["5m"]), len(registry.universe.codes))

    loader.registry.modules["test.daily"] = load_daily
    loader.registry.modules["test.m5"] = load_m5

    first = loader.gen_feature(20200102)
    loader.set_current_ti(100000)
    second = loader.gen_feature(20200102)
    assert torch.equal(first["5m"], second["5m"])

    window = FeatureGroups.stack([first, second], dim=0)
    masked = loader.transform_feature_window(window, stage="train")
    future_mask = torch.as_tensor(np.asarray(CANONICAL_BAR_TIMES["5m"]) > 100000)
    assert torch.isfinite(masked["5m"][0]).all()
    assert torch.isnan(masked["5m"][-1, :, future_mask, :]).all()
    assert torch.isfinite(masked["5m"][-1, :, ~future_mask, :]).all()
    assert torch.isfinite(masked["1d"]).all()


def test_combo_data_loader_prefetch_syncs_current_ti_before_registry_load(monkeypatch) -> None:
    from comb2 import DataLoader as data_loader_module

    class FakeMask:
        date = (20200101,)
        code = ("000001", "000002", "000003")

    monkeypatch.setattr(data_loader_module, "MASK", FakeMask())

    loader = ComboDataLoader(
        LoaderConfig(
            dtype=torch.float32,
            data_start_ds=20200101,
            data_offset=0,
            data_items=(
                DataItem(
                    name="alpha.raw",
                    module="test.tensor",
                    role="factor",
                    params={},
                ),
            ),
        )
    )

    class TiAwareRegistry:
        def __init__(self):
            self.current_ti = 150000
            self.seen_ti = None

        def set_current_ti(self, ti: int):
            self.current_ti = int(ti)

        def _ensure_range(self, names, start_ds, end_ds):
            self.seen_ti = self.current_ti
            return None

    registry = TiAwareRegistry()
    loader.registry = registry
    loader.factor_names = ("alpha.raw",)
    loader.set_current_ti(103000)

    loader.prefetch_features([20200101])

    assert registry.seen_ti == 103000


def test_combo_base_sets_random_seed_from_model_config(monkeypatch) -> None:
    from comb2.ComboBase import ComboBase

    calls = []

    monkeypatch.setattr(random, "seed", lambda v: calls.append(("random", v)))
    monkeypatch.setattr(np.random, "seed", lambda v: calls.append(("numpy", v)))
    monkeypatch.setattr(torch, "manual_seed", lambda v: calls.append(("torch", v)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class DummyLoader:
        def __init__(self, config):
            self.dtype = torch.float32
            self.num_features = 1
            self.freqs = ("1d",)
            self.num_features_by_freq = {"1d": 1}
            self.mask = type("M", (), {"code": ("000001",)})()
            self.codec = PassthroughCodec(torch.float32)

        def set_current_ti(self, ti: int):
            return None

        def feature_group_shapes(self, inst_count: int):
            return {"1d": (int(inst_count), 1)}

    class DummyCombo(ComboBase):
        def _load_research_model_class(self, model_path: str):
            class DummyModel:
                pass
            return DummyModel

        def _load_optional_research_class(self, path, *, class_name, base_cls, default_cls):
            return DummyLoader if class_name == "ResearchLoader" else default_cls

    node = type(
        "Node",
        (),
            {
                "model_path": __file__,
                "research_loader_path": None,
                "research_dataset_path": None,
                "snaptime": "exp",
                "snap_ti": None,
                "seed": 42,
                "deterministic": False,
            "livetrading": False,
            "trainDelay": 0,
            "retDays": 1,
            "tsDays": 2,
            "load_chunk_days": None,
            "model_smooth_rate": 0.7,
            "model_keep_num": 0,
            "max_train_days": 20,
            "checkpoint_root": "",
                "alpha_history": {},
                "loader_config": LoaderConfig(dtype=torch.float32, data_start_ds=20200101, data_items=(DataItem(name="alpha.raw", module="test.tensor", role="factor", params={}),)),
                "model_config": {},
            },
        )()

    DummyCombo(node)
    assert ("random", 42) in calls
    assert ("numpy", 42) in calls
    assert ("torch", 42) in calls


def test_combo_base_need_train_uses_only_train_calendar_and_exact_checkpoint(monkeypatch) -> None:
    from comb2.ComboBase import ComboBase

    class DummyLoader:
        def __init__(self, config):
            self.dtype = torch.float32
            self.num_features = 1
            self.freqs = ("1d",)
            self.num_features_by_freq = {"1d": 1}
            self.mask = type("M", (), {"code": ("000001",), "date": tuple(range(20200101, 20200180))})()
            self.codec = PassthroughCodec(torch.float32)

        def set_current_ti(self, ti: int):
            return None

        def feature_group_shapes(self, inst_count: int):
            return {"1d": (int(inst_count), 1)}

        def date2didx(self, ds: int):
            return int(ds) - 20200101

    class DummyCombo(ComboBase):
        def _load_research_model_class(self, model_path: str):
            class DummyModel:
                pass
            return DummyModel

        def _load_optional_research_class(self, path, *, class_name, base_cls, default_cls):
            return DummyLoader if class_name == "ResearchLoader" else default_cls

    node = type(
        "Node",
        (),
        {
            "model_path": __file__,
            "research_loader_path": None,
            "research_dataset_path": None,
            "snaptime": "exp",
            "snap_ti": None,
            "seed": None,
            "deterministic": False,
            "livetrading": False,
            "trainDelay": 0,
            "retDays": 1,
            "tsDays": 2,
            "load_chunk_days": None,
            "model_smooth_rate": 0.7,
            "model_keep_num": 1,
            "max_train_days": 20,
            "checkpoint_root": "/tmp/checkpoints",
            "alpha_history": {},
            "loader_config": LoaderConfig(
                dtype=torch.float32,
                data_start_ds=20200101,
                data_items=(DataItem(name="alpha.raw", module="test.tensor", role="factor", params={}),),
            ),
            "model_config": {},
        },
    )()

    combo = DummyCombo(node)
    monkeypatch.setattr(combo, "isTrainDay", lambda ds: True)
    monkeypatch.setattr(combo, "_prev_date", lambda ds, offset=1: ds)
    checkpoint_day = 20200110
    monkeypatch.setattr(combo, "LoadCheckpointModel", lambda save_dir, dt: checkpoint_day)

    assert combo.needTrain(20200120) is True
    assert combo.needTrain(20200150) is True

    checkpoint_day = 20200120
    assert combo.needTrain(20200120) is False

    monkeypatch.setattr(combo, "isTrainDay", lambda ds: False)
    assert combo.needTrain(20200150) is False


def test_transform_feature_window_hook_applies_to_train_and_predict(monkeypatch) -> None:
    from comb2 import DataLoader as data_loader_module
    from comb2.ComboBase import ComboBase
    from comb2.DataLoader import ComboTrainDataset

    class FakeMask:
        date = (20200101, 20200102, 20200103)
        code = ("000001", "000002")

    class HookLoader(ComboDataLoader):
        def __init__(self, config):
            super().__init__(config)
            self.transform_calls = []

        def preprocess_feature_group(self, freq: str, feature: torch.Tensor, ds: int) -> torch.Tensor:
            return torch.nan_to_num(feature, nan=0.0).to(self.dtype)

        def transform_feature_window(self, feature_window: FeatureGroups, *, stage: str) -> FeatureGroups:
            self.transform_calls.append((stage, tuple(feature_window["1d"].shape)))
            bump = 10.0 if stage == "train" else 20.0
            return FeatureGroups({freq: value.to(self.dtype) + bump for freq, value in feature_window.items()}, feature_window.freqs)

    class DummyModel:
        def __init__(self, config):
            self.config = config

        def predict(self, x_window):
            return x_window["1d"][-1, :, 0]

        def save(self, path_or_buffer):
            return None

        def load(self, path_or_buffer):
            return self

    class DummyCombo(ComboBase):
        def _load_research_model_class(self, model_path: str):
            return DummyModel

        def _load_optional_research_class(self, path, *, class_name, base_cls, default_cls):
            return HookLoader if class_name == "ResearchLoader" else default_cls

    monkeypatch.setattr(data_loader_module, "MASK", FakeMask())

    loader_config = LoaderConfig(
        dtype=torch.float32,
        data_start_ds=20200101,
        data_offset=0,
        data_items=(
            DataItem(name="alpha.raw", module="test.tensor", role="factor", params={}),
            DataItem(name="label.raw", module="test.tensor", role="label", params={}),
        ),
    )
    loader = HookLoader(loader_config)
    dates = (20200101, 20200102, 20200103)
    values = {
        "alpha.raw": {
            20200101: torch.tensor([1.0, 2.0], dtype=torch.float32),
            20200102: torch.tensor([2.0, 3.0], dtype=torch.float32),
            20200103: torch.tensor([4.0, 5.0], dtype=torch.float32),
        },
        "label.raw": {
            20200101: torch.tensor([0.1, 0.2], dtype=torch.float32),
            20200102: torch.tensor([0.2, 0.3], dtype=torch.float32),
            20200103: torch.tensor([0.3, 0.4], dtype=torch.float32),
        },
    }

    def load_tensor(item, registry, start_ds, end_ds):
        selected = [values[item.name][ds] for ds in dates if start_ds <= ds <= end_ds]
        return torch.stack(selected, dim=0)

    loader.registry.modules["test.tensor"] = load_tensor

    dataset = ComboTrainDataset(loader, end_ds=20200103, ndays=3, x_delay=1, ts_days=2)
    _, train_x, _, _ = dataset[0]
    assert loader.transform_calls == [("train", (2, 2, 1))]
    assert train_x["1d"][0, 1, 0].item() == 12.0

    node = type(
        "Node",
        (),
        {
            "model_path": __file__,
            "research_loader_path": None,
            "research_dataset_path": None,
            "snaptime": "exp",
            "snap_ti": None,
            "seed": None,
            "deterministic": False,
            "livetrading": False,
            "trainDelay": 0,
            "retDays": 1,
            "tsDays": 2,
            "load_chunk_days": None,
            "model_smooth_rate": 1.0,
            "model_keep_num": 0,
            "max_train_days": 20,
            "checkpoint_root": "",
            "alpha_history": {},
            "loader_config": loader_config,
            "model_config": {},
            "monitor": None,
            "alpha": torch.zeros(2, dtype=torch.float32),
        },
    )()

    combo = DummyCombo(node)
    combo.loader.registry.modules["test.tensor"] = load_tensor
    combo.model = DummyModel(combo._model_config())
    combo.GenComboPos(20200103)

    assert combo.loader.transform_calls == [("predict", (2, 2, 1))]
    assert torch.isclose(combo.node.alpha[1], torch.tensor(25.0))
