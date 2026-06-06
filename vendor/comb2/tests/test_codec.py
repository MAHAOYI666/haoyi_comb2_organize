from __future__ import annotations

import math

import pytest
import pandas as pd
import torch

from src.codec import FP4Codec, FP4_VALUES, FP8Codec, PassthroughCodec, build_codec
from src.DataLoader import ComboDataLoader, LoaderConfig
from src.DataRegistry import DataItem, DataRegistry, OpSpec, Universe
from src.op_utils import cs_zscore, nan_to_num, truncate


FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)


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
          <constants factor_root="factors" />
          <combo>
            <data dtype="float16">
              <item name="factor_a" module="builtin.factor" path="factor_a" role="factor">
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
    assert data_items[0]["module"] == "builtin.factor"
    assert data_items[0]["path"].endswith("/factors/factor_a")
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
              <item name="alpha.bad" source="builtin.factor" path="factor_a" role="factor" />
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
            [{"name": "alpha.bad", "source": "builtin.factor", "path": "factor_a", "role": "factor"}],
            universe=Universe(dates=(20200101,), codes=("000001",), dtype=torch.float32),
            data_start_ds=20200101,
            ashare_data_path=None,
            factor_root=None,
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


def test_config_accepts_data_section_roles_and_ops(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        """
        <config>
          <constants factor_root="factors" />
          <combo>
            <data dtype="float32" compression="fp4" data_start_ds="20200101" data_offset="7">
              <import preset="barra" />
              <item name="alpha.turn20" module="builtin.factor" path="turn20" role="factor">
                <op name="neut(barra.size, barra.btop)" />
                <op name="cs_zscore" />
              </item>
              <item name="label.ret1" module="builtin.label" role="label" />
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
    assert items[0]["path"].endswith("/factors/turn20")
    assert items[0]["role"] == "factor"
    assert items[0]["ops"][0]["name"] == "neut(barra.size, barra.btop)"
    assert items[1]["role"] == "label"


def test_combo_data_loader_applies_default_feature_global_preprocess(monkeypatch) -> None:
    from src import DataLoader as data_loader_module

    class FakeMask:
        date = (20200101,)
        code = tuple(f"{idx + 1:06d}" for idx in range(30))

    class FakeCubeSource:
        feature_dim = 1

        def load_day(self, ds: int) -> torch.Tensor:
            cube = torch.arange(30, dtype=torch.float32).unsqueeze(-1)
            cube[1, 0] = torch.nan
            return cube

        def prefetch_days(self, days):
            return None

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
        ),
        cube_source=FakeCubeSource(),
    )

    def load_tensor(item: DataItem, registry: DataRegistry, start_ds: int, end_ds: int) -> torch.Tensor:
        return item.params["values"]

    loader.registry.modules["test.tensor"] = load_tensor
    feature = loader._build_feature(20200101)

    raw_feature = torch.cat([factor_values[0].unsqueeze(-1), FakeCubeSource().load_day(20200101)], dim=-1)
    expected = cs_zscore(raw_feature.transpose(0, 1)).transpose(0, 1)
    expected = nan_to_num(truncate(expected, -4.0, 4.0), 0.0)

    assert torch.allclose(feature, expected, atol=1e-6)
    assert feature[0, 0].item() == 0.0
    assert feature[1, 1].item() == 0.0
    assert feature[-1, 0].item() == 4.0


def test_combo_data_loader_preprocess_feature_hook_can_replace_default(monkeypatch) -> None:
    from src import DataLoader as data_loader_module

    class FakeMask:
        date = (20200101,)
        code = ("000001", "000002", "000003")

    class RankLoader(ComboDataLoader):
        def preprocess_feature(self, feature: torch.Tensor, ds: int) -> torch.Tensor:
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

    assert torch.equal(loader.gen_feature(20200101), torch.tensor([[2.0], [0.0], [1.0]]))


def test_combo_data_loader_preprocess_label_hook_can_replace_default(monkeypatch) -> None:
    from src import DataLoader as data_loader_module

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


def test_barra_preset_registers_all_cne5_styles() -> None:
    registry, _ = _fake_registry([], presets=("barra",))
    expected = {
        "barra.beta",
        "barra.btop",
        "barra.earnyild",
        "barra.growth",
        "barra.industry",
        "barra.leverage",
        "barra.liquidty",
        "barra.momentum",
        "barra.resvol",
        "barra.size",
        "barra.sizenl",
    }

    assert expected.issubset(set(registry.items))
    assert registry._resolve_name("size") == "barra.size"
    assert registry._resolve_name("sizenl") == "barra.sizenl"


def test_config_imports_data_pack(tmp_path) -> None:
    from config import load_config

    pack_path = tmp_path / "pack.xml"
    pack_path.write_text(
        """
        <data-pack>
          <item name="factor.pack_a" module="builtin.factor" path="pack_a" role="factor" />
        </data-pack>
        """,
        encoding="utf-8",
    )
    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        f"""
        <config>
          <constants factor_root="factors" />
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
    assert items[0]["path"].endswith("/factors/pack_a")


def test_config_import_can_filter_roles_from_data_pack(tmp_path) -> None:
    from config import load_config

    pack_path = tmp_path / "source_pack.xml"
    pack_path.write_text(
        """
        <data-pack>
          <import preset="barra" />
          <item name="factor.pack_a" module="builtin.factor" path="pack_a" role="factor">
            <op name="neut(size)" />
          </item>
          <item name="label.ret1" module="builtin.label" path="label1d" role="label" />
        </data-pack>
        """,
        encoding="utf-8",
    )
    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        f"""
        <config>
          <constants factor_root="factors" />
          <combo>
            <data>
              <import path="{pack_path.name}" role="factor" />
              <item name="label.local" module="builtin.label" path="label1d" role="label" />
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


def test_config_import_rejects_full_config(tmp_path) -> None:
    from config import load_config

    source_path = tmp_path / "source_config.xml"
    source_path.write_text(
        """
        <config>
          <combo>
            <data>
              <item name="factor.pack_a" module="builtin.factor" path="pack_a" role="factor" />
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

    registry._ensure_range(("alpha.base",), 20200102, 20200102)
    loaded = registry.get_data("alpha.base")[registry.universe.date2idx(20200102)]

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

    registry._ensure_range(("alpha.base",), 20200102, 20200102)
    loaded = registry.get_data("alpha.base")[registry.universe.date2idx(20200102)]

    assert loaded.dtype == torch.float32
    assert _finite_abs_max(loaded) < 1e-4


def _fake_registry(
    items: list[DataItem],
    presets: tuple[str, ...] = (),
    verbose: bool = False,
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
        ashare_data_path=None,
        factor_root=None,
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


def test_data_registry_get_data_returns_processed_fixed_matrix() -> None:
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
                ops=(OpSpec("delay(1)", {}),),
                params={"values": values},
            )
        ]
    )

    registry._ensure_range(("alpha.raw",), 20200102, 20200106)
    data = registry.get_data("alpha.raw")
    registry._ensure_range(("alpha.raw",), 20200102, 20200106)

    assert data.shape == (4, 3)
    assert torch.isnan(data[0]).all()
    assert torch.equal(data[1], values[0])
    assert torch.equal(data[2], values[1])
    assert torch.equal(data[3], values[2])
    assert calls == [("alpha.raw", 20200101, 20200106)]


def test_data_registry_ts_mean_uses_history_window() -> None:
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
                ops=(OpSpec("ts_mean(2)", {}),),
                params={"values": values},
            )
        ]
    )

    registry._ensure_range(("alpha.mean",), 20200102, 20200106)
    data = registry.get_data("alpha.mean")

    assert torch.isnan(data[0]).all()
    assert torch.equal(data[1], torch.tensor([2.0, 4.0, 6.0]))
    assert torch.equal(data[2], torch.tensor([4.0, 8.0, 12.0]))
    assert torch.equal(data[3], torch.tensor([6.0, 12.0, 18.0]))


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
