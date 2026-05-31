from __future__ import annotations

import math

import pytest
import pandas as pd
import torch

from src.codec import FP4Codec, FP4_VALUES, FP8Codec, PassthroughCodec, build_codec
from src.DataLoader import AlphaParquetItemSource, FeatureSpec, OpSpec


FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)


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


def test_config_accepts_loader_compression_attribute(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        '<config><combo><loader compression="fp4" dtype="float16" /></combo></config>',
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))

    assert load_config()["combo"]["loader"]["compression"] == "none"
    assert loaded["combo"]["loader"]["compression"] == "fp4"


def test_config_accepts_loader_features(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    alpha_path = tmp_path / "alpha.parquet"
    xml_path.write_text(
        f"""
        <config>
          <constants factor_root="factors" />
          <combo>
            <loader dtype="float16">
              <features>
                <factor name="factor_a" path="factor_a">
                  <op name="truncate" min="-2" max="2" />
                </factor>
                <alpha name="base_lgbm" dump_path="{alpha_path}" mode="read_dump">
                  <op name="nan_to_num" value="0" />
                </alpha>
              </features>
            </loader>
          </combo>
        </config>
        """,
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))
    features = loaded["combo"]["loader"]["features"]

    assert len(features) == 2
    assert features[0]["kind"] == "factor"
    assert features[0]["name"] == "factor_a"
    assert features[0]["path"].endswith("/factors/factor_a")
    assert features[0]["ops"][0]["params"] == {"min": -2, "max": 2}
    assert features[1]["kind"] == "alpha"
    assert features[1]["path"] == str(alpha_path.resolve())
    assert loaded["combo"]["loader"]["apply_global_ops"] is True


def test_config_accepts_apply_global_ops_flag(tmp_path) -> None:
    from config import load_config

    xml_path = tmp_path / "config.xml"
    xml_path.write_text(
        '<config><combo><loader apply_global_ops="false" /></combo></config>',
        encoding="utf-8",
    )

    loaded = load_config(str(xml_path))

    assert loaded["combo"]["loader"]["apply_global_ops"] is False


def test_alpha_parquet_item_source_loads_date_and_reindexes_codes(tmp_path) -> None:
    alpha_path = tmp_path / "alpha.parquet"
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=[20200101, 20200102],
        columns=["1", "000002"],
    ).to_parquet(alpha_path)
    source = AlphaParquetItemSource(
        FeatureSpec(
            kind="alpha",
            name="base",
            path=str(alpha_path),
            ops=(OpSpec("nan_to_num", {"value": 0.0}),),
        ),
        dtype=torch.float32,
        codes=["000001", "000002", "000003"],
    )

    loaded = source.load_day(20200102)

    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, torch.tensor([3.0, 4.0, 0.0]))
