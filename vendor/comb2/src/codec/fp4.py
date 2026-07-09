from __future__ import annotations

import torch

from .base import COMPRESSED_FLOAT_DTYPES, CodecMeta, normalize_cpu_device, normalize_shape, validate_input_device, validate_input_dtype, validate_logical_dtype, validate_value_shape


FP4_VALUES = (
    -5.0,
    -4.0,
    -3.0,
    -2.0,
    -1.0,
    -0.5,
    -0.0625,
    0.0,
    0.0625,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
)
FP4_PADDING_CODE = 7
FP4_RESERVED_CODE = 15


class FP4Codec:
    name = "fp4"
    storage_dtype = torch.uint8
    supported_input_dtypes = COMPRESSED_FLOAT_DTYPES

    def __init__(self):
        self.fp4_sorted = torch.tensor(FP4_VALUES, dtype=torch.float32, device="cpu")

    def allocate(self, shape, device="cpu", logical_dtype=None) -> tuple[torch.Tensor, CodecMeta]:
        if logical_dtype is None:
            logical_dtype = torch.float16
        validate_logical_dtype(logical_dtype, self.supported_input_dtypes, "FP4Codec")
        logical_shape = normalize_shape(shape)
        storage_device = normalize_cpu_device(device)
        storage_shape = logical_shape[:-1] + ((logical_shape[-1] + 1) // 2,)
        buf = torch.zeros(storage_shape, dtype=self.storage_dtype, device=storage_device)
        meta = CodecMeta(
            logical_shape=logical_shape,
            storage_shape=storage_shape,
            orig_last_dim=logical_shape[-1],
            logical_dtype=logical_dtype,
            storage_dtype=self.storage_dtype,
            device=storage_device,
        )
        return buf, meta

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(dtype=torch.float32)
        finite = torch.isfinite(x32)

        values = self.fp4_sorted
        safe = torch.where(finite, x32, torch.zeros_like(x32))
        pos = torch.searchsorted(values, safe)
        pos = pos.clamp(min=1, max=14)
        left = values[pos - 1]
        right = values[pos]
        use_right = (safe - left).abs() > (right - safe).abs()
        codes = torch.where(use_right, pos, pos - 1).to(torch.uint8)
        return torch.where(finite, codes, torch.full_like(codes, FP4_RESERVED_CODE))

    def encode_into(self, buf: torch.Tensor, meta: CodecMeta, idx, x: torch.Tensor) -> None:
        validate_input_device(x, meta, "FP4Codec")
        validate_input_dtype(x, self.supported_input_dtypes, "FP4Codec")
        validate_value_shape(x, meta, idx)

        codes = self._quantize(x)
        if meta.orig_last_dim % 2:
            pad_shape = codes.shape[:-1] + (1,)
            padding = torch.full(pad_shape, FP4_PADDING_CODE, dtype=torch.uint8, device=codes.device)
            codes = torch.cat((codes, padding), dim=-1)

        low = codes[..., 0::2].to(torch.int16)
        high = codes[..., 1::2].to(torch.int16)
        packed = ((high << 4) | low).to(torch.uint8)
        buf[idx] = packed.to(device=meta.device)

    def decode(self, buf: torch.Tensor, meta: CodecMeta, idx, out_dtype=None) -> torch.Tensor:
        packed = buf[idx].to(dtype=torch.int16, device="cpu")
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F

        codes_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
        codes = torch.empty(codes_shape, dtype=torch.long, device="cpu")
        codes[..., 0::2] = low.to(torch.long)
        codes[..., 1::2] = high.to(torch.long)
        codes = codes[..., : meta.orig_last_dim]

        reserved = codes == FP4_RESERVED_CODE
        values = self.fp4_sorted[codes.clamp(max=14)]
        if reserved.any().item():
            values = torch.where(reserved, torch.full_like(values, torch.nan), values)

        dtype = meta.logical_dtype if out_dtype is None else out_dtype
        return values.to(dtype=dtype, device=meta.device)
