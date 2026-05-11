from __future__ import annotations

import torch

from .base import COMPRESSED_FLOAT_DTYPES, CodecMeta, normalize_cpu_device, normalize_shape, validate_input_device, validate_input_dtype, validate_logical_dtype, validate_value_shape


def probe_cpu_float8_cast() -> bool:
    storage_dtype = getattr(torch, "float8_e4m3fn", None)
    if storage_dtype is None:
        return False
    try:
        probe = torch.tensor([0.0, 1.0], dtype=torch.float32, device="cpu")
        y = probe.to(storage_dtype)
        _ = y.to(torch.float16)
    except Exception:
        return False
    return True


class FP8Codec:
    name = "fp8"
    supported_input_dtypes = COMPRESSED_FLOAT_DTYPES
    storage_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)

    def __init__(self):
        storage_dtype = getattr(torch, "float8_e4m3fn", None)
        if storage_dtype is None:
            raise RuntimeError("FP8Codec requires torch.float8_e4m3fn, which is not available in this PyTorch build")
        self.storage_dtype = storage_dtype
        if not probe_cpu_float8_cast():
            raise RuntimeError("FP8Codec requires CPU cast support for torch.float8_e4m3fn")

    def allocate(self, shape, device="cpu", logical_dtype=None) -> tuple[torch.Tensor, CodecMeta]:
        if logical_dtype is None:
            logical_dtype = torch.float16
        validate_logical_dtype(logical_dtype, self.supported_input_dtypes, "FP8Codec")
        logical_shape = normalize_shape(shape)
        storage_device = normalize_cpu_device(device)
        buf = torch.zeros(logical_shape, dtype=self.storage_dtype, device=storage_device)
        meta = CodecMeta(
            logical_shape=logical_shape,
            storage_shape=logical_shape,
            orig_last_dim=logical_shape[-1],
            logical_dtype=logical_dtype,
            storage_dtype=self.storage_dtype,
            device=storage_device,
        )
        return buf, meta

    def encode_into(self, buf: torch.Tensor, meta: CodecMeta, idx, x: torch.Tensor) -> None:
        validate_input_device(x, meta, "FP8Codec")
        validate_input_dtype(x, self.supported_input_dtypes, "FP8Codec")
        validate_value_shape(x, meta, idx)
        buf[idx] = x.to(dtype=self.storage_dtype)

    def decode(self, buf: torch.Tensor, meta: CodecMeta, idx, out_dtype=None) -> torch.Tensor:
        dtype = meta.logical_dtype if out_dtype is None else out_dtype
        return buf[idx].to(dtype=dtype, device=meta.device)
