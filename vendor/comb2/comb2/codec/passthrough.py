from __future__ import annotations

import torch

from .base import FLOAT_DTYPES, CodecMeta, normalize_cpu_device, normalize_shape, validate_input_device, validate_input_dtype, validate_logical_dtype, validate_value_shape


class PassthroughCodec:
    name = "passthrough"
    supported_input_dtypes = FLOAT_DTYPES

    def __init__(self, dtype: torch.dtype):
        validate_logical_dtype(dtype, FLOAT_DTYPES, "PassthroughCodec")
        self.storage_dtype = dtype

    def allocate(self, shape, device="cpu", logical_dtype=None) -> tuple[torch.Tensor, CodecMeta]:
        logical_dtype = self.storage_dtype if logical_dtype is None else logical_dtype
        validate_logical_dtype(logical_dtype, FLOAT_DTYPES, "PassthroughCodec")
        if logical_dtype != self.storage_dtype:
            raise TypeError(
                f"PassthroughCodec storage dtype {self.storage_dtype} must match logical_dtype={logical_dtype}"
            )
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
        validate_input_device(x, meta, "PassthroughCodec")
        validate_input_dtype(x, self.supported_input_dtypes, "PassthroughCodec")
        validate_value_shape(x, meta, idx)
        buf[idx] = x.to(dtype=meta.storage_dtype)

    def decode(self, buf: torch.Tensor, meta: CodecMeta, idx, out_dtype=None) -> torch.Tensor:
        dtype = meta.logical_dtype if out_dtype is None else out_dtype
        return buf[idx].to(dtype=dtype, device=meta.device)
