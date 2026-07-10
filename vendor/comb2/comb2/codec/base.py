from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol

import torch


FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
COMPRESSED_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@dataclass(frozen=True)
class CodecMeta:
    logical_shape: tuple[int, ...]
    storage_shape: tuple[int, ...]
    orig_last_dim: int
    logical_dtype: torch.dtype
    storage_dtype: torch.dtype
    device: torch.device


class Codec(Protocol):
    name: str
    storage_dtype: torch.dtype
    supported_input_dtypes: tuple[torch.dtype, ...]

    def allocate(self, shape, device, logical_dtype) -> tuple[torch.Tensor, CodecMeta]:
        ...

    def encode_into(self, buf: torch.Tensor, meta: CodecMeta, idx, x: torch.Tensor) -> None:
        ...

    def decode(self, buf: torch.Tensor, meta: CodecMeta, idx, out_dtype=None) -> torch.Tensor:
        ...


def normalize_cpu_device(device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise ValueError(f"codec storage only supports CPU in this phase, got device={resolved}")
    return resolved


def normalize_shape(shape) -> tuple[int, ...]:
    logical_shape = tuple(int(dim) for dim in shape)
    if not logical_shape:
        raise ValueError("codec shape must have at least one dimension")
    if any(dim < 0 for dim in logical_shape):
        raise ValueError(f"codec shape dimensions must be non-negative, got {logical_shape}")
    return logical_shape


def validate_logical_dtype(logical_dtype: torch.dtype, supported: tuple[torch.dtype, ...], codec_name: str) -> None:
    if logical_dtype not in supported:
        supported_names = ", ".join(str(dtype).replace("torch.", "") for dtype in supported)
        raise TypeError(f"{codec_name} does not support logical_dtype={logical_dtype}; supported: {supported_names}")


def validate_input_dtype(x: torch.Tensor, supported: tuple[torch.dtype, ...], codec_name: str) -> None:
    if x.dtype not in supported:
        supported_names = ", ".join(str(dtype).replace("torch.", "") for dtype in supported)
        raise TypeError(f"{codec_name} does not support input dtype={x.dtype}; supported: {supported_names}")


def validate_input_device(x: torch.Tensor, meta: CodecMeta, codec_name: str) -> None:
    if x.device != meta.device:
        raise ValueError(f"{codec_name} input device {x.device} does not match storage device {meta.device}")


def sliced_logical_shape(logical_shape: tuple[int, ...], idx) -> tuple[int, ...]:
    if isinstance(idx, Integral):
        return logical_shape[1:]
    if isinstance(idx, slice):
        return (len(range(*idx.indices(logical_shape[0]))),) + logical_shape[1:]
    if isinstance(idx, list):
        return (len(idx),) + logical_shape[1:]
    if isinstance(idx, torch.Tensor):
        if idx.dtype != torch.long:
            raise TypeError(f"codec tensor indices must have dtype torch.long, got {idx.dtype}")
        if idx.dim() == 0:
            return logical_shape[1:]
        return tuple(idx.shape) + logical_shape[1:]
    raise TypeError(f"unsupported codec index type: {type(idx).__name__}")


def validate_value_shape(x: torch.Tensor, meta: CodecMeta, idx) -> None:
    expected_shape = sliced_logical_shape(meta.logical_shape, idx)
    if tuple(x.shape) != expected_shape:
        raise ValueError(f"encoded tensor shape {tuple(x.shape)} does not match indexed logical shape {expected_shape}")
