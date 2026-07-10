from __future__ import annotations

import torch

from .base import COMPRESSED_FLOAT_DTYPES, FLOAT_DTYPES, Codec, validate_logical_dtype
from .fp4 import FP4Codec
from .fp8 import FP8Codec
from .passthrough import PassthroughCodec


def build_codec(name: str | None, loader_dtype: torch.dtype) -> Codec:
    normalized = "none" if name is None else str(name).strip().lower()
    if normalized in {"none", "passthrough"}:
        validate_logical_dtype(loader_dtype, FLOAT_DTYPES, "PassthroughCodec")
        return PassthroughCodec(loader_dtype)
    if normalized == "fp4":
        validate_logical_dtype(loader_dtype, COMPRESSED_FLOAT_DTYPES, "FP4Codec")
        return FP4Codec()
    if normalized == "fp8":
        validate_logical_dtype(loader_dtype, COMPRESSED_FLOAT_DTYPES, "FP8Codec")
        return FP8Codec()
    raise ValueError(f"unsupported compression codec: {name!r}")
