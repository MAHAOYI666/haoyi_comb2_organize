# Codec Storage Compression

`src.codec` provides optional CPU-side storage compression for comb2 feature tensors.
It is applied only after the existing feature pipeline has already run:

1. `cs_zscore`
2. `truncate`
3. `nan_to_num`

The codec is not part of preprocessing and does not change label, weight, mask, model,
checkpoint, or backtest logic.

## Configuration

Set the loader attribute in XML:

```xml
<loader dtype="float16" compression="fp4" />
```

Supported values:

| value | behavior |
| --- | --- |
| `none` | `PassthroughCodec(loader.dtype)`; default and backward-compatible |
| `fp8` | stores one raw byte per element in `torch.uint8`; bytes encode `torch.float8_e4m3fn` values |
| `fp4` | stores two 4-bit codes per byte |

Unknown values raise immediately. There is no silent fallback.

## Data Contract

| item | requirement |
| --- | --- |
| input distribution | already z-scored; expected range is approximately `[-5, 5]` |
| NaN | call `nan_to_num` before encode |
| Inf | call `nan_to_num` before encode |
| passthrough input dtype | `float16`, `float32`, `float64`, `bfloat16` |
| fp4/fp8 input dtype | `float16`, `float32`, `bfloat16`; `float64` raises |
| shape | last dimension is the feature dimension |
| storage device | CPU only in this phase |
| device mixing | encode and decode use CPU storage; decode returns CPU tensors |
| FP4 saturation | values below `-5` decode as `-5`; values above `5` decode as `5` |
| FP8 logical encoding | `torch.float8_e4m3fn` |
| FP8 physical storage dtype | `torch.uint8` |
| FP8 requirement | PyTorch must support CPU cast to and from `torch.float8_e4m3fn` |

Model code still receives decoded floating tensors. The existing model-side device move
continues to own CPU-to-GPU transfer.

## FP4 Details

FP4 uses 15 numeric codes, `0..14`. Code `15` is reserved for future NaN/padding
semantics and is not produced by the current data path.

Current values:

```text
[-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.0625, 0.0,
  0.0625, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
```

Odd feature dimensions are padded with code `7`, which decodes to `0.0`. Decode restores
the original feature dimension using `CodecMeta.orig_last_dim`; it does not infer length
from values or from code `15`.

Approximate precision for z-score data:

| range | behavior |
| --- | --- |
| around zero | highest precision; explicit codes at `-0.0625`, `0`, `0.0625` |
| middle range | coarser half-step to unit-step quantization |
| tails | saturated at `-5` and `5` |

## FP8 Environment Check

FP8 uses `torch.float8_e4m3fn` as the logical encoded format, but the physical storage
buffer returned by `allocate()` is `torch.uint8`. `encode_into()` casts input floats to
`torch.float8_e4m3fn` and stores the raw bytes. `decode()` applies `int`, `slice`, list,
or `torch.long` tensor indexing on the `torch.uint8` buffer first, then reinterprets the
selected bytes as `torch.float8_e4m3fn` and casts to the requested output dtype.

This avoids CPU `torch.float8_e4m3fn` advanced indexing kernels such as `index_cpu`,
which are not implemented in some PyTorch builds. Storage remains 1 byte per element.

`compression="fp8"` requires this probe to pass in the runtime environment:

```python
import torch

probe = torch.tensor([0.0, 1.0], dtype=torch.float32)
y = probe.to(torch.float8_e4m3fn)
z = y.to(torch.float16)
```

If the probe fails, `FP8Codec` raises. Use `compression="none"` in environments without
CPU float8 cast support.

## Scope

This phase compresses only:

| owner | compressed field |
| --- | --- |
| `ComboTrainDataset` | `X` |
| `ComboBuffer` | `buffer` |

It does not compress `Y`, `W`, masks, labels, factor files, memmaps, or
`ComboDataLoader._feature_cache`. Process peak RSS will not necessarily shrink in the
same ratio as `Dataset.X` or `ComboBuffer.buffer`.

## Source Runtime and Packaging

Current validation is for source-path execution:

```powershell
$env:PYTHONPATH = "vendor/comb2"
.\.venv\Scripts\python.exe -c "from src.codec import build_codec; print('ok')"
.\.venv\Scripts\python.exe -m pytest vendor/comb2/tests/test_codec.py
```

`vendor/comb2/setup.py` currently cythonizes selected modules and excludes `*.py` package
data. If deployment uses `pip install` or wheels, the release owner must update packaging
before enabling codec in that mode. Viable options are:

1. include `src.codec.*` in the cythonized extension list, or
2. change package data/exclusion rules so the pure-Python `src.codec` package is included.

Until that packaging decision is made, codec support should be treated as source-runtime
only.

## Troubleshooting

| symptom | likely cause | action |
| --- | --- | --- |
| `unsupported compression codec` | typo in XML `compression` | use `none`, `fp8`, or `fp4` |
| `logical_dtype=torch.float64` for fp4/fp8 | compressed codecs do not accept float64 | use `dtype="float16"`/`float32`/`bfloat16`, or `compression="none"` |
| FP8 raises during construction | PyTorch lacks CPU float8 cast support | use `compression="none"` or deploy a compatible PyTorch build |
| FP4 encode raises on non-finite input | NaN/Inf reached codec input | keep `nan_to_num` before encode |
| package install cannot import `src.codec` | setup/wheel packaging does not include codec | update `setup.py` before installed deployment |
