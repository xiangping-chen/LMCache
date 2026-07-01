# SPDX-License-Identifier: Apache-2.0
"""Shared torch dtype <-> string mapping for disk-based storage backends.

Used by ``LightningPosixBackend`` and ``GdsBackend`` to serialize tensor
dtype information in checkpoint files and on-disk metadata.
"""

# Third Party
import torch

TORCH_DTYPES: dict[torch.dtype, str] = {
    torch.half: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.uint8: "U8",
    torch.uint16: "U16",
    torch.uint32: "U32",
    torch.uint64: "U64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float8_e4m3fn: "F8E4M3FN",
    torch.float8_e5m2: "F8E5M2",
}

TORCH_DTYPES_INVERSE: dict[str, torch.dtype] = {
    v: k for k, v in TORCH_DTYPES.items()
}
