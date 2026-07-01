# SPDX-License-Identifier: Apache-2.0
"""Tests for LightningPosixBackend O_DIRECT-aligned, zero-copy POSIX I/O.

O_DIRECT requires the in-memory I/O buffer to be aligned to the device block
size (typically 4 KiB), not just the file offset and length. These tests
exercise the public ``batched_submit_put_task`` / ``get_blocking`` round-trip
with O_DIRECT both enabled and disabled, and with block-aligned and non-aligned
payload lengths, to cover both the zero-copy direct path and the page-aligned
bounce-buffer fallback.

Tests skip automatically on filesystems that do not support O_DIRECT (e.g.
tmpfs / overlayfs commonly used in CI containers).
"""

# Standard
import asyncio
import os
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.lightning_backend import (
    LightningPosixBackend,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# Block alignment for O_DIRECT (must match the backend's internal _ALIGNMENT).
_BLOCK_ALIGNMENT = 4 * 1024  # 4 KiB

# Fixed KV geometry for test chunks.
_LAYERS = 2
_KV = 2
_HEADS = 2
_HEAD_DIM = 64


def _make_config(
    data_path: str, use_odirect: bool, slot_size_mb: int, num_slots: int
) -> LMCacheEngineConfig:
    """Build an LMCacheEngineConfig wired for the lightning plugin."""
    return LMCacheEngineConfig(
        chunk_size=1,
        local_cpu=True,
        max_local_cpu_size=1.0,
        local_disk=None,
        max_local_disk_size=0,
        enable_p2p=False,
        enable_pd=False,
        gds_path=None,
        extra_config={
            "lightning.data_path": data_path,
            "lightning.num_slots": num_slots,
            "lightning.slot_size_mb": slot_size_mb,
            "lightning.io_threads": 1,
            "lightning.use_odirect": use_odirect,
            "lightning.checkpoint_interval_sec": 0,
        },
    )


def _make_metadata(tokens: int) -> LMCacheMetadata:
    """Build test metadata matching the chunk geometry."""
    return LMCacheMetadata(
        model_name="test/model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(_LAYERS, _KV, tokens, _HEADS, _HEAD_DIM),
        role="worker",
        chunk_size=tokens,
    )


def _make_key(idx: int) -> CacheEngineKey:
    """Create a unique CacheEngineKey for testing."""
    return CacheEngineKey(
        model_name="test/model",
        world_size=1,
        worker_id=0,
        chunk_hash=idx,
        dtype=torch.bfloat16,
    )


def _stop_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Stop and close an event loop, waiting for its thread to exit."""
    loop.call_soon_threadsafe(loop.stop)
    time.sleep(0.05)  # let the loop thread exit
    loop.close()


def _build_backend(
    tmp_path: str, use_odirect: bool, tokens: int
) -> tuple[LightningPosixBackend, LocalCPUBackend, asyncio.AbstractEventLoop]:
    """Construct a fully-initialized backend for round-trip testing."""
    data_path = os.path.join(str(tmp_path), "lightning")
    shape = (_LAYERS, _KV, tokens, _HEADS, _HEAD_DIM)
    chunk_bytes = 1
    for d in shape:
        chunk_bytes *= d
    chunk_bytes *= 2  # bfloat16 = 2 bytes
    slot_size_mb = max(1, -(-chunk_bytes // (1024 * 1024)))

    config = _make_config(data_path, use_odirect, slot_size_mb, num_slots=8)
    metadata = _make_metadata(tokens)

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()

    cpu_backend = LocalCPUBackend(config, metadata, "cpu", None)

    try:
        backend = LightningPosixBackend(
            dst_device="cpu",
            config=config,
            metadata=metadata,
            local_cpu_backend=cpu_backend,
            loop=loop,
        )
    except OSError:
        cpu_backend.close()
        _stop_loop(loop)
        pytest.skip("Cannot open slot file with requested flags (O_DIRECT)")

    return backend, cpu_backend, loop


def _roundtrip(
    backend: LightningPosixBackend,
    cpu_backend: LocalCPUBackend,
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Store a random KV chunk and read it back via the public API.

    Returns:
        Tuple of (original_tensor, retrieved_tensor).
    """
    shape = torch.Size((_LAYERS, _KV, tokens, _HEADS, _HEAD_DIM))
    dtype = torch.bfloat16

    # Allocate and fill with random data
    src_obj = cpu_backend.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD)
    if src_obj is None:
        pytest.skip("Failed to allocate MemoryObj from CPU backend")
    src_obj.tensor.copy_(torch.randn(shape, dtype=dtype, device="cpu"))
    original = src_obj.tensor.clone()

    key = _make_key(tokens)

    # Write via public API
    completion = threading.Event()

    def on_complete(k: CacheEngineKey) -> None:
        completion.set()

    try:
        backend.batched_submit_put_task([key], [src_obj], on_complete_callback=on_complete)
    except OSError:
        src_obj.ref_count_down()
        pytest.skip("Filesystem does not support O_DIRECT I/O")

    # Release the caller's reference; the backend holds its own via
    # ref_count_up() inside _submit_single_put.
    src_obj.ref_count_down()

    if not completion.wait(timeout=10.0):
        pytest.fail("Write did not complete in time")

    # Read back via public API
    result = backend.get_blocking(key)
    if result is None:
        pytest.fail("get_blocking returned None for a key that was just stored")

    retrieved = result.tensor.clone()
    result.ref_count_down()
    return original, retrieved


@pytest.mark.parametrize("use_odirect", [False, True])
def test_roundtrip_block_aligned(tmp_path, use_odirect):
    """A block-aligned payload round-trips byte-exact through the public API."""
    # Choose tokens so total bytes are 4 KiB-aligned.
    # bytes = LAYERS * KV * tokens * HEADS * HEAD_DIM * elem_size
    # = 2 * 2 * tokens * 2 * 64 * 2 = 1024 * tokens
    # tokens=16 => 16384 bytes (4 KiB-aligned)
    tokens = 16
    chunk_bytes = _LAYERS * _KV * tokens * _HEADS * _HEAD_DIM * 2
    assert chunk_bytes % _BLOCK_ALIGNMENT == 0

    backend, cpu_backend, loop = _build_backend(tmp_path, use_odirect, tokens)
    try:
        original, retrieved = _roundtrip(backend, cpu_backend, tokens)
        assert torch.equal(original, retrieved)
    finally:
        backend.close()
        cpu_backend.close()
        _stop_loop(loop)


@pytest.mark.parametrize("use_odirect", [False, True])
def test_roundtrip_unaligned_length(tmp_path, use_odirect):
    """A non-block-aligned payload length round-trips (exercises bounce buffer)."""
    # tokens=9 => 9216 bytes, not 4 KiB-aligned
    tokens = 9
    chunk_bytes = _LAYERS * _KV * tokens * _HEADS * _HEAD_DIM * 2
    assert chunk_bytes % _BLOCK_ALIGNMENT != 0

    backend, cpu_backend, loop = _build_backend(tmp_path, use_odirect, tokens)
    try:
        original, retrieved = _roundtrip(backend, cpu_backend, tokens)
        assert torch.equal(original, retrieved)
    finally:
        backend.close()
        cpu_backend.close()
        _stop_loop(loop)
