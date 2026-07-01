# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Lightning backend slot allocation algorithm.

Tests the module-level ``find_contiguous_run`` function for correctness and
edge cases, and tests slot allocation behavior (capacity limits, eviction,
file growth) through the public ``LightningPosixBackend`` interface.
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
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.lightning_backend import (
    LightningPosixBackend,
    find_contiguous_run,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


# ---------------------------------------------------------------------------
# find_contiguous_run tests (module-level public function)
# ---------------------------------------------------------------------------


class TestFindContiguousRun:
    """Tests for the module-level find_contiguous_run function."""

    def test_empty_list(self):
        assert find_contiguous_run([], 1) is None
        assert find_contiguous_run([], 8) is None

    def test_single_slot_needed(self):
        assert find_contiguous_run([5], 1) == 0
        assert find_contiguous_run([0, 10, 20], 1) == 0

    def test_fewer_slots_than_needed(self):
        assert find_contiguous_run([0, 1, 2], 4) is None

    def test_exact_contiguous_run(self):
        assert find_contiguous_run([0, 1, 2], 3) == 0

    def test_run_at_start(self):
        assert find_contiguous_run([0, 1, 2, 3, 10, 20], 4) == 0

    def test_run_in_middle(self):
        assert find_contiguous_run([0, 5, 6, 7, 8, 20], 4) == 1

    def test_run_at_end(self):
        assert find_contiguous_run([0, 5, 10, 11, 12, 13], 4) == 2

    def test_no_contiguous_run(self):
        # Every other slot free: 0, 2, 4, 6, 8
        assert find_contiguous_run([0, 2, 4, 6, 8], 2) is None

    def test_large_run(self):
        slots = list(range(100, 200))  # 100 contiguous
        assert find_contiguous_run(slots, 50) == 0
        assert find_contiguous_run(slots, 100) == 0
        assert find_contiguous_run(slots, 101) is None

    def test_multiple_runs_returns_first(self):
        # Two runs: [2,3,4] and [10,11,12]
        slots = [2, 3, 4, 10, 11, 12]
        assert find_contiguous_run(slots, 3) == 0

    def test_fragmented_with_one_valid_run(self):
        # Fragmented with one 8-slot run at positions 100-107
        slots = [0, 2, 4, 6] + list(range(100, 108)) + [200, 202]
        assert find_contiguous_run(slots, 8) == 4


# ---------------------------------------------------------------------------
# Backend integration tests (via public API)
# ---------------------------------------------------------------------------

# Small geometry so slot files stay tiny during tests.
_LAYERS = 2
_KV = 2
_HEADS = 2
_HEAD_DIM = 64
_TOKENS = 4
_SHAPE = (_LAYERS, _KV, _TOKENS, _HEADS, _HEAD_DIM)
_DTYPE = torch.bfloat16
# bytes = 2*2*4*2*64 * 2(bf16) = 4096 bytes per chunk
_CHUNK_BYTES = _LAYERS * _KV * _TOKENS * _HEADS * _HEAD_DIM * 2


def _make_config(
    data_path: str, num_slots: int, slot_size_mb: int = 1
) -> LMCacheEngineConfig:
    """Build a config for the lightning backend."""
    return LMCacheEngineConfig(
        chunk_size=_TOKENS,
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
            "lightning.use_odirect": False,
            "lightning.checkpoint_interval_sec": 0,
            "lightning.enable_hole_punch": False,
        },
    )


def _make_metadata() -> LMCacheMetadata:
    """Build test metadata."""
    return LMCacheMetadata(
        model_name="test/model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=_DTYPE,
        kv_shape=_SHAPE,
        role="worker",
        chunk_size=_TOKENS,
    )


def _make_key(idx: int) -> CacheEngineKey:
    """Create a CacheEngineKey for testing."""
    return CacheEngineKey(
        model_name="test/model",
        world_size=1,
        worker_id=0,
        chunk_hash=idx,
        dtype=_DTYPE,
    )


def _build_backend(
    tmp_path, num_slots: int
) -> tuple[LightningPosixBackend, LocalCPUBackend, asyncio.AbstractEventLoop]:
    """Create a fully-initialized backend for testing."""
    data_path = os.path.join(str(tmp_path), "lightning")
    config = _make_config(data_path, num_slots)
    metadata = _make_metadata()

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()

    cpu_backend = LocalCPUBackend(config, metadata, "cpu", None)
    backend = LightningPosixBackend(
        dst_device="cpu",
        config=config,
        metadata=metadata,
        local_cpu_backend=cpu_backend,
        loop=loop,
    )
    return backend, cpu_backend, loop


def _stop_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Stop and close an event loop, waiting for its thread to exit."""
    loop.call_soon_threadsafe(loop.stop)
    time.sleep(0.05)  # let the loop thread exit
    loop.close()


def _store_key(
    backend: LightningPosixBackend,
    cpu_backend: LocalCPUBackend,
    key: CacheEngineKey,
) -> bool:
    """Store a random chunk under the given key. Returns True on success."""
    obj = cpu_backend.allocate(
        torch.Size(_SHAPE), _DTYPE, fmt=MemoryFormat.KV_2LTD
    )
    if obj is None:
        return False
    obj.tensor.copy_(torch.randn(_SHAPE, dtype=_DTYPE, device="cpu"))

    completion = threading.Event()

    def on_complete(k: CacheEngineKey) -> None:
        completion.set()

    backend.batched_submit_put_task(
        [key], [obj], on_complete_callback=on_complete
    )
    # Release the caller's reference; the backend holds its own via
    # ref_count_up() inside _submit_single_put.
    obj.ref_count_down()
    return completion.wait(timeout=10.0)


class TestSlotAllocation:
    """Tests for slot allocation behavior via the public backend API."""

    def test_store_and_retrieve(self, tmp_path):
        """A stored chunk can be retrieved."""
        backend, cpu, loop = _build_backend(tmp_path, num_slots=8)
        try:
            key = _make_key(1)
            assert _store_key(backend, cpu, key)
            assert backend.contains(key)
            result = backend.get_blocking(key)
            assert result is not None
            result.ref_count_down()
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)

    def test_fill_capacity(self, tmp_path):
        """Can fill all available slots."""
        num_slots = 8
        backend, cpu, loop = _build_backend(tmp_path, num_slots=num_slots)
        try:
            for i in range(num_slots):
                key = _make_key(i)
                assert _store_key(backend, cpu, key), f"Failed to store key {i}"
                assert backend.contains(key)
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)

    def test_eviction_on_full(self, tmp_path):
        """When storage is full, new writes evict old entries."""
        num_slots = 4
        backend, cpu, loop = _build_backend(tmp_path, num_slots=num_slots)
        try:
            # Fill capacity
            for i in range(num_slots):
                assert _store_key(backend, cpu, _make_key(i))

            # Store one more — should evict the oldest
            new_key = _make_key(100)
            assert _store_key(backend, cpu, new_key)
            assert backend.contains(new_key)

            # At least one old key should be evicted
            old_present = sum(
                backend.contains(_make_key(i)) for i in range(num_slots)
            )
            assert old_present < num_slots
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)

    def test_remove_frees_slot(self, tmp_path):
        """Removing a key frees its slot for reuse."""
        num_slots = 4
        backend, cpu, loop = _build_backend(tmp_path, num_slots=num_slots)
        try:
            # Fill capacity
            for i in range(num_slots):
                assert _store_key(backend, cpu, _make_key(i))

            # Remove one
            assert backend.remove(_make_key(0))
            assert not backend.contains(_make_key(0))

            # Should be able to store a new key without eviction
            new_key = _make_key(100)
            assert _store_key(backend, cpu, new_key)
            assert backend.contains(new_key)

            # All other original keys should still be present
            for i in range(1, num_slots):
                assert backend.contains(_make_key(i))
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)

    def test_sequential_store_retrieve(self, tmp_path):
        """Multiple keys can be stored and retrieved independently."""
        backend, cpu, loop = _build_backend(tmp_path, num_slots=8)
        try:
            keys = [_make_key(i) for i in range(4)]
            for key in keys:
                assert _store_key(backend, cpu, key)

            for key in keys:
                assert backend.contains(key)
                result = backend.get_blocking(key)
                assert result is not None
                result.ref_count_down()
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)

    def test_remove_nonexistent_key(self, tmp_path):
        """Removing a key that doesn't exist returns False."""
        backend, cpu, loop = _build_backend(tmp_path, num_slots=8)
        try:
            assert not backend.remove(_make_key(999))
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)

    def test_get_nonexistent_key(self, tmp_path):
        """Getting a key that doesn't exist returns None."""
        backend, cpu, loop = _build_backend(tmp_path, num_slots=8)
        try:
            assert backend.get_blocking(_make_key(999)) is None
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)

    def test_overwrite_existing_key(self, tmp_path):
        """Storing a key that already exists overwrites it."""
        backend, cpu, loop = _build_backend(tmp_path, num_slots=8)
        try:
            key = _make_key(1)
            assert _store_key(backend, cpu, key)
            assert backend.contains(key)

            # Overwrite with new data
            assert _store_key(backend, cpu, key)
            assert backend.contains(key)

            # Should still be retrievable
            result = backend.get_blocking(key)
            assert result is not None
            result.ref_count_down()
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)

    def test_auto_sized_slot(self, tmp_path):
        """When slot_size_mb is omitted, slot auto-sizes to chunk bytes."""
        data_path = os.path.join(str(tmp_path), "lightning_auto")
        config = LMCacheEngineConfig(
            chunk_size=_TOKENS,
            local_cpu=True,
            max_local_cpu_size=1.0,
            local_disk=None,
            max_local_disk_size=0,
            enable_p2p=False,
            enable_pd=False,
            gds_path=None,
            extra_config={
                "lightning.data_path": data_path,
                "lightning.num_slots": 8,
                "lightning.io_threads": 1,
                "lightning.use_odirect": False,
                "lightning.checkpoint_interval_sec": 0,
                # No lightning.slot_size_mb — should auto-derive
            },
        )
        metadata = _make_metadata()

        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        cpu = LocalCPUBackend(config, metadata, "cpu", None)
        backend = LightningPosixBackend(
            dst_device="cpu",
            config=config,
            metadata=metadata,
            local_cpu_backend=cpu,
            loop=loop,
        )
        try:
            # Slot size should match chunk bytes (4096), aligned to 4K
            assert backend.slot_size == _CHUNK_BYTES

            # Store and retrieve should work
            key = _make_key(1)
            assert _store_key(backend, cpu, key)
            assert backend.contains(key)
            result = backend.get_blocking(key)
            assert result is not None
            result.ref_count_down()
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)


    def test_unbounded_growth(self, tmp_path):
        """With num_slots=0, backend grows without a cap."""
        data_path = os.path.join(str(tmp_path), "lightning_unbounded")
        config = LMCacheEngineConfig(
            chunk_size=_TOKENS,
            local_cpu=True,
            max_local_cpu_size=1.0,
            local_disk=None,
            max_local_disk_size=0,
            enable_p2p=False,
            enable_pd=False,
            gds_path=None,
            extra_config={
                "lightning.data_path": data_path,
                "lightning.num_slots": 0,  # unbounded
                "lightning.slot_size_mb": 1,
                "lightning.io_threads": 1,
                "lightning.use_odirect": False,
                "lightning.checkpoint_interval_sec": 0,
                "lightning.enable_hole_punch": False,
            },
        )
        metadata = _make_metadata()

        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        cpu = LocalCPUBackend(config, metadata, "cpu", None)
        backend = LightningPosixBackend(
            dst_device="cpu",
            config=config,
            metadata=metadata,
            local_cpu_backend=cpu,
            loop=loop,
        )
        try:
            assert backend.num_slots == 0

            # Store more chunks than the old default (1024) would allow
            # to verify there's no artificial cap.  We just store a handful
            # since each is tiny.
            for i in range(16):
                key = _make_key(i)
                assert _store_key(backend, cpu, key), f"Failed to store key {i}"
                assert backend.contains(key)

            # All 16 should be present (no eviction)
            for i in range(16):
                assert backend.contains(_make_key(i))
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)

    def test_disk_budget_eviction(self, tmp_path):
        """With max_disk_usage_gb set, eviction occurs when budget exhausted."""
        data_path = os.path.join(str(tmp_path), "lightning_budget")
        # Set a tiny disk budget: 4 MiB = 4 slots of 1 MiB each
        config = LMCacheEngineConfig(
            chunk_size=_TOKENS,
            local_cpu=True,
            max_local_cpu_size=1.0,
            local_disk=None,
            max_local_disk_size=0,
            enable_p2p=False,
            enable_pd=False,
            gds_path=None,
            extra_config={
                "lightning.data_path": data_path,
                "lightning.num_slots": 0,  # unbounded by slot count
                "lightning.slot_size_mb": 1,
                "lightning.max_disk_usage_gb": 4 / 1024,  # 4 MiB budget
                "lightning.io_threads": 1,
                "lightning.use_odirect": False,
                "lightning.checkpoint_interval_sec": 0,
                "lightning.enable_hole_punch": False,
            },
        )
        metadata = _make_metadata()

        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        cpu = LocalCPUBackend(config, metadata, "cpu", None)
        backend = LightningPosixBackend(
            dst_device="cpu",
            config=config,
            metadata=metadata,
            local_cpu_backend=cpu,
            loop=loop,
        )
        try:
            # Fill to budget (4 slots)
            for i in range(4):
                assert _store_key(backend, cpu, _make_key(i))

            # Store one more — should trigger eviction
            assert _store_key(backend, cpu, _make_key(100))
            assert backend.contains(_make_key(100))

            # At least one old key must have been evicted
            old_present = sum(
                backend.contains(_make_key(i)) for i in range(4)
            )
            assert old_present < 4
        finally:
            backend.close()
            cpu.close()
            _stop_loop(loop)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
