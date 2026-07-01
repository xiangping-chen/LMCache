#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: measure per-phase overhead in Lightning vs LocalDisk get_blocking.

Monkey-patches get_blocking in both backends to measure time spent in:
  - index/dict lookup (under lock)
  - memory allocation
  - I/O (read_file / _read_obj)
  - LRU/policy update

Uses the same config and data sizes as the main benchmark.
"""

import asyncio
import os
import statistics
import threading
import time

import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.lightning_backend import LightningPosixBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

_LAYERS = 2
_KV = 2
_HEADS = 2
_HEAD_DIM = 64
_DTYPE = torch.bfloat16
_MODEL = "test/overhead-diag"


def _make_key(idx):
    return CacheEngineKey(
        model_name=_MODEL, world_size=1, worker_id=0,
        chunk_hash=idx, dtype=_DTYPE,
    )


def _make_metadata(tokens):
    return LMCacheMetadata(
        model_name=_MODEL, world_size=1, local_world_size=1,
        worker_id=0, local_worker_id=0, kv_dtype=_DTYPE,
        kv_shape=(_LAYERS, _KV, tokens, _HEADS, _HEAD_DIM),
        role="worker", chunk_size=tokens,
    )


def _make_config(data_path, tokens, num_slots, use_lightning):
    chunk_bytes = _LAYERS * _KV * tokens * _HEADS * _HEAD_DIM * 2
    slot_mb = max(1, -(-chunk_bytes // (1024 * 1024)))
    extra = {
        "use_odirect": True,
    }
    if use_lightning:
        extra.update({
            "lightning.data_path": os.path.join(data_path, "lightning"),
            "lightning.num_slots": num_slots,
            "lightning.slot_size_mb": slot_mb,
            "lightning.io_threads": 4,
            "lightning.use_odirect": True,
            "lightning.checkpoint_interval_sec": 0,
            "lightning.enable_hole_punch": False,
        })
    return LMCacheEngineConfig(
        chunk_size=tokens,
        local_cpu=True,
        max_local_cpu_size=2.0,
        local_disk=os.path.join(data_path, "disk") if not use_lightning else None,
        max_local_disk_size=2 if not use_lightning else 0,
        enable_p2p=False,
        enable_pd=False,
        gds_path=None,
        extra_config=extra,
    )


def _populate(backend, cpu_backend, tokens, num_keys):
    """Write data so we can read it back."""
    shape = torch.Size((_LAYERS, _KV, tokens, _HEADS, _HEAD_DIM))
    keys = [_make_key(i) for i in range(num_keys)]
    completion = threading.Event()
    remaining = [num_keys]

    def on_complete(k):
        remaining[0] -= 1
        if remaining[0] <= 0:
            completion.set()

    for key in keys:
        obj = cpu_backend.allocate(shape, _DTYPE, fmt=MemoryFormat.KV_2LTD)
        obj.tensor.copy_(torch.randn(shape, dtype=_DTYPE, device="cpu"))
        backend.batched_submit_put_task([key], [obj], on_complete_callback=on_complete)

    completion.wait(timeout=60)
    return keys


def _bench_lightning_phases(backend, keys, tokens):
    """Instrument Lightning get_blocking by calling sub-steps directly."""
    from lmcache.v1.storage_backend.lightning_backend import torch_dtypes_inverse

    t_lookup = []
    t_alloc = []
    t_io = []
    t_total = []

    for key in keys:
        t0 = time.perf_counter()

        # Phase 1: index lookup
        t_l0 = time.perf_counter()
        with backend.index_lock:
            entry = backend.index.get(key)
            if entry is None:
                continue
            backend.lru.touch(key)
        t_l1 = time.perf_counter()
        t_lookup.append(t_l1 - t_l0)

        shape = torch.Size(entry.shape)
        dtype = torch_dtypes_inverse[entry.dtype]
        fmt = MemoryFormat(entry.fmt)

        # Phase 2: allocation
        t_a0 = time.perf_counter()
        memory_obj = backend.local_cpu_backend.allocate(shape, dtype, fmt=fmt)
        t_a1 = time.perf_counter()
        t_alloc.append(t_a1 - t_a0)

        if memory_obj is None:
            continue

        # Phase 3: I/O
        memory_obj.ref_count_up()
        offset = backend._get_file_offset(entry.slot_id)
        t_i0 = time.perf_counter()
        backend._read_obj(offset, memory_obj, entry.payload_len)
        t_i1 = time.perf_counter()
        t_io.append(t_i1 - t_i0)
        memory_obj.ref_count_down()

        t1 = time.perf_counter()
        t_total.append(t1 - t0)

    return t_lookup, t_alloc, t_io, t_total


def _bench_disk_phases(backend, keys):
    """Instrument LocalDisk get_blocking by calling sub-steps directly."""
    t_lookup = []
    t_alloc = []
    t_io = []
    t_total = []

    for key in keys:
        t0 = time.perf_counter()

        # Phase 1: dict lookup
        t_l0 = time.perf_counter()
        with backend.disk_lock:
            if key not in backend.dict:
                continue
            disk_meta = backend.dict[key]
            path = disk_meta.path
            dtype = disk_meta.dtype
            shape = disk_meta.shape
            fmt = disk_meta.fmt
        t_l1 = time.perf_counter()
        t_lookup.append(t_l1 - t_l0)

        # Phase 2: allocation
        t_a0 = time.perf_counter()
        memory_obj = backend.local_cpu_backend.allocate(shape, dtype, fmt=fmt)
        t_a1 = time.perf_counter()
        t_alloc.append(t_a1 - t_a0)

        if memory_obj is None:
            continue

        # Phase 3: I/O
        buffer = memory_obj.byte_array
        t_i0 = time.perf_counter()
        backend.read_file(key, buffer, path)
        t_i1 = time.perf_counter()
        t_io.append(t_i1 - t_i0)

        # Phase 4: policy update
        with backend.disk_lock:
            if key in backend.dict:
                backend.cache_policy.update_on_hit(key, backend.dict)

        t1 = time.perf_counter()
        t_total.append(t1 - t0)

    return t_lookup, t_alloc, t_io, t_total


def _report_phases(label, t_lookup, t_alloc, t_io, t_total, chunk_bytes):
    n = len(t_total)
    if n == 0:
        print(f"  {label}: no data")
        return
    total_gb = n * chunk_bytes / (1024**3)
    wall = sum(t_total)
    thr = total_gb / wall if wall > 0 else 0

    def stats(times):
        p50 = statistics.median(times) * 1e6
        mean = statistics.mean(times) * 1e6
        return f"mean={mean:8.1f}us  p50={p50:8.1f}us"

    print(f"  {label}  thr={thr:.2f} GiB/s  n={n}")
    print(f"    lookup:  {stats(t_lookup)}")
    print(f"    alloc:   {stats(t_alloc)}")
    print(f"    io:      {stats(t_io)}")
    print(f"    total:   {stats(t_total)}")
    io_frac = sum(t_io) / wall * 100
    alloc_frac = sum(t_alloc) / wall * 100
    lookup_frac = sum(t_lookup) / wall * 100
    other = 100 - io_frac - alloc_frac - lookup_frac
    print(f"    breakdown: io={io_frac:.1f}%  alloc={alloc_frac:.1f}%  "
          f"lookup={lookup_frac:.1f}%  other={other:.1f}%")


def main():
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/lmbench/overhead"

    for mb in [4, 8, 16, 32]:
        tokens = mb * 1024 * 1024 // (_LAYERS * _KV * _HEADS * _HEAD_DIM * 2)
        chunk_bytes = _LAYERS * _KV * tokens * _HEADS * _HEAD_DIM * 2
        num_keys = 128

        print(f"\n{'='*70}")
        print(f"  chunk_size={mb} MiB ({tokens} tokens)  num_keys={num_keys}")
        print(f"{'='*70}")

        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        # -- LocalDisk --
        cfg_d = _make_config(data_path, tokens, num_keys, use_lightning=False)
        meta = _make_metadata(tokens)
        cpu_d = LocalCPUBackend(cfg_d, meta, "cpu", None)
        disk = LocalDiskBackend(cfg_d, loop, cpu_d, "cuda:0", meta)
        keys = _populate(disk, cpu_d, tokens, num_keys)

        # Drop caches
        os.sync()
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
        except PermissionError:
            pass

        tl, ta, ti, tt = _bench_disk_phases(disk, keys)
        _report_phases("disk (cold)", tl, ta, ti, tt, chunk_bytes)

        tl, ta, ti, tt = _bench_disk_phases(disk, keys)
        _report_phases("disk (warm)", tl, ta, ti, tt, chunk_bytes)

        disk.close()
        cpu_d.close()

        # -- Lightning --
        cfg_l = _make_config(data_path, tokens, num_keys, use_lightning=True)
        cpu_l = LocalCPUBackend(cfg_l, meta, "cpu", None)
        lit = LightningPosixBackend(
            dst_device="cuda:0", config=cfg_l, metadata=meta,
            local_cpu_backend=cpu_l, loop=loop,
        )
        keys_l = _populate(lit, cpu_l, tokens, num_keys)

        os.sync()
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
        except PermissionError:
            pass

        tl, ta, ti, tt = _bench_lightning_phases(lit, keys_l, tokens)
        _report_phases("lightning (cold)", tl, ta, ti, tt, chunk_bytes)

        tl, ta, ti, tt = _bench_lightning_phases(lit, keys_l, tokens)
        _report_phases("lightning (warm)", tl, ta, ti, tt, chunk_bytes)

        lit.close()
        cpu_l.close()
        loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()
