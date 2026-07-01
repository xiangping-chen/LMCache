# SPDX-License-Identifier: Apache-2.0
"""Local-FS performance benchmark: LocalDiskBackend vs LightningPosixBackend vs GdsBackend.

Compares store/retrieve performance of the ``LocalDiskBackend``, the
``LightningPosixBackend``, and the ``GdsBackend`` on a local filesystem
(e.g. a dedicated NVMe), isolating storage-path behavior from any network/PFS
effects.

All backends are driven only through their public ``StorageBackendInterface``
APIs (``batched_submit_put_task`` + completion callback, ``get_blocking``) so
the comparison is apples-to-apples at the "store/retrieve a KV chunk" level.
Disk and Lightning allocate staging memory from a shared ``LocalCPUBackend``;
GDS manages its own GPU-registered memory via ``CuFileMemoryAllocator``.

Fairness controls implemented here:
- Matched O_DIRECT setting for disk and lightning (``--odirect on|off``).
- Matched I/O parallelism (disk uses 4 workers; ``--io-threads`` for lightning/gds).
- Page-cache control: ``--drop-caches`` runs ``sync`` + drops the page cache
  before every *cold* read phase (requires root).
- Write durability: ``os.sync()`` is issued and timed after the write phase.

Example:
    python disk_vs_posix_local.py --backend both --scenario all \\
        --path-root /mnt/lmbench --chunk-mb 4 --num-keys 512 --batch 32 \\
        --odirect on --io-threads 4 --drop-caches --csv results.csv
"""

# Standard
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import argparse
import asyncio
import csv
import os
import statistics
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey, start_loop_in_thread_with_exceptions
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.gds_backend import GdsBackend
from lmcache.v1.storage_backend.lightning_backend import LightningPosixBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

# Fixed KV-like tensor geometry; only the token dimension is scaled to hit a
# target per-chunk payload size. bytes = 8 * 2 * T * 8 * 128 * elem_size.
_LAYERS = 8
_KV = 2
_HEADS = 8
_HEAD_DIM = 128
_BYTES_PER_TOKEN_UNIT = _LAYERS * _KV * _HEADS * _HEAD_DIM  # * tokens * elem_size


@dataclass
class PerfResult:
    """Aggregated statistics for one measured operation phase."""

    backend: str
    operation: str
    num_samples: int
    total_gb: float
    wall_s: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    throughput_gbps: float
    chunks_per_s: float
    extra: dict = field(default_factory=dict)


class _CompletionBarrier:
    """Counts per-key completion callbacks and signals when all have fired."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._count = 0
        self._lock = threading.Lock()
        self._event = threading.Event()
        if total == 0:
            self._event.set()

    def on_complete(self, key: CacheEngineKey) -> None:
        with self._lock:
            self._count += 1
            if self._count >= self._total:
                self._event.set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)

    @property
    def completed(self) -> int:
        with self._lock:
            return self._count


def _tokens_for_chunk_mb(chunk_mb: float, elem_size: int) -> int:
    """Compute the token dimension that yields ~chunk_mb per chunk."""
    target_bytes = chunk_mb * 1024 * 1024
    tokens = int(round(target_bytes / (_BYTES_PER_TOKEN_UNIT * elem_size)))
    return max(tokens, 1)


def _chunk_shape(tokens: int) -> Tuple[int, int, int, int, int]:
    """KV_2LTD-style shape for the given token count."""
    return (_LAYERS, _KV, tokens, _HEADS, _HEAD_DIM)


def _percentiles(latencies_ms: List[float]) -> Tuple[float, float, float, float, float]:
    """Return (mean, p50, p95, p99, max) in ms for a latency sample."""
    if not latencies_ms:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(latencies_ms)
    n = len(ordered)
    mean = statistics.mean(ordered)
    p50 = ordered[min(int(n * 0.50), n - 1)]
    p95 = ordered[min(int(n * 0.95), n - 1)]
    p99 = ordered[min(int(n * 0.99), n - 1)]
    return (mean, p50, p95, p99, ordered[-1])


def drop_page_cache() -> bool:
    """Flush dirty pages and drop the OS page cache. Requires root.

    Returns:
        True if the cache was dropped, False otherwise (e.g. not root).
    """
    os.sync()
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except (PermissionError, OSError) as e:
        print(f"  [warn] could not drop page cache ({e}); read numbers may be warm")
        return False


def build_config(
    backend: str,
    path: str,
    chunk_size_tokens: int,
    max_cpu_gb: float,
    max_disk_gb: float,
    odirect: bool,
    io_threads: int,
    num_slots: int,
    slot_size_mb: int,
    checkpoint_interval_sec: int,
    gds_buffer_size_mb: int = 0,
) -> LMCacheEngineConfig:
    """Construct an LMCacheEngineConfig wired for the requested backend."""
    base = {
        "chunk_size": chunk_size_tokens,
        "local_cpu": True,
        "max_local_cpu_size": max_cpu_gb,
        "local_disk": None,
        "max_local_disk_size": 0,
        "enable_p2p": False,
        "enable_pd": False,
        "gds_path": None,
    }
    if backend == "disk":
        base["local_disk"] = path
        base["max_local_disk_size"] = max_disk_gb
        base["extra_config"] = {
            "use_odirect": odirect,
            "disk_io_threads": io_threads,
        }
    elif backend == "lightning":
        base["extra_config"] = {
            "lightning.data_path": path,
            "lightning.num_slots": num_slots,
            "lightning.slot_size_mb": slot_size_mb,
            "lightning.io_threads": io_threads,
            "lightning.use_odirect": odirect,
            "lightning.checkpoint_interval_sec": checkpoint_interval_sec,
        }
    elif backend == "gds":
        base["gds_path"] = path
        base["gds_buffer_size"] = gds_buffer_size_mb
        base["use_gds"] = True
        base["extra_config"] = {
            "disk_io_threads": io_threads,
            "use_direct_io": odirect,
        }
    else:
        raise ValueError(f"Unknown backend '{backend}'")
    return LMCacheEngineConfig(**base)


def build_metadata(chunk_size_tokens: int, dtype: torch.dtype) -> LMCacheMetadata:
    """Construct test metadata for the chosen chunk geometry."""
    return LMCacheMetadata(
        model_name="bench/model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=_chunk_shape(chunk_size_tokens),
        role="worker",
        chunk_size=chunk_size_tokens,
    )


def make_keys(num_keys: int, dtype: torch.dtype) -> List[CacheEngineKey]:
    """Generate unique CacheEngineKeys."""
    return [
        CacheEngineKey(
            model_name="bench/model",
            world_size=1,
            worker_id=0,
            chunk_hash=i + 1,
            dtype=dtype,
        )
        for i in range(num_keys)
    ]


def allocate_objs(
    cpu_backend: LocalCPUBackend,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    num_objs: int,
    use_gpu: bool,
) -> List[MemoryObj]:
    """Allocate and fill MemoryObjs from the CPU backend with test data."""
    objs = []
    for i in range(num_objs):
        obj = cpu_backend.allocate(torch.Size(shape), dtype, fmt=MemoryFormat.KV_2LTD)
        if obj is None:
            raise RuntimeError(
                f"Failed to allocate MemoryObj {i}; increase --max-cpu-gb"
            )
        if use_gpu and torch.cuda.is_available():
            g = torch.randn(shape, dtype=dtype, device="cuda:0")
            obj.tensor.copy_(g)
            del g
        else:
            obj.tensor.copy_(torch.randn(shape, dtype=dtype, device="cpu"))
        objs.append(obj)
    return objs


def release(obj: Optional[MemoryObj]) -> None:
    """Best-effort ref-count release for a MemoryObj."""
    if obj is not None and hasattr(obj, "ref_count_down"):
        try:
            obj.ref_count_down()
        except Exception:
            pass


def measure_write(
    backend: StorageBackendInterface,
    backend_name: str,
    keys: List[CacheEngineKey],
    objs: List[MemoryObj],
    batch: int,
    bytes_per_chunk: int,
) -> PerfResult:
    """Write all chunks (in batches, concurrently) and time to completion.

    Stops the timer only after every key's completion callback fires and a
    final ``os.sync()`` makes the data durable.
    """
    barrier = _CompletionBarrier(len(keys))
    t0 = time.perf_counter()
    for i in range(0, len(keys), batch):
        bk = keys[i : i + batch]
        bo = objs[i : i + batch]
        backend.batched_submit_put_task(
            bk, bo, on_complete_callback=barrier.on_complete
        )
    if not barrier.wait(timeout=180.0):
        print(
            f"  [warn] {backend_name} write barrier timed out: "
            f"{barrier.completed}/{len(keys)} completed "
            f"(likely dropped puts, e.g. O_DIRECT EINVAL or eviction failure)"
        )
    os.sync()
    wall = time.perf_counter() - t0

    completed = barrier.completed
    total_gb = completed * bytes_per_chunk / (1024**3)
    # Per-op latency is not meaningful under concurrency; report wall-derived
    # aggregates and leave the distribution fields as the amortized per-op time.
    amortized_ms = wall * 1000.0 / max(completed, 1)
    return PerfResult(
        backend=backend_name,
        operation="write",
        num_samples=completed,
        total_gb=total_gb,
        wall_s=wall,
        mean_ms=amortized_ms,
        p50_ms=amortized_ms,
        p95_ms=amortized_ms,
        p99_ms=amortized_ms,
        max_ms=amortized_ms,
        throughput_gbps=total_gb / wall if wall > 0 else 0.0,
        chunks_per_s=completed / wall if wall > 0 else 0.0,
        extra={"batch": batch, "requested": len(keys)},
    )


def measure_read(
    backend: StorageBackendInterface,
    backend_name: str,
    keys: List[CacheEngineKey],
    operation: str,
    bytes_per_chunk: int,
) -> PerfResult:
    """Retrieve each key with ``get_blocking`` and record per-op latency."""
    latencies_ms: List[float] = []
    misses = 0
    t0 = time.perf_counter()
    for key in keys:
        s = time.perf_counter()
        obj = backend.get_blocking(key)
        latencies_ms.append((time.perf_counter() - s) * 1000.0)
        if obj is None:
            misses += 1
        release(obj)
    wall = time.perf_counter() - t0

    mean, p50, p95, p99, mx = _percentiles(latencies_ms)
    total_gb = (len(keys) - misses) * bytes_per_chunk / (1024**3)
    return PerfResult(
        backend=backend_name,
        operation=operation,
        num_samples=len(keys),
        total_gb=total_gb,
        wall_s=wall,
        mean_ms=mean,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=mx,
        throughput_gbps=total_gb / wall if wall > 0 else 0.0,
        chunks_per_s=len(keys) / wall if wall > 0 else 0.0,
        extra={"misses": misses},
    )


def measure_read_batched(
    backend: StorageBackendInterface,
    backend_name: str,
    keys: List[CacheEngineKey],
    operation: str,
    bytes_per_chunk: int,
    batch: int,
) -> PerfResult:
    """Retrieve keys via ``batched_get_blocking`` in batches and record wall time."""
    misses = 0
    t0 = time.perf_counter()
    for i in range(0, len(keys), batch):
        batch_keys = keys[i : i + batch]
        results = backend.batched_get_blocking(batch_keys)
        for obj in results:
            if obj is None:
                misses += 1
            release(obj)
    wall = time.perf_counter() - t0

    total_gb = (len(keys) - misses) * bytes_per_chunk / (1024**3)
    amortized_ms = wall * 1000.0 / len(keys) if keys else 0.0
    return PerfResult(
        backend=backend_name,
        operation=operation,
        num_samples=len(keys),
        total_gb=total_gb,
        wall_s=wall,
        mean_ms=amortized_ms,
        p50_ms=amortized_ms,
        p95_ms=amortized_ms,
        p99_ms=amortized_ms,
        max_ms=amortized_ms,
        throughput_gbps=total_gb / wall if wall > 0 else 0.0,
        chunks_per_s=len(keys) / wall if wall > 0 else 0.0,
        extra={"misses": misses, "batch": batch},
    )


def make_backend(
    backend: str,
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    cpu_backend: Optional[LocalCPUBackend],
    loop: asyncio.AbstractEventLoop,
    dst_device: str,
) -> StorageBackendInterface:
    """Instantiate the requested storage backend."""
    if backend == "disk":
        assert cpu_backend is not None
        return LocalDiskBackend(
            config=config,
            loop=loop,
            local_cpu_backend=cpu_backend,
            dst_device=dst_device,
            metadata=metadata,
        )
    if backend == "gds":
        return GdsBackend(
            config=config,
            metadata=metadata,
            loop=loop,
            dst_device=dst_device,
        )
    assert cpu_backend is not None
    return LightningPosixBackend(
        dst_device=dst_device,
        config=config,
        metadata=metadata,
        local_cpu_backend=cpu_backend,
        loop=loop,
    )


def _allocate_gds_objs(
    backend: GdsBackend,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    num_objs: int,
) -> List[MemoryObj]:
    """Allocate and fill MemoryObjs from the GDS backend's GPU-registered pool."""
    objs = []
    for i in range(num_objs):
        obj = backend.allocate(torch.Size(shape), dtype, fmt=MemoryFormat.KV_2LTD)
        if obj is None:
            raise RuntimeError(
                f"GDS allocation failed at object {i}; "
                f"increase --gds-buffer-mb"
            )
        obj.tensor.copy_(torch.randn(shape, dtype=dtype, device="cuda:0"))
        objs.append(obj)
    return objs


def run_backend(
    backend_name: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    dst_device: str,
    loop: asyncio.AbstractEventLoop,
) -> List[PerfResult]:
    """Run the selected scenario for a single backend; returns PerfResults."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    tokens = _tokens_for_chunk_mb(args.chunk_mb, elem_size)
    shape = _chunk_shape(tokens)
    bytes_per_chunk = _BYTES_PER_TOKEN_UNIT * tokens * elem_size
    slot_size_mb = max(1, -(-bytes_per_chunk // (1024 * 1024)))  # ceil MiB

    dataset_gb = args.num_keys * bytes_per_chunk / (1024**3)
    is_gds = backend_name == "gds"

    # Generous staging pool: write-set + headroom for read staging.
    max_cpu_gb = args.max_cpu_gb or max(2.0, dataset_gb * 1.5 + 2.0)
    # Capacity: for eviction scenario shrink below dataset; else fit it.
    if args.scenario == "evict":
        cap_keys = max(1, args.num_keys // 2)
    else:
        cap_keys = args.num_keys
    max_disk_gb = max(0.1, cap_keys * bytes_per_chunk / (1024**3) + 0.05)
    # Slot is sized to one chunk, so one chunk occupies exactly one slot.
    num_slots = cap_keys

    path = os.path.join(args.path_root, backend_name)
    os.makedirs(path, exist_ok=True)

    # GDS buffer: dataset + headroom for read staging (alloc before free).
    gds_buffer_mb = max(
        128, int((args.num_keys * 2 + 4) * bytes_per_chunk / (1024 * 1024))
    )

    config = build_config(
        backend=backend_name,
        path=path,
        chunk_size_tokens=tokens,
        max_cpu_gb=max_cpu_gb,
        max_disk_gb=max_disk_gb,
        odirect=(args.odirect == "on"),
        io_threads=args.io_threads,
        num_slots=num_slots,
        slot_size_mb=slot_size_mb,
        checkpoint_interval_sec=0,  # disable periodic checkpoint during timing
        gds_buffer_size_mb=gds_buffer_mb,
    )
    metadata = build_metadata(tokens, dtype)

    # GDS manages its own GPU-registered memory; others use LocalCPUBackend.
    cpu_backend: Optional[LocalCPUBackend] = None
    if not is_gds:
        cpu_backend = LocalCPUBackend(config, metadata, dst_device, None)

    backend = make_backend(
        backend_name, config, metadata, cpu_backend, loop, dst_device
    )

    print(
        f"\n=== {backend_name} | chunk={args.chunk_mb}MiB (tokens={tokens}, "
        f"{bytes_per_chunk / 1024 / 1024:.2f}MiB) | keys={args.num_keys} "
        f"(dataset={dataset_gb:.2f}GiB) | batch={args.batch} | "
        f"odirect={args.odirect} | io_threads={args.io_threads} ==="
    )

    results: List[PerfResult] = []
    use_gpu = torch.cuda.is_available()

    try:
        write_keys = make_keys(args.num_keys, dtype)
        if args.scenario in ("write", "all", "evict"):
            if is_gds:
                objs = _allocate_gds_objs(backend, shape, dtype, args.num_keys)
            else:
                objs = allocate_objs(
                    cpu_backend, shape, dtype, args.num_keys, use_gpu
                )
            res = measure_write(
                backend, backend_name, write_keys, objs, args.batch, bytes_per_chunk
            )
            results.append(res)
            _print_result(res)
            for o in objs:
                release(o)

        if args.scenario in ("read_cold", "read_warm", "all"):
            # Ensure data exists if a read-only scenario was requested.
            if args.scenario in ("read_cold", "read_warm"):
                if is_gds:
                    objs = _allocate_gds_objs(
                        backend, shape, dtype, args.num_keys
                    )
                else:
                    objs = allocate_objs(
                        cpu_backend, shape, dtype, args.num_keys, use_gpu
                    )
                wr = measure_write(
                    backend, backend_name, write_keys, objs, args.batch, bytes_per_chunk
                )
                print(f"  (pre-populated for read scenario in {wr.wall_s:.2f}s)")
                for o in objs:
                    release(o)

        if args.scenario in ("read_cold", "all"):
            if args.drop_caches:
                drop_page_cache()
            res = measure_read(
                backend, backend_name, write_keys, "read_cold", bytes_per_chunk
            )
            results.append(res)
            _print_result(res)

        if args.scenario in ("read_warm", "all"):
            res = measure_read(
                backend, backend_name, write_keys, "read_warm", bytes_per_chunk
            )
            results.append(res)
            _print_result(res)

        if args.scenario == "all":
            if args.drop_caches:
                drop_page_cache()
            res = measure_read_batched(
                backend, backend_name, write_keys, "read_batched_cold",
                bytes_per_chunk, args.batch,
            )
            results.append(res)
            _print_result(res)

            res = measure_read_batched(
                backend, backend_name, write_keys, "read_batched_warm",
                bytes_per_chunk, args.batch,
            )
            results.append(res)
            _print_result(res)
    finally:
        backend.close()
        if cpu_backend is not None:
            cpu_backend.close()

    return results


def _print_result(r: PerfResult) -> None:
    """Pretty-print a single PerfResult row."""
    print(
        f"  [{r.operation:10s}] {r.backend:18s} "
        f"thr={r.throughput_gbps:6.3f} GiB/s  "
        f"chunks/s={r.chunks_per_s:8.1f}  "
        f"p50={r.p50_ms:7.3f}ms p99={r.p99_ms:7.3f}ms max={r.max_ms:7.3f}ms  "
        f"wall={r.wall_s:6.2f}s n={r.num_samples} {r.extra}"
    )


def write_csv(path: str, results: List[PerfResult], args: argparse.Namespace) -> None:
    """Append results to a CSV file (creates header if new)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "backend", "operation", "chunk_mb", "num_keys", "batch",
                    "odirect", "io_threads", "num_samples", "total_gb", "wall_s",
                    "throughput_gbps", "chunks_per_s", "mean_ms", "p50_ms",
                    "p95_ms", "p99_ms", "max_ms", "extra",
                ]
            )
        for r in results:
            w.writerow(
                [
                    r.backend, r.operation, args.chunk_mb, args.num_keys, args.batch,
                    args.odirect, args.io_threads, r.num_samples,
                    f"{r.total_gb:.4f}", f"{r.wall_s:.4f}",
                    f"{r.throughput_gbps:.4f}", f"{r.chunks_per_s:.2f}",
                    f"{r.mean_ms:.4f}", f"{r.p50_ms:.4f}", f"{r.p95_ms:.4f}",
                    f"{r.p99_ms:.4f}", f"{r.max_ms:.4f}", str(r.extra),
                ]
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["disk", "lightning", "gds", "both", "all"],
        default="both",
        help="'both' = disk+lightning, 'all' = disk+lightning+gds",
    )
    p.add_argument(
        "--scenario",
        choices=["write", "read_cold", "read_warm", "all", "evict"],
        default="all",
    )
    p.add_argument("--path-root", default="/mnt/lmbench")
    p.add_argument("--chunk-mb", type=float, default=4.0)
    p.add_argument("--num-keys", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--odirect", choices=["on", "off"], default="on")
    p.add_argument("--io-threads", type=int, default=4)
    p.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    p.add_argument("--max-cpu-gb", type=float, default=0.0, help="0 = auto-size")
    p.add_argument("--drop-caches", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--csv", default="")
    return p.parse_args()


def main() -> None:
    """Entry point: run the benchmark for the selected backend(s)."""
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    dst_device = args.device

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions, args=(loop,), daemon=True
    )
    loop_thread.start()

    if args.backend == "both":
        backends = ["disk", "lightning"]
    elif args.backend == "all":
        backends = ["disk", "lightning", "gds"]
    else:
        backends = [args.backend]
    all_results: List[PerfResult] = []
    try:
        for b in backends:
            all_results.extend(run_backend(b, args, dtype, dst_device, loop))
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)

    if args.csv:
        write_csv(args.csv, all_results, args)
        print(f"\nWrote {len(all_results)} rows to {args.csv}")


if __name__ == "__main__":
    main()
