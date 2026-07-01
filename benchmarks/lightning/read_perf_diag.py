#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: isolate read-path differences between per-file and single-file
I/O for large chunks.

Bypasses LMCache entirely — pure POSIX I/O comparison:
  1. per-file:   N individual files, each read with open()+readinto()
  2. single-file-seq: one big file, preadv at sequential offsets
  3. single-file-rnd: one big file, preadv at shuffled offsets
  4. single-file-posix_fadvise: preadv with POSIX_FADV_SEQUENTIAL

All use O_DIRECT with page-aligned buffers.
"""

import ctypes
import errno
import mmap
import os
import random
import statistics
import sys
import time

_ALIGNMENT = 4096

# Try loading libc for posix_fadvise
try:
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
    _POSIX_FADVISE = _LIBC.posix_fadvise
    _POSIX_FADVISE.argtypes = [
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
    ]
    _POSIX_FADVISE.restype = ctypes.c_int
    POSIX_FADV_SEQUENTIAL = 2
    POSIX_FADV_RANDOM = 1
except Exception:
    _POSIX_FADVISE = None


def _align_up(n, a):
    return (n + a - 1) // a * a


def _alloc_aligned(size):
    """Allocate a page-aligned buffer via mmap."""
    buf = mmap.mmap(-1, _align_up(size, _ALIGNMENT))
    return buf


def _drop_caches():
    os.sync()
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except PermissionError:
        print("WARNING: cannot drop caches (not root)")


def bench_per_file(data_dir, chunk_bytes, num_keys, use_odirect):
    """Read N individual files, each opened fresh."""
    times = []
    flags = os.O_RDONLY
    if use_odirect:
        flags |= os.O_DIRECT
    buf = _alloc_aligned(chunk_bytes)
    mv = memoryview(buf)[:chunk_bytes]

    for i in range(num_keys):
        path = os.path.join(data_dir, f"chunk_{i}.dat")
        t0 = time.perf_counter()
        fd = os.open(path, flags)
        os.preadv(fd, [mv], 0)
        os.close(fd)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    del mv
    buf.close()
    return times


def bench_single_file(data_dir, chunk_bytes, num_keys, use_odirect,
                       sequential=True, use_fadvise=False):
    """Read from a single pre-allocated file at slot offsets."""
    slot_size = _align_up(chunk_bytes, _ALIGNMENT)
    path = os.path.join(data_dir, "slots.dat")
    flags = os.O_RDONLY
    if use_odirect:
        flags |= os.O_DIRECT
    fd = os.open(path, flags)

    if use_fadvise and _POSIX_FADVISE is not None:
        _POSIX_FADVISE(fd, 0, slot_size * num_keys, POSIX_FADV_SEQUENTIAL)

    buf = _alloc_aligned(chunk_bytes)
    mv = memoryview(buf)[:chunk_bytes]
    order = list(range(num_keys))
    if not sequential:
        random.shuffle(order)

    times = []
    for i in order:
        offset = i * slot_size
        t0 = time.perf_counter()
        os.preadv(fd, [mv], offset)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    del mv
    os.close(fd)
    buf.close()
    return times


def setup_files(data_dir, chunk_bytes, num_keys):
    """Create test files: N individual files + 1 slot file."""
    os.makedirs(data_dir, exist_ok=True)
    slot_size = _align_up(chunk_bytes, _ALIGNMENT)

    # Use an aligned mmap buffer for O_DIRECT writes
    wbuf = mmap.mmap(-1, slot_size)
    wbuf[:chunk_bytes] = os.urandom(chunk_bytes)
    if slot_size > chunk_bytes:
        wbuf[chunk_bytes:slot_size] = b"\x00" * (slot_size - chunk_bytes)
    payload = memoryview(wbuf)[:slot_size]

    # Individual files
    for i in range(num_keys):
        path = os.path.join(data_dir, f"chunk_{i}.dat")
        if not os.path.exists(path):
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            os.write(fd, payload)
            os.close(fd)

    # Single slot file
    slot_path = os.path.join(data_dir, "slots.dat")
    if not os.path.exists(slot_path):
        fd = os.open(slot_path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
        for i in range(num_keys):
            os.write(fd, payload)
        os.close(fd)

    del payload
    wbuf.close()


def report(label, times, chunk_bytes):
    total = sum(times)
    n = len(times)
    total_gb = n * chunk_bytes / (1024**3)
    thr = total_gb / total if total > 0 else 0
    p50 = statistics.median(times) * 1000
    p99 = sorted(times)[int(n * 0.99)] * 1000 if n > 1 else times[0] * 1000
    mx = max(times) * 1000
    print(f"  {label:30s}  thr={thr:8.2f} GiB/s  "
          f"p50={p50:7.3f}ms  p99={p99:7.3f}ms  max={mx:7.3f}ms  "
          f"wall={total:.3f}s  n={n}")


def main():
    chunk_mbs = [4, 8, 16, 32]
    num_keys = 128
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/lmbench/diag"
    use_odirect = True

    for mb in chunk_mbs:
        chunk_bytes = int(mb * 1024 * 1024)
        test_dir = os.path.join(data_dir, f"chunk_{mb}mb")
        print(f"\n{'='*60}")
        print(f"  chunk_size={mb} MiB  num_keys={num_keys}  O_DIRECT={use_odirect}")
        print(f"{'='*60}")

        setup_files(test_dir, chunk_bytes, num_keys)

        # 1. per-file cold
        _drop_caches()
        t = bench_per_file(test_dir, chunk_bytes, num_keys, use_odirect)
        report("per-file (cold)", t, chunk_bytes)

        # 2. per-file warm
        t = bench_per_file(test_dir, chunk_bytes, num_keys, use_odirect)
        report("per-file (warm)", t, chunk_bytes)

        # 3. single-file sequential cold
        _drop_caches()
        t = bench_single_file(test_dir, chunk_bytes, num_keys, use_odirect,
                              sequential=True)
        report("single-file seq (cold)", t, chunk_bytes)

        # 4. single-file sequential warm
        t = bench_single_file(test_dir, chunk_bytes, num_keys, use_odirect,
                              sequential=True)
        report("single-file seq (warm)", t, chunk_bytes)

        # 5. single-file random cold
        _drop_caches()
        t = bench_single_file(test_dir, chunk_bytes, num_keys, use_odirect,
                              sequential=False)
        report("single-file random (cold)", t, chunk_bytes)

        # 6. single-file random warm
        t = bench_single_file(test_dir, chunk_bytes, num_keys, use_odirect,
                              sequential=False)
        report("single-file random (warm)", t, chunk_bytes)

        # 7. single-file with fadvise
        if _POSIX_FADVISE:
            _drop_caches()
            t = bench_single_file(test_dir, chunk_bytes, num_keys, use_odirect,
                                  sequential=True, use_fadvise=True)
            report("single-file fadvise (cold)", t, chunk_bytes)

            t = bench_single_file(test_dir, chunk_bytes, num_keys, use_odirect,
                                  sequential=True, use_fadvise=True)
            report("single-file fadvise (warm)", t, chunk_bytes)


if __name__ == "__main__":
    main()
