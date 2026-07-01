# SPDX-License-Identifier: Apache-2.0
"""Lightning PFS storage backend for LMCache (POSIX I/O).

This backend stores KV caches on a Lightning Parallel File System (or any
POSIX filesystem) using standard ``pwrite``/``pread`` I/O with CPU-side
buffers via ``LocalCPUBackend``. Data is serialized through host memory; it
has no dependency on GPU-direct I/O libraries or GPU memory registration,
making it suitable for environments without the Lightning kernel module or
where GPU-direct I/O is not desired.

Key design features:
- Pre-allocated slot files (1 MiB slots by default, configurable)
- Multi-slot chunks for large KV caches
- JSON checkpoint for O(1) restart recovery
- LRU eviction
- Runtime slot-file growth and lazy hole punching
- O_DIRECT I/O (with page-aligned buffers) and a thread pool for performance

The backend is loaded as a configurable storage plugin via
``storage_plugin.<name>.class_name = "LightningPosixBackend"``.
``LightningBackend`` is retained as a backward-compatible alias.
"""

# Standard
from __future__ import annotations

import asyncio
import ctypes
import errno
import json
import mmap
import os
import shutil
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Sequence,
    Union,
)

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import instrumentation
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StoragePluginInterface,
)
from lmcache.v1.storage_backend.dtype_utils import (
    TORCH_DTYPES,
    TORCH_DTYPES_INVERSE,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend.local_cpu_backend import (
        LocalCPUBackend,
    )

logger = init_logger(__name__)

_DEFAULT_SLOT_SIZE_MB = 1
_DEFAULT_NUM_SLOTS = 0  # 0 = unbounded (grow until disk budget is exhausted)
_DEFAULT_MAX_DISK_USAGE_GB = 16 * 1024  # 16 TiB per GPU
_DEFAULT_CHECKPOINT_INTERVAL_SEC = 60
_DEFAULT_IO_THREADS = 4
_DEFAULT_USE_ODIRECT = True
_DEFAULT_SLOTS_BATCH_SIZE = 256
_DEFAULT_ENABLE_HOLE_PUNCH = True
_CHECKPOINT_VERSION = 2

# Lightning alignment requirements
_ALIGNMENT = 4 * 1024  # 4 KiB alignment for O_DIRECT
# Growth granularity for the reusable per-thread O_DIRECT bounce buffer, so it
# is allocated/faulted once and reused across writes/reads instead of per call.
_BOUNCE_GRANULARITY = 8 * 1024 * 1024
_FALLOC_FL_KEEP_SIZE = 0x01
_FALLOC_FL_PUNCH_HOLE = 0x02

try:
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
    _LIBC_FALLOCATE = _LIBC.fallocate
    _LIBC_FALLOCATE.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]
    _LIBC_FALLOCATE.restype = ctypes.c_int
except (AttributeError, OSError):
    _LIBC_FALLOCATE = None


# Backward-compatible aliases; canonical dicts live in dtype_utils.
torch_dtypes: dict[torch.dtype, str] = TORCH_DTYPES
torch_dtypes_inverse: dict[str, torch.dtype] = TORCH_DTYPES_INVERSE


def _align_up(size: int, alignment: int) -> int:
    """Align size up to the nearest multiple of alignment."""
    return (size + alignment - 1) // alignment * alignment


def _align_down(size: int, alignment: int) -> int:
    """Align size down to the nearest multiple of alignment."""
    return size // alignment * alignment


def _fallocate_with_mode(fd: int, mode: int, offset: int, length: int) -> None:
    """Call Linux fallocate with mode flags via libc."""
    if _LIBC_FALLOCATE is None:
        raise OSError(errno.ENOSYS, "fallocate is unavailable")

    ret = _LIBC_FALLOCATE(fd, mode, offset, length)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _find_contiguous_run(
    sorted_free: list[int], num_slots_needed: int
) -> int | None:
    """Find a contiguous run of free slots in a sorted list.

    Uses a single linear scan: walks the sorted free list and tracks the
    current run length.  O(N) time, O(1) extra space.

    Args:
        sorted_free: Sorted list of free slot IDs (must be sorted,
            no duplicates).
        num_slots_needed: Number of contiguous slots required (must be
            >= 1).

    Returns:
        Index into sorted_free where the run starts, or None if no
        contiguous run of the required length exists.
    """
    if len(sorted_free) < num_slots_needed:
        return None

    if num_slots_needed == 1:
        return 0

    run_start_idx = 0
    run_length = 1

    for i in range(1, len(sorted_free)):
        if sorted_free[i] == sorted_free[i - 1] + 1:
            run_length += 1
            if run_length >= num_slots_needed:
                return run_start_idx
        else:
            run_start_idx = i
            run_length = 1

    return None


# Backward-compatible alias for tests that import by the old public name.
find_contiguous_run = _find_contiguous_run


@dataclass
class _IndexEntry:
    """In-memory index entry for a stored chunk."""

    slot_id: int
    num_slots: int
    shape: tuple[int, ...]
    dtype: str
    fmt: str
    payload_len: int

    def to_dict(
        self,
    ) -> dict[str, Union[int, str, list[int]]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slot_id": self.slot_id,
            "num_slots": self.num_slots,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "fmt": self.fmt,
            "payload_len": self.payload_len,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Union[int, str, list[int]]]
    ) -> _IndexEntry:
        """Create from dictionary."""
        return cls(
            slot_id=int(data["slot_id"]),
            num_slots=int(data["num_slots"]),
            shape=tuple(int(x) for x in data["shape"]),  # type: ignore[union-attr]
            dtype=str(data["dtype"]),
            fmt=str(data["fmt"]),
            payload_len=int(data["payload_len"]),
        )


class _LRUCache:
    """Simple LRU cache using OrderedDict.

    **Not thread-safe.** All methods must be called while holding the
    owning backend's ``index_lock``.
    """

    def __init__(self) -> None:
        self._cache: OrderedDict[CacheEngineKey, None] = OrderedDict()

    def touch(self, key: CacheEngineKey) -> None:
        """Mark key as recently used."""
        if key in self._cache:
            self._cache.move_to_end(key)

    def add(self, key: CacheEngineKey) -> None:
        """Add key to cache."""
        self._cache[key] = None
        self._cache.move_to_end(key)

    def remove(self, key: CacheEngineKey) -> None:
        """Remove key from cache."""
        if key in self._cache:
            del self._cache[key]

    def peek_oldest(self) -> CacheEngineKey | None:
        """Return oldest key without removing it."""
        if self._cache:
            return next(iter(self._cache))
        return None

    def __len__(self) -> int:
        return len(self._cache)


class LightningPosixBackend(StoragePluginInterface):
    """Lightning PFS storage backend using standard POSIX I/O.

    Stores KV caches in pre-allocated, fixed-size slots within a single backing
    file. Uses ``pwrite``/``pread`` (optionally with O_DIRECT) and stages data
    through host memory via ``LocalCPUBackend``. A JSON checkpoint enables O(1)
    restart recovery, and an LRU policy evicts cold entries when the slot file
    is full.
    """

    def __init__(
        self,
        dst_device: str = "cuda",
        config: LMCacheEngineConfig | None = None,
        metadata: LMCacheMetadata | None = None,
        local_cpu_backend: LocalCPUBackend | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """Initialize the backend with slot file and checkpoint.

        Args:
            dst_device: Target device for tensor operations (e.g., "cuda:0").
            config: LMCache engine configuration.
            metadata: LMCache metadata describing the model.
            local_cpu_backend: Local CPU backend used for memory allocation.
            loop: Asyncio event loop for background operations.

        Raises:
            ValueError: If config is None or required configuration is missing.
        """
        if config is None:
            raise ValueError("config is required for LightningPosixBackend")

        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )

        # Get device ID from dst_device
        if ":" in dst_device:
            self.device_id = int(dst_device.split(":")[1])
        else:
            self.device_id = 0

        # Configuration from extra_config
        extra_config = config.extra_config or {}

        self.data_path = extra_config.get(
            "lightning.data_path", "/mnt/lightning/lmcache"
        )
        # num_slots: 0 (default) = unbounded, >0 = hard cap on slot count.
        self.num_slots = int(
            extra_config.get("lightning.num_slots", _DEFAULT_NUM_SLOTS)
        )
        # Disk budget: 0 = no limit; >0 = max bytes the slot file may occupy.
        max_disk_gb = float(
            extra_config.get(
                "lightning.max_disk_usage_gb", _DEFAULT_MAX_DISK_USAGE_GB
            )
        )
        self.max_disk_usage_bytes = (
            int(max_disk_gb * 1024 * 1024 * 1024) if max_disk_gb > 0 else 0
        )

        # Slot size: if not explicitly configured, derive from metadata so
        # each KV chunk fits in exactly one slot (eliminates fragmentation).
        if "lightning.slot_size_mb" in extra_config:
            slot_size_bytes = (
                int(extra_config["lightning.slot_size_mb"]) * 1024 * 1024
            )
        else:
            if metadata is None:
                raise ValueError(
                    "metadata is required when lightning.slot_size_mb "
                    "is not configured (needed to auto-size slots)"
                )
            elem_size = torch.finfo(metadata.kv_dtype).bits // 8
            chunk_bytes = 1
            for d in metadata.kv_shape:
                chunk_bytes *= d
            chunk_bytes *= elem_size
            if chunk_bytes < 1:
                raise ValueError(
                    "Computed chunk size is 0 bytes; "
                    "kv_shape=%s has a zero dimension" % (metadata.kv_shape,)
                )
            slot_size_bytes = chunk_bytes
            logger.info(
                "Auto-sized slot to %d bytes (%d MiB) from metadata "
                "kv_shape=%s dtype=%s",
                slot_size_bytes,
                slot_size_bytes // (1024 * 1024),
                metadata.kv_shape,
                metadata.kv_dtype,
            )

        # Ensure slot size is 4 KiB aligned
        self.slot_size = _align_up(slot_size_bytes, _ALIGNMENT)
        if self.slot_size != slot_size_bytes:
            logger.warning(
                "Slot size adjusted from %d to %d bytes for 4 KiB alignment",
                slot_size_bytes,
                self.slot_size,
            )

        self.checkpoint_interval_sec = int(
            extra_config.get(
                "lightning.checkpoint_interval_sec", _DEFAULT_CHECKPOINT_INTERVAL_SEC
            )
        )
        self.io_threads = int(
            extra_config.get("lightning.io_threads", _DEFAULT_IO_THREADS)
        )
        self.use_odirect = bool(
            extra_config.get("lightning.use_odirect", _DEFAULT_USE_ODIRECT)
        )
        self.slots_batch_size = int(
            extra_config.get("lightning.slots_batch_size", _DEFAULT_SLOTS_BATCH_SIZE)
        )
        if self.slots_batch_size < 1:
            raise ValueError(
                f"lightning.slots_batch_size must be >= 1, got {self.slots_batch_size}"
            )
        self.enable_hole_punch = bool(
            extra_config.get("lightning.enable_hole_punch", _DEFAULT_ENABLE_HOLE_PUNCH)
        )
        self._hole_punch_supported = True
        # Set if an O_DIRECT zero-copy transfer hits EINVAL (buffer not
        # block-aligned); thereafter all O_DIRECT I/O uses the bounce buffer.
        self._direct_unaligned = False
        # Per-thread reusable, page-aligned bounce buffer for O_DIRECT.
        self._bounce_tls = threading.local()

        if self.local_cpu_backend is None:
            logger.warning(
                "LightningPosixBackend created without a LocalCPUBackend; "
                "retrieval will fail until one is provided "
                "(enable local_cpu with max_local_cpu_size > 0)."
            )

        # Track keys whose write has been dispatched but not yet indexed.
        self._inflight_puts: set[CacheEngineKey] = set()
        self._inflight_lock = threading.Lock()
        self._closed = False

        # File paths
        os.makedirs(self.data_path, exist_ok=True)
        self.data_file = os.path.join(
            self.data_path, f"lmcache_gpu{self.device_id}.dat"
        )
        self.checkpoint_file = os.path.join(
            self.data_path, f"lmcache_gpu{self.device_id}.idx"
        )

        # Open and pre-allocate slot file
        self._open_slot_file()

        # Guard: if anything below raises, close the fd we just opened.
        try:
            # In-memory index: key -> IndexEntry
            self.index: dict[CacheEngineKey, _IndexEntry] = {}
            self.index_lock = threading.Lock()

            # Free slot management
            self.free_slots: list[int] = list(range(self.current_slots))
            self._free_list_sorted = True
            self.lru = _LRUCache()
            self._pending_punch_slots: set[int] = set()

            # Thread pool for I/O operations
            self._executor = ThreadPoolExecutor(max_workers=self.io_threads)

            # Checkpoint state
            self._checkpoint_dirty = False
            self._checkpoint_lock = threading.Lock()
            self._stop_event = threading.Event()

            # Load checkpoint
            self._load_checkpoint()

            # Start background checkpoint thread
            if self.checkpoint_interval_sec > 0:
                self._checkpoint_thread: threading.Thread | None = (
                    threading.Thread(
                        target=self._checkpoint_worker, daemon=True
                    )
                )
                self._checkpoint_thread.start()
            else:
                self._checkpoint_thread = None
        except Exception:
            if hasattr(self, "fd") and self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            raise

        logger.info(
            "LightningPosixBackend initialized: device=%s, "
            "slots=%d, slot_size=%dMiB, allocated=%dMiB, "
            "capacity=%s, entries=%d, free_slots=%d",
            dst_device,
            self.current_slots,
            self.slot_size // 1024 // 1024,
            self.current_slots * self.slot_size // 1024 // 1024,
            self._capacity_desc(),
            len(self.index),
            len(self.free_slots),
        )

    # ------------------------------------------------------------------
    # Slot file / checkpoint lifecycle
    # ------------------------------------------------------------------

    def _capacity_desc(self) -> str:
        """Return a human-readable string describing the capacity limit."""
        parts: list[str] = []
        if self.num_slots > 0:
            max_mib = self.num_slots * self.slot_size // 1024 // 1024
            parts.append("%d slots (%dMiB)" % (self.num_slots, max_mib))
        if self.max_disk_usage_bytes > 0:
            budget_mib = self.max_disk_usage_bytes // 1024 // 1024
            if budget_mib >= 1024:
                parts.append("%dGiB disk budget" % (budget_mib // 1024))
            else:
                parts.append("%dMiB disk budget" % budget_mib)
        return ", ".join(parts) if parts else "unbounded"

    def _open_slot_file(self) -> None:
        """Open slot file and derive currently allocated slot count."""
        flags = os.O_RDWR | os.O_CREAT
        if self.use_odirect:
            try:
                flags |= os.O_DIRECT
            except AttributeError:
                logger.warning("O_DIRECT not available on this platform")
                self.use_odirect = False

        try:
            self.fd = os.open(self.data_file, flags)
            file_size = os.fstat(self.fd).st_size
            if file_size % self.slot_size != 0:
                aligned_size = _align_down(file_size, self.slot_size)
                logger.warning(
                    "Slot file size %d is not multiple of slot_size %d, "
                    "truncating to %d",
                    file_size,
                    self.slot_size,
                    aligned_size,
                )
                os.ftruncate(self.fd, aligned_size)
                file_size = aligned_size

            self.current_slots = file_size // self.slot_size
            if self.num_slots > 0 and self.current_slots > self.num_slots:
                logger.warning(
                    "Existing file has %d slots exceeding configured limit %d; "
                    "capping to configured limit",
                    self.current_slots,
                    self.num_slots,
                )
                self.current_slots = self.num_slots
                os.ftruncate(self.fd, self.current_slots * self.slot_size)

            max_desc = self._capacity_desc()
            logger.info(
                "Slot file opened: %s, allocated=%dMiB, capacity=%s, "
                "O_DIRECT=%s",
                self.data_file,
                self.current_slots * self.slot_size // 1024 // 1024,
                max_desc,
                self.use_odirect,
            )
        except OSError as e:
            logger.error("Failed to open slot file: %s", e)
            raise

    def _load_checkpoint(self) -> None:
        """Load index from checkpoint file on startup."""
        if not os.path.exists(self.checkpoint_file):
            logger.info("No checkpoint file found, starting fresh")
            return

        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint = json.load(f)

            # Validate version
            version = checkpoint.get("version", 1)
            if version not in (1, _CHECKPOINT_VERSION):
                logger.warning(
                    "Unsupported checkpoint version %d, starting fresh",
                    version,
                )
                return

            # Validate model compatibility if metadata available
            if self.metadata is not None:
                checkpoint_model_id = checkpoint.get("model_id")
                if checkpoint_model_id != self.metadata.model_name:
                    logger.warning(
                        "Checkpoint model mismatch: %s vs %s, starting fresh",
                        checkpoint_model_id,
                        self.metadata.model_name,
                    )
                    return

            if version >= _CHECKPOINT_VERSION:
                # Validate slot_size consistency — mismatched sizes mean the
                # on-disk layout is incompatible with the current config.
                checkpoint_slot_size = checkpoint.get("slot_size")
                if (
                    checkpoint_slot_size is not None
                    and int(checkpoint_slot_size) != self.slot_size
                ):
                    logger.warning(
                        "Checkpoint slot_size %d != current %d, "
                        "starting fresh",
                        checkpoint_slot_size,
                        self.slot_size,
                    )
                    return

                checkpoint_slots = int(
                    checkpoint.get("current_slots", self.current_slots)
                )
                checkpoint_slots = max(checkpoint_slots, 0)
                if self.num_slots > 0:
                    checkpoint_slots = min(checkpoint_slots, self.num_slots)
                if checkpoint_slots > self.current_slots:
                    new_size = checkpoint_slots * self.slot_size
                    os.ftruncate(self.fd, new_size)
                    self.current_slots = checkpoint_slots

            # Rebuild index
            for key_str, entry_dict in checkpoint.get("entries", {}).items():
                try:
                    key = CacheEngineKey.from_string(key_str)
                    entry = _IndexEntry.from_dict(entry_dict)
                    if entry.slot_id + entry.num_slots > self.current_slots:
                        logger.warning(
                            "Skipping checkpoint entry %s (slots %d-%d) outside "
                            "allocated slot range %d",
                            key_str,
                            entry.slot_id,
                            entry.slot_id + entry.num_slots - 1,
                            self.current_slots,
                        )
                        continue
                    self.index[key] = entry
                    self.lru.add(key)
                except Exception as e:
                    logger.warning("Failed to load entry for key %s: %s", key_str, e)

            # Update free slots (produced in sorted order by construction)
            used_slots = set()
            for entry in self.index.values():
                for slot_offset in range(entry.num_slots):
                    used_slots.add(entry.slot_id + slot_offset)
            self.free_slots = [
                s for s in range(self.current_slots) if s not in used_slots
            ]
            self._free_list_sorted = True

            logger.info(
                "Loaded checkpoint with %d entries, %d free slots",
                len(self.index),
                len(self.free_slots),
            )

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load checkpoint: %s", e)
            # Start fresh on error
            self.index.clear()
            self.lru = _LRUCache()
            self.free_slots = list(range(self.current_slots))
            self._free_list_sorted = True

    def _save_checkpoint(self) -> None:
        """Save index to checkpoint file atomically.

        Uses ``_checkpoint_lock`` to serialize concurrent saves (e.g. from
        the background checkpoint worker and ``close()``).
        """
        with self._checkpoint_lock:
            if not self._checkpoint_dirty:
                return
            try:
                with self.index_lock:
                    checkpoint = {
                        "version": _CHECKPOINT_VERSION,
                        "model_id": (
                            self.metadata.model_name if self.metadata else ""
                        ),
                        "slot_size": self.slot_size,
                        "current_slots": self.current_slots,
                        "entries": {
                            k.to_string(): e.to_dict()
                            for k, e in self.index.items()
                        },
                    }

                # Atomic write
                tmp_file = self.checkpoint_file + ".tmp"
                with open(tmp_file, "w") as f:
                    json.dump(checkpoint, f)
                os.rename(tmp_file, self.checkpoint_file)

                self._checkpoint_dirty = False

            except OSError as e:
                logger.error("Failed to save checkpoint: %s", e)

    def _checkpoint_worker(self) -> None:
        """Background thread to periodically save checkpoint and punch holes."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.checkpoint_interval_sec)
            if self._stop_event.is_set():
                break
            with self._checkpoint_lock:
                should_save = self._checkpoint_dirty
            if should_save:
                self._save_checkpoint()
            if self.enable_hole_punch and self._hole_punch_supported:
                self._punch_holes()

    def _mark_checkpoint_dirty(self) -> None:
        """Mark checkpoint as needing save."""
        with self._checkpoint_lock:
            self._checkpoint_dirty = True

    # ------------------------------------------------------------------
    # Slot allocation
    # ------------------------------------------------------------------

    @staticmethod
    def _find_contiguous_run(
        sorted_free: list[int], num_slots_needed: int
    ) -> int | None:
        """Delegate to the module-level :func:`_find_contiguous_run`."""
        return _find_contiguous_run(sorted_free, num_slots_needed)

    def _ensure_free_list_sorted(self) -> None:
        """Sort the free list if it has been modified since the last sort.

        Caller must hold index_lock.
        """
        if not self._free_list_sorted:
            self.free_slots.sort()
            self._free_list_sorted = True

    def _can_grow(self, extra_bytes: int) -> bool:
        """Check whether growing the slot file by *extra_bytes* is allowed.

        Respects both the ``num_slots`` hard cap and the
        ``max_disk_usage_bytes`` disk budget.  Caller must hold index_lock.
        """
        if self.num_slots > 0 and self.current_slots >= self.num_slots:
            return False

        current_size = self.current_slots * self.slot_size
        proposed_size = current_size + extra_bytes

        # Disk budget check
        if self.max_disk_usage_bytes > 0 and proposed_size > self.max_disk_usage_bytes:
            return False

        # Filesystem free-space check (leave a 1 GiB safety margin)
        try:
            usage = shutil.disk_usage(self.data_path)
            if usage.free < extra_bytes + 1024 * 1024 * 1024:
                logger.warning(
                    "Insufficient disk space to grow slot file: "
                    "need %dMiB, free %dMiB",
                    extra_bytes // 1024 // 1024,
                    usage.free // 1024 // 1024,
                )
                return False
        except OSError:
            pass  # best-effort; allow growth if we can't query free space

        return True

    def _grow_file_unlocked(self, min_slots_needed: int) -> bool:
        """Grow the slot file by batch size (or *min_slots_needed*).

        Growth is bounded by ``num_slots`` (if > 0), ``max_disk_usage_bytes``
        (if > 0), and available filesystem space.  When ``num_slots`` is 0 and
        no disk budget is set the file grows without an artificial cap.

        Caller must hold index_lock.
        """
        slots_to_add = max(min_slots_needed, self.slots_batch_size)
        if self.num_slots > 0:
            slots_to_add = min(
                slots_to_add, self.num_slots - self.current_slots
            )
        if slots_to_add <= 0:
            return False

        extra_bytes = slots_to_add * self.slot_size
        if not self._can_grow(extra_bytes):
            # Try a smaller growth (just min_slots_needed)
            if slots_to_add > min_slots_needed:
                slots_to_add = min_slots_needed
                extra_bytes = slots_to_add * self.slot_size
                if not self._can_grow(extra_bytes):
                    return False
            else:
                return False

        old_slots = self.current_slots
        new_slots = old_slots + slots_to_add
        new_size = new_slots * self.slot_size

        try:
            os.ftruncate(self.fd, new_size)
        except OSError as e:
            logger.warning(
                "Failed to grow slot file from %d to %d slots: %s",
                old_slots,
                new_slots,
                e,
            )
            return False

        self.free_slots.extend(range(old_slots, new_slots))
        self.current_slots = new_slots
        # New slot IDs (>= old_slots) go at the end; list stays sorted only
        # if it was sorted before and all existing IDs < old_slots.  Don't
        # assume — let _ensure_free_list_sorted() fix it lazily.
        self._free_list_sorted = False
        self._mark_checkpoint_dirty()
        logger.info(
            "Grew slot file: slots=%d (added=%d, size=%dMiB)",
            self.current_slots,
            slots_to_add,
            new_size // 1024 // 1024,
        )
        return True

    def _queue_punch_slots_unlocked(self, slot_id: int, num_slots: int) -> None:
        """Queue freed slots for lazy hole punching (caller holds index_lock)."""
        if not self.enable_hole_punch or not self._hole_punch_supported:
            return
        for i in range(num_slots):
            self._pending_punch_slots.add(slot_id + i)

    def _allocate_slots(self, num_slots_needed: int) -> int | None:
        """Allocate contiguous slots for a chunk.

        Tries three strategies in order:

        1. **Fast path** -- find a contiguous run in the current free list.
        2. **Growth path** -- extend the slot file (bounded by ``num_slots``,
           ``max_disk_usage_bytes``, and filesystem free space) then retry.
        3. **Eviction path** -- evict LRU entries one at a time until a
           contiguous run forms.

        Args:
            num_slots_needed: Number of contiguous slots required.

        Returns:
            Starting slot ID or None if allocation failed.

        Raises:
            ValueError: If num_slots_needed < 1.
        """
        if num_slots_needed < 1:
            raise ValueError(f"num_slots_needed must be >= 1, got {num_slots_needed}")
        if self.num_slots > 0 and num_slots_needed > self.num_slots:
            logger.error(
                "Chunk requires %d contiguous slots but "
                "total capacity is only %d slots",
                num_slots_needed,
                self.num_slots,
            )
            return None

        with self.index_lock:
            self._ensure_free_list_sorted()

            # --- fast path: find a contiguous run in current free list ---
            run_idx = self._find_contiguous_run(self.free_slots, num_slots_needed)
            if run_idx is not None:
                slot_id = self.free_slots[run_idx]
                del self.free_slots[run_idx : run_idx + num_slots_needed]
                if self.enable_hole_punch and self._hole_punch_supported:
                    for s in range(slot_id, slot_id + num_slots_needed):
                        self._pending_punch_slots.discard(s)
                # Slice delete from a sorted list stays sorted
                return slot_id

            # --- growth path: extend file and retry contiguous allocation ---
            if self._grow_file_unlocked(num_slots_needed):
                run_idx = self._find_contiguous_run(self.free_slots, num_slots_needed)
                if run_idx is not None:
                    slot_id = self.free_slots[run_idx]
                    del self.free_slots[run_idx : run_idx + num_slots_needed]
                    if self.enable_hole_punch and self._hole_punch_supported:
                        for s in range(slot_id, slot_id + num_slots_needed):
                            self._pending_punch_slots.discard(s)
                    return slot_id

            # --- slow path: evict until a contiguous run appears ---
            # Build a set for O(1) adjacency checks during eviction
            free_set = set(self.free_slots)

            max_evictions = len(self.index)
            for _ in range(max_evictions):
                victim_key = self.lru.peek_oldest()
                if victim_key is None:
                    break

                victim_entry = self.index[victim_key]

                # Free victim's slots
                for offset in range(victim_entry.num_slots):
                    free_set.add(victim_entry.slot_id + offset)
                self._queue_punch_slots_unlocked(
                    victim_entry.slot_id, victim_entry.num_slots
                )

                del self.index[victim_key]
                self.lru.remove(victim_key)
                self._mark_checkpoint_dirty()

                # Check if the victim's region now forms a large enough
                # contiguous run (walk left and right from victim's range)
                run_start = victim_entry.slot_id
                while run_start > 0 and (run_start - 1) in free_set:
                    run_start -= 1
                run_end = victim_entry.slot_id + victim_entry.num_slots - 1
                while (run_end + 1) in free_set:
                    run_end += 1

                run_length = run_end - run_start + 1
                if run_length >= num_slots_needed:
                    # Claim directly from the known run_start
                    for s in range(run_start, run_start + num_slots_needed):
                        free_set.remove(s)
                        self._pending_punch_slots.discard(s)
                    self.free_slots = sorted(free_set)
                    self._free_list_sorted = True
                    return run_start

            # Exhausted all entries without finding a contiguous run
            self.free_slots = sorted(free_set)
            self._free_list_sorted = True
            logger.error(
                f"Failed to find {num_slots_needed} contiguous slots "
                f"even after eviction (free_slots={len(self.free_slots)})"
            )
            return None

    def _free_slots_unlocked(self, slot_id: int, num_slots: int) -> None:
        """Free slots back to the free list (caller must hold index_lock)."""
        for i in range(num_slots):
            self.free_slots.append(slot_id + i)
        self._queue_punch_slots_unlocked(slot_id, num_slots)
        self._free_list_sorted = False

    def _free_slots(self, slot_id: int, num_slots: int) -> None:
        """Free slots back to the free list."""
        with self.index_lock:
            self._free_slots_unlocked(slot_id, num_slots)

    def _punch_hole(self, offset: int, length: int) -> None:
        """Punch a hole in the slot file range without changing file size."""
        mode = _FALLOC_FL_KEEP_SIZE | _FALLOC_FL_PUNCH_HOLE
        _fallocate_with_mode(self.fd, mode, offset, length)

    def _punch_holes(self) -> None:
        """Punch queued free-slot ranges in batches.

        The pending set is snapshotted under ``index_lock``, but the actual
        (potentially slow) ``fallocate`` syscalls are issued outside the lock
        so that get/put/contains operations are not blocked on storage I/O.
        """
        with self.index_lock:
            if (
                not self.enable_hole_punch
                or not self._hole_punch_supported
                or not self._pending_punch_slots
            ):
                return
            pending_slots = set(self._pending_punch_slots)
            self._pending_punch_slots.clear()

        sorted_slots = sorted(pending_slots)

        try:
            run_start = sorted_slots[0]
            run_end = sorted_slots[0]
            for slot in sorted_slots[1:]:
                if slot == run_end + 1:
                    run_end = slot
                    continue
                offset = run_start * self.slot_size
                length = (run_end - run_start + 1) * self.slot_size
                self._punch_hole(offset, length)
                run_start = slot
                run_end = slot

            offset = run_start * self.slot_size
            length = (run_end - run_start + 1) * self.slot_size
            self._punch_hole(offset, length)
        except OSError as e:
            unsupported_errno = {
                errno.ENOSYS,
                errno.EOPNOTSUPP,
                errno.ENOTSUP,
            }
            if e.errno in unsupported_errno:
                self._hole_punch_supported = False
                with self.index_lock:
                    self._pending_punch_slots.clear()
                logger.warning(
                    "Hole punching disabled (unsupported by filesystem/kernel): %s",
                    e,
                )
            else:
                with self.index_lock:
                    self._pending_punch_slots.update(pending_slots)
                logger.warning(
                    "Hole punching failed, will retry later: %s",
                    e,
                )

    def _get_file_offset(self, slot_id: int) -> int:
        """Calculate file offset for a given slot."""
        return slot_id * self.slot_size

    # ------------------------------------------------------------------
    # Serialization / POSIX I/O
    # ------------------------------------------------------------------

    def _aligned_bounce(self, min_size: int) -> mmap.mmap:
        """Return a reusable, page-aligned per-thread bounce buffer.

        Anonymous ``mmap`` mappings are page-aligned (satisfying O_DIRECT
        buffer-alignment) and are allocated/faulted once per thread then reused
        across calls, avoiding the per-call ``mmap``/page-fault/``munmap``
        overhead that otherwise dominates the O_DIRECT bounce path.

        Args:
            min_size: Minimum required buffer size in bytes.

        Returns:
            An ``mmap`` object of at least ``min_size`` bytes.
        """
        tls = getattr(self, "_bounce_tls", None)
        if tls is None:
            tls = self._bounce_tls = threading.local()
        buf = getattr(tls, "buf", None)
        if buf is None or getattr(tls, "size", 0) < min_size:
            if buf is not None:
                buf.close()
            size = _align_up(min_size, _BOUNCE_GRANULARITY)
            buf = mmap.mmap(-1, size)
            buf.madvise(mmap.MADV_WILLNEED)
            tls.buf = buf
            tls.size = size
        return buf

    def _can_direct(self, payload_len: int) -> bool:
        """Whether a chunk is eligible for a zero-copy (no-bounce) transfer.

        Buffered I/O always is. Under O_DIRECT the transfer length must be a
        multiple of the block size; the buffer address is trusted to be
        block-aligned (as the pinned-memory allocator provides), with a
        one-shot EINVAL fallback in case it is not.

        Args:
            payload_len: Logical chunk size in bytes.

        Returns:
            True if a direct (zero-copy) transfer should be attempted.
        """
        if not self.use_odirect:
            return True
        return not self._direct_unaligned and payload_len % _ALIGNMENT == 0

    def _write_obj(self, offset: int, memory_obj: MemoryObj) -> int:
        """Write a MemoryObj's bytes to the slot file with no intermediate copy.

        Writes directly from the object's ``byte_array`` (a view over the
        tensor storage). Falls back to a padded bounce buffer only when
        O_DIRECT is enabled and the chunk is not block-aligned (or a direct
        transfer reports EINVAL because the buffer is not block-aligned).

        Args:
            offset: Block-aligned file offset to write at.
            memory_obj: The object whose bytes are persisted.

        Returns:
            The number of logical (unpadded) payload bytes written.

        Raises:
            IOError: If fewer bytes than expected are written.
        """
        buf = memory_obj.byte_array
        payload_len = len(buf)
        if self._can_direct(payload_len):
            try:
                written = os.pwrite(self.fd, buf, offset)
                if written != payload_len:
                    raise IOError(f"Short write: {written} of {payload_len} bytes")
                return payload_len
            except OSError as e:
                if e.errno != errno.EINVAL or not self.use_odirect:
                    raise
                self._direct_unaligned = True
        # Bounce path: copy into a reusable page-aligned buffer, then write.
        aligned_len = _align_up(payload_len, _ALIGNMENT)
        bounce = memoryview(self._aligned_bounce(aligned_len))
        bounce[:payload_len] = buf.cast("B")
        if aligned_len > payload_len:
            bounce[payload_len:aligned_len] = b"\x00" * (aligned_len - payload_len)
        written = os.pwrite(self.fd, bounce[:aligned_len], offset)
        if written != aligned_len:
            raise IOError(f"Short write: {written} of {aligned_len} bytes")
        return payload_len

    def _read_obj(self, offset: int, memory_obj: MemoryObj, payload_len: int) -> None:
        """Read a chunk from the slot file directly into a MemoryObj.

        Reads straight into the object's ``byte_array`` with no intermediate
        copy. Falls back to a bounce buffer plus one copy only when O_DIRECT
        is enabled and the chunk is not block-aligned (or a direct transfer
        reports EINVAL).

        Args:
            offset: Block-aligned file offset to read from.
            memory_obj: Destination object (allocated with matching size).
            payload_len: Number of logical payload bytes to read.

        Raises:
            IOError: If fewer bytes than expected are read.
        """
        buf = memory_obj.byte_array
        if self._can_direct(payload_len):
            try:
                nread = os.preadv(self.fd, [buf], offset)
                if nread < payload_len:
                    raise IOError(f"Short read: {nread} of {payload_len} bytes")
                return
            except OSError as e:
                if e.errno != errno.EINVAL or not self.use_odirect:
                    raise
                self._direct_unaligned = True
        # Bounce path: read into a reusable page-aligned buffer, then copy out.
        aligned_len = _align_up(payload_len, _ALIGNMENT)
        bounce = memoryview(self._aligned_bounce(aligned_len))
        nread = os.preadv(self.fd, [bounce[:aligned_len]], offset)
        if nread < payload_len:
            raise IOError(f"Short read: {nread} of {payload_len} bytes")
        # Normalize the ctypes-backed view to a plain byte format for assignment.
        memory_obj.byte_array.cast("B")[:payload_len] = bounce[:payload_len]

    # ------------------------------------------------------------------
    # Public StorageBackendInterface surface
    # ------------------------------------------------------------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check whether key is in the storage backend.

        Args:
            key: The key to check.
            pin: Whether to pin the key (not implemented).

        Returns:
            True if the key exists.
        """
        with self.index_lock:
            exists = key in self.index
            if exists:
                self.lru.touch(key)
        return exists

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether key has an in-flight (not yet indexed) put task.

        Puts are dispatched to a thread pool and the index is only updated
        once the write completes, so this tracks keys that are mid-write to
        avoid duplicate stores and premature misses.
        """
        with self._inflight_lock:
            return key in self._inflight_puts

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: list[MemoryObj],
        transfer_spec: object = None,
        on_complete_callback: Callable[[CacheEngineKey], None] | None = None,
    ) -> list[Future]:
        """Submit batched put tasks to store KV caches.

        Args:
            keys: The keys of the MemoryObjs.
            memory_objs: The MemoryObjs to store.
            transfer_spec: Optional transfer specification (unused).
            on_complete_callback: Optional callback per key on completion.

        Returns:
            List of Future objects for async tracking.
        """
        futures = []
        for key, memory_obj in zip(keys, memory_objs, strict=True):
            future = self._submit_single_put(key, memory_obj, on_complete_callback)
            futures.append(future)
        return futures

    def _submit_single_put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Callable[[CacheEngineKey], None] | None,
    ) -> Future:
        """Submit a single put task."""
        memory_obj.ref_count_up()

        # Calculate slots needed
        tensor = memory_obj.tensor
        if tensor is None:
            raise ValueError("MemoryObj has no tensor")
        tensor_bytes = tensor.numel() * tensor.element_size()
        num_slots_needed = (tensor_bytes + self.slot_size - 1) // self.slot_size

        # Allocate slots
        slot_id = self._allocate_slots(num_slots_needed)
        if slot_id is None:
            memory_obj.ref_count_down()
            logger.warning("No slot available for key %s", key)
            future: Future = Future()
            future.set_exception(RuntimeError("No slots available"))
            return future

        # Track the key as in-flight until the write completes
        with self._inflight_lock:
            self._inflight_puts.add(key)

        # Submit async write via thread pool
        future = self._executor.submit(
            self._write_slot_sync,
            key,
            memory_obj,
            slot_id,
            num_slots_needed,
            on_complete_callback,
        )
        return future

    def _write_slot_sync(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        slot_id: int,
        num_slots: int,
        on_complete_callback: Callable[[CacheEngineKey], None] | None,
    ) -> None:
        """Synchronous write to slot (runs in thread pool)."""
        try:
            tensor = memory_obj.tensor
            if tensor is None:
                raise ValueError("MemoryObj has no tensor")
            offset = self._get_file_offset(slot_id)
            payload_len = len(memory_obj.byte_array)

            with instrumentation.Timer(
                lambda ms: instrumentation.log_trace(
                    "LIGHTNING_PERF",
                    f"Write key={key.to_string()} "
                    f"slots={num_slots} payload={payload_len}B "
                    f"latency={ms:.2f}ms",
                )
            ):
                self._write_obj(offset, memory_obj)

            # Update index
            entry = _IndexEntry(
                slot_id=slot_id,
                num_slots=num_slots,
                shape=tuple(tensor.shape),
                dtype=torch_dtypes[tensor.dtype],
                fmt=memory_obj.metadata.fmt.value,
                payload_len=payload_len,
            )

            with self.index_lock:
                # Free old slots if key exists
                if key in self.index:
                    old_entry = self.index[key]
                    self._free_slots_unlocked(old_entry.slot_id, old_entry.num_slots)
                    self.lru.remove(key)

                self.index[key] = entry
                self.lru.add(key)

            self._mark_checkpoint_dirty()

            if instrumentation.is_enabled():
                logger.info(
                    "[LIGHTNING_TRACE] Put complete: key=%s "
                    "slot_id=%d num_slots=%d payload=%dB",
                    key.to_string(),
                    slot_id,
                    num_slots,
                    payload_len,
                )

        except Exception as e:
            logger.error("Failed to write slot for key %s: %s", key, e)
            # Free the allocated slots on failure (must acquire lock)
            self._free_slots(slot_id, num_slots)
            raise
        finally:
            memory_obj.ref_count_down()
            with self._inflight_lock:
                self._inflight_puts.discard(key)
            if on_complete_callback:
                try:
                    on_complete_callback(key)
                except Exception as cb_err:
                    logger.error(
                        "on_complete_callback failed for key %s: %s",
                        key, cb_err,
                    )

    def get_blocking(self, key: CacheEngineKey) -> MemoryObj | None:
        """Blocking read from Lightning PFS.

        Allocates a MemoryObj from LocalCPUBackend and deserializes the bytes
        read from the slot file into it.

        Args:
            key: The key to retrieve.

        Returns:
            MemoryObj with the retrieved data, or None if not found.
        """
        with self.index_lock:
            entry = self.index.get(key)
            if entry is None:
                if instrumentation.is_enabled():
                    logger.info("[LIGHTNING_TRACE] Get miss: key=%s", key.to_string())
                return None
            self.lru.touch(key)

        shape = torch.Size(entry.shape)
        dtype = torch_dtypes_inverse[entry.dtype]
        fmt = MemoryFormat(entry.fmt)

        if self.local_cpu_backend is None:
            logger.error("local_cpu_backend not available for allocation")
            return None

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt=fmt)
        if memory_obj is None:
            logger.error("Failed to allocate MemoryObj from LocalCPUBackend")
            return None

        try:
            memory_obj.ref_count_up()

            # Read from slot directly into the allocated MemoryObj.
            offset = self._get_file_offset(entry.slot_id)

            with instrumentation.Timer(
                lambda ms: instrumentation.log_trace(
                    "LIGHTNING_PERF",
                    f"Read key={key.to_string()} "
                    f"slots={entry.num_slots} payload={entry.payload_len}B "
                    f"latency={ms:.2f}ms",
                )
            ):
                self._read_obj(offset, memory_obj, entry.payload_len)

            if instrumentation.is_enabled():
                logger.info(
                    "[LIGHTNING_TRACE] Get hit: key=%s "
                    "slot_id=%d num_slots=%d payload=%dB",
                    key.to_string(),
                    entry.slot_id,
                    entry.num_slots,
                    entry.payload_len,
                )

        except Exception as e:
            logger.error("Failed to read slot for key %s: %s", key, e)
            memory_obj.ref_count_down()
            return None

        memory_obj.ref_count_down()
        return memory_obj

    def batched_get_blocking(
        self,
        keys: list[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        """Batched blocking get with thread pool.

        Args:
            keys: List of keys to retrieve.

        Returns:
            List of MemoryObjs in same order as keys.
        """
        if instrumentation.is_enabled():
            logger.info("[LIGHTNING_TRACE] batched_get_blocking: %d keys", len(keys))

        # Submit all reads to thread pool
        futures = [self._executor.submit(self.get_blocking, key) for key in keys]

        # Collect results
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("Failed to get key: %s", e)
                results.append(None)

        if instrumentation.is_enabled():
            hits = sum(1 for r in results if r is not None)
            logger.info(
                f"[LIGHTNING_TRACE] batched_get result: {hits}/{len(keys)} hits"
            )

        return results

    def touch_cache(self) -> None:
        """Update cache policy with keys accessed during a request.

        No-op: LRU is updated on each access.
        """
        pass

    def get_allocator_backend(self) -> "AllocatorBackendInterface":
        """Return the LocalCPUBackend used for memory allocation.

        Raises:
            RuntimeError: If no LocalCPUBackend was provided.
        """
        if self.local_cpu_backend is None:
            raise RuntimeError("local_cpu_backend not set")
        return self.local_cpu_backend

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a memory object so it will not be evicted.

        Not fully implemented - just marks as recently used.
        """
        with self.index_lock:
            if key not in self.index:
                return False
            self.lru.touch(key)
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a memory object so it can be evicted.

        No-op for this backend.
        """
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a memory object.

        Args:
            key: The key to remove.
            force: Whether to force remove.

        Returns:
            True if removal was successful.
        """
        with self.index_lock:
            entry = self.index.get(key)
            if entry is None:
                return False

            # Free slots
            self._free_slots_unlocked(entry.slot_id, entry.num_slots)

            # Remove from index and LRU
            del self.index[key]
            self.lru.remove(key)

        self._mark_checkpoint_dirty()
        return True

    def close(self) -> None:
        """Close the storage backend and cleanup resources.

        Safe to call multiple times; the second call is a no-op.
        """
        if self._closed:
            return
        self._closed = True

        # Stop checkpoint thread
        if self._checkpoint_thread is not None:
            self._stop_event.set()
            self._checkpoint_thread.join(timeout=5.0)

        # Final hole punching flush
        if self.enable_hole_punch and self._hole_punch_supported:
            self._punch_holes()

        # Final checkpoint save (idempotent, checks _checkpoint_dirty inside)
        self._save_checkpoint()

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        # Close file descriptor
        if hasattr(self, "fd") and self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

        logger.info("LightningPosixBackend closed")

    def __str__(self) -> str:
        return "LightningPosixBackend"


# Backward-compatible alias for the original class name.
LightningBackend = LightningPosixBackend
