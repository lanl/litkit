# src/litkit/cli.py
"""Litkit CLI entrypoint.

This module provides the command-line interface for litkit, including:
- Build/index management (--rebuild, --update, --build-only)
- Two-stage RAG retrieval (paper shortlisting + chunk search)
- LLM-powered question answering

Business logic is delegated to well-organized submodules:
- litkit.db: SQLite operations
- litkit.index: FAISS index operations  
- litkit.build: Build pipeline orchestration
- litkit.retrieval: RAG retrieval pipeline
- litkit.segments: Embedding segment I/O
- litkit.ingest: Tar/XML parsing
"""

import os
import sys

from . import __version__ as LITKIT_VERSION

# -------- simple early quieting (env), used before argparse exists ----------
# Set LITKIT_QUIET=1 to squelch startup banners that print before args are parsed.
# NOTE: Do not use a frozen QUIET variable here - use is_quiet() from progress.py
# which checks os.environ on each call for consistent behavior after --quiet is parsed.
# For --version/--help, argparse exits before any printing, so no special handling needed.


def _version_banner() -> str:
    git = (os.environ.get("LITKIT_SHA") or "").strip()
    if git and len(git) > 12:
        git = git[:12]
    tail = f" (git:{git})" if git else ""
    return f"litkit {LITKIT_VERSION}{tail}"


# -------------------- Standard library imports --------------------
# Only imports needed BEFORE argparse runs (for --help/--version) are at module level.
# Other stdlib imports are deferred to their use sites to minimize import-time side effects.
import argparse
import re
import threading
from pathlib import Path
from typing import Iterator


# Third-party import used early in XML parsing utilities.
# from lxml import etree
from types import SimpleNamespace

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT STRATEGY: Stdlib + lightweight only at module level
# ═══════════════════════════════════════════════════════════════════════════════
# 
# Module-level imports must NOT pull in faiss, numpy, torch, lxml, transformers.
# This ensures `python -m litkit --help` and `python -m litkit --version` work
# even if heavy dependencies are missing.
#
# Heavy imports are deferred to:
#   1. main() - after argparse runs (most imports)
#   2. Function scope - for thin wrappers and build_or_update_indices
#
# What CAN stay at module level:
#   - stdlib (os, sys, pathlib, typing, argparse, etc.)
#   - litkit.progress (pure Python, no heavy deps)
#   - litkit.concurrent (pure Python, uses fcntl which is stdlib)
#   - litkit.config.paths (pure Python)
#   - TYPE_CHECKING blocks for type hints
# ═══════════════════════════════════════════════════════════════════════════════

from typing import TYPE_CHECKING

from litkit.progress import (
    eprint as _eprint,
)
from litkit.concurrent import (
    FileLock as _FileLockBase,
    FLOCK_AVAILABLE,
)

# Type hints only - not imported at runtime
if TYPE_CHECKING:
    from litkit.embeddings.base import Embedder
    from litkit.ingest.ingest import ArticleMeta, TarMemberMeta
# Heavy imports (litkit.db, litkit.segments, litkit.embeddings.*, litkit.ingest.*)
# are deferred via _load_heavy_deps(). This function is idempotent and must be
# called at the top of any function that uses these dependencies.

_heavy_lock = threading.Lock()
_deps: SimpleNamespace | None = None  # Populated by _load_heavy_deps()

def _load_heavy_deps() -> None:
    """Idempotent, thread-safe loader for heavy dependencies.
    
    Must be called at the top of any function that uses:
    - litkit.embeddings.* (torch, transformers)
    - litkit.db.* (sqlite3 wrappers)
    - litkit.segments.* (numpy)
    - litkit.ingest.* (lxml)
    - litkit.formatting.* (answer rendering)
    
    Safe to call multiple times from multiple threads; only loads once.
    Uses double-checked locking to avoid races while minimizing lock contention.
    
    After loading, access dependencies via `_deps.name` (e.g., `_deps.db_connect_db`).
    This consolidates all deferred imports into a single namespace for maintainability.
    """
    global _deps
    if _deps is not None:
        return
    with _heavy_lock:
        if _deps is not None:  # Double-check inside lock
            return
        
        from litkit.embeddings.devices import configure_threads, detect_device
        from litkit.embeddings.factory import make_chunk_embedder, make_paper_embedder
        from litkit.formatting.answers import normalize_answer_and_build_refs, render_references
        from litkit.ingest.ingest import (
            iter_tar_paths,
            iter_tar_xml_streams,
            parallel_iter_tar_articles,
            parse_xml_fileobj,
        )
        from litkit.ingest import is_uncompressed_tar, shard_filter
        from litkit.db import (
            init_db as db_init_db,
            init_shard_db as db_init_shard_db,
            connect_db as db_connect_db,
            shard_db_path as db_shard_db_path,
            chunk_ids_to_paper_ids as db_chunk_ids_to_paper_ids,
            flush_pending_marks as db_flush_pending_marks,
            load_temp_candidates as db_load_temp_candidates,
        )
        from litkit.segments import (
            validate_shard_consistency as seg_validate_shard_consistency,
            write_build_meta as seg_write_build_meta,
            read_build_meta as seg_read_build_meta,
            has_segment_files as seg_has_segment_files,
            SegmentWriter,
            ChunkSegmentWriter,
            ProducerCoordinator as SegProducerCoordinator,
            ConsumerCoordinator as SegConsumerCoordinator,
            ingest_paper_segments as seg_ingest_paper_segments,
            ingest_chunk_segments as seg_ingest_chunk_segments,
        )
        
        # Set deterministic FAISS seed (moved from _init_runtime for conceptual purity)
        # _init_runtime() is now purely filesystem/env; faiss belongs with heavy deps
        #
        # IMPORTANT: Only catch ImportError (missing faiss), NOT other exceptions.
        # A broken faiss install should fail fast, not be silently ignored.
        #
        # NOTE: FAISS is required for ALL litkit operations except --help/--version.
        # Both build AND query paths use FAISS indices. The faiss_available flag
        # enables early fail-fast with a clear error message rather than a cryptic
        # ImportError deep in the call stack.
        faiss_available = False
        try:
            import faiss
            faiss_available = True
            try:
                faiss.cvar.seed = int(os.environ.get("LITKIT_FAISS_SEED", "123456"))
            except AttributeError:
                pass  # faiss.cvar.seed not available in this build (e.g., macOS faiss-cpu)
        except ImportError:
            pass  # Will fail fast via _require_faiss() when actually needed
        
        # Consolidate all imports into a single namespace
        _deps = SimpleNamespace(
            faiss_available=faiss_available,
            # Embeddings
            configure_threads=configure_threads,
            detect_device=detect_device,
            make_paper_embedder=make_paper_embedder,
            make_chunk_embedder=make_chunk_embedder,
            # Formatting
            normalize_answer_and_build_refs=normalize_answer_and_build_refs,
            render_references=render_references,
            # Ingest
            iter_tar_paths=iter_tar_paths,
            iter_tar_xml_streams=iter_tar_xml_streams,
            parallel_iter_tar_articles=parallel_iter_tar_articles,
            parse_xml_fileobj=parse_xml_fileobj,
            is_uncompressed_tar=is_uncompressed_tar,
            shard_filter=shard_filter,
            # DB
            db_init_db=db_init_db,
            db_init_shard_db=db_init_shard_db,
            db_connect_db=db_connect_db,
            db_shard_db_path=db_shard_db_path,
            db_chunk_ids_to_paper_ids=db_chunk_ids_to_paper_ids,
            db_flush_pending_marks=db_flush_pending_marks,
            db_load_temp_candidates=db_load_temp_candidates,
            # Segments
            seg_validate_shard_consistency=seg_validate_shard_consistency,
            seg_write_build_meta=seg_write_build_meta,
            seg_read_build_meta=seg_read_build_meta,
            seg_has_segment_files=seg_has_segment_files,
            SegmentWriter=SegmentWriter,
            ChunkSegmentWriter=ChunkSegmentWriter,
            SegProducerCoordinator=SegProducerCoordinator,
            SegConsumerCoordinator=SegConsumerCoordinator,
            seg_ingest_paper_segments=seg_ingest_paper_segments,
            seg_ingest_chunk_segments=seg_ingest_chunk_segments,
        )



# logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def deps() -> SimpleNamespace:
    """Load heavy dependencies (idempotent) and return the deps namespace.
    
    Use this instead of accessing _deps directly to ensure deps are loaded
    and avoid NoneType crashes if someone forgets _load_heavy_deps().
    
    Example:
        d = deps()
        conn = d.db_connect_db(...)
    """
    _load_heavy_deps()
    assert _deps is not None, "_load_heavy_deps() failed to populate _deps"
    return _deps


def _require_faiss(context: str = "this operation") -> None:
    """Fail fast if FAISS is not available.
    
    FAISS is required for ALL litkit operations except --help/--version:
    - Build paths: creating/updating indices
    - Query paths: shortlist_papers(), search_chunks_constrained()
    - Maintenance: --reconcile-only, --consume-only
    
    Call this early in any code path that touches FAISS indices.
    """
    d = deps()
    if not d.faiss_available:
        raise SystemExit(
            f"[error] FAISS is required for {context}.\n"
            "Install faiss-cpu or faiss-gpu:\n"
            "  pip install faiss-cpu    # CPU-only\n"
            "  pip install faiss-gpu    # CUDA-enabled"
        )


def _maybe_cleanup_own_stale_guard():
    """Best-effort cleanup of a guard file left by THIS process.
    
    Only removes the guard if the recorded PID matches os.getpid().
    This handles the case where the same process tries to re-create
    a guard (e.g., after a soft restart), but does NOT clean up
    guards left by crashed processes with different PIDs - that's
    handled by TTL-based stale detection in _create_writer_guard_or_exit.
    
    Note: PID reuse is theoretically possible after a crash, but rare
    enough that we accept this as a benign edge case.
    """
    get_runtime()  # ensure WRITER_GUARD is bound
    try:
        if WRITER_GUARD.exists():
            # Parse guard file with explicit field extraction (avoids silent truncation)
            parts = WRITER_GUARD.read_text().split()
            pid = parts[0] if len(parts) >= 1 else ""
            # host and ts unused here, but kept for clarity if parsing expands
            if pid.isdigit() and int(pid) == os.getpid():
                WRITER_GUARD.unlink(missing_ok=True)
    except Exception:
        pass


def _create_writer_guard_or_exit(args, *, ttl_sec: int | None = None):
    """Create a writer guard file or exit if another writer is active.
    
    Stdlib imports (atexit, errno, signal, socket, time) are deferred to this function
    to minimize import-time side effects for --help/--version.
    
    Uses a bounded retry loop (max 2 attempts) to handle stale guard cleanup.
    
    DESIGN DECISIONS (cross-host TTL eviction & signal handling):
    
    1. Cross-host TTL eviction:
       - On same host: we check PID liveness via os.kill(pid, 0) before evicting
       - On different host: we cannot check PID liveness, so TTL expiry alone triggers eviction
       - Risk: a long-running build on another host could be evicted if TTL is enabled
       - Mitigations:
         * Default TTL is 0 (DISABLED) for safety in multi-host HPC environments
         * Same-host PID check still works regardless of TTL setting
         * Set LITKIT_WRITER_GUARD_TTL=86400 to enable 24h auto-eviction if desired
         * Users can manually remove stale guards: rm .writer_guard
       - Rationale: >24h HPC jobs are common; false eviction is catastrophic
    
    2. Signal handler uses os._exit(1):
       - On SIGINT/SIGTERM, we clean up the guard file then os._exit(1)
       - This bypasses normal Python shutdown (atexit, finally, destructors)
       - Risk: FAISS indices and SQLite may have unflushed data
       - Why this is safe:
         * reconcile_sqlite_flags_with_faiss() runs on EVERY faiss_writer startup
         * backfill_unindexed_vectors() repairs any missing vectors
         * Guard cleanup is CRITICAL: a leftover guard blocks all future runs
         * Normal shutdown can deadlock if interrupted during a lock hold
       - Accepted tradeoff: rely on reconcile+backfill vs. risk deadlock/blocked runs
    """
    # Deferred imports to minimize module-level side effects
    import atexit
    import errno
    import signal
    import socket
    import time
    
    get_runtime()  # ensure WRITER_GUARD is bound
    _maybe_cleanup_own_stale_guard()
    if not getattr(args, "faiss_writer", False):
        return
    if ttl_sec is None:
        try:
            # Default TTL=0 (disabled) for safety in multi-host HPC environments.
            # Cross-host TTL eviction cannot verify PID liveness, so auto-eviction
            # risks evicting a legitimate long-running build (>24h jobs are common).
            # Same-host: PID check via os.kill(pid, 0) still works regardless of TTL.
            # Set LITKIT_WRITER_GUARD_TTL=86400 (or higher) to enable auto-eviction.
            ttl_sec = int(os.environ.get("LITKIT_WRITER_GUARD_TTL", "0"))
        except ValueError:
            _eprint("[writer] WARNING: invalid LITKIT_WRITER_GUARD_TTL; using 0 (disabled)")
            ttl_sec = 0

    def _cleanup_guard():
        try:
            WRITER_GUARD.unlink(missing_ok=True)
        except Exception:
            pass

    max_attempts = 2  # initial try + one retry after stale cleanup
    for attempt in range(max_attempts):
        try:
            # Use os.fspath() for explicit Path→str conversion (consistent with other os.* calls)
            fd = os.open(os.fspath(WRITER_GUARD), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(
                fd, f"{os.getpid()} {socket.gethostname()} {int(time.time())}\n".encode()
            )
            try:
                os.fsync(fd)
            except Exception:
                pass
            os.close(fd)
            atexit.register(_cleanup_guard)
            try:
                if threading.current_thread() is threading.main_thread():
                    # HARD KILL on SIGINT/SIGTERM: clean up guard file, then os._exit(1).
                    #
                    # WHY os._exit(1) instead of sys.exit():
                    # 1. Avoids deadlock if interrupted while holding FileLock (flock is not reentrant)
                    # 2. Avoids partial writes from half-executed finally/atexit handlers
                    # 3. Guard cleanup is CRITICAL: leftover guard blocks ALL future runs
                    #
                    # WHY this is SAFE despite bypassing normal shutdown:
                    # 1. FAISS saves are ATOMIC (see litkit/index/io.py):
                    #    - write to .tmp, fsync, os.replace, fsync dir
                    #    - index file is either fully old or fully new, never corrupt
                    # 2. reconcile_sqlite_flags_with_faiss() runs on EVERY faiss_writer start
                    # 3. backfill_unindexed_vectors() re-embeds any missing vectors
                    # 4. Producer mode: segments are durable (written before checkpoint advance)
                    # 5. Writer mode: FAISS saves are checkpointed; partial batches are re-embedded
                    #
                    # The invariant: reconcile+backfill ALWAYS runs before any new work.
                    # See build_or_update_indices() near the faiss_writer block.
                    signal.signal(signal.SIGINT,  lambda *_: (_cleanup_guard(), os._exit(1)))
                    signal.signal(signal.SIGTERM, lambda *_: (_cleanup_guard(), os._exit(1)))
            except Exception:
                pass
            return  # Success - guard created
        except FileExistsError:
            info = "unknown"
            stale_cleaned = False
            try:
                info = Path(WRITER_GUARD).read_text().strip()
                parts = info.split()
                ts = 0
                guard_pid = None
                guard_host = None
                if len(parts) >= 1 and parts[0].isdigit():
                    guard_pid = int(parts[0])
                if len(parts) >= 2:
                    guard_host = parts[1]
                if len(parts) >= 3:
                    try: ts = int(parts[2])
                    except ValueError: ts = 0
                
                # TTL ≤ 0 means "never consider guards stale" (manual cleanup required).
                if ttl_sec > 0 and ts and (time.time() - ts) > ttl_sec and attempt == 0:
                    # Guard appears stale by timestamp, but check PID liveness first
                    # to avoid evicting long-running builds that exceed TTL.
                    is_live = False
                    if guard_host == socket.gethostname() and guard_pid is not None:
                        # Same host: can check if PID is still alive
                        try:
                            os.kill(guard_pid, 0)  # Signal 0 = check existence
                            is_live = True
                            _eprint(f"[writer] Guard PID {guard_pid} is still alive (long build?). Not evicting.")
                        except OSError as e:
                            # ESRCH (no such process) or EPERM (exists but we can't signal)
                            if e.errno == errno.ESRCH:
                                is_live = False  # Process dead, safe to evict
                            elif e.errno == errno.EPERM:
                                is_live = True  # Process exists but we can't signal it
                            else:
                                is_live = True  # Unknown error, be conservative
                    # Different host: cannot check PID, use timestamp-based staleness only
                    # ⚠️  CROSS-HOST TTL EVICTION: This is inherently risky!
                    #     We have no way to verify if the remote process is still alive.
                    #     A long-running build on another host WILL be evicted when TTL expires.
                    if guard_host != socket.gethostname() and not is_live:
                        sys.stderr.write(
                            "\n╔══════════════════════════════════════════════════════════════════════════════╗\n"
                            "║  ⚠️  WARNING: CROSS-HOST TTL EVICTION                                         ║\n"
                            "╠══════════════════════════════════════════════════════════════════════════════╣\n"
                            f"║  Guard file: {str(WRITER_GUARD)[:60]:<60s} ║\n"
                            f"║  Guard host: {str(guard_host)[:60]:<60s} ║\n"
                            f"║  This host:  {socket.gethostname()[:60]:<60s} ║\n"
                            "║                                                                              ║\n"
                            "║  Cannot verify if remote process is still alive!                             ║\n"
                            "║  If a build is running on the other host, THIS WILL CORRUPT DATA.            ║\n"
                            "║                                                                              ║\n"
                            "║  Proceeding because TTL expired and LITKIT_WRITER_GUARD_TTL is set.          ║\n"
                            "║  Consider adding heartbeat updates for long builds (see README.md).          ║\n"
                            "╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
                        )
                    
                    if not is_live:
                        _eprint(f"[writer] Guard appears stale (> {ttl_sec}s, process dead): {info}. Attempting exclusive cleanup.")
                        try:
                            stale = WRITER_GUARD.with_name(f"{WRITER_GUARD.name}.stale.{os.getpid()}")
                            # Atomic claim: if this replace fails, someone else is cleaning.
                            os.replace(WRITER_GUARD, stale)
                            stale.unlink(missing_ok=False)
                            stale_cleaned = True  # retry once via continue
                        except Exception as e:
                            _eprint(f"[writer] ERROR: failed to remove guard: {e}.")
                            sys.exit(2)
            except Exception:
                pass
            
            if stale_cleaned:
                continue  # Retry exactly once after successful cleanup
            
            # Not stale, or already retried, or couldn't determine staleness
            sys.stderr.write(
                f"[writer] Another FAISS writer appears active (guard {WRITER_GUARD} exists: {info}).\n"
                "Stop the other job or remove the stale guard if you are sure it is dead.\n"
            )
            sys.exit(2)

# -- Paths / offline env --
# Path discovery and workspace configuration delegated to litkit.config.paths
# WorkspacePaths import is deferred to _init_runtime() for future-proofing:
# if someone later adds numpy/torch to paths.py, --help/--version still work.


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY RUNTIME INITIALIZATION (Phase 2 refactor)
# ═══════════════════════════════════════════════════════════════════════════════
# Path discovery and filesystem I/O are deferred until first access.
# 
# What IS deferred (lazy):
# - Path constant resolution (via __getattr__/get_runtime())
# - Directory creation (sqlite_dir, indices_dir)
# - Environment variable setup (HF_HOME, HF_HUB_OFFLINE, etc.)
# (Note: faiss.cvar.seed is now set in _load_heavy_deps(), not here)
#
# What is NOT deferred (import-time):
# - Third-party imports: faiss, numpy (heavyweight but necessary for type hints)
# - Internal module imports: litkit.build, litkit.retrieval, etc.
#
# Access any path constant (e.g., SQLITE_DIR) to trigger initialization.
# ═══════════════════════════════════════════════════════════════════════════════

_runtime: "WorkspacePaths | None" = None  # String annotation - import deferred
_runtime_lock = threading.Lock()


def get_runtime() -> "WorkspacePaths":
    """Thread-safe lazy initialization of runtime paths and directories.
    
    Side effects (first call only):
    - Creates SQLITE_DIR and INDICES_DIR directories
    - Sets HF_HOME, HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, TOKENIZERS_PARALLELISM env vars
    - Populates module-level globals (ROOT, WORKSPACE, DB_PATH, etc.)
    
    Returns:
        WorkspacePaths: Immutable container with all path constants.
    """
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        _runtime = _init_runtime()
        # Populate module-level globals for internal code that uses bare names
        for attr_name, field_name in _LAZY_PATH_ATTRS.items():
            globals()[attr_name] = getattr(_runtime, field_name)
        return _runtime


def _init_runtime() -> "WorkspacePaths":
    """Perform all one-time initialization. Called only by get_runtime()."""
    from litkit.config.paths import WorkspacePaths  # deferred for future-proofing
    paths = WorkspacePaths.from_env_or_default()
    
    # Set environment variables (safe defaults for HPC/offline use)
    paths.setup_environment()
    
    # Create required directories
    paths.ensure_directories()
    
    return paths


# Backward compatibility: module-level __getattr__ for lazy path access
# Allows both `litkit.cli.SQLITE_DIR` and `from litkit.cli import SQLITE_DIR`
_LAZY_PATH_ATTRS = {
    "ROOT": "root",
    "WORKSPACE": "workspace",
    "HF_HOME": "hf_home",
    "SQLITE_DIR": "sqlite_dir",
    "INDICES_DIR": "indices_dir",
    "EMBED_SEGMENTS_DIR": "embed_segments_dir",
    "DB_PATH": "db_path",
    "CKPT_PATH": "ckpt_path",
    "DB_LOCK": "db_lock",
    "FAISS_LOCK": "faiss_lock",
    "WRITER_GUARD": "writer_guard",
    "PAPER_INDEX_PATH": "paper_index_path",
    "CHUNK_INDEX_PATH": "chunk_index_path",
    "CHUNK_TRAINED_FLAG": "chunk_trained_flag",
    "CKPT_LOCK": "ckpt_lock",
}


def __getattr__(name: str):
    """Module-level __getattr__ for lazy initialization of path constants.
    
    This function is called when an attribute is not found in the module namespace.
    On first access, we populate the module globals so subsequent local lookups work.
    """
    if name in _LAZY_PATH_ATTRS:
        # Trigger full initialization and populate ALL path globals
        rt = get_runtime()
        for attr_name, field_name in _LAZY_PATH_ATTRS.items():
            globals()[attr_name] = getattr(rt, field_name)
        # Return the requested attribute
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _report_paths(
    tar_dir: Path | None,
    workspace: Path,
    tar_manifest: Path | None,
    tar_dir_origin: str | None = None,  # "(from LITKIT_TAR_DIR)" or "(default)"
):
    """Print paths once args are parsed, so banners reflect reality."""
    # Use is_quiet() to check env var at runtime (not frozen at import time)
    from litkit.progress import is_quiet
    if is_quiet():
        return
    if tar_manifest:
        _eprint(f"[paths] using manifest -> " f"{tar_manifest} for paths to tar shards")
    else:
        src = str(tar_dir) if tar_dir else "(unset)"
        origin = f" {tar_dir_origin}" if tar_dir_origin else ""
        _eprint(f"[paths] using {src} as source directory for tar shards{origin}")
    _eprint(f"[paths] using {workspace} as writable directory for job artifacts/outputs")


# Reasonable defaults for large Lustre/NFS runs:
# ~131,072 vectors/segment ≈ 0.4 GB per file (fp32, dim=768). Half that if stored as fp16.
DEFAULT_EMBED_SEGMENT_SIZE = int(os.environ.get("LITKIT_EMBED_SEGMENT_SIZE", "131072"))
DEFAULT_EMBED_SEGMENT_DTYPE = os.environ.get("LITKIT_EMBED_SEGMENT_DTYPE", "fp16")  # fp16|fp32

# Default SQLite busy timeout in milliseconds (tunable for shared filesystems).
DEFAULT_BUSY_TIMEOUT_MS = int(os.environ.get("LITKIT_SQLITE_BUSY_TIMEOUT_MS", "120000"))


PROMPT_HEADROOM_TOKENS = int(os.environ.get("LITKIT_PROMPT_HEADROOM_TOKENS", "200"))


# FileLock wrapper that binds the DB_LOCK and FAISS_LOCK paths at runtime
# for lock depth tracking. Uses litkit.concurrent.FileLock as the base.
class FileLock(_FileLockBase):
    """File lock with runtime binding of DB_LOCK and FAISS_LOCK paths."""
    def __init__(self, path: Path):
        get_runtime()  # ensure DB_LOCK and FAISS_LOCK are initialized
        super().__init__(path, db_lock_path=DB_LOCK, faiss_lock_path=FAISS_LOCK)

# Always acquire in this order to avoid deadlock: DB_LOCK then FAISS_LOCK
# (These constants are now provided via lazy __getattr__ from get_runtime())

# Producer-mode segment writers (set in main)
chunk_seg_writer = None
paper_seg_writer = None


# -------------------- LLM defaults --------------------
# (HPC) production: o3; laptop testing: gpt-oss:20b
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
# default LLM timeout (seconds) for OpenAI client; safe on air-gapped cluster
OPENAI_TIMEOUT_SEC = int(os.environ.get("LITKIT_OPENAI_TIMEOUT_SEC", "15"))


def _default_base_url_for(model: str) -> str:
    """Default base URL for OpenAI-compatible endpoints.
    
    All models currently use the same default; the parameter is retained
    for future model-specific routing if needed.
    """
    return os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")

def _default_api_key_for(model: str) -> str:
    """Provide an API key default appropriate for the target endpoint.

    Local testing (gpt-oss:*) tolerates "no-auth". OpenAI (o3*) expects a key.
    """
    m = model.lower()
    if m.startswith("gpt-oss"):
        return os.environ.get("OPENAI_API_KEY", "no-auth")
    if m.startswith("o3"):
        return os.environ.get("OPENAI_API_KEY", "")
    return os.environ.get("OPENAI_API_KEY", "")


def _is_openai_cloud(url: str | None) -> bool:
    """True only for the official OpenAI cloud endpoint."""
    if not url:
        return True  # be conservative if unknown
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        return host.endswith("api.openai.com")
    except Exception:
        return "openai.com" in url.lower()


# -------------------- Build / search knobs (sane defaults) --------------------
TOP_PAPERS_DEFAULT = 500  # wide shortlist for quality in Stage 1
TOP_CHUNKS_DEFAULT = 30  # final chunks given to LLM in Stage 2
OVERSHOOT_DEFAULT = 20  # widen ANN chunk search before candidate-paper filter

# -------------------- Progress / render defaults --------------------
# How often to refresh per-tar progress (seconds). Higher = fewer lines.
TAR_RENDER_SEC = float(os.environ.get("LITKIT_TAR_RENDER_SEC", "5.0"))

# Optional % gating. 0 => disabled (time-based only).
# Accept both the canonical var and the short alias LITKIT_TAR_RENDER_PCT_STP.
TAR_RENDER_PCT_STEP = float(
    os.environ.get("LITKIT_TAR_RENDER_PCT_STEP", os.environ.get("LITKIT_TAR_RENDER_PCT_STP", "0"))
)

# batching (large batches are OK; flush gated by --faiss-writer)
PAPER_BATCH = 20000  # embed-add papers per batch
CHUNK_BATCH = 20000  # embed-add chunks per batch
TRAIN_CHUNK_SAMPLES = 150_000  # IVF-PQ training sample size (chunk embeddings)
CKPT_EVERY = 500  # checkpoint scan progress every N processed files

# chunking parameters (aggregate paragraphs into ~CHUNK_TARGET_CHARS)
BODY_MIN_CHARS = 300
CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200
BATCH_TRAIN_FLUSH = int(
    os.environ.get("LITKIT_TRAIN_FLUSH", "4000")
)  # how many training chunks per embed flush

# token budgets (approx; ~4 chars/token heuristic used)
# Override via env for local models with different context sizes:
#   LITKIT_BUDGET_O3=128000  (e.g., for 128k context models)
#   LITKIT_BUDGET_OSS20B=8000  (e.g., for larger local models)
BUDGET_TOKENS_O3 = int(os.environ.get("LITKIT_BUDGET_O3", "32000"))
BUDGET_TOKENS_OSS20B = int(os.environ.get("LITKIT_BUDGET_OSS20B", "3000"))

# -------------------- Destructive action confirmation --------------------
def _fmt_bytes(n: int) -> str:
    """Human-ish size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.1f} {units[i]}"


def _file_size_str(p: Path) -> str:
    try:
        st = p.stat()
        return _fmt_bytes(st.st_size)
    except FileNotFoundError:
        return "missing"
    except Exception:
        return "unknown"


def _confirm_rebuild(conn) -> None:
    """Ask the user to confirm --rebuild when in an interactive TTY.
    In non-interactive mode, require --yes or LITKIT_ASSUME_YES=1.
    """
    get_runtime()  # ensure path globals are bound
    # Allow fully non-interactive approvals
    assume_yes = os.environ.get("LITKIT_ASSUME_YES", "0") == "1"
    if assume_yes:
        return
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if not is_tty:
        sys.stderr.write(
            "[rebuild] Refusing to proceed non-interactively without --yes (or LITKIT_ASSUME_YES=1).\n"
        )
        sys.exit(2)

    # Summarize what will be affected
    p_idx_sz = _file_size_str(PAPER_INDEX_PATH)
    c_idx_sz = _file_size_str(CHUNK_INDEX_PATH)
    tf_flag = f"{CHUNK_TRAINED_FLAG} ({'exists' if CHUNK_TRAINED_FLAG.exists() else 'missing'})"
    ckpt_sz = _file_size_str(CKPT_PATH) if CKPT_PATH.exists() else "missing"
    try:
        cur = conn.cursor()
        n_p = cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        n_c = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_f = cur.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        counts = f"DB rows → papers={n_p}, chunks={n_c}, files={n_f}"
    except Exception:
        counts = "DB rows → (unavailable)"

    _eprint("\n[rebuild] You asked to perform a HARD REBUILD. This will:")
    _eprint(f"  • Delete index file: {PAPER_INDEX_PATH} [{p_idx_sz}]")
    _eprint(f"  • Delete index file: {CHUNK_INDEX_PATH} [{c_idx_sz}]")
    _eprint(f"  • Remove trained-flag: {tf_flag}")
    _eprint(f"  • Clear SQLite tables in: {DB_PATH}")
    _eprint(f"  • Remove checkpoint: {CKPT_PATH} [{ckpt_sz}]")
    _eprint(f"  • {counts}")
    # Note: _eprint (litkit.progress.eprint) supports end= in its signature:
    # def eprint(msg: str = "", *, end: str = "\n") -> None
    _eprint("\nType 'yes' to continue (anything else aborts): ", end="")
    resp = input().strip().lower()
    if resp not in ("y", "yes"):
        _eprint("[rebuild] Aborted by user.")
        sys.exit(2)


def _resolve_question(args) -> str | None:
    """Resolve the effective question from one of:
    1) --question-file FILE (or '-' for stdin)
    2) Positional 'question' that is:
        - '@path' shorthand (read from path)
        - a path to an existing file (read from file)
        - a literal string otherwise
    Returns the question text (stripped) or None.
    """
    # 1) explicit flag wins
    if args.question_file is not None:
        if str(args.question_file) == "-":
            return sys.stdin.read().strip()
        return args.question_file.read_text(encoding="utf-8", errors="ignore").strip()

    q = args.question
    if not q:
        return None

    # 2a) '@file' shorthand
    if q.startswith("@") and len(q) > 1:
        p = Path(q[1:])
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="ignore").strip()

    # 2b) bare path to existing file (e.g., ./question.txt)
    p = Path(q)
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="ignore").strip()

    # 2c) treat as literal question
    return q.strip()


# -------------------- Embedders --------------------

# Note: faiss.cvar.seed is set inside _load_heavy_deps() for conceptual purity
# (_init_runtime() is now purely filesystem/env; faiss belongs with heavy deps)


# -------------------- FAISS index helpers --------------------
# (Constants PQ_BITS and USE_DOWNCAST_FALLBACK are now imported from litkit.index)
# (Path constants are now provided via lazy __getattr__ from get_runtime():
#  PAPER_INDEX_PATH, CHUNK_INDEX_PATH, CHUNK_TRAINED_FLAG, CKPT_LOCK)

# ------------------------------------------------------------------
# Producer-mode segment writer handle (set in main(); read elsewhere)
# ------------------------------------------------------------------


# Throttle index saves to reduce I/O on shared filesystems.
_SAVE_MIN_SEC = int(os.environ.get("LITKIT_SAVE_EVERY_SEC", "120"))
_last_save_ts = {"papers": 0.0, "chunks": 0.0}






# -------------------- Embedding segment I/O (producer↔writer) --------------------

def backfill_unindexed_vectors(
    conn,
    paper_embedder,
    chunk_embedder,
    paper_index,
    chunk_index,
    batch=20000,
    paper_bs=None,
    chunk_bs=None,
):
    """Thin wrapper: delegates to litkit.build.backfill with runtime paths."""
    _load_heavy_deps()  # litkit.build pulls in faiss/numpy
    from litkit.build import backfill_unindexed_vectors as build_backfill_unindexed_vectors
    get_runtime()
    return build_backfill_unindexed_vectors(
        conn, paper_embedder, chunk_embedder, paper_index, chunk_index,
        paper_index_path=PAPER_INDEX_PATH,
        chunk_index_path=CHUNK_INDEX_PATH,
        faiss_lock_path=FAISS_LOCK,
        db_lock_path=DB_LOCK,
        FileLock=FileLock,
        batch=batch,
        paper_bs=paper_bs,
        chunk_bs=chunk_bs,
    )




def _post_build_sanity_check(conn, args):
    """Thin wrapper: delegates to litkit.build.post_build_sanity_check with runtime paths."""
    _load_heavy_deps()  # litkit.build pulls in faiss/numpy
    from litkit.build import post_build_sanity_check as build_post_build_sanity_check
    get_runtime()
    return build_post_build_sanity_check(
        conn,
        paper_index_path=PAPER_INDEX_PATH,
        chunk_index_path=CHUNK_INDEX_PATH,
    )


def _auto_top_papers() -> int:
    """Heuristic for Stage-1 shortlist size based on corpus size."""
    d = deps()  # ensures loaded + returns namespace (consistent pattern)
    get_runtime()  # ensure path globals are initialized for library use
    
    def _piecewise_heuristic(n: int) -> int:
        if n < 50_000:
            return 500
        if n < 500_000:
            return 1000
        if n < 2_000_000:
            return 2000
        return 4000
    
    conn = None
    try:
        conn = d.db_connect_db(DB_PATH)
        n = conn.execute("SELECT COUNT(1) FROM papers").fetchone()[0]
        return _piecewise_heuristic(n)
    except Exception:
        return _piecewise_heuristic(0)
    finally:
        if conn is not None:
            conn.close()


# NOTE: load_checkpoint and save_checkpoint moved to litkit.segments.checkpoint
# Use seg_load_checkpoint() and seg_save_checkpoint() from imports above.


def iter_tar_articles(
    tar_path: Path,
    parse_workers: int = 8,
) -> Iterator[tuple["TarMemberMeta | SimpleNamespace", "ArticleMeta"]]:
    """Unified iterator over articles in a tar file.
    
    For uncompressed .tar files (when parse_workers > 1), uses parallel XML parsing.
    For compressed .tar.gz/.tar.bz2 files, uses sequential parsing.
    
    Yields:
        (member_meta, article_meta) tuples where:
        - member_meta has .name, .size, .mtime attributes
        - article_meta is the parsed ArticleMeta dict
    """
    d = deps()  # ensures loaded + returns namespace (consistent pattern)
    use_parallel = parse_workers > 1 and d.is_uncompressed_tar(tar_path)
    
    if use_parallel:
        # Parallel path for uncompressed tars
        from litkit.progress import is_quiet
        if not is_quiet():
            _eprint(f"[scan] using parallel XML parsing ({parse_workers} workers) for {tar_path.name}")
        for member_meta, article_meta in d.parallel_iter_tar_articles(tar_path, workers=parse_workers):
            yield member_meta, article_meta
    else:
        # Sequential path for compressed tars (or when parallel disabled)
        for tarinfo, fobj in d.iter_tar_xml_streams(tar_path):
            try:
                article_meta = d.parse_xml_fileobj(fobj)
                if article_meta is not None:
                    # Wrap TarInfo in SimpleNamespace for consistent interface
                    member_meta = SimpleNamespace(
                        name=tarinfo.name,
                        size=int(getattr(tarinfo, "size", 0)),
                        mtime=float(getattr(tarinfo, "mtime", 0.0) or 0.0),
                    )
                    yield member_meta, article_meta
            finally:
                try:
                    fobj.close()
                except Exception:
                    pass


def build_or_update_indices(args):
    """Build or update indices & DB depending on flags.

    Modes
    -----
    * --rebuild: wipe DB & indices, then rebuild from scratch (writer creates indices).
    * --update : append-only update (skip previously seen files).
    * --build-only: ingest/build but do not run a query.
    * --consume-only: skip tar scanning and embedding, only ingest segments in a polling loop.

    Writer vs Non-writer
    --------------------
    Exactly one process should run with --faiss-writer to add vectors to FAISS and
    save indices. Other processes (possibly using --shard-id/--num-shards) only
    populate SQLite rows and commit; they do not mutate FAISS indices.
    """
    d = deps()  # ensures loaded + returns namespace (consistent pattern)
    _require_faiss("building/updating indices")
    # Lazy imports to defer faiss/numpy loading until actually needed
    from litkit.build import (
        BuildConfig,
        init_empty_indices as build_init_empty_indices,
        run_consume_only_mode as build_run_consume_only_mode,
        load_or_create_paper_index as build_load_or_create_paper_index,
        load_or_create_chunk_index as build_load_or_create_chunk_index,
        train_ivfpq_index as build_train_ivfpq_index,
        process_tar_files as build_process_tar_files,
    )
    from litkit.index import faiss_save_force, kind_and_core

    get_runtime()  # ensure all path globals are bound (required for library use)
    need = args.rebuild or not (
        DB_PATH.exists() and PAPER_INDEX_PATH.exists() and CHUNK_INDEX_PATH.exists()
    )
    if not need and not args.update and not args.build_only and not args.consume_only and not args.init_indices_only and not args.embed_producer:
        # nothing to do
        return

    # Validate build mode and shard count consistency BEFORE any work begins
    # Determine the build mode based on args
    is_multi_node = args.num_shards > 1 or args.embed_producer or args.consume_only
    build_mode = "multi" if is_multi_node else "single"
    seg_dir = args.embed_outdir or EMBED_SEGMENTS_DIR
    
    # Always validate if segment directory exists with prior work
    if d.seg_has_segment_files(seg_dir) or d.seg_read_build_meta(seg_dir) is not None:
        d.seg_validate_shard_consistency(seg_dir, args.num_shards, current_mode=build_mode)
    
    # Write build metadata if this is a fresh start
    # For multi-node: producer 0 writes it; for single-node: the writer writes it
    if is_multi_node:
        if args.embed_producer and args.shard_id == 0:
            meta = d.seg_read_build_meta(seg_dir)
            if meta is None:
                manifest_path = str(args.tar_manifest) if args.tar_manifest else None
                d.seg_write_build_meta(seg_dir, args.num_shards, manifest_path, mode="multi")
    else:
        # Single-node mode: write metadata if fresh start
        if args.faiss_writer:
            meta = d.seg_read_build_meta(seg_dir)
            if meta is None and not args.init_indices_only:
                manifest_path = str(args.tar_manifest) if args.tar_manifest else None
                d.seg_write_build_meta(seg_dir, args.num_shards, manifest_path, mode="single")

    # ═══════════════════════════════════════════════════════════════════════════
    # DATABASE CONCURRENCY CONTRACT
    # ═══════════════════════════════════════════════════════════════════════════
    # 
    # SQLite concurrency model for litkit builds:
    #
    #   1. PRODUCERS (--embed-producer): Each gets its own shard-specific DB file.
    #      No external locking needed; SQLite handles single-writer internally.
    #
    #   2. CONSUMER/WRITER (--faiss-writer): Uses the main DB exclusively.
    #      Pure-DB writes (INSERT, UPDATE) rely on SQLite busy_timeout.
    #      Cross-resource ops (DB + FAISS) use FileLock(DB_LOCK) + FileLock(FAISS_LOCK).
    #
    #   3. QUERIES (search path): Read-only; no locking required.
    #
    # ⚠️  DO NOT run multiple --faiss-writer processes against the same DB!
    #     SQLite handles concurrent readers, but concurrent writers to the same
    #     file WILL cause SQLITE_BUSY errors, especially on NFS/Lustre.
    #
    # See README.md "Concurrency Model" for the correct multi-node setup.
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Use shard-specific DB for producers (lock-free parallel writes)
    if args.embed_producer and not args.faiss_writer:
        _eprint(f"[build] Producer mode: using shard-specific DB for shard {args.shard_id}")
        conn = d.db_init_shard_db(d.db_shard_db_path(SQLITE_DIR, args.shard_id), args.shard_id, args.sqlite_journal_mode, args.sqlite_busy_timeout_ms)
    else:
        _eprint(f"[build] using DB at {DB_PATH}")
        conn = d.db_init_db(DB_PATH, args.sqlite_journal_mode, args.sqlite_busy_timeout_ms)
    cur = conn.cursor()

    if args.init_indices_only:
        # Build a minimal config for the module function
        cfg = BuildConfig(
            faiss_writer=args.faiss_writer,
            papers_index=args.papers_index,
            chunks_index=args.chunks_index,
            hnsw_m=args.hnsw_m,
            efconstruction=args.efconstruction,
            efsearch=args.efsearch,
            ivf_nlist=args.ivf_nlist,
            pq_m=args.pq_m,
            paper_index_path=PAPER_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            faiss_lock=FAISS_LOCK,
        )
        build_init_empty_indices(cfg, FileLock=FileLock)
        conn.close()
        return

    if args.consume_only:
        if not args.faiss_writer:
            raise ValueError("--consume-only requires --faiss-writer")
        
        seg_dir = args.embed_outdir or EMBED_SEGMENTS_DIR
        completed = build_run_consume_only_mode(
            conn,
            seg_dir=seg_dir,
            num_shards=args.num_shards,
            paper_index_path=PAPER_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            faiss_lock_path=FAISS_LOCK,
            db_lock_path=DB_LOCK,
            FileLock=FileLock,
        )
        conn.close()
        if not completed:
            return  # Timeout - already logged in module
        return  # Success - end consume-only mode

    # Embedders
    paper_embedder, _paper_cfg = d.make_paper_embedder()
    chunk_embedder, _chunk_cfg = d.make_chunk_embedder(
        devices=args.embed_devices,
        workers=args.embed_workers,
        force_devices=args.force_embed_devices,
    )

    paper_dim = getattr(paper_embedder, "dim", None) or 768
    chunk_dim = chunk_embedder.dim

    # Prepare / open FAISS indices
    if args.rebuild:
        # Guard: --rebuild with unconsumed segments or active producers can corrupt data
        # This check is here (not in main()) because it requires heavy deps for segment functions
        if not getattr(args, "force_rebuild", False):
            seg_dir = args.embed_outdir or EMBED_SEGMENTS_DIR
            conflict_reasons = []
            
            # Check for unconsumed segment files
            if d.seg_has_segment_files(seg_dir):
                conflict_reasons.append(f"Segment directory {seg_dir} contains unconsumed segment files")
            
            # Check for producer completion markers (indicates multi-node run)
            # Use shard count from build_meta.json (if exists) to correctly interpret markers.
            meta = d.seg_read_build_meta(seg_dir)
            effective_num_shards = meta.get("num_shards", args.num_shards) if meta else args.num_shards
            
            coordinator = d.SegConsumerCoordinator(seg_dir, effective_num_shards)
            completed = coordinator.completed_shards()
            if completed:
                if len(completed) < effective_num_shards:
                    conflict_reasons.append(
                        f"Prior multi-node run (incomplete): {len(completed)}/{effective_num_shards} producer shards marked done"
                    )
                else:
                    conflict_reasons.append(
                        f"Prior multi-node run (not consumed): all {effective_num_shards} producer shards done"
                    )
            
            if conflict_reasons:
                reasons_str = "\n  • ".join(conflict_reasons)
                raise SystemExit(
                    f"[rebuild] BLOCKED: Unconsumed multi-node artifacts detected:\n  • {reasons_str}\n\n"
                    "A rebuild would discard these pending segments/markers.\n"
                    "Options:\n"
                    "  • First consume pending segments: --faiss-writer --consume-only\n"
                    f"  • Or clean up manually: rm -rf {seg_dir}/*\n"
                    "  • Or force (DATA LOSS WARNING): --force-rebuild"
                )
        
        # Confirm before destructive actions
        _confirm_rebuild(conn)
        _eprint(
            "[rebuild] hard reset: deleting FAISS indices, clearing DB tables, removing checkpoint"
        )
        for p in [PAPER_INDEX_PATH, CHUNK_INDEX_PATH, CHUNK_TRAINED_FLAG]:
            if p.exists():
                p.unlink()
        # also clear checkpoint
        if CKPT_PATH.exists():
            CKPT_PATH.unlink()

        # wipe tables (DELETE in transaction, VACUUM outside)
        cur.executescript("DELETE FROM chunks; DELETE FROM papers; DELETE FROM files;")
        conn.commit()
        
        # VACUUM must run outside any transaction (autocommit mode)
        # Without this, sqlite3.OperationalError: cannot VACUUM from within a transaction
        old_isolation = conn.isolation_level
        conn.isolation_level = None  # autocommit
        try:
            conn.execute("VACUUM")
        finally:
            conn.isolation_level = old_isolation  # restore
        _eprint("[rebuild] done")

    # PAPER index (load existing or create new)
    paper_index = build_load_or_create_paper_index(
        paper_index_path=PAPER_INDEX_PATH,
        faiss_lock_path=FAISS_LOCK,
        db_lock_path=DB_LOCK,
        FileLock=FileLock,
        paper_dim=paper_dim,
        papers_index=args.papers_index,
        hnsw_m=args.hnsw_m,
        efconstruction=args.efconstruction,
        efsearch=args.efsearch,
        is_faiss_writer=args.faiss_writer,
    )

    # CHUNK index - load existing or create new
    chunk_index, needs_training = build_load_or_create_chunk_index(
        chunk_index_path=CHUNK_INDEX_PATH,
        chunk_trained_flag=CHUNK_TRAINED_FLAG,
        faiss_lock_path=FAISS_LOCK,
        db_lock_path=DB_LOCK,
        FileLock=FileLock,
        chunk_dim=chunk_dim,
        chunks_index=args.chunks_index,
        ivf_nlist=args.ivf_nlist,
        pq_m=args.pq_m,
        is_faiss_writer=args.faiss_writer,
    )

    # If IVF-PQ needs training, run training pass (one-time)
    if needs_training:
        if not args.faiss_writer:
            _eprint(
                "[train] ERROR: chunks index requires training; "
                "start a writer with --faiss-writer."
            )
            sys.exit(2)
        
        chunk_index = build_train_ivfpq_index(
            chunk_dim=chunk_dim,
            chunk_embedder=chunk_embedder,
            tar_dir=args.tar_dir,
            tar_manifest=args.tar_manifest,
            ivf_nlist=args.ivf_nlist,
            pq_m=args.pq_m,
            nprobe=args.nprobe,
            ivf_nlist_forced=getattr(args, "_ivf_nlist_forced", False),
            chunk_target_chars=int(
                getattr(args, "chunk_target_chars", CHUNK_TARGET_CHARS)
            ),
            chunk_min_chars=int(getattr(args, "chunk_min_chars", BODY_MIN_CHARS)),
            chunk_overlap=int(getattr(args, "chunk_overlap", CHUNK_OVERLAP_CHARS)),
            chunk_embed_bs=args.chunk_embed_bs,
            train_samples=TRAIN_CHUNK_SAMPLES,
            batch_train_flush=BATCH_TRAIN_FLUSH,
            chunk_index_path=CHUNK_INDEX_PATH,
            chunk_trained_flag=CHUNK_TRAINED_FLAG,
            faiss_lock_path=FAISS_LOCK,
            db_lock_path=DB_LOCK,
            FileLock=FileLock,
            is_faiss_writer=args.faiss_writer,
        )

    use_tar = (getattr(args, "tar_dir", None) is not None) or (
        getattr(args, "tar_manifest", None) is not None
    )

    # ----- TAR SHARD PATH (NO EXTRACTION) -----
    # NOTE: We materialize the tar_paths iterator into a list here for two reasons:
    # 1. Progress reporting needs len(tar_paths) for "N of M tars processed"
    # 2. Checkpoint resume logic benefits from knowing total shard count
    #
    # Memory impact is negligible: even 10,000 Path objects ≈ 2MB, compared to
    # FAISS indices (1-10GB), embedding batches (100-500MB), SQLite (10-50MB).
    # PMC-OA corpus has ~600-2000 shard files, so ~200-400KB total.
    tar_paths = list(
        d.shard_filter(
            d.iter_tar_paths(args.tar_dir, args.tar_manifest), args.shard_id, args.num_shards
        )
    )

    _eprint(f"[scan] found {len(tar_paths)} tar shards in shard {args.shard_id}/{args.num_shards}")

    # NOTE: The global --rebuild handling already reset DB/indices/checkpoint
    # near the start of build_or_update_indices(). No extra resets here.

    # Use per-shard checkpoint for producers (multi-process safe)
    # Single-node and consumer use shared checkpoint (backward compatible)
    ckpt_shard_id = args.shard_id if args.embed_producer else None

    # Process all tar files using the extracted module function
    paper_index, chunk_index, papers_added_total, chunks_added_total = build_process_tar_files(
        tar_paths=tar_paths,
        conn=conn,
        paper_embedder=paper_embedder,
        chunk_embedder=chunk_embedder,
        paper_index=paper_index,
        chunk_index=chunk_index,
        paper_seg_writer=paper_seg_writer,
        chunk_seg_writer=chunk_seg_writer,
        ckpt_path=CKPT_PATH,
        ckpt_lock=CKPT_LOCK,
        faiss_lock=FAISS_LOCK,
        db_lock=DB_LOCK,
        paper_index_path=PAPER_INDEX_PATH,
        chunk_index_path=CHUNK_INDEX_PATH,
        sqlite_dir=SQLITE_DIR,
        FileLock=FileLock,
        rebuild=args.rebuild,
        embed_producer=args.embed_producer,
        faiss_writer=args.faiss_writer,
        strict_ingest=getattr(args, "strict_ingest", False),
        paper_batch=PAPER_BATCH,
        chunk_batch=CHUNK_BATCH,
        ckpt_every=CKPT_EVERY,
        paper_embed_bs=args.paper_embed_bs,
        chunk_embed_bs=args.chunk_embed_bs,
        chunk_target_chars=int(getattr(args, "chunk_target_chars", CHUNK_TARGET_CHARS)),
        chunk_min_chars=int(getattr(args, "chunk_min_chars", BODY_MIN_CHARS)),
        chunk_overlap=int(getattr(args, "chunk_overlap", CHUNK_OVERLAP_CHARS)),
        ckpt_shard_id=ckpt_shard_id,
        parse_workers=args.parse_workers,
    )

    # Final commit + ensure on-disk indices are current *before* sanity
    conn.commit()

    # Write completion marker for producer
    if args.embed_producer:
        seg_dir = args.embed_outdir or EMBED_SEGMENTS_DIR
        producer_coordinator = d.SegProducerCoordinator(seg_dir, args.shard_id, args.num_shards)
        producer_coordinator.mark_complete()
        _eprint(f"[producer] Shard {args.shard_id}/{args.num_shards} marked complete")

    if args.faiss_writer and not args.consume_only:
        seg_dir = args.embed_outdir or EMBED_SEGMENTS_DIR

        if args.consume_segments and seg_dir and Path(seg_dir).exists():
            # Ingest segments - functions handle IDMap2 wrapping and return the (possibly wrapped) index
            paper_index, p_added = d.seg_ingest_paper_segments(
                conn, paper_index, seg_dir, FAISS_LOCK, PAPER_INDEX_PATH, DB_LOCK,
                FileLock=FileLock
            )
            chunk_index, c_added = d.seg_ingest_chunk_segments(
                conn, chunk_index, seg_dir, FAISS_LOCK, CHUNK_INDEX_PATH, DB_LOCK,
                FileLock=FileLock
            )
            if c_added or p_added:
                _eprint(
                    f"[segments] ingested {p_added} paper vectors and {c_added} chunk vectors from segments"
                )

        # ═══════════════════════════════════════════════════════════════════════════
        # SAFETY INVARIANT: reconcile+backfill runs on EVERY faiss_writer startup
        # ═══════════════════════════════════════════════════════════════════════════
        # This is the critical repair step that makes os._exit(1) in signal handlers
        # safe (see _create_writer_guard_or_exit). No matter how the previous run
        # terminated (normal exit, SIGINT, SIGTERM, crash, OOM kill), this sequence:
        #
        #   1. reconcile_sqlite_flags_with_faiss() - finds rows with in_index=1 that
        #      are NOT in FAISS (e.g., DB committed but FAISS not saved before kill)
        #      and resets their flags to in_index=0
        #
        #   2. backfill_unindexed_vectors() - re-embeds and re-adds any rows with
        #      in_index=0 to FAISS, then marks them in_index=1
        #
        # guarantees the vector store is consistent before any new work begins.
        #
        # This runs in ALL relevant modes:
        #   - --faiss-writer (normal build): here
        #   - --reconcile-only: explicitly runs reconcile+backfill, then exits
        #   - --consume-only: calls run_consume_only_mode() which has its own guards
        #
        # Producers (--embed-producer) don't touch FAISS, so no reconcile needed.
        # ═══════════════════════════════════════════════════════════════════════════
        
        # Reset any rows marked in_index=1 that are missing from FAISS
        p_reset, c_reset = reconcile_sqlite_flags_with_faiss(conn, paper_index, chunk_index)
        if p_reset or c_reset:
            _eprint(
                f"[reconcile] reset flags for missing vectors — papers={p_reset} chunks={c_reset}"
            )

        backfill_unindexed_vectors(
            conn,
            paper_embedder,
            chunk_embedder,
            paper_index,
            chunk_index,
            paper_bs=args.paper_embed_bs,
            chunk_bs=args.chunk_embed_bs,
        )

        # Respect lock order while forcing saves and flushing marks
        with FileLock(FAISS_LOCK):
            faiss_save_force(paper_index, PAPER_INDEX_PATH)
            faiss_save_force(chunk_index, CHUNK_INDEX_PATH)
        with FileLock(DB_LOCK):
            d.db_flush_pending_marks(conn.cursor())
            conn.commit()

        try:
            k, _ = kind_and_core(paper_index)
            if k == "hnsw":
                _eprint(
                    f"[done] HNSW (papers): build complete — ntotal={int(getattr(paper_index, 'ntotal', 0) or 0)}"
                )
        except Exception:
            pass

        if not args.quiet:
            _eprint("\n=== SUMMARY ================================================")
        _post_build_sanity_check(conn, args)
        if not args.quiet:
            _eprint("=================================================================")
    else:
        if not args.quiet:
            _eprint("\n=== SUMMARY ================================================")
        _post_build_sanity_check(conn, args)
        if not args.quiet:
            _eprint("=================================================================")

    _eprint(f"[done] indexed {papers_added_total} papers and {chunks_added_total} chunks (this run)")
    conn.close()
    return


# -------------------- Retrieval helpers --------------------




def reconcile_sqlite_flags_with_faiss(conn, paper_index, chunk_index) -> tuple[int, int]:
    """Thin wrapper: delegates to litkit.build.backfill."""
    _load_heavy_deps()  # litkit.build pulls in faiss/numpy
    from litkit.build import reconcile_sqlite_flags_with_faiss as build_reconcile_sqlite_flags
    return build_reconcile_sqlite_flags(conn, paper_index, chunk_index)




def shortlist_papers(
    question: str,
    k: int,
    efsearch: int = 128,
    *,
    embedder: "Embedder | None" = None,
) -> list[int]:
    """Thin wrapper: delegates to litkit.retrieval.shortlist_papers."""
    d = deps()  # ensures loaded + returns namespace (consistent pattern)
    from litkit.retrieval import shortlist_papers as retrieval_shortlist_papers
    get_runtime()
    enc = embedder or d.make_paper_embedder()[0]
    return retrieval_shortlist_papers(
        question, k,
        paper_index_path=PAPER_INDEX_PATH,
        embedder=enc,
        efsearch=efsearch,
    )


def search_chunks_constrained(
    question: str,
    candidate_papers: list[int],
    k: int,
    overshoot: int = 20,
    nprobe: int | None = None,
    min_chunks_per_paper: float | None = None,
    lexical_cap: int | None = None,
    lexical_limit: int = 200,
    allow_global_lexical: bool | None = None,
    embedder: "Embedder | None" = None,
    per_paper_cap: int = 0,
) -> tuple[list[int], dict[str, int]]:
    """Thin wrapper: delegates to litkit.retrieval.search_chunks_constrained."""
    d = deps()  # ensures loaded + returns namespace (consistent pattern)
    from litkit.retrieval import search_chunks_constrained as retrieval_search_chunks_constrained
    get_runtime()
    enc = embedder or d.make_chunk_embedder()[0]
    return retrieval_search_chunks_constrained(
        question=question,
        candidate_papers=candidate_papers,
        k=k,
        chunk_index_path=CHUNK_INDEX_PATH,
        db_path=DB_PATH,
        embedder=enc,
        connect_db=d.db_connect_db,
        load_temp_candidates=d.db_load_temp_candidates,
        chunk_ids_to_paper_ids=d.db_chunk_ids_to_paper_ids,
        overshoot=overshoot,
        nprobe=nprobe,
        min_chunks_per_paper=min_chunks_per_paper,
        lexical_cap=lexical_cap,
        lexical_limit=lexical_limit,
        allow_global_lexical=allow_global_lexical,
        per_paper_cap=per_paper_cap,
    )


def get_chunks(conn, ids: list[int]) -> list[dict[str, str]]:
    """Thin wrapper: delegates to litkit.retrieval.get_chunks."""
    _load_heavy_deps()  # litkit.retrieval pulls in faiss/numpy
    from litkit.retrieval import get_chunks as retrieval_get_chunks
    return retrieval_get_chunks(conn, ids)


# -------------------- LLM + token-budgeting --------------------
# OpenAI client is imported lazily inside LLM paths to keep build-only runs offline-safe.

# Define a single system prompt constant at module scope
SYS_PROMPT = (
    "You are a precise scientific assistant. Use ONLY the provided context chunks to answer; "
    "do not use prior knowledge. You MUST include bracketed citations like [1], [2] that refer "
    "to the provided chunks. Cite what you use in your answer and prefer multiple sources when "
    "the claim spans chunks. If context is insufficient, say so briefly."
)


def approx_tokens(s: str) -> int:
    """Very rough char→token approximation used to enforce context budgets.
    Uses ≈4 chars per token, returns at least 1.
    """
    return max(1, len(s) // 4)


def pack_context(
    chunks: list[dict[str, str]],
    question: str,
    model_name: str,
    *,
    sys_prompt: str = SYS_PROMPT,
    max_out_tokens: int = 3000,
) -> tuple[str, list[int], dict]:
    """Assemble a model-aware context window from ranked chunks, respecting an approximate
    token budget determined by the target model. Uses the *actual* system prompt for
    budgeting to avoid drift.

    Args:
        chunks: Ranked list of chunk dicts with 'text', 'paper_title', etc.
        question: User question text.
        model_name: Model identifier for budget selection (e.g., "o3-mini", "gpt-oss:20b").
        sys_prompt: System prompt to budget for.
        max_out_tokens: Output token allowance to reserve (deducted from input budget).

    Returns:
    -------
    (context_text, used_indices, token_meta)
      context_text : str
          Concatenated context blocks prefixed with [i] and paper metadata.
      used_indices : List[int]
          1-based indices of chunks that fit within the budget (in order).
      token_meta : dict
          Token budget metadata for logging:
          - approx_tokens: approximate tokens used (heuristic: 4 chars/token)
          - budget: total token budget for the model
          - input_budget: budget minus max_out_tokens
          - max_out_tokens: reserved output allowance
    """
    # Choose token budget per model family
    budget = BUDGET_TOKENS_O3 if model_name.lower().startswith("o3") else BUDGET_TOKENS_OSS20B

    # Reserve output allowance from the total budget first
    # This prevents overflow exceptions by ensuring we have room for the response
    input_budget = max(0, budget - max_out_tokens)

    # Budget against exactly what you'll send (prompt + "QUESTION:/CONTEXT:" wrappers)
    base_cost = (
        approx_tokens(sys_prompt)
        + approx_tokens("QUESTION:\n")
        + approx_tokens(question)
        + approx_tokens("\n\nCONTEXT:\n")
        + PROMPT_HEADROOM_TOKENS  # safety buffer for tool/SDK scaffolding
    )
    remain = max(0, input_budget - base_cost)

    blocks: list[str] = []
    used: list[int] = []

    for i, ch in enumerate(chunks, 1):
        meta = []
        if ch.get("pmcid"):
            meta.append(f"PMCID:{ch['pmcid']}")
        if ch.get("pmid") and not ch.get("pmcid"):
            meta.append(f"PMID:{ch['pmid']}")
        meta_str = f" ({', '.join(meta)})" if meta else ""

        block = f"[{i}] {ch.get('paper_title','').strip()}{meta_str}\n{ch['text']}"
        cost = approx_tokens(block) + 20  # small join/formatting overhead

        if cost <= remain:
            blocks.append(block)
            used.append(i)
            remain -= cost
        else:
            break

    ctx_text = "\n\n".join(blocks) if blocks else "(no context)"
    approx_used = input_budget - remain  # how many tokens we consumed
    token_meta = {
        "approx_tokens": approx_used,
        "budget": budget,
        "input_budget": input_budget,
        "max_out_tokens": max_out_tokens,
    }
    return ctx_text, used, token_meta


def answer_with_llm(
    question, chunks, model, base_url, api_key, max_out_tokens=None, *, sys_prompt: str = SYS_PROMPT
):
    """Call an OpenAI-compatible endpoint to answer using ONLY the provided context.

    Policy:
      - "o3*" -> Responses API with reasoning.
      - otherwise -> Chat Completions.
      - Fallback: if Responses API is unavailable (e.g., local endpoint), fall back to Chat.

    Overflow handling:
      - Detect a broader set of context/token-limit errors.
      - On each retry, trim chunks AND reduce max_out to actually free room.
    """

    def _clarify_llm_error(model: str, base_url: str, raw: str) -> str:
        msg = (raw or "").strip()
        low = msg.lower()
        # Common local/LM Studio cases
        if (
            ("no models loaded" in low)
            or ("model_not_found" in low)
            or ("404" in low and "model" in low)
        ):
            return (
                "[llm] ERROR: The endpoint is up but has no model loaded (or does not recognize "
                f"{model!r}).\n"
                f"  • Endpoint: {base_url}\n"
                "  • Fix (LM Studio): open LM Studio, load a chat model, and enable the local server "
                "(or run: `lms load <model_name>`). Then rerun your command.\n"
                "  • Alt: use OpenAI — e.g., `--llm-model o3-mini --openai-api-key $OPENAI_API_KEY`.\n"
                "  • Alt: retrieval only — add `--no-llm`.\n"
                f"  • Provider message: {msg}\n"
            )
        if ("unauthorized" in low) or ("invalid api key" in low) or ("401" in low):
            return (
                "[llm] ERROR: Authentication failed for the LLM endpoint.\n"
                f"  • Endpoint: {base_url}\n"
                "  • Fix: pass a valid `--openai-api-key` (OpenAI), or for local servers set a dummy token or none, "
                "depending on the server’s requirements.\n"
                f"  • Provider message: {msg}\n"
            )
        return (
            "[llm] ERROR: LLM call failed.\n"
            f"  • Endpoint: {base_url}\n"
            f"  • Model: {model}\n"
            "  • Try: load a local model, switch to an OpenAI model with a valid API key, "
            "or run with `--no-llm`.\n"
            f"  • Provider message: {msg}\n"
        )

    # Lazy import to avoid hard dependency during build-only runs.
    try:
        import openai
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "[llm] OpenAI client not installed; use --build-only or install 'openai'."
        ) from e

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=OPENAI_TIMEOUT_SEC)
    m = (model or "").lower()
    is_o3 = m.startswith("o3")

    # Preflight check: local servers often support `models.list`.
    # NOTE: Some OpenAI-compatible servers (vLLM, text-generation-inference, etc.)
    # return empty from models.list() but still accept completions. We now treat
    # empty results as a warning, not an error.
    try:
        models_resp = client.models.list()
        available = [
            getattr(x, "id", str(x)) for x in getattr(models_resp, "data", list(models_resp) or [])
        ]
        if ("localhost" in base_url or "127.0.0.1" in base_url) and not available:
            # Some servers don't implement models.list properly - warn but continue
            sys.stderr.write(
                "[llm] WARNING: models.list() returned empty; proceeding anyway.\n"
                "  (Some servers don't implement this endpoint but still accept completions.)\n"
            )
        elif (
            available
            and (model not in available)
            and ("localhost" in base_url or "127.0.0.1" in base_url)
        ):
            # Model not in list - warn but continue (server might accept it anyway)
            sys.stderr.write(
                f"[llm] WARNING: Model {model!r} not in models.list() response.\n"
                f"  Available: {', '.join(available[:8])}{' …' if len(available) > 8 else ''}\n"
                "  Proceeding anyway - server may still accept this model.\n"
            )
    except Exception:
        # Not fatal: some providers don't implement models.list; proceed to the main call.
        pass

    # Default output budgets (conservative)
    max_out = int(max_out_tokens) if max_out_tokens is not None else 3000

    # Work on a local copy so we can trim safely on retry
    working_chunks = list(chunks)

    # Up to 4 tries: progressively trim the number of chunks *and* reduce max_out
    for attempt in range(4):
        ctx_text, _used_idxs, _ = pack_context(working_chunks, question, model, sys_prompt=sys_prompt, max_out_tokens=max_out)
        sys_msg = sys_prompt

        try:
            if is_o3:
                # Primary: Responses API (use string `input` + `instructions`)
                try:
                    resp = client.responses.create(
                        model=model,
                        input=f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}",
                        instructions=sys_msg,
                        max_output_tokens=max_out,
                        reasoning={"effort": "medium"},
                    )
                    text = getattr(resp, "output_text", None)
                    if text is None:
                        # Older SDKs: synthesize from content if needed
                        try:
                            parts = []
                            for item in getattr(resp, "output", []) or []:
                                for c in getattr(item, "content", []) or []:
                                    if getattr(c, "type", "") == "output_text":
                                        parts.append(getattr(c, "text", ""))
                            text = "".join(parts).strip() if parts else ""
                        except Exception:
                            text = ""
                    return (text or "").strip()
                except Exception as ee:
                    # If the endpoint doesn't support Responses, surface a clear error.
                    msg = (str(ee) or "").lower()
                    if any(
                        s in msg
                        for s in (
                            "404",
                            "not found",
                            "405",
                            "method not allowed",
                            "responses.create",
                            "unknown parameter",
                            "unexpected argument",
                            "unrecognized field",
                            "invalid request body",
                            "schema validation",
                            "unsupported field",
                            "does not support reasoning",
                            "unsupported parameter 'reasoning'",
                        )
                    ):
                        raise RuntimeError(
                            f"[llm] The endpoint at {base_url!r} does not support the Responses API "
                            f"for model {model!r}. Use an OpenAI endpoint for o-series (Responses-only), "
                            "switch to a chat-compatible local model (e.g., --llm-model gpt-oss:20b), "
                            "or run retrieval-only with --no-llm on air-gapped systems."
                        ) from ee
                    # Otherwise, bubble up for the outer overflow handler
                    raise

            else:
                # Local/OSS endpoints: Chat Completions
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {
                            "role": "user",
                            "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}",
                        },
                    ],
                    temperature=0.2,
                    max_tokens=max_out,
                )
                text = getattr(resp.choices[0].message, "content", "") or ""
                return text.strip()

        except Exception as e:
            # Overflow detection via message matching only (not exception type)
            # BadRequestError is too broad - also covers "unknown model", "invalid request", etc.
            # We only want to retry-with-trimming for actual context/token overflow errors.
            msg = (str(e) or "").lower()

            is_overflow = (
                "context length" in msg
                or "maximum context length" in msg
                or "exceeds context window" in msg
                or "token limit" in msg
                or "too many tokens" in msg
                or "reduce the length of the messages" in msg
                or "max tokens" in msg
                or "prompt too long" in msg
                or "input too long" in msg
                or "payload too large" in msg
                or "413" in msg  # HTTP 413 Payload Too Large
            )

            if is_overflow and attempt < 3:
                # Trim chunks harder each time and reduce max_out to free space
                if len(working_chunks) > 1:
                    if attempt == 0:
                        new_len = max(1, int(len(working_chunks) * 0.7))
                    else:
                        new_len = max(1, len(working_chunks) // 2)
                    working_chunks = working_chunks[:new_len]
                # reduce output allowance by 25% each retry (floor at 128)
                max_out = max(128, int(max_out * 0.75))
                continue

            # Not an overflow, or no sensible retry left
            raise RuntimeError(_clarify_llm_error(model, base_url, str(e))) from e


# -------------------- Citations: normalize + print only cited --------------------

# Keep these for potential downstream uses (harmless if unused)
_CITATION_BR = re.compile(
    r"(\[(?:\s*\d+(?:\s*,\s*\d+)*\s*)\])"  # [1] or [1, 3]
    r"|"
    r"(【(?:\s*\d+(?:\s*[,、，]\s*\d+)*\s*)】)"  # 【2】 or 【1, 3】 (Chinese/JP commas allowed)
)

# We’ll normalize by extracting only the leading numeric list inside a bracket and
# discarding any trailing “†L1–L8” or similar. Accept -, – or — in those tails.
_CITATION_LINELOC = re.compile(r"([\[【]\s*\d+)\s*†L\d+(?:[–—-]\d+)?(\s*[】\]])")

# General bracket grabber; we’ll rebuild the contents canonically.
_CITATION_ANYBR = re.compile(r"(?P<open>[\[【])(?P<inside>[^\[\]【】]{0,200}?)(?P<close>[】\]])")

# Leading numeric list only (before any tails); supports ASCII/CJK commas.
_LEADING_NUM_LIST = re.compile(r"^\s*(\d+(?:\s*[，、,]\s*\d+)*)")

def _normalize_and_strip_citations(text: str) -> str:
    """Canonically normalize bracketed numeric citations so downstream matching works:
       1) Convert fullwidth brackets to ASCII [ ].
       2) Remove '†Lx–Ly' / location tails by only keeping the leading numeric list.
       3) Normalize commas/spacing, de-duplicate while preserving order.
    """
    import unicodedata  # deferred to minimize module-level side effects
    try:
        s = unicodedata.normalize("NFKC", text)
    except Exception:
        s = text

    # Strip common zero-width junk that can sneak in
    for z in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"):
        s = s.replace(z, "")

    # Unify bracket style so later regexes see ASCII brackets
    s = s.replace("【", "[").replace("】", "]")

    # Collapse explicit †L tails like “[2†L1–L8]” -> “[2]” (still leave contents to the next pass)
    s = _CITATION_LINELOC.sub(r"\1\2", s)

    # Rebuild each bracketed group canonically from just the *leading* numeric list
    def _rebuild(m: re.Match) -> str:
        inside = m.group("inside")
        lead = _LEADING_NUM_LIST.match(inside)
        if not lead:
            # Not a numeric citation; leave untouched (e.g., [Note])
            return m.group(0)

        nums_str = lead.group(1)
        parts = [p.strip() for p in re.split(r"[，、,]", nums_str) if p.strip()]

        out, seen = [], set()
        for p in parts:
            if p.isdigit():
                n = int(p)
                if n not in seen:
                    seen.add(n)
                    out.append(n)

        if not out:
            # Nothing numeric survived; drop the brackets entirely
            return ""

        return "[" + ", ".join(str(n) for n in out) + "]"

    return _CITATION_ANYBR.sub(_rebuild, s)

# Back-compat for existing call sites that invoke _strip_citation_linelocs()
def _strip_citation_linelocs(text: str) -> str:
    return _normalize_and_strip_citations(text)


# -------------------- Main --------------------
def main():
    """CLI entry point."""
    # Check Python version at runtime (moved from import-time for import purity)
    if sys.version_info < (3, 10):
        sys.stderr.write("[env] Python >= 3.10 required.\n")
        sys.exit(2)
    
    # NOTE: get_runtime() is called AFTER --offline handling so that HF_HUB_OFFLINE
    # is set before setup_environment() runs. See below after parse_args().
    
    # declare BEFORE any references to these names in this function (to satisfy Python rule)
    global PAPER_BATCH, CHUNK_BATCH, CKPT_EVERY
    global DEFAULT_BUSY_TIMEOUT_MS
    global paper_seg_writer, chunk_seg_writer

    ap = argparse.ArgumentParser(
        description="An air-gapped HPC two-stage RAG pipeline for the PMC-OA corpus"
    )

    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Squelch startup banners and section headers for batch logs. "
        "Tip: set LITKIT_QUIET=1 to also silence earliest pre-arg prints.",
    )

    # Allow fractional threshold and honor env if provided
    min_cpp_env = os.environ.get("LITKIT_MIN_CHUNKS_PER_PAPER")
    try:
        MIN_CPP_DEFAULT = float(min_cpp_env) if min_cpp_env is not None else 2.0
    except ValueError:
        MIN_CPP_DEFAULT = 2.0
        # logging deferred to main(); use stderr for early warnings
        sys.stderr.write(f"[args] WARNING: Ignoring invalid LITKIT_MIN_CHUNKS_PER_PAPER={min_cpp_env!r}; using 2.0\n")
    ap.add_argument(
        "--min-chunks-per-paper",
        type=float,
        default=MIN_CPP_DEFAULT,
        help="If the Stage-1 shortlist is too sparse (avg chunks/paper below this), "
        "Stage-2 falls back to global chunk search. Default: env LITKIT_MIN_CHUNKS_PER_PAPER or 2.",
    )

    ap.add_argument(
        "--version",
        action="version",
        version=_version_banner(),
        help="Print version and exit",
    )

    ap.add_argument("question", nargs="?", help='Question to ask, e.g. "What is HIV?"')
    ap.add_argument(
        "--question-file",
        type=Path,
        help="Read the question from a text file; use '-' to read from stdin.",
    )

    ap.add_argument(
        "--hnsw-recall",
        choices=["normal", "high"],
        default="normal",
        help="Preset to bump HNSW recall: 'high' sets M=48, efConstruction=300, efSearch=256.",
    )

    ap.add_argument(
        "--chunk-target-chars",
        type=int,
        default=CHUNK_TARGET_CHARS,
        help="Approx target characters per chunk (default 1200).",
    )
    ap.add_argument(
        "--chunk-min-chars",
        type=int,
        default=BODY_MIN_CHARS,
        help="Minimum characters per chunk (default 300).",
    )
    ap.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP_CHARS,
        help="Character overlap between consecutive chunks (default 200).",
    )

    ap.add_argument(
        "--max-out-tokens",
        type=int,
        default=None,
        help="Cap on LLM output tokens. If omitted, defaults to 3000.",
    )

    ap.add_argument(
        "--per-paper-cap",
        type=int,
        default=3,
        help="Max number of chunks per paper in the final context (default: 3)",
    )

    ap.add_argument(
        "--reconcile-only",
        action="store_true",
        help="Do not scan corpus; only reconcile SQLite in_index flags with FAISS and backfill any missing vectors.",
    )

    ap.add_argument(
        "--lexical-cap",
        type=int,
        default=None,
        help="Max lexical-boost items to interleave into top-k (default = max(5, k//3)).",
    )
    ap.add_argument(
        "--allow-global-lexical",
        action="store_true",
        help="Allow lexical boost without a candidate-paper filter (also respects LITKIT_ALLOW_GLOBAL_LEXICAL=1).",
    )
    ap.add_argument(
        "--lexical-limit",
        type=int,
        default=200,
        help="SQL LIMIT for lexical candidate scan (default 200).",
    )

    # stream NXMLs from tar shards (without extraction)
    ap.add_argument(
        "--tar-dir",
        type=Path,
        default=None,
        help="Directory containing tar shards; streamed without extracting. ",
    )
    ap.add_argument(
        "--tar-manifest",
        type=Path,
        default=None,
        help="Optional newline file of absolute tar paths (one per line). Overrides --tar-dir.",
    )

    # run fully offline for HuggingFace (sets HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for this process)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode for HuggingFace (no network); equivalent to HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.",
    )

    # control SQLite journal mode explicitly (TRUNCATE is best on Lustre/NFS)
    ap.add_argument(
        "--sqlite-journal-mode",
        choices=["TRUNCATE", "WAL"],
        default="TRUNCATE",
        help="SQLite journal mode; TRUNCATE recommended for shared filesystems.",
    )

    ap.add_argument(
        "--sqlite-busy-timeout-ms",
        type=int,
        default=DEFAULT_BUSY_TIMEOUT_MS,
        help="SQLite busy timeout in milliseconds (both Python connect() and PRAGMA busy_timeout). "
        "Use higher values (e.g., 120000–300000) on NFS/Lustre with shared writers.",
    )

    ap.add_argument(
        "--build-only", action="store_true", help="Only (re)build indexes; do not run a query"
    )
    ap.add_argument(
        "--init-indices-only",
        action="store_true",
        help="Create empty FAISS indices and exit immediately (for multi-node bootstrap). "
        "Requires --faiss-writer. Does not process any documents.",
    )
    ap.add_argument("--update", action="store_true", help="Append-only update (skip seen files)")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe DB & indices; full rebuild (asks for confirmation in TTY)",
    )
    ap.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force --rebuild even if unconsumed segments or active producers exist. "
        "WARNING: This can cause data corruption if producers are still active.",
    )
    # NOTE: --yes only takes effect AFTER parse_args() returns. If you ever add
    # pre-parse prompts (before argparse runs), check "-y" in sys.argv directly
    # instead of relying on os.environ["LITKIT_ASSUME_YES"].
    ap.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (for automation). Equivalent to LITKIT_ASSUME_YES=1.",
    )
    ap.add_argument(
        "--strict-ingest",
        action="store_true",
        help="Fail fast on ingest errors instead of mark-and-skip (default: skip failed members and continue)",
    )

    # index types / params
    ap.add_argument("--papers-index", choices=["hnsw", "flat"], default="hnsw")
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--efsearch", type=int, default=128)
    ap.add_argument(
        "--efconstruction",
        type=int,
        default=200,
        help="HNSW build-time efConstruction (higher improves recall at build time)",
    )

    ap.add_argument("--chunks-index", choices=["ivfpq", "flat"], default="ivfpq")
    ap.add_argument("--ivf-nlist", type=int, default=16384)
    ap.add_argument("--pq-m", type=int, default=64)
    ap.add_argument(
        "--nprobe",
        type=int,
        default=None,  # None means: auto = ~sqrt(nlist) at search-time
        help="IVF probe count. If omitted, set dynamically to ~sqrt(nlist) (clamped to [8,512]).",
    )

    # batching / limits
    ap.add_argument("--paper-batch", type=int, default=PAPER_BATCH)
    ap.add_argument("--chunk-batch", type=int, default=CHUNK_BATCH)
    ap.add_argument(
        "--ckpt-every", type=int, default=CKPT_EVERY, help="Checkpoint scan progress every N files"
    )

    ap.add_argument(
        "--force-embed-devices",
        action="store_true",
        help="Allow loose device tokens (e.g., 'cuda:1'); otherwise invalid tokens error out.",
    )

    # retrieval sizes
    ap.add_argument(
        "--top-papers",
        type=int,
        default=None,  # adaptive if not provided
        help=f"Stage-1 shortlist size (papers). Default {TOP_PAPERS_DEFAULT} if omitted.",
    )
    ap.add_argument(
        "--top-chunks",
        type=int,
        default=TOP_CHUNKS_DEFAULT,
        help=f"Stage-2 final chunk count passed to LLM (default {TOP_CHUNKS_DEFAULT}).",
    )
    ap.add_argument(
        "--overshoot",
        type=int,
        default=OVERSHOOT_DEFAULT,
        help=f"Stage-2 ANN overshoot multiplier before filtering by candidate papers (default {OVERSHOOT_DEFAULT}).",
    )

    # shard fanout (for builders/readers)
    ap.add_argument("--shard-id", type=int, default=0, help="This process' shard id (0-indexed).")
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")

    # writer / producer role
    ap.add_argument(
        "--faiss-writer",
        action="store_true",
        help="This process is allowed to mutate and save FAISS indices.",
    )
    ap.add_argument(
        "--embed-producer",
        action="store_true",
        help="Producer mode: embed chunks to on-disk segments instead of mutating FAISS.",
    )
    ap.add_argument(
        "--consume-segments",
        action="store_true",
        help="Writer: also consume any pending segment files at the end of a build.",
    )
    ap.add_argument(
        "--consume-only",
        action="store_true",
        help="Consumer-only mode: skip tar scanning and embedding, only ingest segments in a polling loop. "
        "Requires --faiss-writer. Typically used on a dedicated consumer node in multi-node setups.",
    )
    ap.add_argument(
        "--embed-outdir",
        type=Path,
        default=None,
        help="Directory for chunk embedding segments (producer output / writer input). "
        "Defaults to WORKSPACE/emb_segments if omitted.",
    )

    # embedder knobs
    ap.add_argument(
        "--embed-devices",
        type=str,
        default="auto",
        help='Devices for SBERT (e.g. "auto", "cpu", "mps", "cuda:0,cuda:1").',
    )
    ap.add_argument(
        "--embed-workers",
        type=int,
        default=1,
        help="Number of worker processes for multi-GPU SBERT (one typically per CUDA device).",
    )
    ap.add_argument(
        "--paper-embed-bs", type=int, default=16, help="Batch size for SPECTER2 (papers)."
    )
    ap.add_argument("--chunk-embed-bs", type=int, default=64, help="Batch size for SBERT (chunks).")
    ap.add_argument(
        "--parse-workers",
        type=int,
        default=8,
        help="Number of parallel XML parsing workers (default: 8). "
        "Higher values improve tar scanning throughput on multi-core systems.",
    )

    # LLM options
    ap.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"LLM to use (default {DEFAULT_LLM_MODEL}).",
    )
    ap.add_argument(
        "--openai-base-url", type=str, default=None, help="Override OpenAI-compatible base URL."
    )
    ap.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Override OpenAI API key (or local endpoint token).",
    )
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="Run retrieval only and print selected context; do not call an LLM.",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # INVARIANT: parse_args() MUST come before _load_heavy_deps()
    # ═══════════════════════════════════════════════════════════════════════════
    # argparse automatically exits on --version, -h, --help BEFORE returning.
    # This guarantees fast --help/--version even if heavy dependencies (faiss,
    # torch, transformers) are missing or broken.
    #
    # DO NOT move _load_heavy_deps() or any heavy import above this line.
    # ═══════════════════════════════════════════════════════════════════════════
    args = ap.parse_args()

    # ivf_nlist_forced = "--ivf-nlist" in sys.argv
    seen_flags = {s.split("=", 1)[0] for s in sys.argv}
    args._ivf_nlist_forced = ("--ivf-nlist" in seen_flags)

    # Honor --offline FIRST (before any heavy imports that touch transformers/HF)
    # Transformers checks HF_HUB_OFFLINE at import time, so this MUST come before _load_heavy_deps()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ═══════════════════════════════════════════════════════════════════════════
    # DEFERRED HEAVY IMPORTS: _load_heavy_deps() is NOT called here!
    # ═══════════════════════════════════════════════════════════════════════════
    # Heavy dependencies (faiss, torch, transformers) are loaded ONLY when needed:
    # - build_or_update_indices() calls _load_heavy_deps() internally
    # - --reconcile-only calls _require_faiss() which loads deps
    # - Retrieval pipeline calls _require_faiss() which loads deps
    #
    # This ensures fast exits for validation errors (missing tar shards, flag
    # conflicts, etc.) without paying the 2-5 second import penalty.
    # ═══════════════════════════════════════════════════════════════════════════

    # Trigger lazy runtime initialization - populates path globals like WORKSPACE, DB_PATH, etc.
    # This must happen before any code that uses path globals (e.g., _vector_store_exists).
    get_runtime()

    # Do we have a vector store?
    def _vector_store_exists() -> bool:
        """Return True only if ALL three required paths exist:
        - PAPER_INDEX_PATH (FAISS paper index)
        - CHUNK_INDEX_PATH (FAISS chunk index)
        - DB_PATH (SQLite database)
        
        If any component is missing, returns False (treat as "no vector store").
        This is intentional: a partial store (e.g., DB exists but one index missing)
        requires a fresh build or --init-indices-only to bootstrap.
        
        Note: Path.exists() returns False for missing files (doesn't raise).
        OSError can occur for permission denied, path too long, etc.
        """
        try:
            return PAPER_INDEX_PATH.exists() and CHUNK_INDEX_PATH.exists() and DB_PATH.exists()
        except OSError as e:
            # Permission denied, path too long, etc. - fail fast with clear message
            _eprint(f"[error] cannot access vector store paths: {e.__class__.__name__}: {e}")
            raise SystemExit(2)

    # Early guard for --consume-only with missing indices
    if args.consume_only and not _vector_store_exists():
        raise SystemExit(
            "[consumer] FAISS indices/DB not found. Run an initial build "
            "or --init-indices-only first before using --consume-only."
        )

    # Make all later db_connect_db() calls honor the user's timeout setting
    # by setting the env var that litkit.db.connection reads.
    #
    # NOTE: This works because litkit.db.connection reads the env var at CALL TIME
    # (via _get_busy_timeout() inside connect_db()), not at import time. Importing
    # the module just loads function definitions; the env var isn't read until
    # connect_db() is actually invoked. Safe to set here after parse_args().
    os.environ["LITKIT_SQLITE_BUSY_TIMEOUT_MS"] = str(args.sqlite_busy_timeout_ms)

    # Do we need tar shards?
    # Note: --consume-only doesn't need corpus (it only ingests pre-computed segments)
    # Note: --reconcile-only doesn't need corpus (it only fixes in_index flags)
    # Note: --faiss-writer is a role flag (may mutate indices), not "must scan tars"
    needs_corpus = (
        not args.consume_only
        and not args.reconcile_only
        and (
            args.rebuild
            or args.update
            or args.build_only
            or args.embed_producer
            or (not _vector_store_exists())
        )
    ) and not args.init_indices_only  # Bootstrap doesn't need corpus

    # --- helper: determine if a directory contains .tar.gz files ---
    def _has_tars(p: Path) -> bool:
        try:
            return p.exists() and (
                any(p.glob("*.tar"))
                or any(p.glob("*.tar.*"))  # catches .tar.gz, .tar.bz2, etc.
                or any(p.glob("*.tgz"))
            )
        except Exception:
            return False

    # --- resolve path to tar shards (CLI > ENV > defaults) ---
    env_tar = os.getenv("LITKIT_TAR_DIR")

    # Track provenance of tar shard source setting for clarity in logs.
    tar_dir_origin: str | None = None

    if args.tar_manifest is not None:
        effective_tar_dir = None
    elif args.tar_dir is not None:
        effective_tar_dir = Path(args.tar_dir).expanduser().resolve()
    elif env_tar:
        effective_tar_dir = Path(env_tar).expanduser().resolve()
        tar_dir_origin = "(from LITKIT_TAR_DIR)"
    else:
        # Only accept defaults that actually contain tar shards.
        candidates: list[Path] = [
            (ROOT.parent / "litkit_input" / "tar_shards"),
            (WORKSPACE / "tar_shards"),
        ]

        def _pick(cands: list[Path]) -> Path | None:
            for c in cands:
                if _has_tars(c):
                    return c
            return None

        effective_tar_dir = _pick(candidates)
        if effective_tar_dir is not None:
            tar_dir_origin = "(default)"

    args.tar_dir = effective_tar_dir
    args.tar_dir_origin = tar_dir_origin

    if needs_corpus and args.tar_manifest is not None and not args.tar_manifest.exists():
        raise SystemExit(f"Manifest not found: {args.tar_manifest}")

    # Guard: building from scratch requires a writer/producer role
    # Without this, we'd create in-memory indices, do all the DB work, but never persist FAISS.
    if needs_corpus and not args.faiss_writer and not args.embed_producer and not _vector_store_exists():
        raise SystemExit(
            "[build] No vector store found. A fresh build requires one of:\n"
            "  • --faiss-writer           (single-node build, or writer node in multi-node)\n"
            "  • --embed-producer         (producer node in multi-node setup)\n"
            "  • --init-indices-only --faiss-writer (bootstrap empty indices first)\n\n"
            "For query-only usage, first run a build with one of the above flags."
        )

    # No source error
    if needs_corpus and args.tar_dir is None and args.tar_manifest is None:
        raise SystemExit(
            "No source specified for tar shards.\n"
            "Provide --tar-dir or set env LITKIT_TAR_DIR.\n"
            "You can also pass a file with absolute shard paths via --tar-manifest.\n"
            f"Defaults checked: {ROOT.parent / 'litkit_input' / 'tar_shards'} and {WORKSPACE / 'tar_shards'}."
        )

    # If a directory was resolved but has no tar files, exit with error message.
    if needs_corpus and args.tar_manifest is None and args.tar_dir is not None and not _has_tars(args.tar_dir):
        raise SystemExit(
            f"No tar shards found in {args.tar_dir}\n"
            "Expected one of: *.tar, *.tar.gz, *.tar.*, or *.tgz.\n"
            "Specify a directory containing shards via --tar-dir or LITKIT_TAR_DIR, "
            "or pass a file with absolute shard paths via --tar-manifest.\n"
            f"Defaults: {ROOT.parent / 'litkit_input' / 'tar_shards'} or {WORKSPACE / 'tar_shards'}."
        )

    # Report paths
    if not args.quiet:
        if needs_corpus:
            _report_paths(args.tar_dir, WORKSPACE, args.tar_manifest, args.tar_dir_origin)
        else:
            # Report the specific reason why tar shards are being skipped
            if args.reconcile_only:
                _eprint("[paths] skipping tar shards: --reconcile-only mode")
            elif args.consume_only:
                _eprint("[paths] skipping tar shards: --consume-only mode")
            elif args.init_indices_only:
                _eprint("[paths] skipping tar shards: --init-indices-only mode")
            elif _vector_store_exists():
                _eprint("[paths] skipping tar shards: existing vector store found")
            else:
                _eprint("[paths] skipping tar shards: no build action requested")
            _eprint(f"[paths] using {WORKSPACE} as writable directory for job artifacts/outputs")

    # Wire CLI --quiet into the early env-based guard for the rest of the run
    if args.quiet:
        os.environ["LITKIT_QUIET"] = "1"

    # Enforce documented default for max output tokens
    if args.max_out_tokens is None:
        args.max_out_tokens = 3000

    PAPER_BATCH = int(args.paper_batch)
    CHUNK_BATCH = int(args.chunk_batch)
    CKPT_EVERY = int(args.ckpt_every)

    # Windows gating
    if not FLOCK_AVAILABLE and sys.platform.startswith("win"):
        if args.faiss_writer or args.embed_producer or args.consume_segments:
            sys.stderr.write(
                "[lock] This run needs POSIX flock (writer/producer modes), "
                "which is unavailable on Windows. "
                "Please run on Linux/macOS for indexing/segment ingestion.\n"
                "Tip: you can still query with --no-llm or do non-mutating reads on Windows.\n"
            )
            sys.exit(2)
        else:
            # allow read-only flows, --version, help, etc.
            pass

    # ---- Validate mutually exclusive flag combinations ----
    if args.embed_producer and args.faiss_writer:
        sys.exit(
            "[args] ERROR: --embed-producer and --faiss-writer are mutually exclusive.\n"
            "  • Producers write embedding segments to disk (not FAISS).\n"
            "  • Writers mutate FAISS indices directly (or ingest segments with --consume-only).\n"
            "Use --embed-producer for producer nodes, --faiss-writer for the single writer/consumer."
        )
    if args.consume_only and args.embed_producer:
        sys.exit(
            "[args] ERROR: --consume-only and --embed-producer are mutually exclusive.\n"
            "  • Consumers ingest pre-computed segments; they don't produce embeddings.\n"
            "Use --embed-producer on producer nodes, --consume-only on the writer node."
        )
    if args.init_indices_only and args.embed_producer:
        sys.exit(
            "[args] ERROR: --init-indices-only and --embed-producer are mutually exclusive.\n"
            "  • --init-indices-only only creates empty FAISS indices (bootstrap step).\n"
            "Run --init-indices-only first, then start producers separately."
        )
    
    # Destructive / store-defining operations require writer role
    if args.rebuild and not args.faiss_writer:
        sys.exit(
            "[args] ERROR: --rebuild requires --faiss-writer.\n"
            "  • --rebuild wipes DB and indices; only a writer can rebuild them.\n"
            "  • Without --faiss-writer, you'd have a wiped store with no rebuilt indices."
        )
    if args.rebuild and args.consume_only:
        sys.exit(
            "[args] ERROR: --rebuild and --consume-only are mutually exclusive.\n"
            "  • --rebuild performs a full wipe-and-rebuild from tar shards.\n"
            "  • --consume-only only ingests pre-computed segments."
        )
    if args.init_indices_only and not args.faiss_writer:
        sys.exit(
            "[args] ERROR: --init-indices-only requires --faiss-writer.\n"
            "  • --init-indices-only creates empty FAISS indices.\n"
            "  • Only a writer can create/save FAISS index files."
        )

    # NOTE: Segment conflict check for --rebuild moved to build_or_update_indices()
    # where _load_heavy_deps() is already called. This allows fast exits for other
    # validation errors without paying the import penalty.

    _create_writer_guard_or_exit(args)

    if args.hnsw_recall == "high" and args.papers_index == "hnsw":
        args.hnsw_m = max(args.hnsw_m, 48)
        args.efconstruction = max(args.efconstruction, 300)
        args.efsearch = max(args.efsearch, 256)

    if args.reconcile_only:
        _require_faiss("reconcile-only mode")
        d = deps()  # ensures loaded + returns namespace (consistent pattern)
        # Lazy import for reconcile-only path
        from litkit.index import faiss_load, faiss_save_force
        
        conn = d.db_connect_db(DB_PATH)
        try:
            try:
                paper_index = faiss_load(PAPER_INDEX_PATH)
                chunk_index = faiss_load(CHUNK_INDEX_PATH)
            except FileNotFoundError:
                sys.stderr.write(
                    "[reconcile] FAISS index files not found; run a build first (e.g., --faiss-writer --build-only)\n"
                )
                return
            p_reset, c_reset = reconcile_sqlite_flags_with_faiss(conn, paper_index, chunk_index)
            if p_reset or c_reset:
                _eprint(f"[reconcile] reset flags — papers={p_reset} chunks={c_reset}")
            # backfill (uses current embedders)
            paper_embedder, _ = d.make_paper_embedder()
            chunk_embedder, _ = d.make_chunk_embedder(
                devices=args.embed_devices,
                workers=args.embed_workers,
                force_devices=args.force_embed_devices,
            )
            backfill_unindexed_vectors(
                conn,
                paper_embedder,
                chunk_embedder,
                paper_index,
                chunk_index,
                paper_bs=args.paper_embed_bs,
                chunk_bs=args.chunk_embed_bs,
            )
            with FileLock(FAISS_LOCK):
                faiss_save_force(paper_index, PAPER_INDEX_PATH)
                faiss_save_force(chunk_index, CHUNK_INDEX_PATH)
        finally:
            conn.close()
        return

    if args.embed_producer:
        d = deps()  # <-- ensures heavy deps are loaded before accessing writers
        outdir = args.embed_outdir or EMBED_SEGMENTS_DIR

        paper_seg_writer = d.SegmentWriter(
            outdir=outdir,
            segment_size=DEFAULT_EMBED_SEGMENT_SIZE,
            dtype=DEFAULT_EMBED_SEGMENT_DTYPE,
            shard_id=args.shard_id,
            kind="papers",
        )
        chunk_seg_writer = d.ChunkSegmentWriter(
            outdir=outdir,
            segment_size=DEFAULT_EMBED_SEGMENT_SIZE,
            dtype=DEFAULT_EMBED_SEGMENT_DTYPE,
            shard_id=args.shard_id,
        )

    # If user asked to build (explicitly) or a rebuild/update is needed, do that first
    # Honor --yes by setting the env so _confirm_rebuild doesn't prompt.
    if args.yes:
        os.environ["LITKIT_ASSUME_YES"] = "1"
    build_or_update_indices(args)
    # True "build-only": stop after indexing even if a question was provided
    if args.build_only or args.init_indices_only:
        return

    # Resolve question (positional arg or --question-file)
    question = _resolve_question(args)
    if not question:
        # No query: we're done if this was build-only, else print a hint
        if args.build_only or args.update or args.rebuild:
            return
        _eprint('No question provided. Example:\n  python -m litkit "What is BioNetGen?"')
        return

    # -------- Retrieval pipeline --------
    _require_faiss("retrieval")
    # Stage 1: shortlist candidate papers (HNSW on SPECTER2)
    k_papers = args.top_papers if args.top_papers is not None else _auto_top_papers()
    papers = shortlist_papers(
        question=question,
        k=int(k_papers),
        efsearch=int(args.efsearch or 128),
        embedder=None,  # constructed lazily inside
    )

    if not papers:
        _eprint(
            "[info] no candidate papers found; consider enabling --allow-global-lexical or lowering --min-chunks-per-paper"
        )
        return

    chunk_ids, meta = search_chunks_constrained(
        question=question,
        candidate_papers=papers,
        k=int(args.top_chunks),
        overshoot=int(args.overshoot),
        nprobe=(int(args.nprobe) if args.nprobe is not None else None),
        min_chunks_per_paper=args.min_chunks_per_paper,
        lexical_cap=args.lexical_cap,
        lexical_limit=args.lexical_limit,
        allow_global_lexical=args.allow_global_lexical,
        per_paper_cap=args.per_paper_cap,
    )

    if not chunk_ids:
        _eprint(
            "[info] no chunks found; consider lowering --min-chunks-per-paper or enabling --allow-global-lexical"
        )
        return

    d = deps()  # ensures loaded + returns namespace (consistent pattern)
    conn = d.db_connect_db(DB_PATH)
    try:
        chunks = get_chunks(conn, chunk_ids)
    finally:
        conn.close()

    # If the user asked for retrieval only, print the context and exit
    if args.no_llm:
        ctx_text, used_idx, token_meta = pack_context(chunks, question, args.llm_model, max_out_tokens=args.max_out_tokens)
        print("CONTEXT")
        print("=" * 80)
        print(ctx_text)
        print("=" * 80)
        if not args.quiet:
            _eprint(f"[context] packed ~{token_meta['approx_tokens']} tokens (budget={token_meta['budget']}, input_budget={token_meta['input_budget']})")
        _eprint(f"[info] used {len(used_idx)} chunks; meta={meta}")
        return

    # Fail fast for o-series when no API key is available (unless retrieval-only)
    # Only enforce this when targeting OpenAI cloud; local/on-prem endpoints may not need auth.
    base_url = args.openai_base_url or _default_base_url_for(args.llm_model)
    api_key  = args.openai_api_key if args.openai_api_key is not None else _default_api_key_for(args.llm_model)
    if args.llm_model.lower().startswith("o3") and _is_openai_cloud(base_url) and not api_key:
        raise SystemExit(
            "[llm] o-series on OpenAI cloud requires an API key. "
            "Either pass --openai-api-key, set OPENAI_API_KEY, "
            "or point --openai-base-url at your on-prem endpoint."
        )

    # -------- LLM call (strict RAG prompt) --------
    # (re-use the resolved base_url/api_key for the call below)
    # base_url = args.openai_base_url or _default_base_url_for(args.llm_model)
    # api_key = args.openai_api_key or _default_api_key_for(args.llm_model)

    ctx_text, used_idx, token_meta = pack_context(chunks, question, args.llm_model, max_out_tokens=args.max_out_tokens)
    if not args.quiet:
        _eprint(f"[context] packed ~{token_meta['approx_tokens']} tokens (budget={token_meta['budget']}, input_budget={token_meta['input_budget']})")
    selected_chunks = [chunks[i - 1] for i in used_idx]  # 0-based indexing
    try:
        answer = answer_with_llm(
            question=question,
            chunks=selected_chunks,  # pass only packed subset
            model=args.llm_model,
            base_url=base_url,
            api_key=api_key,
            max_out_tokens=args.max_out_tokens,
            sys_prompt=SYS_PROMPT,
        )
    except Exception as e:
        sys.stderr.write(str(e).rstrip() + "\n")
        # Fall back to printing context to unblock usage
        _eprint("\n[llm] Falling back to retrieval-only output.\n")
        answer = ""

    if answer:
        # Normalize oddball citation shapes the model may emit
        answer = _strip_citation_linelocs(answer)
        # Collapse chunk-level citations to doc-level and render a clean bibliography
        answer, doc_refs = d.normalize_answer_and_build_refs(answer, selected_chunks)
        print(answer)
        print()
        print(d.render_references(doc_refs))
        print()
    else:
        # No model output -> print the context
        print("CONTEXT")
        print("=" * 80)
        print(ctx_text)
        print("=" * 80)
        _eprint(f"[info] used {len(used_idx)} chunks; meta={meta}")


if __name__ == "__main__":
    main()

# EOF
