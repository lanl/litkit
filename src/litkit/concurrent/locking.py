# litkit/concurrent/locking.py
"""
File-based locking and concurrency control for litkit.

This module centralizes all locking logic that was previously scattered
throughout cli.py, including:
- FileLock context manager (POSIX flock with NFS/Windows fallback)
- Lock depth tracking for nested acquisition
- Writer guard management for exclusive FAISS writers
- Lock ordering enforcement (DB_LOCK before FAISS_LOCK)

Lock ordering rule: Always acquire db_lock BEFORE faiss_lock.
"""

from __future__ import annotations

import atexit
import errno
import os
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litkit.config.paths import WorkspacePaths

# fcntl is not available on Windows
try:
    import fcntl
    FLOCK_AVAILABLE = True
except ModuleNotFoundError:
    fcntl = None  # type: ignore[assignment]
    FLOCK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lock depth tracking (thread-local for proper nesting detection)
# ---------------------------------------------------------------------------

_FAISS_LOCK_DEPTH = threading.local()
_DB_LOCK_DEPTH = threading.local()


def _faiss_lock_enter() -> None:
    _FAISS_LOCK_DEPTH.n = getattr(_FAISS_LOCK_DEPTH, "n", 0) + 1


def _faiss_lock_exit() -> None:
    _FAISS_LOCK_DEPTH.n = max(0, getattr(_FAISS_LOCK_DEPTH, "n", 0) - 1)


def in_faiss_lock() -> bool:
    """Return True if the current thread holds the FAISS lock."""
    return getattr(_FAISS_LOCK_DEPTH, "n", 0) > 0


def _db_lock_enter() -> None:
    _DB_LOCK_DEPTH.n = getattr(_DB_LOCK_DEPTH, "n", 0) + 1


def _db_lock_exit() -> None:
    _DB_LOCK_DEPTH.n = max(0, getattr(_DB_LOCK_DEPTH, "n", 0) - 1)


def in_db_lock() -> bool:
    """Return True if the current thread holds the DB lock."""
    return getattr(_DB_LOCK_DEPTH, "n", 0) > 0


# Track advisory lock status
_ADVISORY_LOCK_DISABLED = False


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# FileLock context manager
# ---------------------------------------------------------------------------

class FileLock:
    """File-based lock using POSIX flock when available.
    
    Falls back to best-effort (no locking) on Windows or NFS filesystems
    that don't support advisory locks.
    
    Usage:
        with FileLock(path):
            # exclusive access
            ...
    
    When used with the specific lock paths (DB_LOCK, FAISS_LOCK), this class
    also tracks lock depth for nested acquisition detection.
    """
    
    def __init__(self, path: Path, *, db_lock_path: Path | None = None,
                 faiss_lock_path: Path | None = None):
        """Initialize a file lock.
        
        Args:
            path: Path to the lock file
            db_lock_path: If provided, enables DB lock depth tracking when
                          path matches this
            faiss_lock_path: If provided, enables FAISS lock depth tracking
                             when path matches this
        """
        self.path = Path(path)
        self._fd = None
        self._faiss_depth_bumped = False
        self._db_depth_bumped = False
        # Store reference paths for depth tracking
        self._db_lock_path = db_lock_path
        self._faiss_lock_path = faiss_lock_path

    def __enter__(self):
        global _ADVISORY_LOCK_DISABLED
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.path, "w")

        if FLOCK_AVAILABLE and fcntl is not None:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
            except OSError as e:
                # Treat ENOTSUP/EOPNOTSUPP as "best-effort" (no advisory
                # locking), but continue as if acquired.
                enotsup = getattr(errno, "ENOTSUP", 95)
                eopnotsupp = getattr(errno, "EOPNOTSUPP", 95)
                if e.errno not in (enotsup, eopnotsupp):
                    raise
                _eprint(
                    f"[lock] WARNING: flock unsupported on {self.path}; "
                    "proceeding best-effort."
                )
                _ADVISORY_LOCK_DISABLED = True

        # Track depth for known lock paths
        if self._db_lock_path and self.path == self._db_lock_path:
            _db_lock_enter()
            self._db_depth_bumped = True
        if self._faiss_lock_path and self.path == self._faiss_lock_path:
            _faiss_lock_enter()
            self._faiss_depth_bumped = True
            
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if FLOCK_AVAILABLE and fcntl is not None:
                try:
                    fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                except OSError as e:
                    enotsup = getattr(errno, "ENOTSUP", 95)
                    eopnotsupp = getattr(errno, "EOPNOTSUPP", 95)
                    if e.errno not in (enotsup, eopnotsupp):
                        raise
        finally:
            try:
                self._fd.close()
            finally:
                if self._faiss_depth_bumped:
                    _faiss_lock_exit()
                    self._faiss_depth_bumped = False
                if self._db_depth_bumped:
                    _db_lock_exit()
                    self._db_depth_bumped = False


# ---------------------------------------------------------------------------
# LockManager - centralized lock management with WorkspacePaths integration
# ---------------------------------------------------------------------------

@dataclass
class LockManager:
    """Centralized lock management for litkit.
    
    Manages file locks and writer guards for safe concurrent access to
    SQLite databases and FAISS indices.
    
    Lock ordering rule: Always acquire db_lock() BEFORE faiss_lock().
    
    Usage:
        paths = WorkspacePaths.from_env_or_default()
        locks = LockManager.from_paths(paths)
        
        with locks.db_lock():
            # Exclusive DB access
            ...
        
        with locks.db_lock(), locks.faiss_lock():
            # Both locks held (correct order)
            ...
    """
    
    db_lock_path: Path
    faiss_lock_path: Path
    ckpt_lock_path: Path
    writer_guard_path: Path
    
    # Internal state
    _guard_cleanup_registered: bool = field(default=False, init=False)
    _guard_created: bool = field(default=False, init=False)
    
    @classmethod
    def from_paths(cls, paths: "WorkspacePaths") -> "LockManager":
        """Construct LockManager from WorkspacePaths."""
        return cls(
            db_lock_path=paths.db_lock,
            faiss_lock_path=paths.faiss_lock,
            ckpt_lock_path=paths.ckpt_lock,
            writer_guard_path=paths.writer_guard,
        )
    
    @contextmanager
    def db_lock(self):
        """Context manager for exclusive DB access."""
        with FileLock(
            self.db_lock_path,
            db_lock_path=self.db_lock_path,
            faiss_lock_path=self.faiss_lock_path,
        ):
            yield
    
    @contextmanager
    def faiss_lock(self):
        """Context manager for exclusive FAISS index access."""
        with FileLock(
            self.faiss_lock_path,
            db_lock_path=self.db_lock_path,
            faiss_lock_path=self.faiss_lock_path,
        ):
            yield
    
    @contextmanager
    def ckpt_lock(self):
        """Context manager for exclusive checkpoint file access."""
        with FileLock(self.ckpt_lock_path):
            yield
    
    @contextmanager
    def db_and_faiss_lock(self):
        """Context manager for both DB and FAISS locks (correct order).
        
        This ensures the lock ordering rule is always followed.
        """
        with self.db_lock(), self.faiss_lock():
            yield
    
    def assert_faiss_locked(self) -> None:
        """Raise AssertionError if FAISS lock is not held.
        
        Use this at the start of functions that must only be called
        while holding the FAISS lock.
        """
        if not in_faiss_lock():
            raise AssertionError(
                "FAISS save called without holding FAISS_LOCK. "
                "If you also update SQLite, acquire DB_LOCK first."
            )
    
    def _cleanup_writer_guard(self) -> None:
        """Remove the writer guard file (cleanup handler)."""
        try:
            if self.writer_guard_path.exists():
                os.remove(self.writer_guard_path)
        except Exception:
            pass
    
    def _maybe_cleanup_own_stale_guard(self) -> None:
        """Clean up our own stale guard from a prior crash."""
        try:
            if self.writer_guard_path.exists():
                content = self.writer_guard_path.read_text()
                parts = (content.split() + ["", "", "0"])[:3]
                pid, host, ts = parts
                if pid.isdigit() and int(pid) == os.getpid():
                    self.writer_guard_path.unlink(missing_ok=True)
        except Exception:
            pass
    
    def create_writer_guard_or_exit(
        self,
        faiss_writer: bool,
        *,
        ttl_sec: int | None = None
    ) -> None:
        """Create exclusive writer guard or exit if another writer is active.
        
        The guard file prevents multiple FAISS writers from corrupting
        indices. Stale guards (older than ttl_sec) are automatically
        cleaned up.
        
        Args:
            faiss_writer: If False, skip guard creation (no-op)
            ttl_sec: Guard TTL in seconds (default: LITKIT_WRITER_GUARD_TTL
                     env var, or 86400 = 24h)
        
        Raises:
            SystemExit: If another active writer holds the guard
        """
        self._maybe_cleanup_own_stale_guard()
        
        if not faiss_writer:
            return
        
        if ttl_sec is None:
            ttl_sec = int(os.environ.get("LITKIT_WRITER_GUARD_TTL", "86400"))
        
        try:
            fd = os.open(
                str(self.writer_guard_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            os.write(
                fd,
                f"{os.getpid()} {socket.gethostname()} {int(time.time())}\n"
                .encode()
            )
            try:
                os.fsync(fd)
            except Exception:
                pass
            os.close(fd)
            
            self._guard_created = True
            
            # Register cleanup
            if not self._guard_cleanup_registered:
                atexit.register(self._cleanup_writer_guard)
                self._guard_cleanup_registered = True
                
                # Register signal handlers (main thread only)
                try:
                    if threading.current_thread() is threading.main_thread():
                        def _signal_cleanup(*_):
                            self._cleanup_writer_guard()
                            os._exit(1)
                        signal.signal(signal.SIGINT, _signal_cleanup)
                        signal.signal(signal.SIGTERM, _signal_cleanup)
                except Exception:
                    pass
                    
        except FileExistsError:
            info = "unknown"
            try:
                info = self.writer_guard_path.read_text().strip()
                parts = info.split()
                ts = 0
                if len(parts) >= 3:
                    try:
                        ts = int(parts[2])
                    except ValueError:
                        ts = 0
                
                if ts and (time.time() - ts) > ttl_sec:
                    _eprint(
                        f"[writer] Guard appears stale (> {ttl_sec}s): {info}. "
                        "Attempting exclusive cleanup."
                    )
                    try:
                        stale = self.writer_guard_path.with_suffix(
                            ".guard.stale." + str(os.getpid())
                        )
                        os.replace(self.writer_guard_path, stale)
                        stale.unlink(missing_ok=False)
                        # Success: retry once non-recursively
                        return self.create_writer_guard_or_exit(
                            faiss_writer, ttl_sec=ttl_sec
                        )
                    except Exception as e:
                        _eprint(
                            f"[writer] ERROR: failed to remove guard: {e}."
                        )
                        sys.exit(2)
            except Exception:
                pass
            
            sys.stderr.write(
                f"[writer] Another FAISS writer appears active "
                f"(guard {self.writer_guard_path} exists: {info}).\n"
                "Stop the other job or remove the stale guard if you are "
                "sure it is dead.\n"
            )
            sys.exit(2)
    
    def cleanup_writer_guard(self) -> None:
        """Explicitly clean up the writer guard (for graceful shutdown)."""
        if self._guard_created:
            self._cleanup_writer_guard()
            self._guard_created = False


# ---------------------------------------------------------------------------
# Module-level defaults for backward compatibility during migration
# ---------------------------------------------------------------------------

_DEFAULT_LOCK_MANAGER: LockManager | None = None


def get_default_lock_manager() -> LockManager:
    """Get or create the default LockManager instance.
    
    Requires that WorkspacePaths has been initialized first.
    This provides backward compatibility during the migration from cli.py
    globals.
    """
    global _DEFAULT_LOCK_MANAGER
    if _DEFAULT_LOCK_MANAGER is None:
        from litkit.config.paths import get_default_paths
        paths = get_default_paths()
        _DEFAULT_LOCK_MANAGER = LockManager.from_paths(paths)
    return _DEFAULT_LOCK_MANAGER


def reset_default_lock_manager() -> None:
    """Reset the cached default lock manager. Useful for testing."""
    global _DEFAULT_LOCK_MANAGER
    _DEFAULT_LOCK_MANAGER = None


# Convenience aliases for direct use (matching cli.py patterns)
def assert_faiss_locked() -> None:
    """Raise AssertionError if FAISS lock is not held."""
    if not in_faiss_lock():
        raise AssertionError(
            "FAISS save called without holding FAISS_LOCK. "
            "If you also update SQLite, acquire DB_LOCK first."
        )
