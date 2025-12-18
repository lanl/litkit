# litkit/config/paths.py
"""
Workspace paths and environment configuration for litkit.

This module centralizes all path discovery and environment setup that was
previously scattered throughout cli.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


def _find_root(root_override: Path | None = None) -> Path:
    """Return repo root.
    
    Preference:
    1) root_override (if provided)
    2) LITKIT_ROOT env var
    3) CWD or its parents containing .git or pyproject.toml
    4) Package path or its parents containing .git or pyproject.toml
    5) CWD if it has src/litkit
    6) site-packages parent (last resort)
    """
    if root_override is not None:
        return Path(root_override).expanduser().resolve()
    
    env = os.getenv("LITKIT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    for p in [cwd] + list(cwd.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    for p in [here] + list(here.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    if (cwd / "src" / "litkit").exists():
        return cwd
    try:
        return here.parents[2]  # config -> litkit -> src -> root
    except IndexError:
        return here


def _resolve_workspace(root: Path, workspace_override: Path | None = None) -> Path:
    """Return path to the workspace directory."""
    if workspace_override is not None:
        return Path(workspace_override).expanduser().resolve()
    ws = os.getenv("LITKIT_WORKSPACE")
    return Path(ws).expanduser().resolve() if ws else (root / "workspace").resolve()


@dataclass
class WorkspacePaths:
    """Centralized path configuration for litkit.
    
    All paths are absolute and resolved. This class replaces the scattered
    global variables that were previously defined at module level in cli.py.
    
    Usage:
        paths = WorkspacePaths.from_env_or_default()
        # or with overrides:
        paths = WorkspacePaths.from_env_or_default(
            root_override=Path("/custom/root"),
            workspace_override=Path("/custom/workspace")
        )
    """
    
    root: Path
    workspace: Path
    
    # Derived paths (computed from workspace)
    hf_home: Path = field(init=False)
    sqlite_dir: Path = field(init=False)
    indices_dir: Path = field(init=False)
    embed_segments_dir: Path = field(init=False)
    
    # Specific file paths
    db_path: Path = field(init=False)
    ckpt_path: Path = field(init=False)
    paper_index_path: Path = field(init=False)
    chunk_index_path: Path = field(init=False)
    chunk_trained_flag: Path = field(init=False)
    
    # Lock file paths
    db_lock: Path = field(init=False)
    faiss_lock: Path = field(init=False)
    ckpt_lock: Path = field(init=False)
    writer_guard: Path = field(init=False)
    
    # Default DB filename (can be overridden via LITKIT_DB_FILE)
    DB_FILENAME: ClassVar[str] = os.environ.get("LITKIT_DB_FILE", "litkit.sqlite3")
    
    def __post_init__(self):
        """Compute derived paths after initialization."""
        # Directory paths
        self.hf_home = self.workspace / "hf_cache"
        self.sqlite_dir = self.workspace / "sqlite"
        self.indices_dir = self.workspace / "indices"
        self.embed_segments_dir = self.workspace / "emb_segments"
        
        # File paths
        self.db_path = self.sqlite_dir / self.DB_FILENAME
        self.ckpt_path = self.sqlite_dir / "build_checkpoint.json"
        self.paper_index_path = self.indices_dir / "papers.faiss"
        self.chunk_index_path = self.indices_dir / "chunks.faiss"
        self.chunk_trained_flag = self.indices_dir / "chunks.trained.json"
        
        # Lock file paths
        self.db_lock = self.sqlite_dir / "db.writer.lock"
        self.faiss_lock = self.sqlite_dir / "faiss.writer.lock"
        self.ckpt_lock = self.sqlite_dir / "ckpt.writer.lock"
        self.writer_guard = self.sqlite_dir / "faiss_writer.guard"
    
    @classmethod
    def from_env_or_default(
        cls,
        root_override: Path | None = None,
        workspace_override: Path | None = None,
    ) -> "WorkspacePaths":
        """Construct WorkspacePaths from environment variables with sensible defaults.
        
        Args:
            root_override: Override for project root (defaults to auto-detection)
            workspace_override: Override for workspace (defaults to LITKIT_WORKSPACE or root/workspace)
        
        Returns:
            WorkspacePaths instance with all paths resolved.
        """
        root = _find_root(root_override)
        workspace = _resolve_workspace(root, workspace_override)
        return cls(root=root, workspace=workspace)
    
    def ensure_directories(self, quiet: bool = False) -> None:
        """Create necessary directories if they don't exist.
        
        Args:
            quiet: If True, suppress error messages
        
        Raises:
            SystemExit: If directories cannot be created
        """
        try:
            self.sqlite_dir.mkdir(parents=True, exist_ok=True)
            self.indices_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if not quiet:
                sys.stderr.write(
                    f"[paths] ERROR: cannot create {self.sqlite_dir} or {self.indices_dir}: {e}\n"
                    "[paths] Set LITKIT_WORKSPACE to a writable path and re-run.\n"
                )
            sys.exit(2)
    
    def setup_environment(self) -> None:
        """Set up environment variables for offline HuggingFace operation.
        
        This sets defaults that can be overridden by explicit env var settings.
        """
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("LITKIT_SEGMENT_FSYNC_DIR", "1")
        # SQLite busy timeout (in ms) - CLI can override via --sqlite-busy-timeout-ms
        # Use 120s default for shared NFS/Lustre filesystems
        os.environ.setdefault("LITKIT_SQLITE_BUSY_TIMEOUT_MS", "120000")
    
    def shard_db_path(self, shard_id: int) -> Path:
        """Return path to shard-specific SQLite database for producer mode."""
        return self.sqlite_dir / f"litkit_shard_{shard_id:02d}.sqlite3"
    
    def list_shard_dbs(self) -> list[Path]:
        """List all shard SQLite databases in the sqlite directory."""
        return sorted(self.sqlite_dir.glob("litkit_shard_*.sqlite3"))
    
    def report(self, tar_dir: Path | None, tar_manifest: Path | None, 
               tar_dir_origin: str | None = None) -> None:
        """Print path configuration for user visibility.
        
        Args:
            tar_dir: Source directory for tar shards (may be None if using manifest)
            tar_manifest: Path to tar manifest file (may be None if using tar_dir)
            tar_dir_origin: Origin label like "(from LITKIT_TAR_DIR)" or "(default)"
        """
        if tar_manifest:
            sys.stderr.write(f"[paths] using manifest -> {tar_manifest} for paths to tar shards\n")
        else:
            src = str(tar_dir) if tar_dir else "(unset)"
            origin = f" {tar_dir_origin}" if tar_dir_origin else ""
            sys.stderr.write(f"[paths] using {src} as source directory for tar shards{origin}\n")
        sys.stderr.write(f"[paths] using {self.workspace} as writable directory for job artifacts/outputs\n")


# Module-level defaults (for backward compatibility during migration)
# These will be deprecated after cli.py is fully migrated
_DEFAULT_PATHS: WorkspacePaths | None = None


def get_default_paths() -> WorkspacePaths:
    """Get or create the default WorkspacePaths instance.
    
    This provides backward compatibility during the migration from cli.py globals.
    Prefer passing WorkspacePaths explicitly where possible.
    """
    global _DEFAULT_PATHS
    if _DEFAULT_PATHS is None:
        _DEFAULT_PATHS = WorkspacePaths.from_env_or_default()
    return _DEFAULT_PATHS


def reset_default_paths() -> None:
    """Reset the cached default paths. Useful for testing."""
    global _DEFAULT_PATHS
    _DEFAULT_PATHS = None
