"""Build configuration dataclass for litkit.build.orchestrator.

This module defines the BuildConfig dataclass that encapsulates all build parameters,
eliminating the need to pass argparse Namespace objects through the build pipeline.

Design principles:
- All paths are explicit (no globals accessed)
- Sensible defaults match cli.py module constants
- Immutable after construction (frozen=True optional for debugging)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Future: type hints for embedders, etc.


@dataclass
class BuildConfig:
    """Configuration for build_or_update_indices() and related functions.
    
    All paths must be provided explicitly. The cli.py thin wrapper is responsible
    for calling get_runtime() and populating these from bound globals.
    
    Attributes:
        # --- Mode flags ---
        rebuild: Wipe DB & indices, rebuild from scratch
        update: Append-only update (skip seen files)
        build_only: Build indices but don't run a query
        init_indices_only: Create empty FAISS indices and exit
        consume_only: Consumer mode - ingest segments, don't scan tars
        faiss_writer: This process can mutate and save FAISS indices
        embed_producer: Producer mode - write segments instead of mutating FAISS
        consume_segments: Writer should also consume pending segment files
        
        # --- Shard settings ---
        shard_id: This process's shard id (0-indexed)
        num_shards: Total number of shards
        
        # --- Index parameters ---
        papers_index: Index type for papers ("hnsw" or "flat")
        chunks_index: Index type for chunks ("ivfpq" or "flat")
        hnsw_m: HNSW M parameter (graph connectivity)
        efconstruction: HNSW build-time ef
        efsearch: HNSW search-time ef
        ivf_nlist: IVF number of centroids
        pq_m: PQ subquantizer count
        nprobe: IVF probe count (None = auto)
        ivf_nlist_forced: User explicitly set --ivf-nlist
        
        # --- Chunking parameters ---
        chunk_target_chars: Target characters per chunk
        chunk_min_chars: Minimum characters per chunk
        chunk_overlap: Character overlap between chunks
        
        # --- Embedding parameters ---
        embed_devices: Device string for SBERT ("auto", "cpu", "cuda:0", etc.)
        embed_workers: Number of worker processes for multi-GPU
        force_embed_devices: Allow loose device tokens
        paper_embed_bs: Batch size for SPECTER2 (papers)
        chunk_embed_bs: Batch size for SBERT (chunks)
        parse_workers: Number of parallel XML parsing workers
        
        # --- SQLite parameters ---
        sqlite_journal_mode: Journal mode ("TRUNCATE" or "WAL")
        sqlite_busy_timeout_ms: Busy timeout in milliseconds
        
        # --- Paths (all explicit, no globals) ---
        db_path: Path to main SQLite database
        sqlite_dir: Directory containing SQLite files
        paper_index_path: Path to FAISS paper index
        chunk_index_path: Path to FAISS chunk index
        chunk_trained_flag: Path to chunk trained flag file
        faiss_lock: Path to FAISS lock file
        db_lock: Path to DB lock file
        ckpt_path: Path to checkpoint file
        ckpt_lock: Path to checkpoint lock file
        embed_segments_dir: Directory for embedding segments
        
        # --- Optional paths (may be None) ---
        tar_dir: Directory containing tar shards (optional if using manifest)
        tar_manifest: Path to manifest file listing tar paths
        embed_outdir: Override for embed_segments_dir (producer output)
        
        # --- Misc ---
        quiet: Suppress verbose output
    """
    
    # Mode flags
    rebuild: bool = False
    update: bool = False
    build_only: bool = False
    init_indices_only: bool = False
    consume_only: bool = False
    faiss_writer: bool = False
    embed_producer: bool = False
    consume_segments: bool = False
    
    # Shard settings
    shard_id: int = 0
    num_shards: int = 1
    
    # Index parameters
    papers_index: str = "hnsw"
    chunks_index: str = "ivfpq"
    hnsw_m: int = 32
    efconstruction: int = 200
    efsearch: int = 128
    ivf_nlist: int = 16384
    pq_m: int = 64
    nprobe: int | None = None
    ivf_nlist_forced: bool = False
    
    # Chunking parameters
    chunk_target_chars: int = 1200
    chunk_min_chars: int = 300
    chunk_overlap: int = 200
    
    # Embedding parameters
    embed_devices: str = "auto"
    embed_workers: int = 1
    force_embed_devices: bool = False
    paper_embed_bs: int = 16
    chunk_embed_bs: int = 64
    parse_workers: int = 8
    
    # SQLite parameters
    sqlite_journal_mode: str = "TRUNCATE"
    sqlite_busy_timeout_ms: int = 120000
    
    # Paths (required - must be set explicitly)
    db_path: Path = field(default_factory=lambda: Path("litkit.db"))
    sqlite_dir: Path = field(default_factory=lambda: Path("sqlite"))
    paper_index_path: Path = field(default_factory=lambda: Path("papers.faiss"))
    chunk_index_path: Path = field(default_factory=lambda: Path("chunks.faiss"))
    chunk_trained_flag: Path = field(default_factory=lambda: Path(".chunk_trained"))
    faiss_lock: Path = field(default_factory=lambda: Path(".faiss.lock"))
    db_lock: Path = field(default_factory=lambda: Path(".db.lock"))
    ckpt_path: Path = field(default_factory=lambda: Path("checkpoint.json"))
    ckpt_lock: Path = field(default_factory=lambda: Path(".ckpt.lock"))
    embed_segments_dir: Path = field(default_factory=lambda: Path("emb_segments"))
    
    # Optional paths
    tar_dir: Path | None = None
    tar_manifest: Path | None = None
    embed_outdir: Path | None = None
    
    # Misc
    quiet: bool = False
    
    @property
    def is_multi_node(self) -> bool:
        """True if this is a multi-node build (sharded or producer/consumer)."""
        return self.num_shards > 1 or self.embed_producer or self.consume_only
    
    @property
    def build_mode(self) -> str:
        """Return 'multi' or 'single' based on configuration."""
        return "multi" if self.is_multi_node else "single"
    
    @property
    def effective_embed_outdir(self) -> Path:
        """Return the effective embedding output directory."""
        return self.embed_outdir if self.embed_outdir is not None else self.embed_segments_dir
    
    def needs_corpus(self) -> bool:
        """Return True if this build mode requires corpus (tar) access."""
        # consume_only doesn't need corpus (only ingests pre-computed segments)
        # init_indices_only doesn't need corpus (bootstrap only)
        if self.consume_only or self.init_indices_only:
            return False
        # Other modes that modify indices need corpus
        return self.rebuild or self.update or self.build_only or self.embed_producer
    
    def validate(self) -> None:
        """Validate configuration for consistency.
        
        Raises:
            ValueError: If configuration is invalid or inconsistent.
        """
        # Mutual exclusivity checks
        if self.embed_producer and self.faiss_writer:
            raise ValueError(
                "--embed-producer and --faiss-writer are mutually exclusive. "
                "Producers write segments; writers mutate FAISS."
            )
        if self.consume_only and self.embed_producer:
            raise ValueError(
                "--consume-only and --embed-producer are mutually exclusive. "
                "Consumers ingest segments; producers create them."
            )
        if self.init_indices_only and self.embed_producer:
            raise ValueError(
                "--init-indices-only and --embed-producer are mutually exclusive. "
                "Bootstrap creates empty indices; run producers separately."
            )
        if self.consume_only and not self.faiss_writer:
            raise ValueError("--consume-only requires --faiss-writer.")
        if self.init_indices_only and not self.faiss_writer:
            raise ValueError("--init-indices-only requires --faiss-writer.")
        
        # Corpus checks
        if self.needs_corpus() and self.tar_dir is None and self.tar_manifest is None:
            raise ValueError(
                "No corpus source specified. Provide --tar-dir or --tar-manifest."
            )


def build_config_from_args(args, *, paths: dict[str, Path]) -> BuildConfig:
    """Create a BuildConfig from an argparse Namespace and runtime paths.
    
    Args:
        args: argparse Namespace from cli.py
        paths: Dictionary mapping path names to Path objects (from get_runtime())
               Expected keys: db_path, sqlite_dir, paper_index_path, chunk_index_path,
               chunk_trained_flag, faiss_lock, db_lock, ckpt_path, ckpt_lock, 
               embed_segments_dir
    
    Returns:
        BuildConfig: Populated configuration object
    
    Example:
        >>> get_runtime()  # Ensure globals are bound
        >>> paths = {
        ...     "db_path": DB_PATH,
        ...     "sqlite_dir": SQLITE_DIR,
        ...     "paper_index_path": PAPER_INDEX_PATH,
        ...     # ... etc
        ... }
        >>> cfg = build_config_from_args(args, paths=paths)
    """
    return BuildConfig(
        # Mode flags
        rebuild=getattr(args, "rebuild", False),
        update=getattr(args, "update", False),
        build_only=getattr(args, "build_only", False),
        init_indices_only=getattr(args, "init_indices_only", False),
        consume_only=getattr(args, "consume_only", False),
        faiss_writer=getattr(args, "faiss_writer", False),
        embed_producer=getattr(args, "embed_producer", False),
        consume_segments=getattr(args, "consume_segments", False),
        
        # Shard settings
        shard_id=getattr(args, "shard_id", 0),
        num_shards=getattr(args, "num_shards", 1),
        
        # Index parameters
        papers_index=getattr(args, "papers_index", "hnsw"),
        chunks_index=getattr(args, "chunks_index", "ivfpq"),
        hnsw_m=getattr(args, "hnsw_m", 32),
        efconstruction=getattr(args, "efconstruction", 200),
        efsearch=getattr(args, "efsearch", 128),
        ivf_nlist=getattr(args, "ivf_nlist", 16384),
        pq_m=getattr(args, "pq_m", 64),
        nprobe=getattr(args, "nprobe", None),
        ivf_nlist_forced=getattr(args, "_ivf_nlist_forced", False),
        
        # Chunking parameters
        chunk_target_chars=getattr(args, "chunk_target_chars", 1200),
        chunk_min_chars=getattr(args, "chunk_min_chars", 300),
        chunk_overlap=getattr(args, "chunk_overlap", 200),
        
        # Embedding parameters
        embed_devices=getattr(args, "embed_devices", "auto"),
        embed_workers=getattr(args, "embed_workers", 1),
        force_embed_devices=getattr(args, "force_embed_devices", False),
        paper_embed_bs=getattr(args, "paper_embed_bs", 16),
        chunk_embed_bs=getattr(args, "chunk_embed_bs", 64),
        parse_workers=getattr(args, "parse_workers", 8),
        
        # SQLite parameters
        sqlite_journal_mode=getattr(args, "sqlite_journal_mode", "TRUNCATE"),
        sqlite_busy_timeout_ms=getattr(args, "sqlite_busy_timeout_ms", 120000),
        
        # Paths (from runtime)
        db_path=paths["db_path"],
        sqlite_dir=paths["sqlite_dir"],
        paper_index_path=paths["paper_index_path"],
        chunk_index_path=paths["chunk_index_path"],
        chunk_trained_flag=paths["chunk_trained_flag"],
        faiss_lock=paths["faiss_lock"],
        db_lock=paths["db_lock"],
        ckpt_path=paths["ckpt_path"],
        ckpt_lock=paths["ckpt_lock"],
        embed_segments_dir=paths["embed_segments_dir"],
        
        # Optional paths (from args)
        tar_dir=getattr(args, "tar_dir", None),
        tar_manifest=getattr(args, "tar_manifest", None),
        embed_outdir=getattr(args, "embed_outdir", None),
        
        # Misc
        quiet=getattr(args, "quiet", False),
    )
