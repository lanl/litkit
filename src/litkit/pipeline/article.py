# litkit/pipeline/article.py
"""Article processing for document ingestion pipeline."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import faiss
import numpy as np

from litkit.pipeline.helpers import dedupe_ids_and_texts, check_file_processed
from litkit.pipeline.context import ProcessingContext
from litkit.ingest import pack_paragraphs, ArticleMeta
from litkit.db.queries import register_file
from litkit.index.dedup import add_with_ids_dedup
from litkit.index.io import faiss_save
from litkit.concurrent.locking import FileLock

if TYPE_CHECKING:
    import sqlite3


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def process_article(
    ctx: ProcessingContext,
    meta: ArticleMeta,
    file_path: str,
    file_size: int,
    file_mtime: float,
    *,
    doc_id: str = "",
) -> tuple[int, int]:
    """Process a single article: insert to DB and buffer for embedding.
    
    This function handles:
    1. Checking if file was already processed
    2. Inserting paper row into database
    3. Chunking body text and inserting chunk rows
    4. Adding paper/chunks to embedding buffers
    
    Does NOT:
    - Flush buffers (caller decides when)
    - Embed or write to FAISS (done by flush functions)
    
    Args:
        ctx: Processing context with DB, embedders, buffers
        meta: Parsed article metadata
        file_path: Canonical file path for deduplication
        file_size: File size in bytes
        file_mtime: File modification time
        doc_id: Optional document ID for segment storage
    
    Returns:
        Tuple of (papers_added, chunks_added) for this article
    """
    cur = ctx.cursor()
    
    # Check if already processed
    if check_file_processed(cur, file_path):
        return 0, 0
    
    title = (meta.get("title") or "").strip()
    abstract = (meta.get("abstract") or "").strip()
    pmid = meta.get("pmid", "")
    pmcid = meta.get("pmcid", "")
    
    # Skip articles without meaningful content
    if not title and not abstract:
        return 0, 0
    
    # Use doc_id from meta or parameter
    if not doc_id:
        doc_id = pmcid or pmid or file_path
    
    # Register file
    register_file(cur, file_path, file_size, file_mtime)
    
    # Insert paper
    cur.execute(
        """
        INSERT INTO papers (doc_id, pmid, pmcid, title, abstract)
        VALUES (?, ?, ?, ?, ?)
        """,
        (doc_id, pmid, pmcid, title, abstract),
    )
    paper_id = cur.lastrowid
    
    # Prepare paper text for embedding
    paper_text = f"{title} {abstract}".strip()
    
    # Add to paper buffer
    ctx.paper_buffer.add_paper(paper_id, paper_text, doc_id)
    
    # Process chunks from body paragraphs
    paragraphs = meta.get("paragraphs", [])
    chunks_added = 0
    
    if paragraphs:
        chunks = pack_paragraphs(paragraphs)
        
        for ord_, chunk_text in enumerate(chunks):
            cur.execute(
                """
                INSERT INTO chunks (paper_id, ord, body)
                VALUES (?, ?, ?)
                """,
                (paper_id, ord_, chunk_text),
            )
            chunk_id = cur.lastrowid
            
            # Add to chunk buffer
            ctx.chunk_buffer.add_chunk(chunk_id, chunk_text, doc_id, ord_)
            chunks_added += 1
    
    ctx.papers_added += 1
    ctx.chunks_added += chunks_added
    ctx.files_processed += 1
    
    return 1, chunks_added


def flush_paper_buffer(ctx: ProcessingContext) -> int:
    """Flush paper embeddings to FAISS or segment writer.
    
    This function embeds the buffered paper texts and writes them to
    either FAISS indices (single-node mode) or segment files (producer mode).
    
    Args:
        ctx: Processing context with buffers, embedders, and outputs
    
    Returns:
        Number of papers successfully processed
    """
    if ctx.paper_buffer.is_empty():
        return 0
    
    ids, texts, doc_ids = ctx.paper_buffer.pop_all()
    
    # Dedupe by ID
    u_ids, u_texts = dedupe_ids_and_texts(ids, texts)
    
    if not u_ids:
        return 0
    
    # Embed
    Xp = ctx.paper_embedder.encode(
        u_texts,
        progress_label=f"Embedding papers ({len(u_texts)})",
    )
    Xp = np.asarray(Xp, dtype="float32")
    faiss.normalize_L2(Xp)
    
    # Route to appropriate output
    if ctx.paper_seg_writer is not None:
        # Producer mode: write to segment file
        # Align doc_ids with deduped indices
        seen = set()
        u_doc_ids = []
        for i, did in zip(ids, doc_ids):
            if i not in seen:
                seen.add(i)
                u_doc_ids.append(did)
        
        ctx.paper_seg_writer.add_batch(u_doc_ids, Xp)
        
    elif ctx.is_faiss_writer and ctx.paper_index is not None:
        # Single-node or consumer mode: write to FAISS
        with FileLock(ctx.faiss_lock_path):
            added, _ = add_with_ids_dedup(ctx.paper_index, u_ids, Xp)
            if ctx.paper_index_path:
                faiss_save(ctx.paper_index, ctx.paper_index_path)
        
    return len(u_ids)


def flush_chunk_buffer(ctx: ProcessingContext) -> int:
    """Flush chunk embeddings to FAISS or segment writer.
    
    This function embeds the buffered chunk texts and writes them to
    either FAISS indices (single-node mode) or segment files (producer mode).
    
    Args:
        ctx: Processing context with buffers, embedders, and outputs
    
    Returns:
        Number of chunks successfully processed
    """
    if ctx.chunk_buffer.is_empty():
        return 0
    
    ids, texts, doc_ids, ords = ctx.chunk_buffer.pop_all()
    
    # Dedupe by ID
    u_ids, u_texts = dedupe_ids_and_texts(ids, texts)
    
    if not u_ids:
        return 0
    
    # Embed
    Xc = ctx.chunk_embedder.encode(
        u_texts,
        progress_label=f"Embedding chunks ({len(u_texts)})",
    )
    Xc = np.asarray(Xc, dtype="float32")
    faiss.normalize_L2(Xc)
    
    # Route to appropriate output
    if ctx.chunk_seg_writer is not None:
        # Producer mode: write to segment file
        # Align doc_ids and ords with deduped indices
        seen = set()
        u_doc_ids = []
        u_ords = []
        for i, did, o in zip(ids, doc_ids, ords):
            if i not in seen:
                seen.add(i)
                u_doc_ids.append(did)
                u_ords.append(o)
        
        ctx.chunk_seg_writer.add_batch(u_doc_ids, u_ords, Xc)
        
    elif ctx.is_faiss_writer and ctx.chunk_index is not None:
        # Single-node or consumer mode: write to FAISS
        with FileLock(ctx.faiss_lock_path):
            added, _ = add_with_ids_dedup(ctx.chunk_index, u_ids, Xc)
            if ctx.chunk_index_path:
                faiss_save(ctx.chunk_index, ctx.chunk_index_path)
        
    return len(u_ids)
