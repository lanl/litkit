# src/litkit/build/ingest_loop.py
"""Tar processing loop for litkit build pipeline.

This module contains the main tar ingestion loop extracted from cli.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import faiss
import numpy as np

from litkit.progress import eprint as _eprint
from litkit.embeddings.base import (
    progress_newline as _progress_newline,
    progress_write as _progress_write,
)
from litkit.ingest.ingest import (
    ArticleMeta,
    TarMemberMeta,
    count_tar_xml_members,
    iter_tar_xml_streams,
    parallel_iter_tar_articles,
    parse_xml_fileobj,
)
from litkit.ingest import is_uncompressed_tar
from litkit.index import (
    make_id_selector,
    safe_remove_ids,
    add_with_ids_dedup,
    faiss_save,
    faiss_save_force,
)
from litkit.db import (
    already_processed as db_already_processed,
    register_file as db_register_file,
    mark_in_index as db_mark_in_index,
    flush_pending_marks as db_flush_pending_marks,
)
from litkit.segments import (
    load_checkpoint as seg_load_checkpoint,
    save_checkpoint as seg_save_checkpoint,
)
from litkit.build.helpers import (
    pack_paragraphs,
    dedupe_papers_with_doc_ids,
    dedupe_chunks_with_doc_ids,
)

if TYPE_CHECKING:
    import sqlite3
    from litkit.embeddings.base import Embedder
    from litkit.segments import SegmentWriter, ChunkSegmentWriter


# Progress rendering defaults
TAR_RENDER_SEC = float(os.environ.get("LITKIT_TAR_RENDER_SEC", "5.0"))
TAR_RENDER_PCT_STEP = float(
    os.environ.get("LITKIT_TAR_RENDER_PCT_STEP", os.environ.get("LITKIT_TAR_RENDER_PCT_STP", "0"))
)


def iter_tar_articles(
    tar_path: Path,
    parse_workers: int = 8,
) -> Iterator[tuple[TarMemberMeta | SimpleNamespace, ArticleMeta]]:
    """Unified iterator over articles in a tar file.
    
    For uncompressed .tar files (when parse_workers > 1), uses parallel XML parsing.
    For compressed .tar.gz/.tar.bz2 files, uses sequential parsing.
    
    Yields:
        (member_meta, article_meta) tuples where:
        - member_meta has .name, .size, .mtime attributes
        - article_meta is the parsed ArticleMeta dict
    """
    from litkit.progress import is_quiet
    
    use_parallel = parse_workers > 1 and is_uncompressed_tar(tar_path)
    
    if use_parallel:
        if not is_quiet():
            _eprint(f"[scan] using parallel XML parsing ({parse_workers} workers) for {tar_path.name}")
        for member_meta, article_meta in parallel_iter_tar_articles(tar_path, workers=parse_workers):
            yield member_meta, article_meta
    else:
        # Sequential path for compressed tars (or when parallel disabled)
        for tarinfo, fobj in iter_tar_xml_streams(tar_path):
            try:
                article_meta = parse_xml_fileobj(fobj)
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


def process_tar_files(
    *,
    tar_paths: Sequence[Path],
    conn: "sqlite3.Connection",
    # Embedders
    paper_embedder: "Embedder",
    chunk_embedder: "Embedder",
    # Indices (mutable, returned)
    paper_index: faiss.Index,
    chunk_index: faiss.Index,
    # Segment writers (producer mode)
    paper_seg_writer: "SegmentWriter | None",
    chunk_seg_writer: "ChunkSegmentWriter | None",
    # Paths
    ckpt_path: Path,
    ckpt_lock: Path,
    faiss_lock: Path,
    db_lock: Path,
    paper_index_path: Path,
    chunk_index_path: Path,
    sqlite_dir: Path,
    # Lock factory
    FileLock: type,
    # Build flags
    rebuild: bool,
    embed_producer: bool,
    faiss_writer: bool,
    strict_ingest: bool,
    # Batching config
    paper_batch: int,
    chunk_batch: int,
    ckpt_every: int,
    paper_embed_bs: int,
    chunk_embed_bs: int,
    # Chunking config
    chunk_target_chars: int,
    chunk_min_chars: int,
    chunk_overlap: int,
    # Checkpoint shard ID (None for single-node/consumer)
    ckpt_shard_id: int | None,
    # Parse workers
    parse_workers: int,
) -> tuple[faiss.Index, faiss.Index, int, int]:
    """Process tar files and ingest articles into DB and indices.
    
    This is the main tar processing loop extracted from cli.py's build_or_update_indices().
    
    Args:
        tar_paths: Sequence of tar file paths to process
        conn: SQLite connection (already open)
        paper_embedder: Embedder for paper abstracts (SPECTER2)
        chunk_embedder: Embedder for chunks (SBERT)
        paper_index: FAISS paper index (mutable)
        chunk_index: FAISS chunk index (mutable)
        paper_seg_writer: Segment writer for papers (producer mode)
        chunk_seg_writer: Segment writer for chunks (producer mode)
        ckpt_path: Path to checkpoint file
        ckpt_lock: Path to checkpoint lock file
        faiss_lock: Path to FAISS lock file
        db_lock: Path to DB lock file
        paper_index_path: Path to save paper index
        chunk_index_path: Path to save chunk index
        sqlite_dir: Directory containing SQLite DB
        FileLock: Lock class to use (runtime-bound)
        rebuild: Whether this is a rebuild (skip already_processed checks)
        embed_producer: Whether in producer mode (write segments, not FAISS)
        faiss_writer: Whether in writer mode (mutate FAISS directly)
        strict_ingest: Whether to fail fast on ingest errors
        paper_batch: Batch size for paper embeddings
        chunk_batch: Batch size for chunk embeddings
        ckpt_every: Checkpoint frequency (members)
        paper_embed_bs: Batch size for paper embedder.encode()
        chunk_embed_bs: Batch size for chunk embedder.encode()
        chunk_target_chars: Target characters per chunk
        chunk_min_chars: Minimum characters per chunk
        chunk_overlap: Overlap characters between chunks
        ckpt_shard_id: Shard ID for per-shard checkpoints (None for shared)
        parse_workers: Number of parallel XML parsing workers
        
    Returns:
        Tuple of (paper_index, chunk_index, papers_added_total, chunks_added_total)
        Indices may be wrapped in IDMap2 during processing.
    """
    cur = conn.cursor()
    
    # Load checkpoint
    ckpt = seg_load_checkpoint(ckpt_path, shard_id=ckpt_shard_id)
    ckpt_stream = ckpt.get("build_stream", {})
    
    # Buffers for batch processing
    paper_ids_buf: list[int] = []
    paper_texts_buf: list[str] = []
    paper_doc_ids_buf: list[str] = []  # Track doc_ids for content-addressed segments
    chunk_ids_buf: list[int] = []
    chunk_texts_buf: list[str] = []
    chunk_paper_doc_ids_buf: list[str] = []  # Track parent paper doc_ids for chunks
    chunk_ords_buf: list[int] = []  # Track chunk ordinals within papers
    
    papers_added_total = 0
    chunks_added_total = 0
    
    # Per-cycle timing instrumentation
    cycle_scan_start = time.time()  # Reset after each batch flush
    cycle_docs = 0  # Docs processed since last flush
    cycle_chunks = 0  # Chunks generated since last flush
    total_docs_for_avg = 0  # Total docs (for chunks/doc rolling avg)
    total_chunks_for_avg = 0  # Total chunks (for chunks/doc rolling avg)
    
    for tpath in tar_paths:
        # Number of *persisted* members previously processed for this tar shard
        start_persisted = int(ckpt_stream.get(str(tpath), 0))
        processed_count = start_persisted  # increments after each successfully handled member
        persisted_count = start_persisted  # last value safely fsynced via commit + checkpoint
        
        # Set LITKIT_TAR_PRESCAN=0 to skip counting members
        total_members = (
            count_tar_xml_members(tpath)
            if os.environ.get("LITKIT_TAR_PRESCAN", "1") == "1"
            else None
        )
        
        if total_members is not None:
            _eprint(
                f"[scan] shard {tpath} (resume=#{start_persisted}{'' if total_members is None else f', total≈{total_members}'})"
            )
        
        # ---- compact single-line progress for large shards ----
        start_ts = time.time()
        last_render = 0.0
        render_every = TAR_RENDER_SEC
        render_pct_step = TAR_RENDER_PCT_STEP
        next_pct = 0.0  # next threshold to print (0, 1, 2, ... if step=1)
        
        # For instantaneous rate calculation
        last_render_count = start_persisted
        last_render_ts = start_ts
        
        def _render(force: bool = False):
            nonlocal last_render, next_pct, last_render_count, last_render_ts
            now = time.time()
            done = processed_count
            
            # time gate
            if not force and (now - last_render) < render_every:
                # allow percent gate to break the time gate if we crossed a threshold
                if not (render_pct_step > 0 and total_members):
                    return
            
            # percent gate (optional)
            if (not force) and render_pct_step > 0 and total_members:
                cur_pct = 100.0 * done / max(1, total_members)
                if cur_pct + 1e-9 < next_pct and (now - last_render) < render_every:
                    return
                while cur_pct + 1e-9 >= next_pct:
                    next_pct += render_pct_step
            
            elapsed = max(1e-3, now - start_ts)
            avg_rate = done / elapsed
            
            # Instantaneous rate from delta since last render
            delta_time = now - last_render_ts
            delta_count = done - last_render_count
            if delta_time > 0.1 and delta_count >= 0:
                inst_rate = delta_count / delta_time
            else:
                inst_rate = avg_rate  # Fall back if no meaningful delta
            
            total_str = str(total_members) if total_members is not None else "?"
            pct_str = f"  ({100.0*done/total_members:.1f}%)" if total_members else ""
            msg = (
                f"[progress] [scan] {tpath.name}: {done}/{total_str}{pct_str}"
                f"  {inst_rate:.1f}/s now, {avg_rate:.1f}/s avg"
            )
            # Print scan progress as a discrete line to avoid fighting with other
            # in-place tickers (embedder, heartbeats).
            _progress_newline(sys.stderr)
            _progress_write(msg, sys.stderr)
            _progress_newline(sys.stderr)
            
            # Update tracking for next instantaneous calculation
            last_render_count = done
            last_render_ts = now
            last_render = now
        
        # show initial 0/N state (or ? if unknown)
        _render(force=True)
        
        # Skip exactly 'start_persisted' members (they are guaranteed persisted)
        skipped = 0
        
        # Use iter_tar_articles for parallel XML parsing (--parse-workers)
        for m, meta in iter_tar_articles(tpath, parse_workers=parse_workers):
            if skipped < start_persisted:
                skipped += 1
                if skipped == start_persisted:
                    _render(force=True)  # render resume point
                continue
            
            f = f"tar://{tpath}!/{m.name}"
            st = SimpleNamespace(
                st_size=int(getattr(m, "size", 0)),
                st_mtime=float(getattr(m, "mtime", 0.0) or 0.0),
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # FAST PATH - No savepoint overhead for already-processed or unparsable
            # ═══════════════════════════════════════════════════════════════════
            
            # inline heartbeat/progress refresh
            _render()
            
            # Fast path 1: already processed (vast majority in --update mode)
            if not rebuild and db_already_processed(cur, str(f), st):
                processed_count += 1
                continue
            
            # Fast path 2: unparsable XML (meta is None)
            if meta is None:
                processed_count += 1
                continue
            
            # ═══════════════════════════════════════════════════════════════════
            # SLOW PATH - Savepoint-protected writes
            # ═══════════════════════════════════════════════════════════════════
            
            handled_ok = False
            
            # Snapshot buffer lengths BEFORE processing so we can truncate on rollback
            buf_snapshot = (
                len(paper_ids_buf),
                len(paper_texts_buf),
                len(paper_doc_ids_buf),
                len(chunk_ids_buf),
                len(chunk_texts_buf),
                len(chunk_paper_doc_ids_buf),
                len(chunk_ords_buf),
            )
            
            conn.execute("SAVEPOINT member_sp")
            try:
                # ---------- BEGIN INGEST BODY ----------
                pmcid = (meta["pmcid"] or "").strip()
                pmid = (meta["pmid"] or "").strip()
                
                # Content-addressed doc_id: use DB's canonical doc_id when reusing
                # an existing paper row.
                existing_row = None
                if pmcid:
                    existing_row = cur.execute(
                        "SELECT id, doc_id FROM papers WHERE pmcid=?", (pmcid,)
                    ).fetchone()
                if (existing_row is None) and pmid:
                    existing_row = cur.execute(
                        "SELECT id, doc_id FROM papers WHERE pmid=?", (pmid,)
                    ).fetchone()
                
                if existing_row:
                    pid, existing_doc_id = existing_row
                    if existing_doc_id:
                        canon_doc_id = existing_doc_id
                    else:
                        canon_doc_id = str(f)
                        cur.execute(
                            "UPDATE papers SET doc_id = ? WHERE id = ?",
                            (canon_doc_id, pid)
                        )
                else:
                    # New paper: use current path as canonical doc_id
                    canon_doc_id = str(f)
                    cur.execute(
                        "INSERT INTO papers(doc_id, pmid, pmcid, title, abstract) VALUES (?,?,?,?,?)",
                        (canon_doc_id, pmid, pmcid, meta["title"], meta["abstract"]),
                    )
                    pid = cur.lastrowid
                
                seen_this_path = (
                    cur.execute("SELECT 1 FROM files WHERE path=?", (str(f),)).fetchone()
                    is not None
                )
                if seen_this_path:
                    if faiss_writer:
                        with FileLock(faiss_lock):
                            if not isinstance(paper_index, faiss.IndexIDMap2):
                                paper_index = faiss.IndexIDMap2(paper_index)
                            selp = make_id_selector([pid])
                            safe_remove_ids(paper_index, selp)
                            faiss_save_force(paper_index, paper_index_path)
                        
                        old_ids = [
                            row[0]
                            for row in cur.execute(
                                "SELECT id FROM chunks WHERE paper_id=?", (pid,)
                            )
                        ]
                        if old_ids:
                            with FileLock(faiss_lock):
                                if not isinstance(chunk_index, faiss.IndexIDMap2):
                                    chunk_index = faiss.IndexIDMap2(chunk_index)
                                selc = make_id_selector(old_ids)
                                safe_remove_ids(chunk_index, selc)
                                faiss_save_force(chunk_index, chunk_index_path)
                    with FileLock(db_lock):
                        cur.execute("DELETE FROM chunks WHERE paper_id=?", (pid,))
                        cur.execute("UPDATE papers SET in_index=0 WHERE id=?", (pid,))
                
                db_register_file(cur, str(f), pid, st)
                
                ta = (meta["title"] or "").strip()
                ab = (meta["abstract"] or "").strip()
                ta_ab = (ta + " " + ab).strip() or (
                    meta["paragraphs"][0][:800] if meta["paragraphs"] else "untitled"
                )
                paper_ids_buf.append(pid)
                paper_texts_buf.append(ta_ab)
                paper_doc_ids_buf.append(canon_doc_id)
                
                proposed_chunks = []
                if ta_ab:
                    proposed_chunks.append((-1, ta_ab))
                
                paras = meta["paragraphs"] or ([ab] if ab else [])
                chunks = (
                    pack_paragraphs(
                        paras,
                        max_chars=chunk_target_chars,
                        min_chars=chunk_min_chars,
                        overlap_chars=chunk_overlap,
                    )
                    if paras
                    else []
                )
                for ord_i, ch in enumerate(chunks):
                    proposed_chunks.append((ord_i, ch))
                
                for ord_i, text_i in proposed_chunks:
                    cur.execute(
                        "INSERT INTO chunks(paper_id, ord, text) VALUES (?,?,?)",
                        (pid, ord_i, text_i),
                    )
                    cid = cur.lastrowid
                    chunk_ids_buf.append(cid)
                    chunk_texts_buf.append(text_i)
                    chunk_paper_doc_ids_buf.append(canon_doc_id)
                    chunk_ords_buf.append(ord_i)
                
                # ---------- END INGEST BODY ----------
                
                handled_ok = True
                
                # Track docs and chunks for cycle timing
                cycle_docs += 1
                
                # Release savepoint on success
                conn.execute("RELEASE SAVEPOINT member_sp")
            
            except Exception as e:
                _eprint(f"[ingest] ERROR tar://{tpath}!/{m.name}: {e.__class__.__name__}: {e}")
                # Rollback ONLY this member's changes via savepoint
                try:
                    conn.execute("ROLLBACK TO SAVEPOINT member_sp")
                    conn.execute("RELEASE SAVEPOINT member_sp")
                except Exception:
                    pass
                
                # Truncate in-memory buffers back to pre-member state
                (p_len, pt_len, pd_len, c_len, ct_len, cp_len, co_len) = buf_snapshot
                del paper_ids_buf[p_len:]
                del paper_texts_buf[pt_len:]
                del paper_doc_ids_buf[pd_len:]
                del chunk_ids_buf[c_len:]
                del chunk_texts_buf[ct_len:]
                del chunk_paper_doc_ids_buf[cp_len:]
                del chunk_ords_buf[co_len:]
                
                if strict_ingest:
                    raise
                
                # Mark-and-skip: log failure and continue
                try:
                    failure_log = sqlite_dir / "ingest_failures.jsonl"
                    failure_entry = {
                        "tar": str(tpath),
                        "member": m.name,
                        "error": f"{e.__class__.__name__}: {e}",
                        "ts": int(time.time()),
                    }
                    with open(failure_log, "a", encoding="utf-8") as ff:
                        ff.write(json.dumps(failure_entry) + "\n")
                except Exception as log_err:
                    _eprint(f"[ingest] WARNING: could not log failure: {log_err}")
                
                _eprint(f"[ingest] SKIP: {m.name} (use --strict-ingest to fail fast)")
                handled_ok = True
            
            # ═══════════════════════════════════════════════════════════════════
            # BATCH FLUSH - OUTSIDE SAVEPOINT REGION
            # ═══════════════════════════════════════════════════════════════════
            
            if handled_ok:
                processed_count += 1
            
            if handled_ok and len(paper_ids_buf) >= paper_batch:
                u_ids, u_texts, u_doc_ids = dedupe_papers_with_doc_ids(
                    paper_ids_buf, paper_texts_buf, paper_doc_ids_buf
                )
                
                if paper_seg_writer is not None:
                    # Producer mode: embed and write segments
                    try:
                        Xp = paper_embedder.encode(
                            u_texts,
                            progress_label=f"Embedding papers (producer, {len(u_texts)})",
                            batch_size=paper_embed_bs,
                            progress_done_summary=False,
                        )
                        paper_seg_writer.write(doc_ids=u_doc_ids, vecs=Xp)
                        conn.commit()
                        papers_added_total += len(u_ids)
                        
                        ckpt_stream[str(tpath)] = processed_count
                        ckpt["build_stream"] = ckpt_stream
                        seg_save_checkpoint(ckpt, ckpt_path, ckpt_lock, shard_id=ckpt_shard_id)
                        persisted_count = processed_count
                    except Exception as e:
                        conn.rollback()
                        _eprint(f"[flush] FATAL: paper segment flush failed: {e.__class__.__name__}: {e}")
                        raise
                
                elif faiss_writer:
                    # Writer mode: embed and update FAISS
                    try:
                        Xp = paper_embedder.encode(
                            u_texts,
                            progress_label=f"Embedding papers (batch of {len(u_texts)})",
                            batch_size=paper_embed_bs,
                            progress_done_summary=False,
                        )
                        prior_ntotal = int(getattr(paper_index, "ntotal", 0) or 0)
                        with FileLock(faiss_lock):
                            # skip_dedup=rebuild: in rebuild mode, index starts empty, no dedup needed
                            added, ids_added = add_with_ids_dedup(paper_index, u_ids, Xp, skip_dedup=rebuild)
                            saved = False
                            if added:
                                if prior_ntotal == 0:
                                    saved = faiss_save_force(paper_index, paper_index_path)
                                else:
                                    saved = faiss_save(paper_index, paper_index_path)
                        if added:
                            if saved:
                                with FileLock(db_lock):
                                    db_mark_in_index(cur, "papers", [int(i) for i in ids_added])
                                    db_flush_pending_marks(cur)
                                    conn.commit()
                        papers_added_total += int(added)
                    except Exception as e:
                        conn.rollback()
                        _eprint(f"[flush] FATAL: paper FAISS flush failed: {e.__class__.__name__}: {e}")
                        raise
                
                else:
                    conn.commit()
                    papers_added_total += len(u_ids)
                
                paper_ids_buf.clear()
                paper_texts_buf.clear()
                paper_doc_ids_buf.clear()
            
            if handled_ok and len(chunk_ids_buf) >= chunk_batch:
                # --- CYCLE TIMING: capture scan phase duration ---
                t_scan = time.time() - cycle_scan_start
                
                u_ids, u_texts, u_paper_doc_ids, u_ords = dedupe_chunks_with_doc_ids(
                    chunk_ids_buf, chunk_texts_buf, chunk_paper_doc_ids_buf, chunk_ords_buf
                )
                
                if chunk_seg_writer is not None:
                    # Producer mode: embed and write segments
                    try:
                        t_embed_start = time.time()
                        Xc = chunk_embedder.encode(
                            u_texts,
                            progress_label=f"Embedding chunks (producer, {len(u_texts)})",
                            batch_size=chunk_embed_bs,
                            progress_done_summary=False,
                        )
                        t_embed = time.time() - t_embed_start
                        
                        t_seg_start = time.time()
                        chunk_seg_writer.write(
                            paper_doc_ids=u_paper_doc_ids, ords=u_ords, vecs=Xc
                        )
                        t_seg = time.time() - t_seg_start
                        
                        t_db_start = time.time()
                        conn.commit()
                        t_db = time.time() - t_db_start
                        
                        chunks_added_total += len(u_ids)
                        
                        ckpt_stream[str(tpath)] = processed_count
                        ckpt["build_stream"] = ckpt_stream
                        seg_save_checkpoint(ckpt, ckpt_path, ckpt_lock, shard_id=ckpt_shard_id)
                        persisted_count = processed_count
                        
                        # --- CYCLE TIMING: emit summary ---
                        total_docs_for_avg += cycle_docs
                        total_chunks_for_avg += len(u_ids)
                        if total_docs_for_avg > 0:
                            avg_cpd = total_chunks_for_avg / total_docs_for_avg
                        else:
                            avg_cpd = 0.0
                        _eprint(
                            f"[cycle] docs={cycle_docs} chunks={len(u_ids)} | "
                            f"scan={t_scan:.1f}s embed={t_embed:.1f}s "
                            f"seg={t_seg:.2f}s db={t_db:.2f}s | "
                            f"chunks/doc={avg_cpd:.1f}"
                        )
                        # Reset cycle tracking
                        cycle_scan_start = time.time()
                        cycle_docs = 0
                        cycle_chunks = 0
                    except Exception as e:
                        conn.rollback()
                        _eprint(f"[flush] FATAL: chunk segment flush failed: {e.__class__.__name__}: {e}")
                        raise
                
                elif faiss_writer:
                    # Writer mode: embed and update FAISS
                    try:
                        Xc = chunk_embedder.encode(
                            u_texts,
                            progress_label=f"Embedding chunks (batch of {len(u_texts)})",
                            batch_size=chunk_embed_bs,
                            progress_done_summary=False,
                        )
                        prior_ntotal = int(getattr(chunk_index, "ntotal", 0) or 0)
                        with FileLock(faiss_lock):
                            # skip_dedup=rebuild: in rebuild mode, index starts empty, no dedup needed
                            added, ids_added = add_with_ids_dedup(chunk_index, u_ids, Xc, skip_dedup=rebuild)
                            saved = False
                            if added:
                                if prior_ntotal == 0:
                                    saved = faiss_save_force(chunk_index, chunk_index_path)
                                else:
                                    saved = faiss_save(chunk_index, chunk_index_path)
                        if added:
                            if saved:
                                with FileLock(db_lock):
                                    db_mark_in_index(cur, "chunks", [int(i) for i in ids_added])
                                    db_flush_pending_marks(cur)
                                    conn.commit()
                        chunks_added_total += int(added)
                    except Exception as e:
                        conn.rollback()
                        _eprint(f"[flush] FATAL: chunk FAISS flush failed: {e.__class__.__name__}: {e}")
                        raise
                
                else:
                    conn.commit()
                    chunks_added_total += len(u_ids)
                
                chunk_ids_buf.clear()
                chunk_texts_buf.clear()
                chunk_paper_doc_ids_buf.clear()
                chunk_ords_buf.clear()
            
            # Update progress & checkpoint only after a successful handle
            if handled_ok:
                _render()
                
                # Persist every ckpt_every handled members
                if (processed_count - persisted_count) >= ckpt_every:
                    if embed_producer:
                        pass  # Producer checkpoints only after segment flush
                    else:
                        conn.commit()
                        ckpt_stream[str(tpath)] = processed_count
                        ckpt["build_stream"] = ckpt_stream
                        seg_save_checkpoint(ckpt, ckpt_path, ckpt_lock, shard_id=ckpt_shard_id)
                        persisted_count = processed_count
                        _render(force=True)
        
        # End of this tar: force flush remainder buffers + checkpoint
        _render(force=True)
        _progress_newline(sys.stderr)
        
        # ═══════════════════════════════════════════════════════════════════
        # PRODUCER MODE: Two-phase durability for tar-boundary flush
        # ═══════════════════════════════════════════════════════════════════
        if embed_producer:
            try:
                # Phase 1: Write all segments to .pending files
                if paper_ids_buf:
                    u_ids, u_texts, u_doc_ids = dedupe_papers_with_doc_ids(
                        paper_ids_buf, paper_texts_buf, paper_doc_ids_buf
                    )
                    Xp = paper_embedder.encode(
                        u_texts,
                        progress_label=f"Embedding papers (producer tar-boundary, {len(u_texts)})",
                        batch_size=paper_embed_bs,
                        progress_done_summary=False,
                    )
                    paper_seg_writer.write(doc_ids=u_doc_ids, vecs=Xp)
                    papers_added_total += len(u_ids)
                    paper_ids_buf.clear()
                    paper_texts_buf.clear()
                    paper_doc_ids_buf.clear()
                
                if chunk_ids_buf:
                    u_ids, u_texts, u_paper_doc_ids, u_ords = dedupe_chunks_with_doc_ids(
                        chunk_ids_buf, chunk_texts_buf, chunk_paper_doc_ids_buf, chunk_ords_buf
                    )
                    Xc = chunk_embedder.encode(
                        u_texts,
                        progress_label=f"Embedding chunks (producer tar-boundary, {len(u_texts)})",
                        batch_size=chunk_embed_bs,
                        progress_done_summary=False,
                    )
                    chunk_seg_writer.write(
                        paper_doc_ids=u_paper_doc_ids, ords=u_ords, vecs=Xc
                    )
                    chunks_added_total += len(u_ids)
                    chunk_ids_buf.clear()
                    chunk_texts_buf.clear()
                    chunk_paper_doc_ids_buf.clear()
                    chunk_ords_buf.clear()
                
                # Phase 2: Commit DB (rows now durable)
                conn.commit()
                
                # Phase 3: Finalize segments (atomic rename .pending → .npz)
                paper_seg_writer.finalize()
                chunk_seg_writer.finalize()
                
                persisted_count = processed_count
            
            except Exception as e:
                # Compensating transaction: clean up any .pending files
                paper_seg_writer.cleanup_pending()
                chunk_seg_writer.cleanup_pending()
                conn.rollback()
                _eprint(f"[flush] FATAL: producer tar-boundary flush failed: {e.__class__.__name__}: {e}")
                raise
        else:
            conn.commit()
        
        # Checkpoint: now safe for both modes
        if embed_producer:
            ckpt_stream[str(tpath)] = persisted_count
        else:
            ckpt_stream[str(tpath)] = processed_count
        ckpt["build_stream"] = ckpt_stream
        seg_save_checkpoint(ckpt, ckpt_path, ckpt_lock, shard_id=ckpt_shard_id)
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINAL BUFFER FLUSH - After all tars processed
    # ═══════════════════════════════════════════════════════════════════════
    # Writer mode may have remaining items in buffers; producer mode
    # should be empty after tar-boundary flush but handle for safety.
    
    if faiss_writer:
        if paper_ids_buf:
            u_ids, u_texts, _ = dedupe_papers_with_doc_ids(
                paper_ids_buf, paper_texts_buf, paper_doc_ids_buf
            )
            Xp = paper_embedder.encode(
                u_texts,
                progress_label=f"Embedding papers (batch of {len(u_texts)})",
                batch_size=paper_embed_bs,
                progress_done_summary=False,
            )
            if not isinstance(paper_index, faiss.IndexIDMap2):
                paper_index = faiss.IndexIDMap2(paper_index)
            
            with FileLock(db_lock), FileLock(faiss_lock):
                prior_ntotal = int(getattr(paper_index, "ntotal", 0) or 0)
                # skip_dedup=rebuild: in rebuild mode, index starts empty, no dedup needed
                added, ids_added = add_with_ids_dedup(paper_index, u_ids, Xp, skip_dedup=rebuild)
                if added:
                    if prior_ntotal == 0:
                        faiss_save_force(paper_index, paper_index_path)
                        db_mark_in_index(cur, "papers", [int(i) for i in ids_added])
                    else:
                        if faiss_save(paper_index, paper_index_path):
                            db_mark_in_index(cur, "papers", [int(i) for i in ids_added])
                            db_flush_pending_marks(cur)
                conn.commit()
            papers_added_total += int(added)
        
        paper_ids_buf.clear()
        paper_texts_buf.clear()
        
        if chunk_ids_buf:
            u_ids, u_texts, _, _ = dedupe_chunks_with_doc_ids(
                chunk_ids_buf, chunk_texts_buf, chunk_paper_doc_ids_buf, chunk_ords_buf
            )
            Xc = chunk_embedder.encode(
                u_texts,
                progress_label=f"Embedding chunks (batch of {len(u_texts)})",
                batch_size=chunk_embed_bs,
                progress_done_summary=False,
            )
            if not isinstance(chunk_index, faiss.IndexIDMap2):
                chunk_index = faiss.IndexIDMap2(chunk_index)
            
            with FileLock(db_lock), FileLock(faiss_lock):
                prior_ntotal = int(getattr(chunk_index, "ntotal", 0) or 0)
                # skip_dedup=rebuild: in rebuild mode, index starts empty, no dedup needed
                added, ids_added = add_with_ids_dedup(chunk_index, u_ids, Xc, skip_dedup=rebuild)
                if added:
                    if prior_ntotal == 0:
                        faiss_save_force(chunk_index, chunk_index_path)
                        db_mark_in_index(cur, "chunks", [int(i) for i in ids_added])
                    else:
                        if faiss_save(chunk_index, chunk_index_path):
                            db_mark_in_index(cur, "chunks", [int(i) for i in ids_added])
                            db_flush_pending_marks(cur)
                conn.commit()
            chunks_added_total += int(added)
        
        chunk_ids_buf.clear()
        chunk_texts_buf.clear()
    
    elif embed_producer:
        # Producer: final embed and write segment files (should be empty after tar-boundary)
        if paper_ids_buf:
            u_ids, u_texts, u_doc_ids = dedupe_papers_with_doc_ids(
                paper_ids_buf, paper_texts_buf, paper_doc_ids_buf
            )
            Xp = paper_embedder.encode(
                u_texts,
                progress_label=f"Embedding papers (producer, {len(u_texts)})",
                batch_size=paper_embed_bs,
            )
            paper_seg_writer.write(doc_ids=u_doc_ids, vecs=Xp)
            conn.commit()
            papers_added_total += len(u_ids)
        paper_ids_buf.clear()
        paper_texts_buf.clear()
        paper_doc_ids_buf.clear()
        
        if chunk_ids_buf:
            u_ids, u_texts, u_paper_doc_ids, u_ords = dedupe_chunks_with_doc_ids(
                chunk_ids_buf, chunk_texts_buf, chunk_paper_doc_ids_buf, chunk_ords_buf
            )
            Xc = chunk_embedder.encode(
                u_texts,
                progress_label=f"Embedding chunks (producer, {len(u_texts)})",
                batch_size=chunk_embed_bs,
            )
            chunk_seg_writer.write(paper_doc_ids=u_paper_doc_ids, ords=u_ords, vecs=Xc)
            conn.commit()
            chunks_added_total += len(u_ids)
        chunk_ids_buf.clear()
        chunk_texts_buf.clear()
        chunk_paper_doc_ids_buf.clear()
        chunk_ords_buf.clear()
    
    else:
        # Plain reader: DB only
        conn.commit()
        papers_added_total += len(paper_ids_buf)
        paper_ids_buf.clear()
        paper_texts_buf.clear()
        chunks_added_total += len(chunk_ids_buf)
        chunk_ids_buf.clear()
        chunk_texts_buf.clear()
    
    # Final commit
    conn.commit()
    
    # Return indices and counts (indices may have been wrapped)
    return paper_index, chunk_index, papers_added_total, chunks_added_total


def flush_final_buffers(
    *,
    conn: "sqlite3.Connection",
    paper_embedder: "Embedder",
    chunk_embedder: "Embedder",
    paper_index: faiss.Index,
    chunk_index: faiss.Index,
    paper_seg_writer: "SegmentWriter | None",
    chunk_seg_writer: "ChunkSegmentWriter | None",
    paper_ids_buf: list[int],
    paper_texts_buf: list[str],
    paper_doc_ids_buf: list[str],
    chunk_ids_buf: list[int],
    chunk_texts_buf: list[str],
    chunk_paper_doc_ids_buf: list[str],
    chunk_ords_buf: list[int],
    faiss_lock: Path,
    db_lock: Path,
    paper_index_path: Path,
    chunk_index_path: Path,
    FileLock: type,
    faiss_writer: bool,
    embed_producer: bool,
    paper_embed_bs: int,
    chunk_embed_bs: int,
) -> tuple[faiss.Index, faiss.Index, int, int]:
    """Flush any remaining buffers at the end of processing.
    
    This handles the final flush after all tars are processed.
    
    Returns:
        Tuple of (paper_index, chunk_index, papers_added, chunks_added)
    """
    cur = conn.cursor()
    papers_added = 0
    chunks_added = 0
    
    if faiss_writer:
        if paper_ids_buf:
            u_ids, u_texts, _ = dedupe_papers_with_doc_ids(
                paper_ids_buf, paper_texts_buf, paper_doc_ids_buf
            )
            Xp = paper_embedder.encode(
                u_texts,
                progress_label=f"Embedding papers (batch of {len(u_texts)})",
                batch_size=paper_embed_bs,
                progress_done_summary=False,
            )
            if not isinstance(paper_index, faiss.IndexIDMap2):
                paper_index = faiss.IndexIDMap2(paper_index)
            
            with FileLock(db_lock), FileLock(faiss_lock):
                prior_ntotal = int(getattr(paper_index, "ntotal", 0) or 0)
                # add_with_ids_dedup handles removal internally - no explicit safe_remove_ids needed
                added, ids_added = add_with_ids_dedup(paper_index, u_ids, Xp)
                if added:
                    if prior_ntotal == 0:
                        faiss_save_force(paper_index, paper_index_path)
                        db_mark_in_index(cur, "papers", [int(i) for i in ids_added])
                    else:
                        if faiss_save(paper_index, paper_index_path):
                            db_mark_in_index(cur, "papers", [int(i) for i in ids_added])
                            db_flush_pending_marks(cur)
                conn.commit()
            papers_added += int(added)
        
        paper_ids_buf.clear()
        paper_texts_buf.clear()
        
        if chunk_ids_buf:
            u_ids, u_texts, _, _ = dedupe_chunks_with_doc_ids(
                chunk_ids_buf, chunk_texts_buf, chunk_paper_doc_ids_buf, chunk_ords_buf
            )
            Xc = chunk_embedder.encode(
                u_texts,
                progress_label=f"Embedding chunks (batch of {len(u_texts)})",
                batch_size=chunk_embed_bs,
                progress_done_summary=False,
            )
            if not isinstance(chunk_index, faiss.IndexIDMap2):
                chunk_index = faiss.IndexIDMap2(chunk_index)
            
            with FileLock(db_lock), FileLock(faiss_lock):
                prior_ntotal = int(getattr(chunk_index, "ntotal", 0) or 0)
                # add_with_ids_dedup handles removal internally - no explicit safe_remove_ids needed
                added, ids_added = add_with_ids_dedup(chunk_index, u_ids, Xc)
                if added:
                    if prior_ntotal == 0:
                        faiss_save_force(chunk_index, chunk_index_path)
                        db_mark_in_index(cur, "chunks", [int(i) for i in ids_added])
                    else:
                        if faiss_save(chunk_index, chunk_index_path):
                            db_mark_in_index(cur, "chunks", [int(i) for i in ids_added])
                            db_flush_pending_marks(cur)
                conn.commit()
            chunks_added += int(added)
        
        chunk_ids_buf.clear()
        chunk_texts_buf.clear()
    
    elif embed_producer:
        # Producer: embed and write segment files
        if paper_ids_buf:
            u_ids, u_texts, u_doc_ids = dedupe_papers_with_doc_ids(
                paper_ids_buf, paper_texts_buf, paper_doc_ids_buf
            )
            Xp = paper_embedder.encode(
                u_texts,
                progress_label=f"Embedding papers (producer, {len(u_texts)})",
                batch_size=paper_embed_bs,
            )
            paper_seg_writer.write(doc_ids=u_doc_ids, vecs=Xp)
            conn.commit()
            papers_added += len(u_ids)
        paper_ids_buf.clear()
        paper_texts_buf.clear()
        paper_doc_ids_buf.clear()
        
        if chunk_ids_buf:
            u_ids, u_texts, u_paper_doc_ids, u_ords = dedupe_chunks_with_doc_ids(
                chunk_ids_buf, chunk_texts_buf, chunk_paper_doc_ids_buf, chunk_ords_buf
            )
            Xc = chunk_embedder.encode(
                u_texts,
                progress_label=f"Embedding chunks (producer, {len(u_texts)})",
                batch_size=chunk_embed_bs,
            )
            chunk_seg_writer.write(paper_doc_ids=u_paper_doc_ids, ords=u_ords, vecs=Xc)
            conn.commit()
            chunks_added += len(u_ids)
        chunk_ids_buf.clear()
        chunk_texts_buf.clear()
        chunk_paper_doc_ids_buf.clear()
        chunk_ords_buf.clear()
    
    else:
        # Plain reader: DB only
        conn.commit()
        papers_added += len(paper_ids_buf)
        paper_ids_buf.clear()
        paper_texts_buf.clear()
        chunks_added += len(chunk_ids_buf)
        chunk_ids_buf.clear()
        chunk_texts_buf.clear()
    
    conn.commit()
    return paper_index, chunk_index, papers_added, chunks_added
