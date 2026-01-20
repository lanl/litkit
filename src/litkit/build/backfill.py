# src/litkit/build/backfill.py
"""Backfill and reconciliation functions for FAISS/SQLite consistency.

Functions:
- backfill_unindexed_vectors: Embed and add rows missing from FAISS
- reconcile_sqlite_flags_with_faiss: Sync in_index flags with FAISS state
- post_build_sanity_check: Summary report after build
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import faiss

from litkit.index import (
    add_with_ids_dedup,
    faiss_present_ids,
    faiss_save,
    faiss_save_force,
    make_id_selector,
    report_faiss_index,
    safe_remove_ids,
)
from litkit.db import mark_in_index, flush_pending_marks
from litkit.progress import eprint

if TYPE_CHECKING:
    from litkit.embeddings.base import Embedder


def backfill_unindexed_vectors(
    conn,
    paper_embedder: "Embedder",
    chunk_embedder: "Embedder",
    paper_index,
    chunk_index,
    *,
    paper_index_path: Path,
    chunk_index_path: Path,
    faiss_lock_path: Path,
    db_lock_path: Path,
    FileLock,  # class to construct locks
    batch: int = 20000,
    paper_bs: int | None = None,
    chunk_bs: int | None = None,
) -> None:
    """Embed and add any rows that exist in SQLite but were never added
    to FAISS (in_index=0).
    
    Args:
        conn: SQLite connection
        paper_embedder: Embedder for papers (SPECTER2)
        chunk_embedder: Embedder for chunks (SBERT)
        paper_index: FAISS paper index
        chunk_index: FAISS chunk index
        paper_index_path: Path to paper FAISS index file
        chunk_index_path: Path to chunk FAISS index file
        faiss_lock_path: Path to FAISS lock file
        db_lock_path: Path to DB lock file
        FileLock: Lock class (with context manager protocol)
        batch: Batch size for DB queries
        paper_bs: Batch size for paper embedding
        chunk_bs: Batch size for chunk embedding
    """
    cur = conn.cursor()

    # Papers
    while True:
        rows = cur.execute(
            "SELECT id, (COALESCE(title,'') || ' ' || COALESCE(abstract,'')) "
            "AS txt FROM papers WHERE in_index=0 LIMIT ?",
            (batch,),
        ).fetchall()
        if not rows:
            break
        ids = [r[0] for r in rows]
        texts = [(r[1] or "untitled").strip() for r in rows]
        Xp = paper_embedder.encode(
            texts,
            progress_label=f"Embedding papers (backfill, {len(texts)})",
            batch_size=paper_bs,
            progress_done_summary=False,
        )

        if not isinstance(paper_index, faiss.IndexIDMap2):
            paper_index = faiss.IndexIDMap2(paper_index)

        prior_ntotal = int(getattr(paper_index, "ntotal", 0) or 0)
        with FileLock(faiss_lock_path):
            # add_with_ids_dedup handles removal internally - no explicit safe_remove_ids needed
            added, ids_added = add_with_ids_dedup(paper_index, ids, Xp)
            saved = False
            if added:
                if prior_ntotal == 0:
                    saved = faiss_save_force(paper_index, paper_index_path)
                else:
                    saved = faiss_save(paper_index, paper_index_path)
        if added and saved:
            with FileLock(db_lock_path):
                mark_in_index(cur, "papers", [int(i) for i in ids_added])
                flush_pending_marks(cur)
                conn.commit()
        else:
            # Save failed - vectors are in FAISS but not marked in DB.
            # reconcile_sqlite_flags_with_faiss() will fix this on next run.
            conn.commit()

    # Chunks
    while True:
        rows = cur.execute(
            "SELECT id, text FROM chunks WHERE in_index=0 LIMIT ?", (batch,)
        ).fetchall()
        if not rows:
            break
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        Xc = chunk_embedder.encode(
            texts,
            progress_label=f"Embedding chunks (backfill, {len(texts)})",
            batch_size=chunk_bs,
            progress_done_summary=False,
        )
        if not isinstance(chunk_index, faiss.IndexIDMap2):
            chunk_index = faiss.IndexIDMap2(chunk_index)

        prior_ntotal = int(getattr(chunk_index, "ntotal", 0) or 0)
        with FileLock(faiss_lock_path):
            # add_with_ids_dedup handles removal internally - no explicit safe_remove_ids needed
            added, ids_added = add_with_ids_dedup(chunk_index, ids, Xc)
            saved = False
            if added:
                if prior_ntotal == 0:
                    saved = faiss_save_force(chunk_index, chunk_index_path)
                else:
                    saved = faiss_save(chunk_index, chunk_index_path)
        if added and saved:
            with FileLock(db_lock_path):
                mark_in_index(cur, "chunks", [int(i) for i in ids_added])
                flush_pending_marks(cur)
                conn.commit()
        else:
            # Save failed - vectors are in FAISS but not marked in DB.
            # reconcile_sqlite_flags_with_faiss() will fix this on next run.
            conn.commit()


def reconcile_sqlite_flags_with_faiss(
    conn, paper_index, chunk_index
) -> tuple[int, int]:
    """For each table, if FAISS lacks some IDs that SQLite thinks are in
    the index, reset those rows to in_index=0 so the normal backfill can
    re-add them.
    
    Returns:
        Tuple of (papers_reset, chunks_reset)
    """
    cur = conn.cursor()
    reset_p = reset_c = 0

    # Papers
    ids_present = faiss_present_ids(paper_index)
    if ids_present is not None:
        cur.execute("SELECT id, in_index FROM papers")
        bad = [
            row[0]
            for row in cur.fetchall()
            if row[0] not in ids_present and row[1] == 1
        ]
        good_missing_flag = [
            row[0]
            for row in cur.execute(
                "SELECT id FROM papers WHERE in_index=0"
            ).fetchall()
            if row[0] in ids_present
        ]
        if bad:
            cur.executemany(
                "UPDATE papers SET in_index=0 WHERE id=?", [(i,) for i in bad]
            )
            reset_p = len(bad)
        if good_missing_flag:
            cur.executemany(
                "UPDATE papers SET in_index=1 WHERE id=?",
                [(i,) for i in good_missing_flag],
            )

    # Chunks
    ids_present = faiss_present_ids(chunk_index)
    if ids_present is not None:
        cur.execute("SELECT id, in_index FROM chunks")
        bad = [
            row[0]
            for row in cur.fetchall()
            if row[0] not in ids_present and row[1] == 1
        ]
        good_missing_flag = [
            row[0]
            for row in cur.execute(
                "SELECT id FROM chunks WHERE in_index=0"
            ).fetchall()
            if row[0] in ids_present
        ]
        if bad:
            cur.executemany(
                "UPDATE chunks SET in_index=0 WHERE id=?", [(i,) for i in bad]
            )
            reset_c = len(bad)
        if good_missing_flag:
            cur.executemany(
                "UPDATE chunks SET in_index=1 WHERE id=?",
                [(i,) for i in good_missing_flag],
            )

    conn.commit()
    return reset_p, reset_c


def post_build_sanity_check(
    conn,
    *,
    paper_index_path: Path,
    chunk_index_path: Path,
) -> None:
    """Sanity print after build: DB vs FAISS counts and index types.
    
    Args:
        conn: SQLite connection
        paper_index_path: Path to paper FAISS index file
        chunk_index_path: Path to chunk FAISS index file
    """
    # ----- papers -----
    try:
        p_idx = faiss.read_index(str(paper_index_path))
    except Exception:
        p_idx = None
    report_faiss_index("papers", paper_index_path)

    cur = conn.cursor()
    n_db_p = cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    n_in_p = cur.execute(
        "SELECT COUNT(*) FROM papers WHERE in_index=1"
    ).fetchone()[0]
    n_faiss_p = int(getattr(p_idx, "ntotal", 0) or 0)
    eprint(
        f"[summary] papers: db={n_db_p} in_index={n_in_p} "
        f"faiss_ntotal={n_faiss_p}"
    )

    # ----- chunks -----
    try:
        c_idx = faiss.read_index(str(chunk_index_path))
    except Exception:
        c_idx = None
    report_faiss_index("chunks", chunk_index_path)

    n_db_c = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_in_c = cur.execute(
        "SELECT COUNT(*) FROM chunks WHERE in_index=1"
    ).fetchone()[0]
    n_faiss_c = int(getattr(c_idx, "ntotal", 0) or 0)
    eprint(
        f"[summary] chunks: db={n_db_c} in_index={n_in_c} "
        f"faiss_ntotal={n_faiss_c}"
    )
