# src/litkit/retrieval/stages.py
"""Two-stage RAG retrieval functions.

Stage 1: shortlist_papers - SPECTER2 + HNSW paper shortlisting
Stage 2: search_chunks_constrained - SBERT + IVF-PQ with lexical boost
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np

from litkit.retrieval.helpers import (
    avg_chunks_for_papers,
)
from litkit.retrieval.search import faiss_search
from litkit.retrieval.lexical import (
    LexicalConfig,
    LexicalResult,
    SqliteLexicalBackend,
    find_rare_terms,
    merge_lexical_and_ann,
)
from litkit.progress import eprint as _eprint

if TYPE_CHECKING:
    from litkit.embeddings.base import Embedder


# Optional kill-switch for lexical prefilter (env var)
DISABLE_LEXICAL = os.environ.get("LITKIT_NO_LEXICAL", "0") == "1"


def shortlist_papers(
    question: str,
    k: int,
    *,
    paper_index_path: Path,
    embedder: "Embedder",
    efsearch: int = 128,
) -> list[int]:
    """Stage 1: Shortlist candidate papers using SPECTER2 + HNSW.
    
    Encodes the question with SPECTER2 and retrieves top-k paper IDs
    from the paper index (HNSW by default).
    
    Args:
        question: The user's question
        k: Number of papers to retrieve
        paper_index_path: Path to the paper FAISS index
        embedder: SPECTER2 embedder instance
        efsearch: HNSW efSearch parameter
    
    Returns:
        List of paper IDs (up to k)
    """
    q = embedder.encode([question]).astype("float32", copy=False)
    faiss.normalize_L2(q)
    ids, _, meta = faiss_search(paper_index_path, q, k, efSearch=efsearch)
    try:
        es = meta.get("efSearch")
        if es is not None:
            _eprint(f"[retrieve] papers: HNSW efSearch={es} k={k}")
    except Exception:
        pass
    return ids


def search_chunks_constrained(
    question: str,
    candidate_papers: list[int],
    k: int,
    *,
    chunk_index_path: Path,
    db_path: Path,
    embedder: "Embedder",
    connect_db,
    load_temp_candidates,
    chunk_ids_to_paper_ids,
    overshoot: int = 20,
    nprobe: int | None = None,
    min_chunks_per_paper: float | None = None,
    lexical_cap: int | None = None,
    lexical_limit: int = 200,
    allow_global_lexical: bool | None = None,
    per_paper_cap: int = 0,
) -> tuple[list[int], dict[str, Any]]:
    """Stage 2: SBERT ANN + lexical front-loading.
    
    1) Wide ANN search (K = max(k*overshoot, 100))
    2) Optional filter to candidate_papers
    3) Front-load chunks that lexically match rare query terms
    4) Return top-k ids (lexical-first, de-duped, then ANN order)
    
    Args:
        question: The user's question
        candidate_papers: Paper IDs from Stage 1
        k: Final number of chunks to return
        chunk_index_path: Path to chunk FAISS index
        db_path: Path to SQLite database
        embedder: SBERT embedder instance
        connect_db: Function to connect to DB
        load_temp_candidates: Function to load temp candidates
        chunk_ids_to_paper_ids: Function to map chunk IDs to paper IDs
        overshoot: ANN overshoot multiplier
        nprobe: IVF nprobe parameter (None = auto)
        min_chunks_per_paper: Threshold for Stage-2 fallback
        lexical_cap: Max lexical items to include
        lexical_limit: SQL LIMIT for lexical scan
        allow_global_lexical: Allow lexical without candidate filter
        per_paper_cap: Max chunks per paper (0 = unlimited)
    
    Returns:
        (chunk_ids, meta) where meta has effective search params
    """
    global _LEXICAL_WARN_ONCE
    
    enc = embedder
    q = enc.encode([question]).astype("float32", copy=False)
    faiss.normalize_L2(q)

    K = max(k * overshoot, 100)
    ids, dists, meta = faiss_search(chunk_index_path, q, K, nprobe=nprobe)
    if not ids:
        return [], meta

    ranked = list(zip(ids, dists, strict=False))
    db_conn = connect_db(db_path)

    try:
        cand: set | None = None
        did_fallback = False

        env_thr = float(os.environ.get("LITKIT_MIN_CHUNKS_PER_PAPER", "2.0"))
        thr = float(min_chunks_per_paper) if min_chunks_per_paper is not None else env_thr
        
        if candidate_papers:
            avg_c = avg_chunks_for_papers(
                candidate_papers,
                db_path=db_path,
                connect_db=connect_db,
                load_temp_candidates=load_temp_candidates,
            )
            if avg_c >= thr:
                cand = set(candidate_papers)
            else:
                src = "param" if min_chunks_per_paper is not None else "env"
                _eprint(
                    f"[retrieve] global Stage-2 fallback: "
                    f"avg chunks/paper={avg_c:.2f} (<{thr} via {src}), "
                    f"shortlist_papers={len(candidate_papers)}, K={K}"
                )
                cand = None
            did_fallback = cand is None

        if cand:
            ann_chunk_to_paper = chunk_ids_to_paper_ids(db_conn, ids)
            ranked = [
                (cid, dist) for cid, dist in ranked
                if ann_chunk_to_paper.get(cid) in cand
            ]

        # Lexical front-loading via modular lexical module
        force_global = (
            allow_global_lexical
            if allow_global_lexical is not None
            else os.environ.get("LITKIT_ALLOW_GLOBAL_LEXICAL", "0") == "1"
        ) or did_fallback

        if did_fallback:
            _eprint("[lexical] enabling global lexical front-load (Stage-2 fallback).")

        # Build lexical config
        lexical_config = LexicalConfig(
            enabled=not DISABLE_LEXICAL,
            cap=lexical_cap,
            limit=lexical_limit,
            allow_global=force_global,
        )

        # Find rare terms and execute lexical search
        rare_terms = find_rare_terms(question, lexical_config)
        
        lexical_result = LexicalResult()  # empty by default
        if rare_terms and lexical_config.enabled:
            # Determine scope: global if forced or candidates available
            if force_global or cand is None:
                candidate_scope = None  # global
            else:
                candidate_scope = cand
            
            # Only search if we have scope
            if candidate_scope is not None or force_global:
                backend = SqliteLexicalBackend(db_conn, load_temp_candidates)
                lexical_result = backend.search(
                    rare_terms, candidate_scope, lexical_config
                )

        # Merge lexical + ANN with cap
        LEX_CAP = lexical_cap if lexical_cap is not None else max(5, k // 3)
        out = merge_lexical_and_ann(lexical_result, ranked, k, LEX_CAP)

        # Per-paper cap
        if per_paper_cap:
            def _batched_map(ids_list: list[int]) -> dict[int, int]:
                if not ids_list:
                    return {}
                out_map: dict[int, int] = {}
                B = 800
                cur = db_conn.cursor()
                for s in range(0, len(ids_list), B):
                    batch = ids_list[s:s + B]
                    qmarks = ",".join("?" for _ in batch)
                    rows = cur.execute(
                        f"SELECT id, paper_id FROM chunks WHERE id IN ({qmarks})",
                        batch
                    ).fetchall()
                    out_map.update({row[0]: row[1] for row in rows})
                return out_map

            from litkit.frontload.cap import cap_chunks_per_paper

            selected_chunk_to_paper: dict[int, int] = _batched_map(out) if out else {}
            if out:
                out = cap_chunks_per_paper(
                    out, selected_chunk_to_paper, max_per_paper=per_paper_cap
                )

            def _top_up_from_ann(ann_ranked: list[tuple[int, float]]) -> None:
                nonlocal out
                selected = set(out)
                ann_pool = [cid for (cid, _) in ann_ranked if cid not in selected]
                if not ann_pool or len(out) >= k:
                    return
                pool_map = _batched_map(ann_pool)

                counts = defaultdict(int)
                for cid in out:
                    pid = selected_chunk_to_paper.get(cid)
                    if pid is not None:
                        counts[pid] += 1

                for cid in ann_pool:
                    if len(out) >= k:
                        break
                    pid = pool_map.get(cid)
                    if pid is None:
                        continue
                    if counts[pid] < per_paper_cap:
                        out.append(cid)
                        counts[pid] += 1
                        if cid not in selected_chunk_to_paper and pid is not None:
                            selected_chunk_to_paper[cid] = pid

            if len(out) < k:
                _top_up_from_ann(ranked)

            # Widen search if still short
            widen_rounds = 2
            seen_ranked_ids = {cid for (cid, _) in ranked}
            curK = K
            for _ in range(widen_rounds):
                if len(out) >= k:
                    break
                curK = min(curK * 2, 10000)
                more_ids, more_dists, _ = faiss_search(
                    chunk_index_path, q, curK, nprobe=nprobe
                )
                new_ranked = [
                    (cid, dist)
                    for cid, dist in zip(more_ids, more_dists, strict=False)
                    if cid not in seen_ranked_ids
                ]
                if not new_ranked:
                    break
                ranked.extend(new_ranked)
                seen_ranked_ids.update(cid for cid, _ in new_ranked)
                _top_up_from_ann(new_ranked)

    finally:
        db_conn.close()

    return out, meta


def get_chunks(conn, ids: list[int]) -> list[dict[str, str]]:
    """Retrieve chunk rows joined with paper metadata.
    
    Preserves input `ids` order.
    
    Args:
        conn: SQLite connection
        ids: List of chunk IDs
    
    Returns:
        List of dicts with keys:
        id, paper_id, ord, text, paper_title, pmid, pmcid
    """
    if not ids:
        return []
    marks = ",".join("?" for _ in ids)
    cur = conn.cursor()
    cur.execute(
        f"""SELECT c.id, c.paper_id, c.ord, c.text, p.title, p.pmid, p.pmcid
            FROM chunks c JOIN papers p ON p.id=c.paper_id
            WHERE c.id IN ({marks})""",
        ids,
    )
    rows = cur.fetchall()
    rowmap = {row[0]: row for row in rows}
    out = []
    for cid in ids:
        row = rowmap.get(cid)
        if not row:
            continue
        out.append({
            "id": row[0],
            "paper_id": row[1],
            "ord": row[2],
            "text": row[3],
            "paper_title": row[4] or "",
            "pmid": row[5] or "",
            "pmcid": row[6] or "",
        })
    return out
