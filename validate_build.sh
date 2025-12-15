#!/bin/bash
# validate_build.sh - Validate litkit vector store build results
#
# Usage:
#   ./validate_build.sh                     # Uses default workspace path
#   ./validate_build.sh /path/to/workspace  # Use custom workspace path
#
# Run this after a multi-node build completes to verify:
#   1. All segment files were consumed
#   2. FAISS indices exist with non-trivial size
#   3. SQLite database has all papers/chunks indexed
#
# Note: Duplicate detection for chunks
#   When the same paper appears in multiple tar files (e.g., cross-referenced),
#   duplicate chunks with the same (paper_id, ord) may exist in the database.
#   Only unique (paper_id, ord) combinations can be indexed. This script
#   detects duplicates and reports them separately from actual errors.

set -euo pipefail

# Default workspace path (HPC)
WORKSPACE="${1:-/path/to/litkit/workspace}"

echo "========================================"
echo "LitKit Build Validation"
echo "========================================"
echo "Workspace: $WORKSPACE"
echo ""

ERRORS=0

# --- Check 1: Segment directory should be empty ---
echo "=== Check 1: Segment Files ==="
EMB_DIR="${WORKSPACE}/emb_segments"

if [[ ! -d "$EMB_DIR" ]]; then
    echo "  ⚠️  Segment directory not found: $EMB_DIR"
else
    SEGMENT_COUNT=$(find "$EMB_DIR" -name "*.npz" -type f 2>/dev/null | wc -l)
    if [[ "$SEGMENT_COUNT" -eq 0 ]]; then
        echo "  ✅ All segments consumed (directory empty)"
    else
        echo "  ❌ Found $SEGMENT_COUNT unconsumed segment files"
        echo "     These should have been processed by the consumer:"
        find "$EMB_DIR" -name "*.npz" -type f | head -5
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# --- Check 2: FAISS indices exist and have reasonable size ---
echo "=== Check 2: FAISS Indices ==="
INDICES_DIR="${WORKSPACE}/indices"

if [[ ! -d "$INDICES_DIR" ]]; then
    echo "  ❌ Indices directory not found: $INDICES_DIR"
    ERRORS=$((ERRORS + 1))
else
    # Check papers index
    PAPERS_INDEX="${INDICES_DIR}/papers.faiss"
    if [[ -f "$PAPERS_INDEX" ]]; then
        SIZE=$(ls -lh "$PAPERS_INDEX" | awk '{print $5}')
        BYTES=$(stat -f%z "$PAPERS_INDEX" 2>/dev/null || stat -c%s "$PAPERS_INDEX" 2>/dev/null || echo 0)
        if [[ "$BYTES" -gt 1000 ]]; then
            echo "  ✅ papers.faiss exists ($SIZE)"
        else
            echo "  ⚠️  papers.faiss exists but is very small ($SIZE)"
        fi
    else
        echo "  ❌ papers.faiss not found"
        ERRORS=$((ERRORS + 1))
    fi

    # Check chunks index
    CHUNKS_INDEX="${INDICES_DIR}/chunks.faiss"
    if [[ -f "$CHUNKS_INDEX" ]]; then
        SIZE=$(ls -lh "$CHUNKS_INDEX" | awk '{print $5}')
        BYTES=$(stat -f%z "$CHUNKS_INDEX" 2>/dev/null || stat -c%s "$CHUNKS_INDEX" 2>/dev/null || echo 0)
        if [[ "$BYTES" -gt 1000 ]]; then
            echo "  ✅ chunks.faiss exists ($SIZE)"
        else
            echo "  ⚠️  chunks.faiss exists but is very small ($SIZE)"
        fi
    else
        echo "  ❌ chunks.faiss not found"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# --- Check 3: SQLite database consistency ---
echo "=== Check 3: SQLite Database ==="
DB_PATH="${WORKSPACE}/sqlite/litkit.sqlite3"

if [[ ! -f "$DB_PATH" ]]; then
    echo "  ❌ Database not found: $DB_PATH"
    ERRORS=$((ERRORS + 1))
else
    echo "  Database: $DB_PATH"
    
    # Get paper counts
    PAPER_STATS=$(sqlite3 "$DB_PATH" "SELECT COUNT(*), COALESCE(SUM(in_index), 0) FROM papers;" 2>/dev/null || echo "0|0")
    PAPER_TOTAL=$(echo "$PAPER_STATS" | cut -d'|' -f1)
    PAPER_INDEXED=$(echo "$PAPER_STATS" | cut -d'|' -f2)
    
    # Get chunk counts - both total rows AND unique (paper_id, ord) combinations
    CHUNK_TOTAL=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM chunks;" 2>/dev/null || echo "0")
    CHUNK_INDEXED=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM chunks WHERE in_index = 1;" 2>/dev/null || echo "0")
    CHUNK_UNIQUE=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM (SELECT DISTINCT paper_id, ord FROM chunks);" 2>/dev/null || echo "0")
    CHUNK_DUPES=$((CHUNK_TOTAL - CHUNK_UNIQUE))
    
    echo "  Papers: $PAPER_INDEXED / $PAPER_TOTAL indexed"
    
    # Report chunks with duplicate awareness
    if [[ "$CHUNK_DUPES" -gt 0 ]]; then
        echo "  Chunks: $CHUNK_INDEXED / $CHUNK_UNIQUE unique indexed (${CHUNK_DUPES} duplicate rows detected)"
    else
        echo "  Chunks: $CHUNK_INDEXED / $CHUNK_TOTAL indexed"
    fi
    
    # Check if all papers are indexed
    if [[ "$PAPER_TOTAL" -gt 0 && "$PAPER_TOTAL" -eq "$PAPER_INDEXED" ]]; then
        echo "  ✅ All papers indexed"
    elif [[ "$PAPER_TOTAL" -eq 0 ]]; then
        echo "  ⚠️  No papers in database"
    else
        UNINDEXED=$((PAPER_TOTAL - PAPER_INDEXED))
        echo "  ❌ $UNINDEXED papers not indexed"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check if all unique chunks are indexed (compare against unique, not total)
    if [[ "$CHUNK_UNIQUE" -gt 0 && "$CHUNK_UNIQUE" -eq "$CHUNK_INDEXED" ]]; then
        echo "  ✅ All unique chunks indexed"
        if [[ "$CHUNK_DUPES" -gt 0 ]]; then
            echo "  ℹ️  $CHUNK_DUPES duplicate chunk rows exist (same paper_id+ord) - this is expected with overlapping corpus data"
        fi
    elif [[ "$CHUNK_TOTAL" -eq 0 ]]; then
        echo "  ⚠️  No chunks in database"
    else
        # Some chunks are not indexed - check if it's just duplicates or a real problem
        UNINDEXED_UNIQUE=$((CHUNK_UNIQUE - CHUNK_INDEXED))
        if [[ "$UNINDEXED_UNIQUE" -le 0 ]]; then
            # All unique chunks are indexed; the "missing" are just duplicates
            echo "  ✅ All unique chunks indexed"
            echo "  ℹ️  $CHUNK_DUPES duplicate chunk rows exist (same paper_id+ord) - only one per position can be indexed"
        else
            echo "  ❌ $UNINDEXED_UNIQUE unique chunks not indexed"
            if [[ "$CHUNK_DUPES" -gt 0 ]]; then
                echo "  ℹ️  Additionally, $CHUNK_DUPES duplicate rows exist (expected)"
            fi
            ERRORS=$((ERRORS + 1))
        fi
    fi
fi
echo ""

# --- Summary ---
echo "========================================"
if [[ "$ERRORS" -eq 0 ]]; then
    echo "✅ BUILD VALIDATION PASSED"
    echo "========================================"
    exit 0
else
    echo "❌ BUILD VALIDATION FAILED ($ERRORS errors)"
    echo "========================================"
    exit 1
fi
