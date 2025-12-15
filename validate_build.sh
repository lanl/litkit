#!/bin/bash
# validate_build.sh - Validate litkit vector store build results
#
# Usage:
#   ./validate_build.sh                         # Uses default workspace path
#   ./validate_build.sh /path/to/workspace      # Use custom workspace path
#   ./validate_build.sh --quick                 # Quick smoke test with tiny_test.manifest
#   ./validate_build.sh --quick /path/to/ws     # Quick test with custom workspace
#
# Run this after a multi-node build completes to verify:
#   1. All segment files were consumed
#   2. FAISS indices exist with non-trivial size
#   3. SQLite database has all papers/chunks indexed
#   4. (Quick mode) Expected counts match for tiny_test.manifest
#
# Exit codes:
#   0 - All checks passed
#   1 - Validation errors detected
#   2 - Usage error or missing prerequisites
#
# Note: Duplicate detection for chunks
#   When the same paper appears in multiple tar files (e.g., cross-referenced),
#   duplicate chunks with the same (paper_id, ord) may exist in the database.
#   Only unique (paper_id, ord) combinations can be indexed. This script
#   detects duplicates and reports them separately from actual errors.

set -euo pipefail

# --- Parse arguments ---
QUICK_MODE=0
WORKSPACE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick|-q)
            QUICK_MODE=1
            shift
            ;;
        --help|-h)
            grep '^#' "$0" | grep -v '^#!' | sed 's/^# //' | head -25
            exit 0
            ;;
        *)
            WORKSPACE="$1"
            shift
            ;;
    esac
done

# Default workspace path (HPC cluster)
WORKSPACE="${WORKSPACE:-/path/to/litkit/workspace}"

# --- Quick mode: run a fresh build with tiny_test.manifest ---
if [[ "$QUICK_MODE" -eq 1 ]]; then
    echo "========================================"
    echo "LitKit Quick Smoke Test"
    echo "========================================"
    
    # Find script directory and project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$SCRIPT_DIR"
    MANIFEST="${PROJECT_ROOT}/workspace/tiny_test.manifest"
    
    if [[ ! -f "$MANIFEST" ]]; then
        echo "❌ tiny_test.manifest not found at $MANIFEST"
        exit 2
    fi
    
    # Create temp workspace for quick test
    QUICK_WORKSPACE="${WORKSPACE}/quick_test_$(date +%s)"
    echo "Using temp workspace: $QUICK_WORKSPACE"
    mkdir -p "$QUICK_WORKSPACE"
    
    # Run a single-node build with tiny_test.manifest
    echo ""
    echo "=== Running single-node build with tiny_test.manifest ==="
    LITKIT_WORKSPACE="$QUICK_WORKSPACE" \
    LITKIT_ASSUME_YES=1 \
    python -m litkit \
        --tar-manifest "$MANIFEST" \
        --faiss-writer \
        --build-only \
        --quiet \
        2>&1 | tail -20 || {
            echo "❌ Build failed"
            rm -rf "$QUICK_WORKSPACE"
            exit 1
        }
    
    echo ""
    echo "=== Validating quick build results ==="
    WORKSPACE="$QUICK_WORKSPACE"
fi

echo "========================================"
echo "LitKit Build Validation"
echo "========================================"
echo "Workspace: $WORKSPACE"
echo ""

ERRORS=0
WARNINGS=0

# --- Check 1: Segment directory should be empty ---
echo "=== Check 1: Segment Files ==="
EMB_DIR="${WORKSPACE}/emb_segments"

if [[ ! -d "$EMB_DIR" ]]; then
    echo "  ⚠️  Segment directory not found: $EMB_DIR"
    echo "     (This is OK for single-node builds)"
else
    SEGMENT_COUNT=$(find "$EMB_DIR" -name "*.npz" -type f 2>/dev/null | wc -l | tr -d ' ')
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

get_file_size_bytes() {
    local file="$1"
    if [[ -f "$file" ]]; then
        # Try stat -f%z (macOS) then stat -c%s (Linux)
        stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

if [[ ! -d "$INDICES_DIR" ]]; then
    echo "  ❌ Indices directory not found: $INDICES_DIR"
    ERRORS=$((ERRORS + 1))
else
    # Check papers index
    PAPERS_INDEX="${INDICES_DIR}/papers.faiss"
    if [[ -f "$PAPERS_INDEX" ]]; then
        SIZE=$(ls -lh "$PAPERS_INDEX" | awk '{print $5}')
        BYTES=$(get_file_size_bytes "$PAPERS_INDEX")
        if [[ "$BYTES" -gt 1000 ]]; then
            echo "  ✅ papers.faiss exists ($SIZE, $BYTES bytes)"
        else
            echo "  ⚠️  papers.faiss exists but is very small ($SIZE)"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "  ❌ papers.faiss not found"
        ERRORS=$((ERRORS + 1))
    fi

    # Check chunks index
    CHUNKS_INDEX="${INDICES_DIR}/chunks.faiss"
    if [[ -f "$CHUNKS_INDEX" ]]; then
        SIZE=$(ls -lh "$CHUNKS_INDEX" | awk '{print $5}')
        BYTES=$(get_file_size_bytes "$CHUNKS_INDEX")
        if [[ "$BYTES" -gt 1000 ]]; then
            echo "  ✅ chunks.faiss exists ($SIZE, $BYTES bytes)"
        else
            echo "  ⚠️  chunks.faiss exists but is very small ($SIZE)"
            WARNINGS=$((WARNINGS + 1))
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
        WARNINGS=$((WARNINGS + 1))
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
        WARNINGS=$((WARNINGS + 1))
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
    
    # --- Check 4: Minimum expected counts (for regression detection) ---
    echo ""
    echo "=== Check 4: Regression Baseline ==="
    
    # These are minimum expected counts; adjust if tiny_test.manifest changes
    # Current tiny_test.manifest should produce at least 1 paper and some chunks
    MIN_PAPERS=1
    MIN_CHUNKS=1
    
    if [[ "$PAPER_TOTAL" -ge "$MIN_PAPERS" ]]; then
        echo "  ✅ Paper count ($PAPER_TOTAL) meets minimum ($MIN_PAPERS)"
    else
        echo "  ❌ Paper count ($PAPER_TOTAL) below minimum ($MIN_PAPERS) - possible regression"
        ERRORS=$((ERRORS + 1))
    fi
    
    if [[ "$CHUNK_UNIQUE" -ge "$MIN_CHUNKS" ]]; then
        echo "  ✅ Chunk count ($CHUNK_UNIQUE unique) meets minimum ($MIN_CHUNKS)"
    else
        echo "  ❌ Chunk count ($CHUNK_UNIQUE unique) below minimum ($MIN_CHUNKS) - possible regression"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

# --- Check 5: CLI is importable (syntax check) ---
echo "=== Check 5: CLI Import Test ==="
if python -c "from litkit import cli" 2>/dev/null; then
    echo "  ✅ litkit.cli imports successfully"
else
    echo "  ❌ litkit.cli failed to import"
    echo "     This likely indicates a syntax error or missing dependency"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# --- Cleanup for quick mode ---
if [[ "$QUICK_MODE" -eq 1 && -n "${QUICK_WORKSPACE:-}" ]]; then
    echo "=== Cleanup ==="
    if [[ "$ERRORS" -eq 0 ]]; then
        rm -rf "$QUICK_WORKSPACE"
        echo "  Removed temp workspace: $QUICK_WORKSPACE"
    else
        echo "  ⚠️  Keeping temp workspace for debugging: $QUICK_WORKSPACE"
    fi
    echo ""
fi

# --- Summary ---
echo "========================================"
if [[ "$ERRORS" -eq 0 ]]; then
    if [[ "$WARNINGS" -gt 0 ]]; then
        echo "✅ BUILD VALIDATION PASSED ($WARNINGS warnings)"
    else
        echo "✅ BUILD VALIDATION PASSED"
    fi
    echo "========================================"
    exit 0
else
    echo "❌ BUILD VALIDATION FAILED ($ERRORS errors, $WARNINGS warnings)"
    echo "========================================"
    exit 1
fi
