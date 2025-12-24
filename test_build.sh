#!/bin/bash
# test_build.sh - Run a test build on Mac during development
#
# Usage:
#   ./test_build.sh --clean              # Clean workspace and run test build
#   ./test_build.sh --clean --verbose    # Same, with full log output
#
# This script is for Mac development testing after code changes.
# It builds a small vector store using workspace/mac_test.manifest.
#
# Requirements:
#   - macOS with Python environment activated
#   - litkit installed: pip install -e .
#   - Test tar files present (see mac_test.manifest)
#
# Exit codes:
#   0 - Build and validation passed
#   1 - Build or validation failed
#   2 - Usage error

set -euo pipefail

# --- Parse arguments ---
CLEAN_MODE=0
VERBOSE_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean|-c)
            CLEAN_MODE=1
            shift
            ;;
        --verbose|-v)
            VERBOSE_MODE=1
            shift
            ;;
        --help|-h)
            echo "Usage: ./test_build.sh --clean [--verbose]"
            echo ""
            echo "Options:"
            echo "  --clean, -c     Required. Delete existing workspace data before building."
            echo "  --verbose, -v   Show full build log (default: quiet)"
            echo ""
            echo "This script runs a test build using workspace/mac_test.manifest."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './test_build.sh --help' for usage."
            exit 2
            ;;
    esac
done

# --- Setup paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
WORKSPACE="${PROJECT_ROOT}/workspace"
MANIFEST="${WORKSPACE}/mac_test.manifest"

# Directories that constitute "data" in workspace
DATA_DIRS=(
    "${WORKSPACE}/indices"
    "${WORKSPACE}/sqlite"
    "${WORKSPACE}/emb_segments"
)

# --- Check for existing data ---
has_data() {
    for dir in "${DATA_DIRS[@]}"; do
        if [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
            return 0
        fi
    done
    return 1
}

if has_data; then
    if [[ "$CLEAN_MODE" -eq 0 ]]; then
        echo "❌ ERROR: Workspace has existing data."
        echo ""
        echo "Found data in:"
        for dir in "${DATA_DIRS[@]}"; do
            if [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
                echo "  - $dir"
            fi
        done
        echo ""
        echo "Use --clean to delete this data and run a fresh build:"
        echo "  ./test_build.sh --clean"
        echo ""
        echo "This will delete:"
        echo "  - ${WORKSPACE}/indices/"
        echo "  - ${WORKSPACE}/sqlite/"
        echo "  - ${WORKSPACE}/emb_segments/"
        exit 2
    fi
fi

# --- Clean if requested ---
if [[ "$CLEAN_MODE" -eq 1 ]]; then
    echo "Cleaning workspace..."
    for dir in "${DATA_DIRS[@]}"; do
        if [[ -d "$dir" ]]; then
            rm -rf "$dir"
            echo "  Removed: $dir"
        fi
    done
fi

# --- Check prerequisites ---
if [[ ! -f "$MANIFEST" ]]; then
    echo "❌ ERROR: Manifest not found: $MANIFEST"
    exit 2
fi

# Check that tar files in manifest exist
echo "Checking manifest files..."
MISSING=0
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ ! -f "$line" ]]; then
        echo "  ❌ Missing: $line"
        MISSING=$((MISSING + 1))
    fi
done < "$MANIFEST"

if [[ "$MISSING" -gt 0 ]]; then
    echo ""
    echo "❌ ERROR: $MISSING tar file(s) not found."
    echo "Update $MANIFEST with correct paths."
    exit 2
fi

# --- Set environment for Mac ---
# FAISS hangs on macOS with multi-threading + MPS
export FAISS_NUM_THREADS="${FAISS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# litkit reads LITKIT_WORKSPACE for output location
export LITKIT_WORKSPACE="$WORKSPACE"
export LITKIT_ASSUME_YES=1

# --- Run build ---
echo ""
echo "========================================"
echo "LitKit Test Build"
echo "========================================"
echo "Manifest: $(basename "$MANIFEST")"
echo "Workspace: $WORKSPACE"
echo ""

QUIET_FLAG=""
if [[ "$VERBOSE_MODE" -eq 0 ]]; then
    QUIET_FLAG="--quiet"
fi

echo "=== Running build ==="
if [[ "$VERBOSE_MODE" -eq 1 ]]; then
    python -m litkit \
        --tar-manifest "$MANIFEST" \
        --faiss-writer \
        --build-only \
        $QUIET_FLAG || {
            echo "❌ Build failed"
            exit 1
        }
else
    python -m litkit \
        --tar-manifest "$MANIFEST" \
        --faiss-writer \
        --build-only \
        $QUIET_FLAG \
        2>&1 | tail -20 || {
            echo "❌ Build failed"
            exit 1
        }
fi

echo ""
echo "=== Validating build ==="
# Run the validation script
"${SCRIPT_DIR}/validate_build.sh" "$WORKSPACE"
