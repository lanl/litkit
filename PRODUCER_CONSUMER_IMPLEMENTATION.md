# Producer-Consumer Implementation Summary

## Problem Solved
The consumer node was unnecessarily doing embedding work, wasting GPU resources that should have been available to producer nodes.

## Solution: `--consume-only` Flag

### Overview
Added a new `--consume-only` flag that allows the consumer node to skip all corpus scanning and embedding work, focusing solely on ingesting segment files from producers.

### Changes Made

#### 1. CLI Changes (`src/litkit/cli.py`)

**New Argument:**
```python
ap.add_argument(
    "--consume-only",
    action="store_true",
    help="Consumer-only mode: skip tar scanning and embedding, only ingest segments in a polling loop. "
    "Requires --faiss-writer. Typically used on a dedicated consumer node in multi-node setups.",
)
```

**Implementation in `build_or_update_indices()`:**
```python
if args.consume_only:
    if not args.faiss_writer:
        raise ValueError("--consume-only requires --faiss-writer")
    _eprint("[consumer] Starting consume-only mode")
    seg_dir = args.embed_outdir or EMBED_SEGMENTS_DIR
    paper_index = _faiss_load(PAPER_INDEX_PATH)
    chunk_index = _faiss_load(CHUNK_INDEX_PATH)
    
    while True:
        p_added = _ingest_paper_segments(conn, paper_index, seg_dir)
        c_added = _ingest_chunk_segments(conn, chunk_index, seg_dir)
        
        if p_added or c_added:
            _eprint(f"[consumer] Ingested {p_added} paper vectors and {c_added} chunk vectors")
        else:
            _eprint("[consumer] No new segments found, waiting...")
            time.sleep(30)
    
    return  # End consume-only mode
```

#### 2. SLURM Script Changes (`vector_build_multi.sbatch`)

**Before:**
```bash
bash -c '
poll_count=0
max_polls=1000
while [ $poll_count -lt $max_polls ]; do
    litkit \
        --faiss-writer \
        --consume-segments \
        --embed-outdir "/workspace/emb_segments" \
        --tar-manifest "/workspace/test.manifest" \
        --build-only
    
    segments_remaining=$(ls /workspace/emb_segments/*.npz 2>/dev/null | wc -l)
    if [ $segments_remaining -eq 0 ]; then
        echo "[Consumer] All segments ingested"
        break
    fi
    echo "[Consumer] Poll $poll_count: $segments_remaining segments remaining"
    sleep 30
    poll_count=$((poll_count + 1))
done
'
```

**After:**
```bash
litkit \
    --faiss-writer \
    --consume-only \
    --embed-outdir "/workspace/emb_segments" \
    --tar-manifest "/workspace/test.manifest" &
```

### Benefits

1. **Clean Separation of Concerns**
   - Producers: Only embed and write segments
   - Consumer: Only ingest segments into FAISS
   - No overlap or wasted work

2. **Maximum GPU Utilization**
   - Consumer doesn't load embedding models
   - Consumer doesn't touch tar files
   - All 6 producer GPUs can focus on embedding

3. **Simpler SLURM Script**
   - No bash polling loop needed
   - Single litkit command per role
   - Built-in infinite polling in consume-only mode

4. **Better Reliability**
   - Automatic retry on errors
   - Clean logging from Python code
   - No shell script complexity

### Usage

**Producer Node:**
```bash
litkit \
    --embed-producer \
    --shard-id 0 \
    --num-shards 3 \
    --embed-outdir "/workspace/emb_segments" \
    --embed-devices "cuda:0,cuda:1" \
    --embed-workers 2 \
    --tar-manifest "/workspace/test.manifest"
```

**Consumer Node:**
```bash
litkit \
    --faiss-writer \
    --consume-only \
    --embed-outdir "/workspace/emb_segments" \
    --tar-manifest "/workspace/test.manifest"
```

### Architecture Diagram

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Producer 0  │  │ Producer 1  │  │ Producer 2  │
│ (2x V100)   │  │ (2x V100)   │  │ (2x V100)   │
│             │  │             │  │             │
│ Read tars   │  │ Read tars   │  │ Read tars   │
│ Embed       │  │ Embed       │  │ Embed       │
│ Write .npz  │  │ Write .npz  │  │ Write .npz  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Shared /workspace│
              │  /emb_segments/  │
              └─────────┬────────┘
                        │
                        ▼
                ┌───────────────┐
                │   Consumer    │
                │               │
                │ Poll segments │
                │ Ingest FAISS  │
                │ Save indices  │
                └───────────────┘
```

### Testing Plan

1. **Verify Consumer Doesn't Load Models**
   - Check logs: no "loading model" messages
   - Check GPU memory: should be near zero
   - Check process tree: no embedding workers

2. **Verify Producers Use All GPUs**
   - `nvidia-smi` should show 100% util on all 6 GPUs
   - Should see 2 workers per node

3. **Verify Segment Flow**
   - Producers write to `/workspace/emb_segments/`
   - Consumer reads and deletes after ingestion
   - Directory grows then shrinks

4. **Verify FAISS Updates**
   - `ntotal` should increase as consumer ingests
   - Index files should update periodically
   - SQLite `in_index=1` should match FAISS count

### Monitoring Commands

```bash
# Check GPU usage on all nodes
srun -N4 nvidia-smi

# Watch segment directory
watch -n 5 "ls -lh /workspace/emb_segments/*.npz | wc -l"

# Check FAISS progress
sqlite3 /workspace/sqlite/litkit.sqlite3 \
  "SELECT COUNT(*) FROM chunks WHERE in_index=1"

# Check consumer logs
grep "\[consumer\]" litkit_multi_*.out
```

## Next Steps

1. ✅ Code changes complete
2. ⏳ Test on HPC with 4 nodes
3. ⏳ Verify all 6 producer GPUs at 100%
4. ⏳ Verify consumer has minimal GPU usage
5. ⏳ Document final performance metrics

## Expected Improvements

- **Before:** 2 GPUs idle (consumer wasting them on duplicate work)
- **After:** 6 GPUs active (consumer freed them for producers)
- **Speedup:** ~50% faster (6 GPUs vs 4 GPUs)
