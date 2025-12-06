# Producer/Consumer Architecture Fix

## Problem Identified

The consumer (gpu-node4) is doing GPU embedding work when it should only be ingesting pre-computed segments from producers.

### Current Behavior (WRONG)
- **Producers (gpu-node1-830)**: `--embed-producer` → Write segments to disk ✅
- **Consumer (gpu-node4)**: `--faiss-writer --consume-segments` → **BUT ALSO runs full tar scan + embedding** ❌

### Root Cause
In `build_or_update_indices()`, the code always:
1. Scans tar files
2. Embeds papers and chunks
3. THEN at the end, if `--consume-segments`, ingests segment files

This means the consumer is duplicating the work of the producers!

## Required Fix

The consumer should have a **consume-only mode** that:
1. Does NOT scan tar files
2. Does NOT embed anything
3. ONLY ingests segments in a polling loop

### Implementation Options

**Option A: Add `--consume-only` flag**
```python
if args.consume_only:
    # Skip tar scan entirely
    # Just poll and ingest segments
    while True:
        c_added = _ingest_chunk_segments(conn, chunk_index, seg_dir)
        p_added = _ingest_paper_segments(conn, paper_index, seg_dir)
        if no_segments_remaining:
            break
        sleep(30)
```

**Option B: Make `--consume-segments` without `--embed-producer` mean consume-only**

Current flags don't clearly express "ingest only, don't produce":
- `--embed-producer`: Produce segments ✅
- `--faiss-writer`: Write to FAISS ✅  
- `--consume-segments`: Ingest segments at end of build ❌ (ambiguous)

## Testing on HPC

Current GPU usage shows the problem:
```
gpu-node1 (producer): GPU 0=0%, GPU 1=0%  ← Should be 100%
gpu-node2 (producer): GPU 0=0%, GPU 1=0%  ← Should be 100%
gpu-node3 (producer): GPU 0=0%, GPU 1=0%  ← Should be 100%
gpu-node4 (consumer): GPU 0=100%, GPU 1=0%  ← Should be 0% (CPU only for FAISS)
```

## Next Steps

1. Decide on flag design
2. Implement consume-only mode
3. Add clear logging: `[producer]`, `[consumer]`, `[writer]` prefixes
4. Test on HPC with 4 nodes
5. Verify all producer GPUs are utilized
6. Verify consumer only does FAISS ingestion
