# Producer-Consumer Bug Analysis and Fixes

## Executive Summary

After deep code analysis of `src/litkit/cli.py`, I've identified several critical bugs in the producer-consumer implementation that can cause data corruption, race conditions, and incomplete indexing in multi-node scenarios.

## Critical Bugs Identified

### 1. **Race Condition in Segment File Writing (CRITICAL)**

**Location**: Lines ~1107-1130 in `_SegmentWriter.write()` and `_ChunkSegmentWriter.write()`

**Bug**: 
```python
# Current code writes to temp file, then does atomic rename
os.replace(tmp, final)
_maybe_fsync_dir(final)
```

The problem is that **producers commit to SQLite BEFORE the segment file is fully written and fsynced**. If a consumer reads the DB and tries to ingest the segment before the file is complete, it will fail or get corrupted data.

**Evidence from code** (line ~2818):
```python
elif getattr(args, "embed_producer", False) and not getattr(args, "faiss_writer", False):
    # ---- producer: DO NOT touch FAISS, DO NOT set in_index=1 ----
    
    if paper_ids_buf:
        # ... embed ...
        paper_seg_writer.write(ids=..., vecs=Xp)
        conn.commit()  # <-- BUG: commits BEFORE segment file is guaranteed durable
```

**Fix**: Move `conn.commit()` to AFTER the segment writer confirms the file is written and fsynced.

### 2. **No Segment File State Management**

**Location**: Throughout the segment writing logic

**Bug**: There's no mechanism to mark segments as "complete" vs "in-progress". A consumer might:
- Start reading a segment while it's still being written
- Skip segments that are complete but haven't been processed yet
- Process the same segment multiple times if it crashes and restarts

**Fix**: Implement a state machine:
1. Producer writes to `.tmp` file
2. Producer fsyncs and renames to `.writing` 
3. Producer commits to DB
4. Producer renames to final name (no extension)
5. Consumer only processes files without extensions

### 3. **Lack of Coordination Between Producer and Consumer**

**Location**: `_ingest_paper_segments()` and `_ingest_chunk_segments()` (~lines 1640-1780)

**Bug**: The consumer scans for segment files periodically, but:
- There's no signal when new segments are ready
- No way to know when producers are done
- Consumers might exit before all segments are processed

**Evidence**:
```python
def _ingest_paper_segments(conn, paper_index, outdir: Path, *, save_every: int = 2):
    outdir = Path(outdir)
    if not outdir.exists():
        return 0
    cand = sorted(list(outdir.glob("papers_*.npz"))  # Just scans directory
```

**Fix**: Implement a coordination mechanism:
- Producers write a "shard_N_complete" marker when done
- Consumer polls for these markers
- Consumer exits only when all shards have completion markers

### 4. **Incomplete Error Recovery**

**Location**: Lines ~1700-1780 in segment ingestion

**Bug**: When a segment fails to ingest:
```python
except Exception as e:
    if not is_ingesting:
        try:
            os.replace(tmp, p)  # Rename back for retry
        except Exception:
            pass
    _eprint(f"[segments] ERROR ingesting {p.name}: ...")
```

The code tries to rename back, but:
- If the file is corrupted, it will retry forever
- No maximum retry count
- No dead-letter queue for persistently failing segments

**Fix**: 
- Add a `.failed` extension for segments that fail N times
- Log detailed error information
- Implement exponential backoff for retries

### 5. **Marker Flag Inconsistency**

**Location**: Throughout the code, particularly in producer mode

**Bug**: In producer mode, the code explicitly avoids setting `in_index=1`:
```python
# ---- producer: DO NOT touch FAISS, DO NOT set in_index=1 ----
```

But later in the same function (~line 2840), there's no explicit flag setting. This means:
- If a producer crashes, those rows will never be marked as indexed
- The backfill logic might try to re-embed already-embedded papers/chunks
- Potential duplicate work across restarts

**Fix**: 
- Producers should mark rows with a `producer_shard_id` column
- Consumers should update `in_index=1` only after successful FAISS ingestion
- Implement reconciliation logic on startup

## Proposed Fixes

### Fix 1: Atomic Segment Writing with State Tracking

```python
class _SegmentWriter:
    def write(self, ids: np.ndarray, vecs: np.ndarray):
        if ids.size == 0:
            return []
        
        written_paths = []
        for start in range(0, ids.shape[0], self.segment_size):
            end = min(ids.shape[0], start + self.segment_size)
            ids_i = np.ascontiguousarray(ids[start:end], dtype=np.int64)
            vecs_i = np.ascontiguousarray(vecs[start:end])
            
            # Step 1: Write to .tmp
            tmp = self._next_path().with_suffix(".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            
            with open(tmp, "wb") as fh:
                np.savez(fh, ids=ids_i, vecs=vecs_i, ...)
                if os.environ.get("LITKIT_SEGMENT_FSYNC_FILE", "1") == "1":
                    fh.flush()
                    os.fsync(fh.fileno())
            
            # Step 2: Rename to .writing (atomic claim)
            writing = tmp.with_suffix(".writing")
            os.replace(tmp, writing)
            _maybe_fsync_dir(writing)
            
            # Step 3: After DB commit (caller's responsibility), rename to final
            # Store path for caller to finalize
            written_paths.append((writing, writing.with_suffix("")))
        
        return written_paths  # Caller must finalize after commit

    def finalize_segments(self, path_pairs):
        """Rename .writing files to final names after DB commit."""
        for writing, final in path_pairs:
            try:
                os.replace(writing, final)
                _maybe_fsync_dir(final)
            except Exception as e:
                _eprint(f"[segment] WARNING: failed to finalize {writing}: {e}")
```

### Fix 2: Producer-Consumer Coordination Protocol

```python
class ProducerCoordinator:
    def __init__(self, outdir: Path, shard_id: int, num_shards: int):
        self.outdir = Path(outdir)
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.marker_file = self.outdir / f".shard_{shard_id:02d}_complete"
    
    def mark_complete(self):
        """Signal that this producer has finished."""
        self.outdir.mkdir(parents=True, exist_ok=True)
        tmp = self.marker_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps({
                "shard_id": self.shard_id,
                "timestamp": time.time(),
                "hostname": socket.gethostname(),
                "pid": os.getpid()
            }))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.marker_file)
        _maybe_fsync_dir(self.marker_file)

class ConsumerCoordinator:
    def __init__(self, outdir: Path, num_shards: int):
        self.outdir = Path(outdir)
        self.num_shards = num_shards
    
    def all_producers_complete(self) -> bool:
        """Check if all producer shards have completed."""
        if not self.outdir.exists():
            return False
        complete = 0
        for i in range(self.num_shards):
            marker = self.outdir / f".shard_{i:02d}_complete"
            if marker.exists():
                complete += 1
        return complete == self.num_shards
    
    def wait_for_completion(self, poll_interval: float = 10.0, timeout: float = 14400.0):
        """Wait for all producers to signal completion."""
        start = time.time()
        while time.time() - start < timeout:
            if self.all_producers_complete():
                return True
            time.sleep(poll_interval)
        return False
```

### Fix 3: Enhanced Segment Ingestion with Retry Logic

```python
def _ingest_segments_with_retry(
    conn, 
    index, 
    outdir: Path, 
    kind: str,
    max_retries: int = 3,
    save_every: int = 2
) -> int:
    """Enhanced segment ingestion with retry logic and state tracking."""
    outdir = Path(outdir)
    if not outdir.exists():
        return 0
    
    # Only process segments without extensions (fully written)
    pattern = f"{kind}_sh*.npz"
    cand = sorted(outdir.glob(pattern))
    
    added_total = 0
    batch_counter = 0
    
    for p in cand:
        # Check retry count
        retry_marker = p.with_suffix(".retry_count")
        retry_count = 0
        if retry_marker.exists():
            try:
                retry_count = int(retry_marker.read_text().strip())
            except:
                retry_count = 0
        
        if retry_count >= max_retries:
            # Move to failed directory
            failed = p.with_suffix(".failed")
            _eprint(f"[segments] Max retries exceeded for {p.name}, marking as failed")
            try:
                os.replace(p, failed)
            except:
                pass
            continue
        
        # Try to claim the segment
        processing = p.with_suffix(".processing")
        try:
            os.replace(p, processing)
        except FileNotFoundError:
            continue
        except Exception:
            continue
        
        try:
            # Load and ingest
            with np.load(processing, mmap_mode="r") as z:
                ids = np.ascontiguousarray(z["ids"].astype(np.int64))
                X = np.ascontiguousarray(z["vecs"].astype(np.float32))
            
            if ids.size == 0:
                os.remove(processing)
                continue
            
            # Ingest into FAISS
            with FileLock(FAISS_LOCK):
                sel = _make_id_selector(ids)
                _safe_remove_ids(index, sel)
                added, ids_added = _add_with_ids_dedup(index, ids, X)
                if added:
                    _faiss_save(index, get_index_path(kind))
            
            # Update DB flags
            if added:
                with FileLock(DB_LOCK):
                    _mark_in_index(conn.cursor(), kind, [int(i) for i in ids_added])
                    conn.commit()
            
            added_total += int(added)
            batch_counter += 1
            
            # Success! Remove the processing file
            os.remove(processing)
            if retry_marker.exists():
                retry_marker.unlink()
            
        except Exception as e:
            # Increment retry count
            retry_count += 1
            retry_marker.write_text(str(retry_count))
            
            # Rename back to original
            try:
                os.replace(processing, p)
            except:
                pass
            
            _eprint(f"[segments] ERROR ingesting {p.name} (attempt {retry_count}/{max_retries}): {e}")
    
    return added_total
```

## Testing Strategy

### Phase 1: Single-Node Testing
1. Test producer-only mode with segment writing
2. Verify segment file states (.tmp → .writing → final)
3. Test consumer-only mode with pre-written segments
4. Verify correct marking of `in_index` flags

### Phase 2: Two-Node Testing  
1. Run 1 producer + 1 consumer simultaneously
2. Verify coordination markers work correctly
3. Test crash recovery (kill producer mid-run)
4. Verify no data loss or corruption

### Phase 3: Multi-Node Testing
1. Run N producers + 1 consumer
2. Verify all shards are processed
3. Test concurrent segment ingestion
4. Measure throughput and identify bottlenecks

## Implementation Priority

1. **HIGH**: Fix 1 (Atomic segment writing) - Prevents data corruption
2. **HIGH**: Fix 2 (Coordination protocol) - Ensures all work completes
3. **MEDIUM**: Fix 3 (Enhanced ingestion) - Improves reliability
4. **MEDIUM**: Fix 5 (Marker flag consistency) - Prevents duplicate work
5. **LOW**: Fix 4 (Error recovery) - Nice to have, but not critical

## Estimated Implementation Time

- Fix 1: 2-3 hours
- Fix 2: 3-4 hours  
- Fix 3: 2-3 hours
- Testing: 4-6 hours
- Total: ~15 hours

## Next Steps

1. Implement fixes in priority order
2. Create unit tests for each fix
3. Run diagnostic SLURM script to validate fixes
4. Create production-ready multi-node SLURM script
5. Document the new producer-consumer protocol
