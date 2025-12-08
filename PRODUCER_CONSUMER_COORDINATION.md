# Producer-Consumer Coordination Implementation

## Overview

This document describes the implementation of graceful coordination between producer and consumer processes in litkit's multi-node build system. The coordination mechanism ensures the consumer exits cleanly when all producers have completed their work.

## Problem Statement

Previously, the consumer process would run indefinitely after producers completed, requiring manual intervention or job timeout to terminate. This wasted resources and complicated job management.

## Solution: Completion Marker Files

### Architecture

The solution uses filesystem-based completion markers:

1. **Producers**: Each producer writes a completion marker file after processing its shard
2. **Consumer**: Polls for completion markers and exits when all producers signal completion

### Implementation Details

#### 1. ProducerCoordinator Class

Located in `src/litkit/cli.py`:

```python
class ProducerCoordinator:
    """Coordinates producer completion signaling."""
    
    def __init__(self, workspace_dir: Path, shard_id: int):
        self.markers_dir = workspace_dir / "producer_markers"
        self.shard_id = shard_id
        self.markers_dir.mkdir(parents=True, exist_ok=True)
    
    def mark_complete(self):
        """Mark this producer shard as complete."""
        marker_file = self.markers_dir / f"producer_{self.shard_id}.done"
        marker_file.write_text(f"completed at {time.time()}")
        print(f"[ProducerCoordinator] Marked shard {self.shard_id} as complete")
```

**Key Features:**
- Creates marker directory if it doesn't exist
- Writes timestamped completion marker
- Simple, filesystem-based (works on Lustre)

#### 2. ConsumerCoordinator Class

Located in `src/litkit/cli.py`:

```python
class ConsumerCoordinator:
    """Coordinates consumer waiting for producer completion."""
    
    def __init__(self, workspace_dir: Path, num_shards: int):
        self.markers_dir = workspace_dir / "producer_markers"
        self.num_shards = num_shards
        self.poll_interval = 10  # seconds
    
    def wait_for_producers(self):
        """Poll for producer completion markers."""
        print(f"[ConsumerCoordinator] Waiting for {self.num_shards} producers to complete...")
        
        while True:
            if not self.markers_dir.exists():
                time.sleep(self.poll_interval)
                continue
            
            completed = []
            for shard_id in range(self.num_shards):
                marker = self.markers_dir / f"producer_{shard_id}.done"
                if marker.exists():
                    completed.append(shard_id)
            
            print(f"[ConsumerCoordinator] Producers completed: {len(completed)}/{self.num_shards}")
            
            if len(completed) == self.num_shards:
                print("[ConsumerCoordinator] All producers complete!")
                return
            
            time.sleep(self.poll_interval)
```

**Key Features:**
- Polls every 10 seconds (Lustre-friendly)
- Tracks completion count
- Returns when all producers complete
- Provides progress updates

#### 3. Producer Integration

In `build_or_update_indices()` function, after main processing:

```python
if args.embed_producer and args.shard_id is not None:
    coordinator = ProducerCoordinator(workspace_dir, args.shard_id)
    coordinator.mark_complete()
```

#### 4. Consumer Integration

In `build_or_update_indices()` function, after segment ingestion:

```python
if args.consume_only and args.num_shards is not None:
    coordinator = ConsumerCoordinator(workspace_dir, args.num_shards)
    coordinator.wait_for_producers()
    print("Consumer exiting cleanly - all producers complete.")
```

#### 5. SLURM Script Update

Added `--num-shards` parameter to consumer command:

```bash
litkit \
    --faiss-writer \
    --consume-only \
    --embed-outdir "/workspace/emb_segments" \
    --tar-manifest "/workspace/test.manifest" \
    --papers-index hnsw \
    --chunks-index ivfpq \
    --num-shards $((SLURM_NNODES - 1))  # NEW: coordination parameter
```

## Testing on HPC

### Steps to Test

1. **Pull latest code:**
   ```bash
   cd /path/to/litkit
   git pull
   ```

2. **Rebuild container:**
   ```bash
   # Allocate interactive node
   salloc -N1 --partition=gpu-v100 --gres=gpu:2 --time=2:00:00
   
   # SSH to allocated node
   ssh <node-name>
   
   # Build container
   cd /path/to/litkit
   just build-arm
   ```

3. **Clean workspace:**
   ```bash
   cd workspace
   rm -rf emb_segments/* sqlite/* indices/* producer_markers/
   ```

4. **Submit multi-node job:**
   ```bash
   sbatch vector_build_multi.sbatch
   ```

### Expected Behavior

1. **Bootstrap phase:** Creates empty FAISS indices (~30 seconds)
2. **Producer phase:** 3 producers generate embedding segments in parallel
3. **Consumer phase:** Consumer ingests segments from all producers
4. **Coordination phase:** Consumer polls for completion markers
5. **Clean exit:** Consumer exits when all 3 producers signal completion

### Monitoring

Watch job output:
```bash
tail -f litkit_multi_<JOBID>.out
```

Look for these key messages:

**Producer completion:**
```
[ProducerCoordinator] Marked shard 0 as complete
[ProducerCoordinator] Marked shard 1 as complete
[ProducerCoordinator] Marked shard 2 as complete
```

**Consumer coordination:**
```
[ConsumerCoordinator] Waiting for 3 producers to complete...
[ConsumerCoordinator] Producers completed: 1/3
[ConsumerCoordinator] Producers completed: 2/3
[ConsumerCoordinator] Producers completed: 3/3
[ConsumerCoordinator] All producers complete!
Consumer exiting cleanly - all producers complete.
```

## Benefits

1. **No manual intervention:** Job completes automatically
2. **Resource efficiency:** Consumer exits promptly when work is done
3. **Simple implementation:** Filesystem-based, no external dependencies
4. **Lustre-friendly:** 10-second polling interval reduces metadata load
5. **Debuggable:** Clear progress messages in job output

## Files Modified

- `src/litkit/cli.py`: Added coordinator classes and integration
- `vector_build_multi.sbatch`: Added `--num-shards` parameter to consumer

## Commit

```
git commit 7bb14a4
Author: hlavacek
Date: 2025-12-08

Implement producer-consumer coordination with completion markers

- Add ProducerCoordinator class to mark shard completion
- Add ConsumerCoordinator class to poll for producer completion
- Update producer logic to write completion markers after processing
- Update consumer logic to poll for markers and exit when all producers complete
- Update SLURM script to pass num-shards to consumer for coordination
- Consumer now exits gracefully when all producers signal completion
```

## Related Documentation

- [PRODUCER_CONSUMER_IMPLEMENTATION.md](PRODUCER_CONSUMER_IMPLEMENTATION.md) - Overall implementation details
- [MULTI_NODE_TESTING_GUIDE.md](MULTI_NODE_TESTING_GUIDE.md) - Testing procedures
- [HPC_DEPLOYMENT_GUIDE.md](HPC_DEPLOYMENT_GUIDE.md) - HPC-specific deployment
- [REBUILD_CONTAINER_HPC.md](REBUILD_CONTAINER_HPC.md) - Container rebuild process
