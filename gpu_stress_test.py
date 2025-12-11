#!/usr/bin/env python3
"""
GPU Stress Test for LitKit Embedding Pipeline

This script bypasses XML parsing and directly stress-tests the multi-GPU 
embedding path to verify that both GPUs on a node are being utilized.

Usage (inside container):
    python gpu_stress_test.py --sentences 100000 --devices cuda:0,cuda:1

Usage (with Charliecloud on HPC):
    sbatch gpu_stress_test.sbatch
"""

import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="GPU stress test for LitKit embeddings")
    parser.add_argument(
        "--sentences", "-n", type=int, default=100_000,
        help="Number of synthetic sentences to embed (default: 100000)"
    )
    parser.add_argument(
        "--devices", "-d", type=str, default="cuda:0,cuda:1",
        help="Comma-separated device list (default: cuda:0,cuda:1)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64,
        help="Batch size for embedding (default: 64)"
    )
    parser.add_argument(
        "--rounds", "-r", type=int, default=3,
        help="Number of test rounds (default: 3)"
    )
    parser.add_argument(
        "--single-gpu-baseline", action="store_true",
        help="Also run single-GPU baseline for comparison"
    )
    args = parser.parse_args()

    # Parse devices
    devices = [d.strip() for d in args.devices.split(",")]
    
    print("=" * 60)
    print("LitKit GPU Stress Test")
    print("=" * 60)
    
    # Check CUDA availability
    try:
        import torch
        print(f"\n[cuda] PyTorch version: {torch.__version__}")
        print(f"[cuda] CUDA available: {torch.cuda.is_available()}")
        print(f"[cuda] CUDA version: {torch.version.cuda}")
        print(f"[cuda] Device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"[cuda]   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    except Exception as e:
        print(f"[cuda] ERROR: {e}")
        sys.exit(1)
    
    # Import LitKit components
    print("\n[import] Loading LitKit embedding modules...")
    try:
        from litkit.embeddings.pool import EmbeddingPool
        from litkit.embeddings.hf_local import local_snapshot_dir
        
        model_id = "sentence-transformers/all-mpnet-base-v2"
        model_path = local_snapshot_dir(model_id)
        print(f"[import] Model path: {model_path}")
        
        if not model_path.exists():
            print(f"[import] ERROR: Model not found at {model_path}")
            print(f"[import] Set HF_HOME to point to cached models.")
            sys.exit(1)
            
    except ImportError as e:
        print(f"[import] ERROR: {e}")
        print("[import] Make sure litkit is installed: pip install -e .")
        sys.exit(1)
    
    # Generate synthetic data
    print(f"\n[data] Generating {args.sentences:,} synthetic sentences...")
    sentences = [
        f"This is synthetic test sentence number {i}. It contains enough words to be representative "
        f"of typical scientific text chunks that would be processed during PMC-OA corpus ingestion. "
        f"The goal is to stress test GPU memory bandwidth and compute throughput. Index: {i}"
        for i in range(args.sentences)
    ]
    
    # Estimate memory
    avg_len = sum(len(s) for s in sentences) / len(sentences)
    print(f"[data] Average sentence length: {avg_len:.0f} chars")
    print(f"[data] Total text size: {sum(len(s) for s in sentences) / 1e6:.1f} MB")
    
    # Single-GPU baseline (optional)
    if args.single_gpu_baseline:
        print("\n" + "=" * 60)
        print("Single-GPU Baseline Test")
        print("=" * 60)
        
        single_device = [devices[0]]
        print(f"\n[baseline] Using single device: {single_device}")
        
        pool_single = EmbeddingPool(model_path, single_device)
        
        # Warmup
        print("[baseline] Warmup...")
        _ = pool_single.encode(sentences[:1000], batch_size=args.batch_size)
        
        # Timed run
        print(f"[baseline] Encoding {args.sentences:,} sentences...")
        start = time.time()
        _ = pool_single.encode(sentences, batch_size=args.batch_size)
        elapsed_single = time.time() - start
        
        throughput_single = args.sentences / elapsed_single
        print(f"\n[baseline] Single-GPU Results:")
        print(f"[baseline]   Time: {elapsed_single:.1f}s")
        print(f"[baseline]   Throughput: {throughput_single:,.0f} sentences/sec")
        
        pool_single.close()
        del pool_single
        
        # Clear GPU memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
    
    # Multi-GPU test
    print("\n" + "=" * 60)
    print(f"Multi-GPU Test ({len(devices)} devices)")
    print("=" * 60)
    
    print(f"\n[multi] Devices: {devices}")
    print(f"[multi] Batch size: {args.batch_size}")
    print(f"[multi] Test rounds: {args.rounds}")
    
    pool = EmbeddingPool(model_path, devices)
    
    # Warmup
    print("\n[multi] Warmup (loading models on all GPUs)...")
    warmup_start = time.time()
    _ = pool.encode(sentences[:2000], batch_size=args.batch_size)
    warmup_time = time.time() - warmup_start
    print(f"[multi] Warmup complete in {warmup_time:.1f}s")
    
    # Print GPU memory after warmup
    print("\n[multi] GPU memory after warmup:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"[multi]   GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print(f"\n[multi] >>> NOW MONITOR WITH: nvidia-smi -l 1 <<<")
    print(f"[multi] >>> Both GPUs should show high utilization <<<")
    print()
    time.sleep(3)  # Give user time to start nvidia-smi
    
    # Timed rounds
    results = []
    for r in range(args.rounds):
        print(f"[multi] Round {r+1}/{args.rounds}: Encoding {args.sentences:,} sentences...")
        
        start = time.time()
        embeddings = pool.encode(sentences, batch_size=args.batch_size)
        elapsed = time.time() - start
        
        throughput = args.sentences / elapsed
        results.append({
            "round": r + 1,
            "time": elapsed,
            "throughput": throughput,
            "shape": embeddings.shape
        })
        
        print(f"[multi]   Time: {elapsed:.1f}s | Throughput: {throughput:,.0f} sentences/sec")
    
    pool.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    avg_throughput = sum(r["throughput"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)
    
    print(f"\n[summary] Multi-GPU ({len(devices)} devices):")
    print(f"[summary]   Average time: {avg_time:.1f}s")
    print(f"[summary]   Average throughput: {avg_throughput:,.0f} sentences/sec")
    print(f"[summary]   Output shape: {results[0]['shape']}")
    
    if args.single_gpu_baseline:
        speedup = avg_throughput / throughput_single
        print(f"\n[summary] Speedup vs single GPU: {speedup:.2f}x")
        if speedup > 1.5:
            print(f"[summary] ✓ Multi-GPU scaling is working properly!")
        else:
            print(f"[summary] ⚠ Multi-GPU speedup is lower than expected.")
    
    # Expected throughput guidance
    print(f"\n[guidance] Expected throughput benchmarks (V100, batch=64):")
    print(f"[guidance]   Single GPU:  ~3,000-4,000 sentences/sec")
    print(f"[guidance]   Dual GPU:    ~5,500-7,000 sentences/sec")
    print(f"[guidance]   Quad GPU:    ~10,000-12,000 sentences/sec")
    
    if avg_throughput < 2000:
        print(f"\n[warning] ⚠ Throughput seems low. Check:")
        print(f"[warning]   - CUDA drivers and PyTorch CUDA version match")
        print(f"[warning]   - GPUs are not being used by other processes")
        print(f"[warning]   - Model is actually loaded on GPU (check nvidia-smi memory)")
    
    print("\n[done] GPU stress test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
