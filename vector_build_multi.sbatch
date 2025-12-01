#!/bin/bash
#SBATCH --job-name=litkit_multi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=litkit_multi_%j.out
#SBATCH --error=litkit_multi_%j.err

# Load necessary modules (adjust as needed for your HPC environment)
module load python/3.10
module load cuda/11.7

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set environment variables
export LITKIT_WORKSPACE="/path/to/shared/workspace"
export LITKIT_TAR_DIR="/path/to/tar/shards"

# Function to run producer
run_producer() {
    local node_id=$1
    srun -N1 -n1 python -m litkit \
        --embed-producer \
        --shard-id $node_id \
        --num-shards $SLURM_NNODES \
        --embed-outdir "${LITKIT_WORKSPACE}/emb_segments" \
        --paper-embed-bs 16 \
        --chunk-embed-bs 64 \
        --embed-devices "cuda:0,cuda:1" \
        --embed-workers 2
}

# Function to run consumer
run_consumer() {
    srun -N1 -n1 python -m litkit \
        --faiss-writer \
        --consume-segments \
        --embed-outdir "${LITKIT_WORKSPACE}/emb_segments"
}

# Run producers on all nodes except the last one
for ((i=0; i<SLURM_NNODES-1; i++))
do
    run_producer $i &
done

# Run consumer on the last node
run_consumer &

# Wait for all background jobs to finish
wait

echo "All jobs completed"
