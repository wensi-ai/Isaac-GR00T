#!/bin/bash
#SBATCH --job-name="gr00t_train"
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpro6000:2
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/sc/gr00t_%j.log
#SBATCH --error=outputs/sc/gr00t_%j.log

# Calculate total GPUs across all nodes
NUM_GPUS=$((${SLURM_GPUS_ON_NODE:-1} * ${SLURM_NNODES:-1}))

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_CPU_PER_TASK="$SLURM_CPUS_PER_TASK
echo "SLURM_MEM_PER_NODE="$SLURM_MEM_PER_NODE
echo "Number of nodes: ${SLURM_NNODES:-1}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE:-1}"
echo "Total GPUs: $NUM_GPUS"
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_NTASKS_PER_NODE"=$SLURM_NTASKS_PER_NODE
echo "working directory="$SLURM_SUBMIT_DIR

source /vision/u/$(whoami)/libs/gr00t/.venv/bin/activate

DATE=$(date +%Y%m%d-%H%M%S)

echo "Current time: $(date)"
echo "Running with args: $@"

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 scripts/b1k/train_b1k.py \
    --experiment-name gr00t-${DATE} \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /vision/u/wsai/data/lerobot/i3l/books \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/i3l/real_r1pro.py \
    --num-gpus $NUM_GPUS \
    --output-dir outputs/b1k-${DATE} \
    --save-total-limit 5 \
    --save-steps 1500 \
    --max-steps 150000 \
    --global-batch-size 128 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4

echo "Job finished."
exit 0
