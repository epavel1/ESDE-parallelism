#!/bin/bash
#SBATCH --job-name=task5
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --gres=gpu:4                   # Request 4 GPUs per node
#SBATCH --partition=booster            # Partition name
#SBATCH --account=deepacf              # account name
#SBATCH --output=job_output_%j.log     # Output file (%j is replaced by the job ID)
#SBATCH --error=job_error_%j.log       # Error file (%j is replaced by the job ID)

# Get the master node address
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=7010

# Set environment variables for PyTorch distributed training
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
export RANK=$SLURM_PROCID

# Print the environment variables for debugging
#echo "MASTER_ADDR=$MASTER_ADDR"
#echo "MASTER_PORT=$MASTER_PORT"
#echo "WORLD_SIZE=$WORLD_SIZE"
#echo "RANK=$RANK"
#echo $SCRATCH
#echo $SCRATCH/vasireddy1/MNIST/

# Run the PyTorch Lightning script with srun
srun -v python task5.py
