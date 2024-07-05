#!/bin/bash
#SBATCH --job-name=task4            # Job name
#SBATCH --partition=booster         # Partition name
#SBATCH --ntasks=4                  # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=64          # Number of CPU cores per task
#SBATCH --account=deepacf           # Account name
#SBATCH --output=job_output_%j.log     # Output file (%j is replaced by the job ID)
#SBATCH --error=job_error_%j.log       # Error file (%j is replaced by the job ID)
#SBATCH --time=00:30:00             # Time limit of 30 minutes

# Set CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Enable CUDA Multi-Process Service (MPS)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_logs_$SLURM_JOB_ID
nvidia-cuda-mps-control -d

# Run the task
srun -v --cuda-mps python3 task4.py

# Disable CUDA MPS after the job is done
echo quit | nvidia-cuda-mps-control