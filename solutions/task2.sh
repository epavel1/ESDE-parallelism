#!/bin/bash
#SBATCH --job-name=task2            # Job name
#SBATCH --partition=booster         # Partition name
#SBATCH --ntasks=4                  # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=64          # Number of CPU cores per task
#SBATCH --gres=gpu:4                # Number of GPUs
#SBATCH --account=deepacf           # Account name
#SBATCH --output=job_output_task2_%j.log  # Output file (%j is replaced by the job ID)
#SBATCH --error=job_error_task2_%j.log    # Error file (%j is replaced by the job ID)
#SBATCH --time=00:30:00             # Time limit of 30 minutes

# Run the task
srun -v python3 task2.py
