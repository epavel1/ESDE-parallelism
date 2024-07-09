# Treasure Hunt HPC Parallelization 

## Introduction

Welcome to the HPC Parallelization Strategies Treasure Hunt! This repository is designed to provide hands-on experience with parallelization strategies using High-Performance Computing (HPC). You'll navigate through five tasks, each demonstrating different parallelization techniques.

## Repository Structure

The repository is organized as follows:


- **tasks/task1.py to tasks/task5.py**: Python scripts for each task.
- **tasks/task1.sh to tasks/task5.sh**: Job submission scripts for each task.
- **solutions/task1.py to tasks/task5.py**: Python script solutions for each task.
- **C_task_1_3/task1.c to C_task_1_3/task3.c**: C script solutions for task 1 to 3.
- **env/environment.sh**: File to set up the environment.
- **ESDE Parallelism.pdf**: Presentation slides.
- **README.md**: This file.

## Tasks Overview

### Task 1 - Parallel Processing (Multi-processing Single Node)
**File**: `task1.py`  
**Job Script**: `task1.sh`  
**Description**: Demonstrates basic parallel processing using multi-processing within a single node. Part of the code is missing and needs to be filled by participants to achieve parallel execution.

### Task 2 - Message Passing Interface (Multi-processing Multiple Nodes)
**File**: `task2.py`  
**Job Script**: `task2.sh`  
**Description**: Implements message passing interface for multi-processing on multiple nodes. Complete the code to effectively use multiple processors for parallel computation.

### Task 3 - CUDA (GPU Memory Computation)
**File**: `task3.py`  
**Job Script**: `task3.sh`  
**Description**: Introduces GPU acceleration using CUDA. Participants will complete the code to utilize GPU resources for memory-intensive computations.

### Task 4 - CUDA Aware MPI (GPU with Multi-processing)
**File**: `task4.py`  
**Job Script**: `task4.sh`  
**Description**: Combines GPU acceleration with MPI for multi-processing. The task requires filling in the missing code to distribute GPU tasks across multiple nodes.

### Task 5 - DDP Parallelization
**File**: `task5.py`  
**Job Script**: `task5.sh`  
**Description**: Focuses on Distributed Data Parallel (DDP) strategies. This task is unrelated to the first four but demonstrates another parallelization technique. Participants must complete the code to achieve parallel execution using DDP.

## Instructions

1. **Clone the Repository**:
```sh
git clone https://github.com/epavel1/ESDE-parallelism.git
cd ESDE-parallelism
```

2. **Setting up the Environment**:
Navigate to /env directory and source the environment.sh file.
```sh
cd env/
source environment.sh
```

3. **Navigate to the Tasks**:
Each task's Python script and corresponding job script are located in the /tasks directory.

4. **Complete the Tasks**:
Open each taskX.py file and fill in the missing code. Use the corresponding taskX.sh script to submit the job and test your solution.
```sh
sbatch taskX.sh
```


