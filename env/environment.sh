#!/bin/bash

ml --force purge
ml use $OTHERSTAGES
ml Stages/2024

ml GCC/12.3.0
ml OpenMPI/4.1.5
ml mpi4py/3.1.4
ml GCCcore/.12.3.0
ml torchvision/0.16.2
ml PyTorch/2.1.2
ml numba/0.58.1
ml CuPy/12.2.0-CUDA-12

