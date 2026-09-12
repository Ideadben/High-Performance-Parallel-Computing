#!/usr/bin/env bash
#SBATCH --job-name=TaskFarm
#SBATCH --partition=modi_HPPC
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --threads-per-core=1
#SBATCH --exclusive

mpiexec apptainer exec \
   ~/modi_images/ucphhpc/hpc-notebook:latest \
   ./task_farm_HEP
