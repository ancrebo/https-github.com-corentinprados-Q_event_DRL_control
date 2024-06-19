#!/bin/bash
#SBATCH --job-name=3Dcylinder
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=46
#SBATCH --ntasks-per-socket=23
#SBATCH --time=02:00:00
#SBATCH --qos=debug

module unload python
module load gmsh
module load ALYA-MPIO-TOOLS

source /apps/INTEL/oneapi/2021.3/intelpython/python3.7/etc/profile.d/conda.sh

conda activate tf260-case

#scontrol show hostnames > hostlist

python3 PARALLEL_TRAINING_3D.py
