#!/bin/bash -l
#SBATCH -A snic2022-3-25
#SBATCH -J multi001_3Dcylinder
#SBATCH -t 00:40:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH --mem=250GB
#SBATCH -p main
#SBATCH --output "debug_cylinder.out"
#SBATCH --error "debug_cylinder.err"
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=polsm@kth.se

## load environment
. /cfs/klemming/home/p/polsm/activate_tensorflow.sh

## check all slurm variables 
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Memory per node Allocated      = $SLURM_MEM_PER_NODE"
echo "Memory per cpu Allocated       = $SLURM_MEM_PER_CPU"

# launch training in 3D
echo "Script initiated at `date` on `hostname`"
python3 PARALLEL_TRAINING_3D.py
echo "Script initiated at `date` on `hostname`"

