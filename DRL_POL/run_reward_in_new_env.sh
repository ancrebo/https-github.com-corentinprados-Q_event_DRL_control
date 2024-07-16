#!/bin/bash

# Activate the new conda environment
source /scratch/pietero/miniconda3/condabin/activate mesh_env

# Execute the necessary commands
python "$@"

# Deactivate the conda environment
conda deactivate
