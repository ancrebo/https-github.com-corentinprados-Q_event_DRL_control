#!/bin/bash

# Activate the new conda environment
source /path/to/anaconda3/bin/activate mesh_env

# Execute the necessary commands
python "$@"

# Deactivate the conda environment
conda deactivate
