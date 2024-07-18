#!/bin/bash

# Activate the new conda environment
source /scratch/pietero/miniconda3/bin/activate mesh_env

# Execute the necessary commands
"$@"

# Capture the exit status of the Python script
exit_status=$?

# Deactivate the conda environment
conda deactivate

# Check if the Python script was successful
if [ $exit_status -eq 0 ]; then
    echo "Python script \`calc_reward.py\` completed successfully."
else
    echo "Python script \`calc_reward.py\` failed with exit status $exit_status."
fi

# Exit with the status of the Python script
exit $exit_status
