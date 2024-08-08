#!/bin/bash


# Debug: Print the current working directory
echo "Current working directory: $(pwd)"

# Debug: Print the current environment
echo "Current environment: $(conda info | grep 'active environment')"

# Activate the new conda environment
source /scratch/pietero/miniconda3/bin/activate mesh_env

# Debug: Verify conda activation
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment"
    exit 1
fi

# Debug: Print the new environment
echo "Activated conda environment: $(conda info | grep 'active environment')"

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
