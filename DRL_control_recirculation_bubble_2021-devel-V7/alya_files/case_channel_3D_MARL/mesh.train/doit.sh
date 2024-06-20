#!/bin/bash
#
# Generate mesh and initial condition for channelflow

gmsh -3 cylinder.geo

#conda activate
# 1. Run conversor from GMSH (can be parallel)
# Boundaries 3,4,5,6 are periodic so skip them
srun -n 100 pyalya_gmsh2alya -c cylinder -p 1,2 cylinder

# 2. Generate periodicity in z directions
# Use the new MPIO format
srun pyalya_periodic -c cylinder -d z --mpio cylinder

# 3. Generate initial condition
#python3 pyalya_initialCondition.py 3Dcylinder

cp *.mpio.bin ../
