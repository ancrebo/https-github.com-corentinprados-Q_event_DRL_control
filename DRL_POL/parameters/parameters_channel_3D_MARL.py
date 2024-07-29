#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# CHANNEL 3D parameters
#
# Just copy this code to the main folder and rename it
# to parameters.py to use it
#
# Pol Suarez, Fran Alcantara, Arnau Miro

# TODO: IN GENERAL - @pietero, @canordq Update for channel parameters!! - Pieter
# TODO: @pietero clean up commented cylinder code - Pieter

from __future__ import print_function, division
from typing import List, Tuple, Dict, Any, Union


import numpy as np
import math
import os
import shutil

from jets import build_jets, JetChannel
from env_utils import agent_index_2d_to_1d, agent_index_1d_to_2d
from witness import calculate_channel_witness_coordinates, write_witness_file

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s\n", __name__, logger.level)

### CASE NAME ************************************************

training_case: str = (
    "channel_3D_MARL"  # This is the DRL training case name, like "channel_3D_MARL" or "cylinder_3D_WS_TEST" - Pieter
)
case: str = (
    "channel"  # This is the ALYA case name, so "channel" or "cylinder", NOT "channel_3D_MARL" or "cylinder_3D_WS_TEST" - Pieter
)
simu_name: str = "3DChan"
dimension: int = 3
reward_function: str = "q_event_volume"

Re_case: int = 6
slices_probes_per_jet: int = 1  # Unused for channel case - Pieter
neighbor_state: bool = False
h_qevent_sensitivity: float = (
    3.0  # Used to identify the Q events, sensitivity to the Q events
)

n_agents_x: int = 2  # Number of agents along x direction
n_agents_z: int = 2  # Number of agents along z direction

quick_episodes: bool = False
quick_episode_length: int = 3

# THESE VALUES ARE CALCULATED FROM `coco_small_load_post_processing.py` and are specific to THIS BASELINE! - Pieter
# Check `calculated_values.csv` in baseline folder for these values
u_tau: float = 0.04871566388520865
delta_tau: float = 0.007280998149274599
Re_tau: float = 137.343806370783
t_tau: float = 0.14945907678547107
t_i: float = 891.744
t_i_plus: float = 5966.47603597861
t_f: float = 1376.61
t_f_plus: float = 9210.614902806761
U_b: float = 0.8975133576311723
t_w: float = 2.974886086427719
t_w_plus: float = 19.9043520835993


#### Reynolds cases
#### 0 --> Re = 100
#### 1 --> Re = 200
#### 2 --> Re = 300
#### 3 --> Re = 400
#### 4 --> Re = 1000
#### 5 --> Re = 3900 (high Re case--FTaC)
#### 6 --> Re = 180

### *****************************************************
### RUN BASELINE ****************************************

# Whether to run the baseline simulation
# False if baseline already exists
run_baseline: bool = False

# Whether to restart the episode from the end of the last checkpoint or baseline
# False if always restarting from baseline
bool_restart: bool = False

### **********************************************************
### DOMAIN BOX ***********************************************
# TODO: @canordq Update for channel parameters!! - Pieter
# These parameters need to match the specific case mesh

# The parameters below are based on the `minimal channel - Jimenez` paper
h = 1.0
Lx = 2.67 * h
Ly = 2 * h
Lz = 0.8 * h


### **********************************************************
### UNIVERSAL MULTI-ENVIRONMENT SETUP ************************

num_episodes = 2000  # Total number of episodes
if Re_case != 5:
    nb_actuations = 120  # Number of actuation of the neural network for each episode (ACTIONS/EPISODE)
else:
    nb_actuations = 200

if quick_episodes:
    nb_actuations = quick_episode_length

nb_actuations_deterministic = nb_actuations * 10


### **********************************************************
### SLURM SPECIFIC SETUP *************************************

nb_proc = 360  # Number of calculation processors
num_servers = 3  # number of environment in parallel

proc_per_node = 128
# proc_per_node = int(os.getenv('SLURM_NTASKS_PER_NODE'))*int(os.getenv('SLURM_CPUS_PER_TASK'))

mem_per_node = 200000  # MB RAM in each node
# mem_per_node   = int(os.getenv('SLURM_MEM_PER_NODE'))

mem_per_cpu = mem_per_node // proc_per_node
# mem_per_cpu   = int(os.getenv('SLURM_MEM_PER_CPU'))

# mem_per_srun  = int(nb_proc*mem_per_cpu) # partition in memory allocation
mem_per_srun = mem_per_node

if Re_case != 5:
    num_nodes_srun = 3
else:
    num_nodes_srun = 3


### **********************************************************
### WORKSTATION SPECIFIC SETUP *******************************

# Number of calculation processors (for local, number of threads on workstation)
nb_proc_ws = 18

# Number of environment in parallel (number of SEPARATE CFD environments) (default 1 for workstation)
num_servers_ws = 1


### *****************************************************
### RUN BASELINE ****************************************
use_MARL = True

nb_inv_per_CFD = 4  # same as nz_Qs¿¿¿ # TODO: @pietero is this necessary with nTotal_Qs defined later? - Pieter
actions_per_inv = 1  # how many actions to control per pseudoenvironment
batch_size = nb_inv_per_CFD * num_servers
# frontal_area = Lz / nb_inv_per_CFD

if num_servers == 1 and not use_MARL:
    sync_episodes = False
else:
    sync_episodes = True


### *****************************************************
### TIMESCALES ******************************************

baseline_duration = 1228.6  # to converge with velocity max = 1
baseline_time_start = 0.0

delta_t_smooth = 0.25  # ACTION DURATION smooth law duration
delta_t_converge = 0.0  # Total time that the DRL waits before executing a new action
smooth_func = "EXPONENTIAL"  # 'LINEAR', 'EXPONENTIAL', 'CUBIC' # TODO: cubic is still not coded - Pol
short_spacetime_func = False  # override smooth func --> TODO: need to fix string size --> FIXED in def_kintyp_functions.f90 (waiting for merging) - Pol

### *****************************************************
### FLUID PROPERTIES ************************************

# TODO: @pietero @canordq Update mu value for channel case! Placeholder now - Pieter
mu_list = [10e-3, 50e-4, 33e-4, 25e-4, 10e-4, 0.00025641025, 10e-4]
mu = mu_list[Re_case]
rho = 1.0

### *****************************************************
### POSTPROCESS OPTIONS *********************************

# TODO: @pietero Update ALL?? for channel parameters!! - Pieter

## Not needed for channel currently! - Pieter
norm_reward = 5  # like PRESS, try to be between -1,1
penal_cl = 0.6  # avoid asymmetrical strategies
alpha_rew = 0.80  # balance between global and local reward
norm_press = 2.0
if Re_case != 4:
    time_avg = 5.00
else:
    time_avg = 5.65  # 5.65 #corresponds to the last Tk (avg Cd Cl, not witness)
post_process_steps = 50  # TODO: put this into a include - Pol

# Could be needed, is determined by baseline and test training runs - Pieter
offset_reward_list = [
    1.381,
    1.374,
    1.325,
    1.267,
    1.079,
    1.20,
    1.2,  # TODO: @pietero @canordq temporary placeholder value for Re=180 - Pieter
]  # for re3900, still working on it
offset_reward = offset_reward_list[Re_case]

### *****************************************************
### JET SETUP *******************************************

# TODO: Update for channel parameters!! @canordq
# Keeping same norm_Q, can be changed at a later date as needed -Chriss
# This should be adjusted based on training run data

norm_Q = 0.176  # (0.088/2)/5 asa said in papers, limited Q for no momentum or discontinuities in the CFD solver

# location jet over the cylinder 0 is top centre
jet_angle = 0  # Unused for channel case - Pieter

# Re-naming number of agents for backwards compatibility - Pieter
nz_Qs: int = n_agents_z
nx_Qs: int = n_agents_x

nTotal_Qs: int = nz_Qs * nx_Qs  # total number of agents

# Create a grid of agents (2D indices)
index2d_Qs: List[Tuple[int, int]] = [(i, j) for i in range(nx_Qs) for j in range(nz_Qs)]

# Convert the list of tuples to a NumPy array
index2d_Qs: np.ndarray = np.array(index2d_Qs)

# Create the 1D index array from the 2D index array
index1d_Qs: np.ndarray = np.array(
    [agent_index_2d_to_1d(i, j, nz_Qs) for i, j in index2d_Qs]
)

delta_Q_z: float = Lz / nz_Qs
delta_Q_x: float = Lx / nx_Qs

Qs_position_z_array: np.ndarray = np.linspace(delta_Q_z / 2, Lz - delta_Q_z / 2, nz_Qs)
Qs_position_x_array: np.ndarray = np.linspace(delta_Q_x / 2, Lx - delta_Q_x / 2, nx_Qs)

Qs_position_z: List[float] = Qs_position_z_array.tolist()
Qs_position_x: List[float] = Qs_position_x_array.tolist()

jet_coordinates: np.ndarray = np.array(
    [(x, z) for x in Qs_position_x for z in Qs_position_z]
).reshape(nx_Qs, nz_Qs, 2)

logger.info("parameters: Jet coordinates calculated successfully!\n")
logger.debug(
    "parameters: Jets are placed in the following X, Z coordinates with their indices:"
)
# print("Jets are placed in the following X, Z coordinates with their indices:\n")
for i in range(nx_Qs):
    for j in range(nz_Qs):
        x, z = jet_coordinates[i, j]
        logger.debug("Agent (%d, %d): X: %.2f, Z: %.2f", i, j, x, z)
        # print(f"Agent ({i}, {j}): X: {x:.2f}, Z: {z:.2f}")


# TODO: @canordq Update for channel parameters!! - Pieter
# Commented out all content. Can add inputs as needed later. Combined the two into one JET_TOP_BOTTOM -Chriss

jets_definition = {
    "JET_TOP_BOTTOM": {
        #        "width": 10,
        #        "radius": radius,
        #        "angle": jet_angle,
        #        "positions_angle": 90
        #        + jet_angle,  # make sure the width doesn't not coincide with 0,90,180 or 270
        #        "positions": [cylinder_coordinates[0], cylinder_coordinates[1] + radius],
        #        "remesh": False,
    },
    #    "JET_BOTTOM": {
    #        "width": 10,
    #        "radius": radius,
    #        "angle": jet_angle,
    #        "positions_angle": 270
    #        - jet_angle,  # make sure the width doesn't not coincide with 0,90,180 or 270
    #        "positions": [cylinder_coordinates[0], cylinder_coordinates[1] - radius],
    #        "remesh": False,
    #    },
}

# Build the jets
# Removed jets_definition as it only (currently) applies to cylinder case. Can be used in the future for adding parameters -Chriss
jets = build_jets(JetChannel, jets_definition, delta_t_smooth)

# Commented out MOST OF section as it only applies to cylinder case -Chriss
# TO BE DELETED!!! Except jet_name -Chriss juli 2024
geometry_params = {  # Kept for legacy purposes but to be deleted when reworking the mesh script
    #        "output": ".".join(["cylinder", "geo"]),
    #        "jet_width": 10,
    #        "jet_angle": jet_angle,
    "jet_name": ["JET_TOP_BOTTOM"],
    #        "jet_positions_angle": [
    #            90 + jet_angle,
    #            270 - jet_angle,
    #        ],  # make sure the width doesn't not coincide with 0,90,180 or 270
    #        "jet_positions": [
    #            [cylinder_coordinates[0], cylinder_coordinates[1] + radius],
    #            [cylinder_coordinates[0], cylinder_coordinates[1] - radius],
    #        ],
    "remesh": False,
}
# assert (
#    jet_angle != geometry_params["jet_width"] / 2
# )  # Maybe to check during mesh construction?


### ***************************************************************
### BL option? ****************************************************

# boundary_layer = (
#     False  # TODO: For what is this used? It is imported in geo_file_maker.py - Pol
# )
# dp_left = 0
# if boundary_layer:
#     outer_radius = 1.3 * radius + dp_left  # Boundary Layer radius
#     Transfinite_number = 20
#     Progression_number = 1.025


### ****************************************************
### STATE OBSERVATION -- WITNESS MAP ******************

## HERE WE HAVE 3 CHOICES TO LOCATE PROBES:
## 1-- S85 ETMM14 //
## 2-- S99 ETMM and NATURE MI //
## 4-- 5 probes experiment from jean //
## 3-- working on it with re3900 (at the same time witness to postprocess: fft, wake profiles, pressure distribution, etc)
## 5-- 3D channel (PER LOCAL: X pattern, 8 layers, no skipping)
## 6 -- 3D channel  (PER LOCAL: X pattern, 20 layers, X every 4 layers)
## 7 -- 3D channel  (PER LOCAL: X pattern, 10 layers, X every 3 layers)
## 8 -- 3D channel  (PER LOCAL: X pattern, 50 layers, X every 5 layers)
## TODO: explain criteria of ordering history points, to "call" them quickly - Pol


probes_location = 5  # 5, 6, 7, 8

list_position_probes = []

witness_parameters: Dict[str, Any] = {
    "nx_Qs": nx_Qs,
    "nz_Qs": nz_Qs,
    "Lx": Lx,
    "Ly": Ly,
    "Lz": Lz,
}

if probes_location == 5:
    probe_type: str = "velocity"  # Probe type ('pressure' or 'velocity')
    pattern: str = "X"  # Pattern type ('X' or '+')
    y_value_density: int = 8  # Number of y values total
    y_skipping: bool = False  # Whether to skip full pattern placement on certain layers
    y_skip_values: int = 3  # Number of layers to skip if y_skipping is True

elif probes_location == 6:
    probe_type: str = "velocity"  # Probe type ('pressure' or 'velocity')
    pattern: str = "X"  # Pattern type ('X' or '+')
    y_value_density: int = 20  # Number of y values total
    y_skipping: bool = True  # Whether to skip full pattern placement on certain layers
    y_skip_values: int = 4  # Number of layers to skip if y_skipping is True

elif probes_location == 7:
    probe_type: str = "velocity"  # Probe type ('pressure' or 'velocity')
    pattern: str = "X"  # Pattern type ('X' or '+')
    y_value_density: int = 10  # Number of y values total
    y_skipping: bool = True  # Whether to skip full pattern placement on certain layers
    y_skip_values: int = 3  # Number of layers to skip if y_skipping is True

elif probes_location == 8:
    probe_type: str = "velocity"  # Probe type ('pressure' or 'velocity')
    pattern: str = "X"  # Pattern type ('X' or '+')
    y_value_density: int = 50  # Number of y values total
    y_skipping: bool = True  # Whether to skip full pattern placement on certain layers
    y_skip_values: int = 5  # Number of layers to skip if y_skipping is True

else:
    raise ValueError(
        "Invalid probes_location value! Must be 5, 6, 7, or 8, NOT %s", probes_location
    )

# Add to witness parameter dictionary for specific version parameters
witness_parameters["probe_type"] = probe_type
witness_parameters["pattern"] = pattern
witness_parameters["y_value_density"] = y_value_density
witness_parameters["y_skipping"] = y_skipping
witness_parameters["y_skip_values"] = y_skip_values

# Save the witness coordinates to dictionary
output_params: Dict[str, Any] = calculate_channel_witness_coordinates(
    witness_parameters
)


## CREATION OF WITNESS FILE
# Whether to overwrite the witness file if exists, True overwrites existing file
need_witness_file_override: bool = False
logger.debug("Witness file override: %s", need_witness_file_override)

case_folder = f"alya_files/case_{training_case}"
witness_file_path = os.path.join(case_folder, "witness.dat")
witness_version_file_path = os.path.join(case_folder, "witness_version.txt")

if os.path.exists(witness_file_path):
    if need_witness_file_override:
        # Create a backup of the existing witness.dat AND witness_version.txt file IF IT EXISTS
        if os.path.exists(witness_version_file_path):
            backup_file_path = witness_file_path + ".backup"
            shutil.copyfile(witness_file_path, backup_file_path)
            os.remove(witness_file_path)
        if os.path.exists(witness_version_file_path):
            backup_version_file_path = witness_version_file_path + ".backup"
            shutil.copyfile(witness_version_file_path, backup_version_file_path)
            os.remove(witness_version_file_path)
            logger.info(
                "Existing witness.dat and witness_version.txt files backed up in %s\n",
                case_folder,
            )

        # Import only when needed!
        from witness import write_witness_file_and_visualize

        # Write the new witness.dat file and visualize
        write_witness_file_and_visualize(
            case_folder,
            output_params,
            probes_location=probes_location,
            pattern=pattern,
            y_value_density=y_value_density,
            y_skipping=y_skipping,
            y_skip_values=y_skip_values,
            nx_Qs=nx_Qs,
            nz_Qs=nz_Qs,
        )
        need_witness_file_override = False

        logger.info("Witness creation override parameter set to False\n")
    else:
        logger.info(
            "witness.dat already exists in %s - No creation needed.\n",
            case_folder,
        )

else:
    # Create and write the witness.dat file if it does not exist
    logger.info("No existing witness.dat found in %s", case_folder)

    # Import only when needed!
    from witness import write_witness_file_and_visualize

    write_witness_file_and_visualize(
        case_folder,
        output_params,
        probes_location=probes_location,
        pattern=pattern,
        y_value_density=y_value_density,
        y_skipping=y_skipping,
        y_skip_values=y_skip_values,
        nx_Qs=nx_Qs,
        nz_Qs=nz_Qs,
    )
    logger.info("New witness.dat has been created in %s\n", case_folder)

############################################################################################
####    These are the probe positions for S85   ####
############################################################################################

# if probes_location == 1:
#     list_radius_around = [radius + 0.2, radius + 0.5]
#     list_angles_around = np.arange(0, 360, 10)
#     # positions_probes_for_grid_z = [2.5]  #beta
#
#     # positions_probes_for_grid_x_1 = [0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
#     positions_probes_for_grid_x_1 = [1, 2, 3]
#     # positions_probes_for_grid_y_1 = [-1.5, -1, -0.5, 0.0, 0.5, 1, 1.5]
#     positions_probes_for_grid_y_1 = [-1, 0.0, 1]
#
#     positions_probes_for_grid_x_2 = [0.25, 0.5]
#     positions_probes_for_grid_y_2 = [-1.5, 1.5]
#     # positions_probes_for_grid_x_2 = [-0.25, 0.0, 0.25, 0.5]
#     # positions_probes_for_grid_y_2 = [-1.5, -1, 1, 1.5]
#
#     for crrt_z in positions_probes_for_grid_z:
#         for crrt_radius in list_radius_around:
#             for crrt_angle in list_angles_around:
#                 angle_rad = np.pi * crrt_angle / 180.0
#                 list_position_probes.append(
#                     np.array(
#                         [
#                             crrt_radius * math.cos(angle_rad) + cylinder_coordinates[0],
#                             crrt_radius * math.sin(angle_rad) + cylinder_coordinates[1],
#                             crrt_z,
#                         ]
#                     )
#                 )
#
#         for crrt_x in positions_probes_for_grid_x_1:
#             for crrt_y in positions_probes_for_grid_y_1:
#                 list_position_probes.append(
#                     np.array(
#                         [
#                             crrt_x + cylinder_coordinates[0],
#                             crrt_y + cylinder_coordinates[1],
#                             crrt_z,
#                         ]
#                     )
#                 )
#
#         for crrt_x in positions_probes_for_grid_x_2:
#             for crrt_y in positions_probes_for_grid_y_2:
#                 list_position_probes.append(
#                     np.array(
#                         [
#                             crrt_x + cylinder_coordinates[0],
#                             crrt_y + cylinder_coordinates[1],
#                             crrt_z,
#                         ]
#                     )
#                 )
#
#     len_left_positions_probes = 0
#     len_cylinder_position_probes = 72
#     len_karman_position_probes = 79
#
#     cylinder_range_min = len_left_positions_probes + 1
#     cylinder_range_max = len_left_positions_probes + len_cylinder_position_probes
#
#     karman_range_min = cylinder_range_max + 1
#     karman_range_max = cylinder_range_max + len_karman_position_probes
#     tag_probs = {
#         "left": [1, len_left_positions_probes],
#         "cylinder": [cylinder_range_min, cylinder_range_max],
#         "karman": [karman_range_min, karman_range_max],
#     }
#
#     output_params = {
#         "locations": list_position_probes,
#         "tag_probs": tag_probs,
#         "probe_type": "pressure",
#     }

############################################################################################
####    These are the probe positions for S99   ####
############################################################################################

# if probes_location == 2:
#     list_radius_around = [radius + 0.2, radius + 0.5]
#     list_angles_around = np.arange(-90, 90, 10)
#
#     positions_probes_for_grid_x = [0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
#     positions_probes_for_grid_y = [-1.5, -1, -0.5, 0.0, 0.5, 1, 1.5]
#
#     for crrt_z in positions_probes_for_grid_z:
#         for crrt_radius in list_radius_around:
#             for crrt_angle in list_angles_around:
#                 angle_rad = np.pi * crrt_angle / 180.0
#                 list_position_probes.append(
#                     np.array(
#                         [
#                             crrt_radius * math.cos(angle_rad) + cylinder_coordinates[0],
#                             crrt_radius * math.sin(angle_rad) + cylinder_coordinates[1],
#                             crrt_z,
#                         ]
#                     )
#                 )
#
#         for crrt_x in positions_probes_for_grid_x:
#             for crrt_y in positions_probes_for_grid_y:
#                 list_position_probes.append(
#                     np.array(
#                         [
#                             crrt_x + cylinder_coordinates[0],
#                             crrt_y + cylinder_coordinates[1],
#                             crrt_z,
#                         ]
#                     )
#                 )
#
#     output_params = {"locations": list_position_probes, "probe_type": "pressure"}

############################################################################################
####    These are the probe positions for SXX for RE=3900  ####
####    inspired from results of Moin.2000
############################################################################################

# if probes_location == 3:
#     ## generate around cylinder info (first array for pressure distribution)
#     list_radius_around = [radius + 0, radius + 0.5]  # 2 concentric rings
#     list_angles_around = np.arange(0, 360, 9)  ## each ring has 36 values
#
#     ## wake region + wake profile refs Moin 0.56, 1.04, 1.52, 4.5, 5.5, 6.5, 9.5.
#     positions_probes_for_grid_x = [
#         0.56,
#         1.04,
#         1.52,
#         2.0,
#         2.5,
#         3,
#         3.5,
#         4,
#         4.5,
#         5.5,
#         6.5,
#         9.5,
#     ]
#     positions_probes_for_grid_y = [
#         -3,
#         -2.5,
#         -2.0,
#         -1.5,
#         -1.25,
#         -1,
#         -0.5,
#         -0.25,
#         0.0,
#         0.25,
#         0.5,
#         1,
#         1.25,
#         1.5,
#         2.0,
#         2.5,
#         3,
#     ]
#
#     for crrt_z in positions_probes_for_grid_z:
#         for crrt_radius in list_radius_around:
#             for crrt_angle in list_angles_around:
#                 angle_rad = np.pi * crrt_angle / 180.0
#                 list_position_probes.append(
#                     np.array(
#                         [
#                             crrt_radius * math.cos(angle_rad) + cylinder_coordinates[0],
#                             crrt_radius * math.sin(angle_rad) + cylinder_coordinates[1],
#                             crrt_z,
#                         ]
#                     )
#                 )
#
#         for crrt_x in positions_probes_for_grid_x:
#             for crrt_y in positions_probes_for_grid_y:
#                 list_position_probes.append(
#                     np.array(
#                         [
#                             crrt_x + cylinder_coordinates[0],
#                             crrt_y + cylinder_coordinates[1],
#                             crrt_z,
#                         ]
#                     )
#                 )
#
#     output_params = {"locations": list_position_probes, "probe_type": "pressure"}


#####################################################################################
####   These are the probe positions from Jean paper with only 5 probes in 3D    ####
#####################################################################################

# if probes_location == 4:
#     list_radius_around = [radius + 0.2]
#     list_angles_around = np.arange(90, 271, 180)
#     list_in_z = [1.0, 2.5, 3.0]
#
#     for crrt_radius in list_radius_around:
#         for crrt_angle in list_angles_around:
#             for crrt_z in list_in_z:
#                 angle_rad = np.pi * crrt_angle / 180.0
#                 list_position_probes.append(
#                     np.array(
#                         [
#                             crrt_radius * math.cos(angle_rad) + cylinder_coordinates[0],
#                             crrt_radius * math.sin(angle_rad) + cylinder_coordinates[1],
#                             crrt_z,
#                         ]
#                     )
#                 )
#
#     positions_probes_for_grid_x = [0.85]
#     positions_probes_for_grid_y = [-1, 0.0, 1]
#
#     for crrt_x in positions_probes_for_grid_x:
#         for crrt_y in positions_probes_for_grid_y:
#             for crrt_z in list_in_z:
#                 list_position_probes.append(
#                     np.array(
#                         [
#                             crrt_x + cylinder_coordinates[0],
#                             crrt_y + cylinder_coordinates[1],
#                             crrt_z,
#                         ]
#                     )
#                 )
#
#     len_left_positions_probes = 0
#     len_cylinder_position_probes = 2
#     len_karman_position_probes = 3
#
#     cylinder_range_min = len_left_positions_probes + 1
#     cylinder_range_max = len_left_positions_probes + len_cylinder_position_probes
#
#     karman_range_min = cylinder_range_max + 1
#     karman_range_max = cylinder_range_max + len_karman_position_probes
#     tag_probs = {
#         "left": [1, len_left_positions_probes],
#         "cylinder": [cylinder_range_min, cylinder_range_max],
#         "karman": [karman_range_min, karman_range_max],
#     }
#
#     output_params = {
#         "locations": list_position_probes,
#         "tag_probs": tag_probs,
#         "probe_type": "pressure",
#     }

###############################################################################
######################################################################

simulation_params: Dict[str, Any] = {
    "simulation_duration": baseline_duration,  # the time the simulation is in permanant regime
    "simulation_timeframe": [
        baseline_time_start,
        baseline_time_start + baseline_duration,
    ],
    "delta_t_smooth": delta_t_smooth,
    "delta_t_converge": delta_t_converge,
    "smooth_func": smooth_func,
    # 'dt': dt,
    "mu": mu,
    "rho": rho,
    "post_process_steps": post_process_steps,
    "u_tau": u_tau,
}

# Variational input
variational_input: Dict[str, Any] = {
    "filename": "channel",  # basename
    "bound": [
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
    ],  # Boundaries to postprocess. Comma separated
    "porous": False,  # Variational boundaries to postprocess. Comma separated
    "density": rho,  # Fluid density #TODO: repeated parameter - Pol
    "veloc": 1,  # average velocity of a parabolic inflow
    # "scale_area": frontal_area,  # Projected frontal area. Scale if it is need it
    "d": 0,  # Distance for momentum calculation
    "time": -time_avg,  # Time to average. Negative to average from last time, 0 to average over total time, positive to postprocess form initial time
    "initial_time": None,  # initial_time required only if positive "time averaging" used
    "rotation_vector": [
        1,
        0,
        0,
    ],  # Vector rotation in case rotation between axis is needed
    # "phi": 90,  # Rotation angle
    # "D_exp": 1.42,  # SCHAFER Experimental Drag
    # "S_exp": 0,  # Experimental Side
    # "L_exp": 1.01,  # SCHAFER Experimental Lift
    # "R_exp": 0,  # Experimental Roll
    # "P_exp": 0,  # Experimental Pitch
    # "Y_exp": 0,  # Experimental yaw
}

# Normalization factors
# TODO: @pietero @canordq Get actual values for these!!! - Pieter
norm_factors: Dict[str, float] = {
    "PRESS": norm_press,  # original norm_press value for backwards compatibility
    "VELOX": 10.0,  # example value, replace with actual value
    "VELOY": 10.0,  # example value, replace with actual value
    "VELOZ": 10.0,  # example value, replace with actual value
}

# Optimization
optimization_params: Dict[str, Any] = {
    "num_steps_in_pressure_history": 1,
    "min_value_jet_MFR": -1,
    "max_value_jet_MFR": 1,
    "norm_Q": norm_Q,  # (0.088/2)/5 asa said in papers, limited Q for no momentum or discontinuities in the CFD solver
    "norm_reward": norm_reward,  # like PRESS, try to be between -1,1
    "penal_cl": penal_cl,  # avoid asymmetrical strategies
    "alpha_rew": alpha_rew,  # weights for local vs global reward
    "norm_press": norm_press,  # force the PRESS comms to be between -1,1 (seggestion for better performance DRL)
    "offset_reward": offset_reward,  # start the reward from 0
    "avg_TIME_PROBE": 0.25,  # % del Tk to avg probe comms
    "zero_net_Qs": False,
    "random_start": False,
}

# TODO: @pietero @canordq Update for channel parameters!! (if necessary??) - Pieter
history_parameters: Dict[str, List[Union[float, int]]] = {
    "drag": [],
    "lift": [],
    "drag_GLOBAL": [],
    "lift_GLOBAL": [],
    "time": [],
    "episode_number": [],
}

inspection_params: Dict[str, Any] = {
    "plot": False,  # TODO: inspection_params is never used - Pol
    "step": 50,
    "dump": 100,
    "range_pressure_plot": [-2.0, 1],
    "range_drag_plot": [-0.175, -0.13],
    "range_lift_plot": [-0.2, +0.2],
    "line_drag": -0.1595,
    "line_lift": 0,
    "show_all_at_reset": True,
    "single_run": False,
}

reward_params: Dict[str, str] = {
    "reward_function": reward_function,
    "neighbor_state": str(neighbor_state),
    "Lx": str(Lx),
    "Ly": str(Ly),
    "Lz": str(Lz),
    "H": str(h_qevent_sensitivity),
    "nx_Qs": str(nx_Qs),
    "nz_Qs": str(nz_Qs),
    "u_tau": str(u_tau),
}
