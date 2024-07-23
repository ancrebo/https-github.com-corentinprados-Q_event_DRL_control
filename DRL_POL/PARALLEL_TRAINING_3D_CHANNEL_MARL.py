#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Parallel training launcher
#
# Pol Suarez, Francisco Alcantara
# 21/02/2022

# TODO: ANY CHANNEL-SPECIFIC EDITS

from __future__ import print_function, division

import os
import sys
import copy as cp
import time
from typing import List
import argparse

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from env_utils import (
    run_subprocess,
    generate_node_list,
    read_node_list,
    detect_system,
    agent_index_1d_to_2d,
    agent_index_2d_to_1d,
)
from configuration import ALYA_ULTCL

from logging_config import configure_logger

# Set up logger
logger = configure_logger(__name__, default_level="WARNING")

# Parser for command line arguments
# example use: `python3 PARALLEL_TRAINING.py --case cylinder_2D`
parser = argparse.ArgumentParser(
    description="Run parallel training for 3D channel MARL.",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--case",
    type=str,
    required=True,
    help=(
        "Specify the training case to run. Options:\n"
        "  - cylinder_2D\n"
        "  - airfoil_2D\n"
        "  - cylinder_3D\n"
        "  - cylinder_3D_WS_test\n"
        "  - channel_3D_MARL"
    ),
)
args = parser.parse_args()

# Run the cleaner
run_subprocess("./", ALYA_ULTCL, "", preprocess=True)

# Set up which case to run
training_case = args.case
logger.info("PARALLEL_TRAINING: Running case: %s", training_case)
logger.debug("PARALLEL_TRAINING: Cleaning up old files...")
run_subprocess(
    "./", "rm -f", "parameters.py", preprocess=True
)  # Ensure deleting old parameters
logger.debug("PARALLEL_TRAINING: Copying parameters for %s ...", training_case)
run_subprocess(
    "./",
    "ln -s",
    f"parameters/parameters_{training_case}.py parameters.py",
    preprocess=True,
)
logger.debug("PARALLEL_TRAINING: Copying case files for %s ...", training_case)
run_subprocess("alya_files", "cp -r", f"case_{training_case} case", preprocess=True)

from Env3D_MARL_channel import Environment

# Import universal parameters
from parameters import (
    nb_inv_per_CFD,
    sync_episodes,
    batch_size,
    nb_actuations,
    num_episodes,
    simu_name,
    run_baseline,
)

# Import system-specific parameters
if detect_system() == "LOCAL":
    logger.debug("PARALLEL_TRAINING: Detected LOCAL system.")
    from parameters import (
        num_servers_ws as num_servers,
        nb_proc_ws as nb_proc,
    )
else:
    logger.debug("PARALLEL_TRAINING: Detected SLURM system.")
    from parameters import (
        num_servers,
        nb_proc,
    )

# Import case-specific parameters
if training_case == "cylinder_3D_WS_test":
    from parameters import (
        nz_Qs,
    )
elif training_case == "channel_3D_MARL":
    from parameters import (
        nz_Qs,
        nx_Qs,
        nTotal_Qs,  # TODO: necessary or just nb_inv_per_CFD?
    )

from cr import cr_reset, cr_info, cr_report
import time

# Ensure the chronometer is reset
cr_reset()

## Run
initial_time = time.time()

# Generate the list of nodes
# TODO --- ADD NUM_CFD (MARL)
logger.debug("PARALLEL_TRAINING: Generating node list...")
generate_node_list(num_servers=num_servers, num_cores_server=nb_proc)
# TODO: check if this works in MN!
# TODO: Update to local nodelists with num_servers

# Read the list of nodes
nodelist = read_node_list()

# IMPORTANT: this environment base is needed to do the baseline, the main one
logger.debug("PARALLEL_TRAINING: Creating base environment...")
environment_base = Environment(simu_name=simu_name, node=nodelist[0])  # Baseline
logger.debug(
    "PARALLEL_TRAINING: Created base environment with ENV_ID %s",
    environment_base.ENV_ID,
)
# print(f"\nDEBUG: Environment Base ENV_ID: {environment_base.ENV_ID}\n")

if run_baseline:
    logger.info("PARALLEL_TRAINING: `run_baseline` is TRUE, running baseline...")
    run_subprocess("alya_files", "rm -rf", "baseline")  # Ensure deleting old parameters
    environment_base.run_baseline(True)

network = [dict(type="dense", size=512), dict(type="dense", size=512)]

logger.debug("PARALLEL_TRAINING: Creating agent...")
agent = Agent.create(
    # Agent + Environment
    agent="ppo",
    environment=environment_base,
    max_episode_timesteps=nb_actuations,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=batch_size,
    learning_rate=1e-3,
    subsampling_fraction=0.2,
    multi_step=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2,
    predict_terminal_values=True,
    # TODO: gae_lambda=0.97 doesn't currently exist
    # Critic
    # TODO -- memory ?
    baseline=network,
    baseline_optimizer=dict(
        type="multi_step", num_steps=5, optimizer=dict(type="adam", learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    # TODO -- change parallel interaction_> how many calls to NN ?
    parallel_interactions=num_servers * nb_inv_per_CFD,
    # TensorFlow etc
    saver=dict(
        directory=os.path.join(os.getcwd(), "saver_data"),
        frequency=1,
        max_checkpoints=1,
    ),  # parallel_interactions=number_servers,
    summarizer=dict(
        directory="data/summaries",
        # list of labels, or 'all'
        summaries=["entropy", "kl-divergence", "loss", "reward", "update-norm"],
    ),
)
logger.debug("PARALLEL_TRAINING: Created agent.")


def split(
    environment: Environment, np: int, nz_Qs: int, nx_Qs: int = 1
) -> List[Environment]:
    """
    Creates individual (local) agent environments for parallel training, per node.

    Parameters:
        environment (Environment): The base environment to be copied. From `Env3D_MARL_channel.py`.
        np (int): The number of the parallel environment (node).
        nz_Qs (int): The number of sections in the z direction.
        nx_Qs (int, optional): The number of sections in the x direction. Defaults to 1.

    Returns:
        List[Environment]: A list of local environments in a single node with updated ENV_IDs for parallel training.
    """
    list_inv_envs = []
    for i in range(nx_Qs):
        for j in range(nz_Qs):
            env = cp.copy(environment)
            env.ENV_ID = [
                np,
                agent_index_2d_to_1d(i, j, nz_Qs),  # Convert 2D agent index to 1D
            ]
            env.host = f"environment{np}"
            list_inv_envs.append(env)
    return list_inv_envs


# def split_old(environment, np):  # called 1 time in PARALLEL_TRAINING.py
#     # np:= number of the parallel environment. e.g. between [1,4] for 4 parallel CFDenvironments
#     # (ni, nj):= env_ID[1]:= 'position'/'ID-card' of the 'pseudo-parallel' invariant environment (a tuple in 3d, in which we have a grid of actuators. A scalar in 2D, in which we have a line of actuators)
#     # nb_inv_envs:= total number of 'pseudo-parallel' invariant environments. e.g. 10
#     """input: one of the parallel environments (np); output: a list of nb_inv_envs invariant environments identical to np. Their ID card: (np, ni)"""
#     list_inv_envs = []
#     for j in range(nz_Qs):
#         env = cp.copy(environment)
#         env.ENV_ID = [
#             np,
#             (j + 1),
#         ]  # Adjust for two dimensions? or translate two to one dimension (n, m) -> (j)
#         env.host = f"environment{np}"
#         list_inv_envs.append(env)
#     return list_inv_envs

logger.info("PARALLEL_TRAINING: nodelist: %s", nodelist)
# print("Here is the nodelist: ", nodelist)

# here the array of environments is defined, will be n-1 host (the 1st one is MASTER) #TODO: assign more nodes to an environment
logger.info(
    "PARALLEL_TRAINING: Creating %d parallel environments for %d separate CFD environments...",
    num_servers,
    num_servers,
)
parallel_environments = [
    Environment(
        simu_name=simu_name,
        ENV_ID=[i, 0],
        host=f"environment{i + 1}",
        node=nodelist[i + 1],
    )
    for i in range(num_servers)
]

if "nx_Qs" not in globals():
    nx_Qs = 1

logger.info(
    "PARALLEL_TRAINING: Splitting environments into %d local environments...",
    nx_Qs * nz_Qs,
)
environments = [
    split(parallel_environments[i], i + 1, nx_Qs, nz_Qs)[j]
    for i in range(num_servers)
    for j in range(nx_Qs * nz_Qs)  # Adjusted for 2D grid
]

for env in environments:
    env_id_2d = agent_index_1d_to_2d(env.ENV_ID[1], nz_Qs)
    logger.info(
        "Verif:   Host: %s ID: %s Agent 1D Index: %s 2D Index: %s",
        env.host,
        env.ENV_ID,
        env.ENV_ID[1],
        env_id_2d,
    )
    # print(
    #     f"Verif : Host: {env.host:<20} ID: {str(env.ENV_ID):<10} Agent 1D Index: {env.ENV_ID[1]:<5} 2D Index: {str(env_id_2d):<10}"
    # )

time.sleep(1.0)

# environments = [Environment(simu_name=simu_name, ENV_ID=i, host="environment{}".format(i+1),node=nodelist[i+1]) for i in range(num_servers)]
for e in environments:
    e.start()
    time.sleep(2)

# start all environments at the same time
# TODO: needs a toy case for the start class a 'light' baseline for everyone which is useless
logger.info("PARALLEL_TRAINING: Starting all environments at the same time...")
runner = Runner(agent=agent, environments=environments, remote="multiprocessing")

logger.info("PARALLEL_TRAINING: Running training for %d episodes...", num_episodes)
# now start the episodes and sync_episodes is very useful to update the DANN efficiently
runner.run(num_episodes=num_episodes, sync_episodes=sync_episodes)
runner.close()
logger.info("PARALLEL_TRAINING: Training completed!!!")

logger.info("PARALLEL_TRAINING: Saving model data in model-numpy format...")
# saving all the model data in model-numpy format
agent.save(
    directory=os.path.join(os.getcwd(), "model-numpy"),
    format="numpy",
    append="episodes",
)

logger.info("PARALLEL_TRAINING: Closing agent...")
agent.close()

end_time = time.time()

logger.info("PARALLEL_TRAINING: Start at: %s", initial_time)
logger.info("PARALLEL_TRAINING: End at: %s", end_time)
logger.info("PARALLEL_TRAINING: Done in: %s", end_time - initial_time)
# print(
#     f"DRL simulation :\nStart at : {initial_time}.\nEnd at {end_time}\nDone in : {end_time - initial_time}"
# )

logger.info("PARALLEL_TRAINING: Creating CR report...")
cr_info()
cr_report("DRL_TRAINING.csv")
