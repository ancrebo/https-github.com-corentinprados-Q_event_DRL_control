"""
PARALLEL_TRAINING_3D_CHANNEL_MARL.py
====================================

DEEP REINFORCEMENT LEARNING WITH ALYA
-------------------------------------

PARALLEL TRAINING LAUNCHER
--------------------------
This script launches the parallel training for 3D channel Multi-Agent
Reinforcement Learning (MARL) with ALYA. It initializes the necessary
environments, agents, and training configurations and starts the training
process.

The script performs the following steps:
1. Cleans up old files and sets up the specified training case.
2. Detects the system configuration (local or SLURM).
3. Creates a base environment for baseline calculations and parallel
   environments for each agent.
4. Initializes the TensorForce agent with specified network configurations.
5. Splits the global environments into local environments for each agent.
6. Starts all environments and runs the training for the specified number of
   episodes.
7. Saves the model data in model-numpy format and generates a CR report.

Examples
--------
Example usage:
python PARALLEL_TRAINING_3D_CHANNEL_MARL.py --case channel_3D_MARL
python PARALLEL_TRAINING_3D_CHANNEL_MARL.py --case cylinder_WS_test

Authors
-------
- Pol Suarez
- Francisco Alcantara
- Pieter Orlandini

Version History
---------------
- Major update in February 2022.
- Major update for channel features in August 2024.
"""

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

from logging_config import (
    configure_logger,
    DEFAULT_LOGGING_LEVEL,
    clear_old_logs,
    DEFAULT_LOG_DIR,
)

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
parser.add_argument(
    "--clearlogs",
    type=bool,
    required=False,
    default=False,
    help="Clear the logs in the default folder"
    "before starting the training."
    "see `logging_config.py` for details.",
)
parser.add_argument(
    "--logdir",
    type=str,
    required=False,
    default=None,
    help="Specify the directory for the log files."
    "see `logging_config.py` for details.",
)

args = parser.parse_args()

# Custom log directory to be used in `logging_config.py` (and so in every module)
CUSTOM_LOG_DIR: str = args.logdir

# Set up logger
logger = configure_logger(
    "PARALLEL_TRAINING_3D_CHANNEL_MARL", default_level=DEFAULT_LOGGING_LEVEL
)

logger.info(
    "PARALLEL_TRAINING_3D_CHANNEL_MARL.py: Logging level set to %s\n", logger.level
)

# Run the cleaner
run_subprocess("./", ALYA_ULTCL, "", preprocess=True)

# Clean old logs if specified
if args.clearlogs:
    clear_old_logs(DEFAULT_LOG_DIR)

# Set up which case to run
training_case = args.case
logger.info("Running case: %s\n", training_case)
logger.debug("Cleaning up old files...\n")
run_subprocess(
    "./", "rm -f", "parameters.py", preprocess=True
)  # Ensure deleting old parameters
logger.debug("Copying parameters for %s ...\n", training_case)
run_subprocess(
    "./",
    "ln -s",
    f"parameters/parameters_{training_case}.py parameters.py",
    preprocess=True,
)
logger.debug("Copying case files for %s ...\n", training_case)
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
    logger.debug("Detected LOCAL system.\n")
    from parameters import (
        num_servers_ws as num_servers,
        nb_proc_ws as nb_proc,
    )
else:
    logger.debug("Detected SLURM system.\n")
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
        nTotal_Qs,  # TODO: necessary or just nb_inv_per_CFD? - Pieter
    )

from cr import cr_reset, cr_info, cr_report
import time

# Ensure the chronometer is reset
cr_reset()

## Run
initial_time = time.time()

# Generate the list of nodes
# TODO --- ADD NUM_CFD (MARL) - Pol (RESOLVED??)
logger.debug("Generating node list...\n")
generate_node_list(num_servers=num_servers, num_cores_server=nb_proc)
# TODO: check if this works in MN! - Pol (RESOLVED??)
# TODO: Update to local nodelists with num_servers - Pol (RESOLVED??)

# Read the list of nodes
nodelist = read_node_list()
logger.info("nodelist: %s\n", nodelist)

# IMPORTANT: this environment base is needed to do the baseline, the main one
logger.debug("Creating base environment...\n")
environment_base = Environment(simu_name=simu_name, node=nodelist[0])  # Baseline
logger.debug(
    "Created base environment with ENV_ID %s\n",
    environment_base.ENV_ID,
)

if run_baseline:
    logger.info("`run_baseline` is TRUE, running baseline...\n")
    run_subprocess("alya_files", "rm -rf", "baseline")  # Ensure deleting old parameters
    environment_base.run_baseline(True)
    logger.info("Baseline completed.\n")

network = [dict(type="dense", size=512), dict(type="dense", size=512)]

logger.debug("Creating agent...\n")
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
logger.debug("Created agent.\n")


def split_CFD_environment_to_invariant_environments(
    environment: Environment, np: int, nz_Qs: int, nx_Qs: int = 1
) -> List[Environment]:
    """
    Creates individual (local) agent environments for parallel training, per node.

    Parameters
    ----------
    environment : Environment
        The base environment to be copied. From `Env3D_MARL_channel.py`.
    np : int
        The number of the parallel environment (node).
    nz_Qs : int
        The number of sections in the z direction.
    nx_Qs : int, optional
        The number of sections in the x direction. Defaults to 1.

    Returns
    -------
    List[Environment]
        A list of local environments in a single node with updated ENV_IDs for parallel training.
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


logger.info("nodelist: %s\n", nodelist)

# here the array of environments is defined, will be n-1 host (the 1st one is MASTER) #TODO: assign more nodes to an environment
logger.info(
    "Creating %d parallel environments for separate CFD simulations...\n",
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
    "Splitting each global environment into %d local environments...\n",
    nx_Qs * nz_Qs,
)
environments = [
    split_CFD_environment_to_invariant_environments(
        parallel_environments[i], i + 1, nx_Qs, nz_Qs
    )[j]
    for i in range(num_servers)
    for j in range(nx_Qs * nz_Qs)  # Adjusted for 2D grid
]

for env in environments:
    env_id_2d = agent_index_1d_to_2d(env.ENV_ID[1], nz_Qs)
    logger.info(
        "Verif:   Host: %s    ID: %s    Agent 1D Index: %s    2D Index: %s",
        env.host,
        env.ENV_ID,
        env.ENV_ID[1],
        env_id_2d,
    )

time.sleep(1.0)

# environments = [Environment(simu_name=simu_name, ENV_ID=i, host="environment{}".format(i+1),node=nodelist[i+1]) for i in range(num_servers)]
for e in environments:
    e.start()
    time.sleep(2)

# start all environments at the same time
# TODO: needs a toy case for the start class a 'light' baseline for everyone which is useless
logger.info("Starting all environments at the same time...\n")
runner = Runner(agent=agent, environments=environments, remote="multiprocessing")

logger.info("Running training for %d episodes...\n", num_episodes)
# now start the episodes and sync_episodes is very useful to update the DANN efficiently
runner.run(num_episodes=num_episodes, sync_episodes=sync_episodes)
runner.close()
logger.info("Training completed!!!\n\n\n")

logger.info("Saving model data in model-numpy format...\n")
# saving all the model data in model-numpy format
agent.save(
    directory=os.path.join(os.getcwd(), "model-numpy"),
    format="numpy",
    append="episodes",
)

logger.info("Closing agent...\n")
agent.close()

end_time = time.time()

logger.info("Start at: %s", initial_time)
logger.info("End at: %s", end_time)
logger.info("Done in: %s", end_time - initial_time)
# print(
#     f"DRL simulation :\nStart at : {initial_time}.\nEnd at {end_time}\nDone in : {end_time - initial_time}"
# )

logger.info("Creating CR report...\n")
cr_info()
cr_report("DRL_TRAINING.csv")
