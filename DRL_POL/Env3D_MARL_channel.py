# -*- coding: utf-8 -*-
"""
multi- agent VERSION 18/04/2024 

AUTHORS ->  POL

"""

###-----------------------------------------------------------------------------
## Import section

## IMPORT PYTHON LIBRARIES
import os, csv, numpy as np
import logging
import sys
import inspect
import shutil
import time
from typing import List, Tuple, Union, Any, Dict, Optional

# IMPORT TENSORFLOW
from tensorforce.environments import Environment

# IMPORT INTERNAL LIBRARIES
from configuration import (
    ALYA_BIN,
    ALYA_GMSH,
    ALYA_SETS,
    ALYA_CLEAN,
    ALYA_VTK,
    OVERSUBSCRIBE,
    DEBUG,
    USE_SLURM,
)
from parameters import (
    bool_restart,
    neighbor_state,
    actions_per_inv,
    nb_inv_per_CFD,
    nz_Qs,
    Qs_position_z,
    delta_Q_z,
    mem_per_srun,
    dimension,
    case,
    simulation_params,
    num_nodes_srun,
    reward_params,
    jets,
    norm_factors,  # added for normalization of multiple components
    optimization_params,
    output_params,
    history_parameters,
    nb_actuations,
    nb_actuations_deterministic,
)
from env_utils import (
    run_subprocess,
    detect_system,
    find_highest_timestep_file,
    copy_mpio2vtk_required_files,
    printDebug,
)

# Import system-specific parameters
if detect_system() == "LOCAL":
    from parameters import (
        num_servers_ws as num_servers,
        nb_proc_ws as nb_proc,
    )
else:
    from parameters import (
        num_servers,
        nb_proc,
    )
from alya import (
    write_case_file,
    write_physical_properties,
    write_time_interval,
    write_run_type,
    detect_last_timeinterval,
)

from extract_forces import compute_avg_lift_drag
from witness import read_last_wit, write_witness_file
from cr import cr_start, cr_stop

# from wrapper3D        import Wrapper
import copy as cp

from logging_config import configure_env_logger

# Set up logger
primary_logger, file_only_logger = configure_env_logger()

primary_logger.info(
    "%s.py: Primary logging level set to %s\n", __name__, primary_logger.level
)
file_only_logger.info(
    "%s.py: File-only logging level set to %s\n", __name__, file_only_logger.level
)

###-------------------------------------------------------------------------###
###-------------------------------------------------------------------------###


### Environment definition - this is the LOCAL ENVIRONMENT of each individual agent
class Environment(Environment):

    ###---------------------------------------------------------------------###
    ###---------------------------------------------------------------------###

    ## Initialization of the environment
    ## only one time in multienvironment # TODO: @pietero check if this is correct - Pieter
    def __init__(
        self,
        simu_name: str,
        number_steps_execution: int = 1,
        continue_training: bool = False,
        deterministic: bool = False,
        ENV_ID: Optional[List[int]] = None,
        host: str = "",
        node: Union[str, None] = None,
        check_id: bool = False,
    ):

        if ENV_ID is None:
            ENV_ID = [-1, -1]
        primary_logger.debug("ENV_ID %s: Env3D.init: Initialization", ENV_ID)

        cr_start("ENV.init", 0)

        self.simu_name: str = simu_name
        self.case: str = case
        self.ENV_ID: List[int] = ENV_ID
        self.host: str = f"enviroment{self.ENV_ID[0]}"
        self.nodelist: Union[str, None] = node
        # self.nodelist     = [n for n in node.split(',')]
        self.do_baseline: bool = (
            True  # This parameter was being overwritten so it is no point to have it optional
        )
        self.action_count: int = 0
        self.check_id: bool = check_id
        self.dimension: int = dimension

        self.number_steps_execution: int = number_steps_execution
        self.reward_function: str = reward_params["reward_function"]
        self.reward_params: Dict[str, str] = reward_params
        self.output_params: Dict[str, Any] = output_params
        self.norm_factors: Dict[str, float] = norm_factors
        self.optimization_params: Dict[str, Union[int, float]] = optimization_params
        self.Jets: Dict[str, Any] = jets
        self.n_jets: int = len(jets)
        self.nz_Qs: int = nz_Qs
        self.Qs_position_z: List[float] = Qs_position_z
        self.delta_Q_z: float = delta_Q_z
        if self.case == "channel":
            from parameters import nx_Qs, Qs_position_x, delta_Q_x

            primary_logger.debug(
                "ENV_ID %s: Env3D.init: Channel-specific parameters imported",
                ENV_ID,
            )
            self.nx_Qs: int = nx_Qs
            self.Qs_position_x: List[float] = Qs_position_x
            self.delta_Q_x: float = delta_Q_x
        self.actions_per_inv: int = actions_per_inv
        self.nb_inv_per_CFD: int = nb_inv_per_CFD
        self.bound_inv: int = 6 + self.ENV_ID[1]  # 6 is from ALYA boundary code ??
        self.neighbor_state: bool = neighbor_state

        self.probes_values_global: np.ndarray = np.ndarray([])
        self.probes_values_global_dict: Dict[str, np.ndarray] = {}

        self.simulation_timeframe: List[float] = simulation_params[
            "simulation_timeframe"
        ]
        self.last_time: float = round(self.simulation_timeframe[1], 3)
        self.delta_t_smooth: float = simulation_params["delta_t_smooth"]
        self.smooth_func: str = simulation_params["smooth_func"]

        self.previous_action_global: np.ndarray = np.zeros(self.nb_inv_per_CFD)
        self.action_global: np.ndarray = np.zeros(self.nb_inv_per_CFD)

        if self.case == "cylinder":
            self.action: np.ndarray = np.zeros(self.actions_per_inv * 2)
        elif self.case == "channel":
            self.action: np.ndarray = np.zeros(self.actions_per_inv)

        # postprocess values
        self.history_parameters: Dict[str, Any] = history_parameters

        name: str = "output.csv"
        # if we start from other episode already done
        last_row: Union[List[str], None] = None
        if os.path.exists("saved_models/" + name):
            with open("saved_models/" + name, "r") as f:
                for row in reversed(
                    list(csv.reader(f, delimiter=";", lineterminator="\n"))
                ):
                    last_row = row
                    break
        if not last_row is None:
            self.episode_number: int = int(last_row[0])
            self.last_episode_number: int = int(last_row[0])
        else:
            self.last_episode_number: int = 0
            self.episode_number: int = 0

        # these are for cylinder case
        if self.case == "cylinder":
            self.episode_drags: np.ndarray = np.array([])
            self.episode_lifts: np.ndarray = np.array([])
            self.episode_drags_GLOBAL: np.ndarray = np.array([])
            self.episode_lifts_GLOBAL: np.ndarray = np.array([])

        # need to get some for two boundary case

        self.continue_training: bool = continue_training
        self.deterministic: bool = deterministic

        if self.deterministic:
            self.host: str = "deterministic"

        # check if the actual environment has to run cfd or not
        # quick way --> if the 2nd component of the ENVID[] is 1...
        primary_logger.debug(
            "ENV_ID %s: Env3D.init: Calling parent class constructor (TENSORFORCE)",
            ENV_ID,
        )
        # Call parent class constructor
        super().__init__()
        primary_logger.debug(
            "ENV_ID %s: Env3D.init: Parent class constructor called (TENSORFORCE)\n",
            ENV_ID,
        )
        cr_stop("ENV.init", 0)

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def log(self, level, message, *args, **kwargs):
        if self.ENV_ID[1] == 1:
            # Use primary logger to log to both console and file
            if primary_logger.isEnabledFor(level):
                primary_logger.log(level, message, *args, **kwargs)
        else:
            # Use file-only logger to log only to file
            if file_only_logger.isEnabledFor(level):
                file_only_logger.log(level, message, *args, **kwargs)

    def start(self) -> None:
        cr_start("ENV.start", 0)
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.start: Beginning `start` method...",
            self.ENV_ID,
        )
        # Get the new avg drag and lift and SAVE
        temp_id: str = (
            "{}".format(self.host)
            if self.continue_training == True or self.deterministic == True
            else ""
        )

        if self.case == "cylinder":
            if self.continue_training:
                averages: Dict[str, float] = {
                    "drag": 0.0,
                    "lift": 0.0,
                    "drag_GLOBAL": 0.0,
                    "lift_GLOBAL": 0.0,
                }
            else:
                # Compute average drag and lift
                averages: Dict[str, float] = {}

                averages["drag"], averages["lift"] = compute_avg_lift_drag(
                    self.episode_number, cpuid=temp_id, nb_inv=self.ENV_ID[1]
                )  # NOTE: add invariant code! not the same BC

                averages["drag_GLOBAL"], averages["lift_GLOBAL"] = (
                    compute_avg_lift_drag(
                        self.episode_number,
                        cpuid=temp_id,
                        nb_inv=self.nb_inv_per_CFD,
                        global_rew=True,
                    )
                )  # NOTE: add invariant code! not the same BC

            # Update history parameters dynamically
            for key in self.history_parameters.keys():
                if key not in ["time", "episode_number"]:
                    self.history_parameters[key].append(averages.get(key, None))

            self.history_parameters["time"].append(self.last_time)
            self.history_parameters["episode_number"].append(self.episode_number)

            # Save history parameters using the new method
            self.save_history_parameters_all(nb_actuations)

            # Print results dynamically
            results = "\n".join(
                f"\tAverage {key}: {value}" for key, value in averages.items()
            )
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.start: Cylinder Results:\n%s",
                self.ENV_ID,
                results,
            )
            # print(f"Results : \n{results}")

        elif self.case == "channel":
            if self.continue_training:
                # TODO: @pietero update with appropriate values! - Pieter
                pass
            else:
                pass

        if case == "cylinder":
            # Initialize action
            self.action = np.zeros(self.actions_per_inv * 2)  #
        elif case == "channel":
            self.action = np.zeros(self.actions_per_inv)

        self.check_id = True  # check if the folder with cpuid number is created
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.start: Finished `start` method!\n",
            self.ENV_ID,
        )
        cr_stop("ENV.start", 0)

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def clean(self, full: bool = False) -> None:
        cr_start("ENV.clean", 0)
        self.log(
            logging.DEBUG, "ENV_ID %s: Env3D.clean: Beginning `clean`...", self.ENV_ID
        )
        if full:
            # saved_models contains the .csv of all cd and cl agt the end of each episode
            if os.path.exists("saved_models"):
                run_subprocess("./", "rm -rf", "saved_models")
            # Best model at the end of each episode
            if os.path.exists("best_model"):
                run_subprocess("./", "rm -rf", "best_model")
        # si no hemos acabado el episodio, continuamos sumando actions
        self.action_count = 1
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.clean: Finished `clean` method.\n",
            self.ENV_ID,
        )
        cr_stop("ENV.clean", 0)

    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------

    def create_mesh(
        self,
    ) -> None:  # TODO: Flag para que no tenga que volver a hacer la malla
        cr_start("ENV.mesh", 0)
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.create_mesh: Beginning `create_mesh`...",
            self.ENV_ID,
        )
        if self.do_baseline == True:
            if self.dimension == 2:
                run_subprocess(
                    "gmsh", "python3", "geo_file_maker.py"
                )  # TODO: this should be a library and be called within this function
                run_subprocess("alya_files/case/mesh", ALYA_GMSH, "-2 %s" % self.case)
            for jet in self.Jets.values():
                jet.update_file("alya_files/case")
            write_witness_file("alya_files/case", output_params["locations"])
            run_subprocess("alya_files/case", ALYA_CLEAN, "")
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.create_mesh: Finished `create_mesh`\n",
            self.ENV_ID,
        )
        cr_stop("ENV.mesh", 0)

    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------

    def run_baseline(self, clean: bool = True) -> None:
        cr_start("ENV.run_baseline", 0)
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.run_baseline: Beginning `run_baseline`...",
            self.ENV_ID,
        )
        # Do a full clean
        if clean:
            self.clean(True)
        # Create the mesh
        self.create_mesh()
        # Setup alya files
        run_subprocess(
            "alya_files", "cp -r", "case baseline"
        )  # TODO: substitute for correct case
        # run_subprocess('alya_files/baseline/mesh','mv','*mpio.bin ..')
        if self.dimension == 2:
            run_subprocess(
                "alya_files/baseline",
                "python3",
                "initialCondition.py {0} 1. 0.".format(self.case),
            )
        # Run alya
        self.log(
            logging.INFO,
            "ENV_ID %s: Env3D.run_baseline: \n\n----STARTING ALYA FOR BASELINE...----\n",
            self.ENV_ID,
        )

        self.run(which="reset")

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.run_baseline: Finished `run_baseline`\n",
            self.ENV_ID,
        )
        cr_stop("ENV.run_baseline", 0)

    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------

    def run(self, which: str) -> None:
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.run: Beginning `run` method...",
            self.ENV_ID,
        )

        self.log(
            logging.INFO,
            "ENV_ID %s: Env3D.run: time interval: [%f, %f]",
            self.ENV_ID,
            self.simulation_timeframe[0],
            self.simulation_timeframe[1],
        )
        # print("Simulation on : ", self.simulation_timeframe)

        logssets = os.path.join("logs", "log_sets.log")
        if which == "reset":
            self.log(
                logging.INFO, "ENV_ID %s: Env3D.run: Starting reset...", self.ENV_ID
            )
            # Baseline run
            if self.do_baseline == True:  # necessary? better?
                self.log(
                    logging.INFO,
                    "ENV_ID %s: Env3D.run: Starting baseline run preparation...",
                    self.ENV_ID,
                )
                # printDebug(
                #     "\n \n Alya has started the baseline run! (Env2D-->run-->reset)\n \n"
                # )
                filepath = os.path.join("alya_files", "baseline")

                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.run: Writing case %s.dat file...",
                    self.ENV_ID,
                    self.case,
                )  # TODO: @pietero check if `self.case` is correct for this - Pieter
                write_case_file(filepath, self.case, self.simu_name)

                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.run: Writing run_type.dat file...",
                    self.ENV_ID,
                )
                write_run_type(filepath, "NONCONTI", freq=1000)

                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.run: Writing time_interval.dat file...",
                    self.ENV_ID,
                )
                write_time_interval(
                    filepath, self.simulation_timeframe[0], self.simulation_timeframe[1]
                )

                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.run: Writing physical_properties.dat file...",
                    self.ENV_ID,
                )
                write_physical_properties(
                    filepath, simulation_params["rho"], simulation_params["mu"]
                )

                # Run Alya
                casepath = os.path.join("alya_files", "baseline")
                logsrun = os.path.join(
                    "logs", "log_last_reset_run.log"
                )  # TODO: @pietero can this work with logging? - Pieter
                # Run subprocess
                if self.dimension == 2:
                    self.log(
                        logging.DEBUG,
                        "ENV_ID %s: Env3D.run: 2D: Creating logs folder...",
                        self.ENV_ID,
                    )
                    run_subprocess(casepath, "mkdir -p", "logs")  # Create logs folder

                    self.log(
                        logging.INFO,
                        "ENV_ID %s: Env3D.run: 2D: \n\n----STARTING ALYA BASELINE RUN!!!...----\n",
                        self.ENV_ID,
                    )
                    run_subprocess(
                        casepath,
                        ALYA_BIN,
                        f"{case}",
                        nprocs=nb_proc,
                        oversubscribe=OVERSUBSCRIBE,
                        nodelist=self.nodelist,
                        log=logsrun,
                    )  # ,parallel=True)

                    self.log(
                        logging.DEBUG,
                        "ENV_ID %s: Env3D.run: 2D: Running ALYA sets...",
                        self.ENV_ID,
                    )  # TODO: @pieter is this correct/necessary? - Pieter
                    run_subprocess(
                        casepath,
                        ALYA_SETS,
                        f"{self.case}-boundary.nsi.set 3",
                        log=logssets,
                    )  # TODO: Boundary hardcoded!!

                if self.dimension == 3:
                    self.log(
                        logging.DEBUG,
                        "ENV_ID %s: Env3D.run: 3D: Creating logs folder...",
                        self.ENV_ID,
                    )
                    run_subprocess(
                        casepath, "mkdir -p", "logs", preprocess=True
                    )  # Create logs folder

                    self.log(
                        logging.INFO,
                        "ENV_ID %s: Env3D.run: 3D: \n\n----STARTING ALYA BASELINE RUN!!!...----\n",
                        self.ENV_ID,
                    )
                    run_subprocess(
                        casepath,
                        ALYA_BIN,
                        f"{self.case}",
                        nprocs=nb_proc,
                        mem_per_srun=mem_per_srun,
                        num_nodes_srun=num_nodes_srun,
                        host=self.nodelist,
                        log=logsrun,
                    )

                    self.log(
                        logging.DEBUG,
                        "ENV_ID %s: Env3D.run: 3D: Running ALYA sets...",
                        self.ENV_ID,
                    )  # TODO: @pieter is this correct/necessary? - Pieter
                    run_subprocess(
                        casepath,
                        ALYA_SETS,
                        f"{self.case}-boundary.nsi.set 3",
                        log=logssets,
                        preprocess=True,
                    )
            self.log(
                logging.DEBUG, "ENV_ID %s: Env3D.run: ALYA Baseline done!", self.ENV_ID
            )
            self.do_baseline = False  # Baseline done, no need to redo it

        elif which == "execute":
            # Actions run
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.run: Starting to execute a new action!!!",
                self.ENV_ID,
            )
            # printDebug(
            #     "\n \n Alya has started executing an action! (Env3D-->run-->execute) \n \n"
            # )
            cr_start("ENV.run_actions", 0)
            filepath = os.path.join(
                "alya_files",
                f"{self.host}",
                f"{self.ENV_ID[1]}",
                f"EP_{self.episode_number}",
            )

            filepath_flag_sync = os.path.join(
                "alya_files",
                f"{self.host}",
                "1",
                f"EP_{self.episode_number}",
                "flags_MARL",
            )
            action_end_flag_path = os.path.join(
                filepath_flag_sync, f"action_end_flag_{self.action_count}"
            )
            time.sleep(0.1)

            if self.ENV_ID[1] == 1:
                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.run: Starting 'main' environment tasks...",
                    self.ENV_ID,
                )

                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.run: Writing run_type.dat file...",
                    self.ENV_ID,
                )
                write_run_type(filepath, "CONTI", freq=1000)

                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.run: Writing time_interval.dat file...",
                    self.ENV_ID,
                )
                write_time_interval(
                    filepath, self.simulation_timeframe[0], self.simulation_timeframe[1]
                )

                casepath = os.path.join(
                    "alya_files",
                    f"{self.host}",
                    f"{self.ENV_ID[1]}",
                    f"EP_{self.episode_number}",
                )
                logsrun = os.path.join(
                    "logs",
                    (
                        "log_last_execute_run.log"
                        if not DEBUG
                        else f"log_execute_run_{self.action_count}.log"
                    ),
                )

                # Run subprocess
                if self.dimension == 2:
                    self.log(
                        logging.DEBUG,
                        "ENV_ID %s: Env3D.run: 2D: Creating logs folder...",
                        self.ENV_ID,
                    )
                    run_subprocess(casepath, "mkdir -p", "logs")  # Create logs folder

                    self.log(
                        logging.INFO,
                        "ENV_ID %s: Env3D.run: 2D: \n\n----STARTING ALYA RUN WITH UPDATED ACTIONS!!!----\n"
                        "-----------------Action #%d\n",
                        self.ENV_ID,
                        self.action_count,
                    )
                    run_subprocess(
                        casepath,
                        ALYA_BIN,
                        f"{self.case}",
                        nprocs=nb_proc,
                        oversubscribe=OVERSUBSCRIBE,
                        nodelist=self.nodelist,
                        log=logsrun,
                    )  # ,parallel=True)

                    self.log(
                        logging.DEBUG,
                        "ENV_ID %s: Env3D.run: 2D: Updating ALYA sets...",
                        self.ENV_ID,
                    )
                    run_subprocess(
                        casepath,
                        ALYA_SETS,
                        f"{self.case}-boundary.nsi.set 3",
                        log=logssets,
                    )  # TODO: Boundary hardcoded!!

                if self.dimension == 3:
                    self.log(
                        logging.DEBUG,
                        "ENV_ID %s: Env3D.run: 3D: Creating logs folder...",
                        self.ENV_ID,
                    )
                    run_subprocess(
                        casepath, "mkdir -p", "logs", preprocess=True
                    )  # Create logs folder

                    self.log(
                        logging.INFO,
                        "ENV_ID %s: Env3D.run: 3D: \n\n----STARTING ALYA RUN WITH UPDATED ACTIONS!!!----\n"
                        "-----------------Action #%d\n",
                        self.ENV_ID,
                        self.action_count,
                    )
                    run_subprocess(
                        casepath,
                        ALYA_BIN,
                        f"{self.case}",
                        nprocs=nb_proc,
                        mem_per_srun=mem_per_srun,
                        num_nodes_srun=num_nodes_srun,
                        host=self.nodelist,
                        log=logsrun,
                    )

                    self.log(
                        logging.DEBUG,
                        "ENV_ID %s: Env3D.run: 3D: Updating ALYA sets...",
                        self.ENV_ID,
                    )
                    run_subprocess(
                        casepath,
                        ALYA_SETS,
                        f"{self.case}-boundary.nsi.set 3",
                        log=logssets,
                        preprocess=True,
                    )

                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.run: create directory as sync flag",
                    self.ENV_ID,
                )
                # CREATE A FILE THAT WORKS AS FLAG TO THE OTHERS ENVS
                run_subprocess(
                    filepath_flag_sync,
                    "mkdir ",
                    f"action_end_flag_{self.action_count}",
                )  # Create dir? not so elegant I think

                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.run: 'main' environment tasks done!",
                    self.ENV_ID,
                )

            else:
                self.log(
                    logging.INFO,
                    "ENV_ID %s: Env3D.run: Waiting for 'main' environment to finish tasks...\n",
                    self.ENV_ID,
                )
                count_wait = 1
                if not self.deterministic:
                    while not os.path.exists(action_end_flag_path) or not os.path.isdir(
                        action_end_flag_path
                    ):
                        if count_wait % 1000 == 0:
                            primary_logger.info(
                                "ENV_ID %s: Env3D.run: Waiting for action #%d ...",
                                self.ENV_ID,
                                self.action_count,
                            )
                            # print(
                            #     f"Inv: {self.ENV_ID} is waiting for the action #{self.action_count}"
                            # )
                        time.sleep(0.05)
                        count_wait += 1

                time.sleep(1)
                self.log(
                    logging.INFO,
                    "ENV_ID %s: Env3D.run: Actions are in sync!\n",
                    self.ENV_ID,
                )
                # print(f"Actions in {self.ENV_ID} are sync")

            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.run: Finished `run` method.\n",
                self.ENV_ID,
            )
            cr_stop("ENV.run_actions", 0)

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def save_history_parameters_all(
        self, nb_actuations: int, name: str = "output.csv"
    ) -> None:

        cr_start("ENV.save_history_parameters", 0)
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.save_history_parameters_all: Beginning method...",
            self.ENV_ID,
        )

        # Save at the end of every episode
        # all `history_parameters` except `time` and `episode_number`
        for key, value in self.history_parameters.items():
            if key not in ["time", "episode_number"]:
                setattr(
                    self,
                    f"episode_{key}",
                    np.append(getattr(self, f"episode_{key}"), value),
                )

        # Check if it is the end of the episode
        if self.action_count == nb_actuations or self.episode_number == 0:
            file = os.path.join("saved_models", name)

            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.save_history_parameters_all: Saving parameters in file %s...",
                self.ENV_ID,
                file,
            )
            # print(f"Action : saving history parameters in {file}")
            self.last_episode_number = self.episode_number

            # Calculate averages
            averages = {}
            for key, value in self.history_parameters.items():
                if key not in ["time", "episode_number"]:
                    averages[key] = np.mean(value[-1:])

            # Prepare the directory and file
            os.makedirs("saved_models", exist_ok=True)
            if not os.path.exists(file):
                with open(file, "w") as csv_file:
                    spam_writer = csv.writer(
                        csv_file, delimiter=";", lineterminator="\n"
                    )
                    spam_writer.writerow(
                        ["Episode"]
                        + [f"Avg{key.capitalize()}" for key in averages.keys()]
                    )
                    spam_writer.writerow(
                        [self.last_episode_number]
                        + [averages[key] for key in averages.keys()]
                    )
            else:
                with open(file, "a") as csv_file:  # Append to the file
                    spam_writer = csv.writer(
                        csv_file, delimiter=";", lineterminator="\n"
                    )
                    spam_writer.writerow(
                        [self.last_episode_number]
                        + [averages[key] for key in averages.keys()]
                    )

            # Reset the episode parameters
            for key in self.history_parameters.keys():
                if key not in ["time", "episode_number"]:
                    setattr(self, f"episode_{key}", np.array([]))

            # Writes all the parameters in .csv
            if os.path.exists(file):
                run_subprocess("./", "cp -r", "saved_models best_model")
            else:
                if os.path.exists("saved_models/output.csv"):
                    if not os.path.exists("best_model"):
                        shutil.copytree("saved_models", "best_model")
                    else:
                        best_file = os.path.join("best_model", name)
                        last_iter = np.genfromtxt(file, skip_header=1, delimiter=";")[
                            -1, 1
                        ]
                        best_iter = np.genfromtxt(
                            best_file, skip_header=1, delimiter=";"
                        )[-1, 1]
                        if float(best_iter) < float(last_iter):
                            self.log(
                                logging.DEBUG,
                                "ENV_ID %s: Env3D.save_history_parameters_all: Best model updated",
                                self.ENV_ID,
                            )
                            # print("best_model updated")
                            run_subprocess("./", "rm -rf", "best_model")
                            run_subprocess("./", "cp -r", "saved_models best_model")

            # TODO: update what channel parameters are being saved? - Pieter @pietero
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.save_history_parameters_all: Saving parameters...",
                self.ENV_ID,
            )
            # printDebug(
            #     f"\n \n Saving parameters, [INSERT CHANNEL PARAMETERS HERE], which are the input of the neural network! (Env3D_MARL_channel-->execute-->save_history_parameters_all)\n \n"
            # )
            # print("Done.")
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.save_history_parameters_all: Done!\n",
                self.ENV_ID,
            )
        cr_stop("ENV.save_history_parameters", 0)

    def save_history_parameters(
        self, nb_actuations: int, name: str = "output.csv"
    ) -> None:
        # TODO: @pietero delete this!!! - Pieter

        cr_start("ENV.save_cd_cl", 0)

        # Save at the end of every episode
        self.episode_drags = np.append(
            self.episode_drags, self.history_parameters["drag"]
        )
        self.episode_lifts = np.append(
            self.episode_lifts, self.history_parameters["lift"]
        )
        self.episode_drags_GLOBAL = np.append(
            self.episode_drags_GLOBAL, self.history_parameters["drag_GLOBAL"]
        )
        self.episode_lifts_GLOBAL = np.append(
            self.episode_lifts_GLOBAL, self.history_parameters["lift_GLOBAL"]
        )

        if self.action_count == nb_actuations or self.episode_number == 0:
            file = os.path.join("saved_models", name)

            print(f"Task : saving history parameters in {file}")
            self.last_episode_number = self.episode_number

            avg_drag = np.mean(self.history_parameters["drag"][-1:])
            avg_lift = np.mean(self.history_parameters["lift"][-1:])
            avg_drag_GLOBAL = np.mean(self.history_parameters["drag_GLOBAL"][-1:])
            avg_lift_GLOBAL = np.mean(self.history_parameters["lift_GLOBAL"][-1:])

            os.makedirs("saved_models", exist_ok=True)
            if not os.path.exists("saved_models/" + name):
                with open(file, "w") as csv_file:
                    spam_writer = csv.writer(
                        csv_file, delimiter=";", lineterminator="\n"
                    )
                    spam_writer.writerow(
                        [
                            "Episode",
                            "AvgDrag",
                            "AvgLift",
                            "AvgDrag_GLOBAL",
                            "AvgLift_GLOBAL",
                        ]
                    )
                    spam_writer.writerow(
                        [
                            self.last_episode_number,
                            avg_drag,
                            avg_lift,
                            avg_drag_GLOBAL,
                            avg_lift_GLOBAL,
                        ]
                    )
            else:
                with open(file, "a") as csv_file:
                    spam_writer = csv.writer(
                        csv_file, delimiter=";", lineterminator="\n"
                    )
                    spam_writer.writerow(
                        [
                            self.last_episode_number,
                            avg_drag,
                            avg_lift,
                            avg_drag_GLOBAL,
                            avg_lift_GLOBAL,
                        ]
                    )
            self.episode_drags = np.array([])
            self.episode_lifts = np.array([])
            self.episode_drags_GLOBAL = np.array([])
            self.episode_lifts_GLOBAL = np.array([])

            # Writes all the cl and cd in .csv
            # IS THIS NECESSARY? I THINK WE DO NOT USE THE BEST MODEL
            if os.path.exists(file):
                run_subprocess("./", "cp -r", "saved_models best_model")
            else:
                if os.path.exists("saved_models/output.csv"):
                    if not os.path.exists("best_model"):
                        shutil.copytree("saved_models", "best_model")
                    else:
                        best_file = os.path.join("best_model", name)
                        last_iter = np.genfromtxt(file, skip_header=1, delimiter=";")[
                            -1, 1
                        ]
                        best_iter = np.genfromtxt(
                            best_file, skip_header=1, delimiter=";"
                        )[-1, 1]
                        if float(best_iter) < float(last_iter):
                            print("best_model updated")
                            run_subprocess("./", "rm -rf", "best_model")
                            run_subprocess("./", "cp -r", "saved_models best_model")
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.save_history_parameters: Saving parameters AVG DRAG & AVG LIFT",
                self.ENV_ID,
            )
            # printDebug(
            #     "\n \n Saving parameters, AVG DRAG & AVG LIFT, which are the input of the neural network! (Env2D-->execute-->save_history_parameters)\n \n"
            # )
            # print("Done.")
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.save_history_parameters: Done!\n",
                self.ENV_ID,
            )
        cr_stop("ENV.save_cd_cl", 0)

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def save_this_action(self) -> None:

        cr_start("ENV.save_action", 0)
        self.log(
            logging.INFO,
            "ENV_ID %s: Env3D.save_this_action: Saving action N° %d ...",
            self.ENV_ID,
            self.action_count,
        )
        # print("Saving a new action : N°", self.action_count)

        name_a = "output_actions.csv"
        if not os.path.exists("actions"):
            os.mkdir("actions")

        if not os.path.exists(f"actions/{self.host}"):
            os.mkdir(f"actions/{self.host}")

        if not os.path.exists(f"actions/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}"):
            os.mkdir(f"actions/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}")

        if not os.path.exists(
            f"actions/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}/ep_{self.episode_number}"
        ):
            os.mkdir(
                f"actions/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}/ep_{self.episode_number}"
            )

        path_a = f"actions/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}/ep_{self.episode_number}/"

        action_line = f"{self.action_count}"

        for i in range(self.actions_per_inv):
            action_line += f"; {self.action[i]}"

        if not os.path.exists(path_a + name_a):
            header_line = "Action"
            for i in range(self.actions_per_inv):
                header_line += f"; Jet_{i + 1}"

            with open(path_a + name_a, "w") as csv_file:
                spam_writer = csv.writer(csv_file, lineterminator="\n")
                spam_writer.writerow([header_line])
                spam_writer.writerow([action_line])
        else:
            with open(path_a + name_a, "a") as csv_file:
                spam_writer = csv.writer(csv_file, lineterminator="\n")
                spam_writer.writerow([action_line])

        self.log(
            logging.INFO, "ENV_ID %s: Env3D.save_this_action: Done!\n", self.ENV_ID
        )
        # print("Done.")
        cr_stop("ENV.save_action", 0)

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def save_reward(self, reward: float) -> None:
        # TODO: @pietero is the reward always input as a float? - Pieter

        cr_start("ENV.save_reward", 0)
        primary_logger.info(
            "ENV_ID %s: Env3D.save_reward: ENV_ID %s Saving reward N° %d: %f ...",
            self.ENV_ID,
            self.ENV_ID,
            self.action_count,
            reward,
        )
        # print("Saving a new reward: N°", reward)

        name_a = "output_rewards.csv"

        if not os.path.exists("rewards"):
            os.mkdir("rewards")

        if not os.path.exists(f"rewards/{self.host}"):
            os.mkdir(f"rewards/{self.host}")

        if not os.path.exists(f"rewards/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}"):
            os.mkdir(f"rewards/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}")

        if not os.path.exists(
            f"rewards/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}/ep_{self.episode_number}"
        ):
            os.mkdir(
                f"rewards/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}/ep_{self.episode_number}"
            )

        path_a = f"rewards/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}/ep_{self.episode_number}/"

        if not os.path.exists(path_a + name_a):
            with open(path_a + name_a, "w") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["Action", "Reward"])  # , "AvgRecircArea"])
                spam_writer.writerow([self.action_count, reward])
        else:
            with open(path_a + name_a, "a") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([self.action_count, reward])

        primary_logger.info(
            "ENV_ID %s: Env3D.save_reward: ENV_ID %s reward saved!\n",
            self.ENV_ID,
            self.ENV_ID,
        )
        # print("Done.")

        cr_stop("ENV.save_reward", 0)

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def save_final_reward(self, reward: float) -> None:

        primary_logger.info(
            "ENV_ID %s: Env3D.save_final_reward: Saving the last reward from episode %d: %f ...",
            self.ENV_ID,
            self.episode_number,
            reward,
        )
        # print(f"Saving the last reward from episode {self.episode_number}: {reward}")

        name_a = "output_final_rewards.csv"

        if not os.path.exists("final_rewards"):
            os.mkdir("final_rewards")

        if not os.path.exists(f"final_rewards/{self.host}"):
            os.mkdir(f"final_rewards/{self.host}")

        if not os.path.exists(
            f"final_rewards/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}"
        ):
            os.mkdir(f"final_rewards/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}")

        path_a = f"final_rewards/{self.host}/{self.ENV_ID[0]}_{self.ENV_ID[1]}/"

        if not os.path.exists(path_a + name_a):
            with open(path_a + name_a, "w") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["EPISODE", "REWARD"])  # , "AvgRecircArea"])
                spam_writer.writerow([self.episode_number, reward])
        else:
            with open(path_a + name_a, "a") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([self.episode_number, reward])

        primary_logger.info("ENV_ID %s: Env3D.save_final_reward: Done!\n", self.ENV_ID)
        # print("Done.")

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def save_comms_probes(
        self,
    ) -> None:  # TODO: This function is not used. May be eliminated

        primary_logger.info(
            "ENV_ID %s: Env3D.save_comms_probes: Saving probes inputs...", self.ENV_ID
        )
        # print(f"Saving probes inputs: N° {self.action_count}")

        name_a = "output_probes_comms.csv"

        if not os.path.exists("probes_comms"):
            os.mkdir("probes_comms")

        if not os.path.exists(f"probes_comms/ep_{self.episode_number}/"):
            os.mkdir(f"probes_comms/ep_{self.episode_number}/")

        path_a = f"probes_comms/ep_{self.episode_number}/"

        if not os.path.exists(path_a + name_a):
            with open(path_a + name_a, "w") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                array_acts = np.linspace(1, 24, dtype=int)
                spam_writer.writerow(
                    ["Action", *array_acts]
                )  # Unpack array_acts to write each value separately
                spam_writer.writerow([self.action_count, self.probes_values])
        else:
            with open(path_a + name_a, "a") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([self.action_count, self.probes_values])

        primary_logger.info("ENV_ID %s: Env3D.save_comms_probes: Done!\n", self.ENV_ID)
        # print("Done.")

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    ### AQUI DEBEMOS ANULAR EL RECUPERAR EL BASELINE SI YA EXISTE EL QUE TOCA
    def recover_start(self) -> None:

        cr_start("ENV.recover_start", 0)
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.recover_start: Beginning `recover_start` method...",
            self.ENV_ID,
        )

        runpath = "alya_files"

        # flag to sync the cp times... then the other pseudo can read properly witness
        # path example: /alya_files/environment1/1/EP_1/.
        filepath_flag_sync_cp = os.path.join(
            runpath, f"{self.host}", "1", f"EP_{self.episode_number}"
        )

        # lowcost mode --> CLEAN everytime olderfiles

        # TODO --> check the rm if we need to restart from the last episode! - Pol
        # TODO --> bool_restart_prevEP HAS TO BE 80-20 but not in parameters - Pol

        if not DEBUG and self.episode_number > 0:
            if not self.bool_restart_prevEP:
                runbin = "rm -rf"
                runargs = os.path.join(
                    f"{self.host}",
                    f"{self.ENV_ID[1]}",
                    f"EP_{self.episode_number}",  # was "EP_*"
                )
                # avoid checks in deterministic
                if not self.deterministic:
                    run_subprocess(runpath, runbin, runargs)

        if self.bool_restart_prevEP and self.episode_number > 1:
            runbin = "mv"
            # runargs = '%s %s' %(os.path.join('%s'%self.host,'%s'%self.ENV_ID[1],'EP_*'),os.path.join('%s'%self.host,'%s'%self.ENV_ID[1],'EP_%d'%self.episode_number))
        else:
            runbin = "cp -r"
            runargs = "baseline " + os.path.join(
                f"{self.host}", f"{self.ENV_ID[1]}", f"EP_{self.episode_number}"
            )
            logs = os.path.join(
                "baseline",
                "logs",
                f"log_restore_last_episode_{self.episode_number}.log",
            )
            run_subprocess(runpath, runbin, runargs, log=logs)

        run_subprocess(
            filepath_flag_sync_cp, "mkdir", "flags_MARL"
        )  # Create dir? not so elegant I think
        run_subprocess(
            os.path.join(filepath_flag_sync_cp, "flags_MARL"),
            "mkdir",
            "action_end_flag_cp",
        )  # Create dir? not so elegant I think

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.recover_start: Finished `recover_start` method.\n",
            self.ENV_ID,
        )
        cr_stop("ENV.recover_start", 0)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # create folder for each cpu id in parallel and folder per invariants inside
    def create_cpuID(self) -> None:
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.create_cpuID: Beginning `create_cpuID` method...",
            self.ENV_ID,
        )
        runpath = "alya_files"
        runbin = "mkdir -p"
        # if self.deterministic == False:
        runargs = f"{self.host}"
        runpath2 = f"alya_files/{self.host}"
        run_subprocess(runpath, runbin, runargs)

        for inv_i in range(
            1, self.nz_Qs + 1
        ):  # TODO: replace `nz_Qs` with `nb_inv_per_CFD` or `nTotal_Qs` ? @pietero
            runargs2 = f"{inv_i}"
            run_subprocess(runpath2, runbin, runargs2)

        # Write the nodes running this environmment
        name = "nodes"
        node_path = f"alya_files/{self.host}/{self.ENV_ID[1]}/{name}"

        if not os.path.exists(node_path):
            with open(node_path, "w") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["Nodes in this learning"])
                spam_writer.writerow(self.nodelist)

        else:
            with open(node_path, "a") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["Nodes in this learning"])
                spam_writer.writerow(self.nodelist)
        # else:
        #    runargs = 'deterministic'
        #    run_subprocess(runpath,runbin,runargs,check_return=False)

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.create_cpuID: Folder created for CPU ID: %s/%s",
            self.ENV_ID,
            self.host,
            self.ENV_ID[1],
        )
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.create_cpuID: Finished `create_cpuID` method.\n",
        )
        # print(f"Folder created for CPU ID: {self.host}/{self.ENV_ID[1]}")

    # Optional
    def close(self) -> None:
        self.log(
            logging.DEBUG, "ENV_ID %s: Env3D.close: Closing environment...", self.ENV_ID
        )
        super().close()
        self.log(
            logging.DEBUG, "ENV_ID %s: Env3D.close: Environment closed!\n", self.ENV_ID
        )

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    ## Default function required for the DRL

    def list_observation_updated(self) -> np.ndarray:
        """
        Generate a 1D array of probe values for the current environment.

        This method slices the global dictionary of probe values based on the current environment ID.
        For pressure data, it directly extracts the relevant slice. For velocity data, it concatenates
        and flattens the slices of VELOX, VELOY, and VELOZ components in column-major order.

        Returns:
            np.ndarray: A 1D numpy array of probe values for the current (local) environment.

        Raises:
            NotImplementedError: If the probe type is not supported.
            NotImplementedError: If the neighbor state is True.
        """
        self.log(
            logging.DEBUG, "ENV_ID %s: Env3D.list_observation_updated: starting ..."
        )
        # print(f"Env3D.list_observation_updated: {self.ENV_ID}: starting ...")

        if not self.neighbor_state:
            probe_type = self.output_params["probe_type"]
            batch_size_probes = int(
                len(
                    self.probes_values_global_dict[
                        next(iter(self.probes_values_global_dict))
                    ]
                )
                / self.nb_inv_per_CFD
            )

            if probe_type == "pressure":
                data = self.probes_values_global_dict["PRESSURE"]
                probes_values_2 = data[
                    ((self.ENV_ID[1] - 1) * batch_size_probes) : (
                        self.ENV_ID[1] * batch_size_probes
                    )
                ]
            elif probe_type == "velocity":
                vel_components = ["VELOX", "VELOY", "VELOZ"]
                probes_values_2 = []
                for comp in vel_components:
                    data = self.probes_values_global_dict[comp]
                    slice_data = data[
                        ((self.ENV_ID[1] - 1) * batch_size_probes) : (
                            self.ENV_ID[1] * batch_size_probes
                        )
                    ]
                    probes_values_2.append(slice_data)
                # Flatten the array in column-major order
                probes_values_2 = np.array(probes_values_2).flatten(order="F")
            else:
                raise NotImplementedError(
                    f"Env3D_MARL_channel: list_obervation_update: Probe type {probe_type} not implemented yet"
                )

        else:
            raise NotImplementedError(
                "Env3D_MARL_channel: list_obervation_update: Neighbor state True not implemented yet"
            )

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.list_observation_updated: Probes filtered for this specific environment!\n",
        )
        return probes_values_2

    def list_observation(self) -> np.ndarray:
        # TODO: @pietero delete this!!! - Pieter

        if not self.neighbor_state:
            # TODO: filter this observation state to each invariant and its neighbours:
            batch_size_probes = int(
                len(self.probes_values_global) / self.nb_inv_per_CFD
            )
            probes_values_2 = self.probes_values_global[
                ((self.ENV_ID[1] - 1) * batch_size_probes) : (
                    self.ENV_ID[1] * batch_size_probes
                )
            ]

        else:
            # TODO: filter this observation state to each invariant and its neighbours:
            batch_size_probes = int(
                len(self.probes_values_global) / self.nb_inv_per_CFD
            )

            if self.ENV_ID[1] == 1:
                probes_values_halo = self.probes_values_global[
                    ((self.nb_inv_per_CFD - 1) * batch_size_probes) : (
                        self.nb_inv_per_CFD * batch_size_probes
                    )
                ]
                probes_values = self.probes_values_global[
                    ((self.ENV_ID[1] - 1) * batch_size_probes) : (
                        (self.ENV_ID[1] + 1) * batch_size_probes
                    )
                ]
                probes_values_2 = np.concatenate((probes_values_halo, probes_values))
                # print("POOOOOOOL len() line656:", probes_values_2)
                print("POOOOOOOL len() line657:", len(probes_values_2))
                print("POOOOOOOOOOOOOOOOL ---> TYPE PROBES ", type(probes_values_2))

            elif self.ENV_ID[1] == self.nb_inv_per_CFD:
                probes_values = self.probes_values_global[
                    ((self.ENV_ID[1] - 2) * batch_size_probes) : (
                        self.ENV_ID[1] * batch_size_probes
                    )
                ]
                probes_values_halo = self.probes_values_global[0:batch_size_probes]
                probes_values_2 = np.concatenate((probes_values, probes_values_halo))
                # print("POOOOOOOL len() line664:", probes_values_2)
                print("POOOOOOOL len() line665:", len(probes_values_2))
                print("POOOOOOOOOOOOOOOOL ---> TYPE PROBES ", type(probes_values_2))

            else:
                probes_values_2 = self.probes_values_global[
                    ((self.ENV_ID[1] - 2) * batch_size_probes) : (
                        (self.ENV_ID[1] + 1) * batch_size_probes
                    )
                ]
                # print("POOOOOOOL len() line669:", probes_values_2)
                print("POOOOOOOL len() line670:", len(probes_values_2))
                print("POOOOOOOOOOOOOOOOL ---> TYPE PROBES ", type(probes_values_2))

        return probes_values_2

    def states(self) -> Dict[str, Any]:
        """
        Define the state space for the TensorForce agent.

        This method calculates the state size based on the probe type and the number of locations in the global environment.
        It adjusts for the number of agents in the environment and handles different probe types (velocity and pressure).

        Returns:
            dict: A dictionary defining the state type and shape for the TensorForce agent.

        Raises:
            NotImplementedError: If the probe type is not supported.
        """
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.states: Calculating state space...",
            self.ENV_ID,
        )
        if not self.neighbor_state:
            if self.output_params["probe_type"] == "velocity":
                # 3 columns VELOX VELOY VELOZ flattened to 1
                state_size = (
                    int(len(self.output_params["locations"]) / self.nb_inv_per_CFD) * 3
                )
            elif self.output_params["probe_type"] == "pressure":
                # 1 column of witness data
                state_size = int(
                    len(self.output_params["locations"]) / self.nb_inv_per_CFD
                )
            else:
                raise NotImplementedError(
                    f"Env3D_MARL_channel: states: Probe type {self.output_params['probe_type']} not implemented yet, state space cannot be calculated"
                )
        else:
            # TODO: introduce neighbours in parameters!
            # NOW IS JUST 1 EACH SIDE 85*3
            state_size = int(
                len(self.output_params["locations"]) / self.nb_inv_per_CFD
            ) * (3)

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.states: State space calculated!\n",
            self.ENV_ID,
        )
        return dict(type="float", shape=(state_size,))

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def actions(self) -> Dict[str, Any]:
        """Action is a list of n_jets-1 capped values of Q"""
        """UPDATE --> now with multiple Q per jet slot --> use nz_Qs"""
        """UPDATE 2 --> NOW WITH MARL --> ACTIONS_PER_INV = 1"""

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.actions: Defining action space...",
            self.ENV_ID,
        )
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.actions: Action space defined!\n",
            self.ENV_ID,
        )
        return dict(
            type="float",
            shape=(self.actions_per_inv),
            min_value=self.optimization_params["min_value_jet_MFR"],
            max_value=self.optimization_params["max_value_jet_MFR"],
        )

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the initial state.

        This method is used ONLY at the beginning of each episode,
        and is followed by the TensorForce agent calling the `execute` method.

        Returns: the initial actions based on the baseline (or previous episode if `. TODO: @pietero finish documentation - Pieter
        """
        self.log(
            logging.INFO,
            "ENV_ID %s: Env3D.reset: Resetting environment to initialize a new episode...",
            self.ENV_ID,
        )
        if self.ENV_ID[1] != 1:
            time.sleep(4)

        """Reset state"""
        # print(
        #     "\n \n Reset to initalize each episode (copy baseline, clean action count...)! (Env3D_MARL_channel-->reset)\n \n"
        # )
        # Create a folder for each environment
        # print("POOOOL --> CHECK_ID = ", self.check_id)
        # print("POOOOL --> ENV_ID   = ", self.ENV_ID[1])
        if self.check_id == True and self.ENV_ID[1] == 1:
            self.create_cpuID()
            self.check_id = False

        # Clean
        self.log(
            logging.INFO,
            "ENV_ID %s: Env3D.reset: Cleaning the environment using `self.clean`...",
            self.ENV_ID,
        )
        # print("\n\nLocation: Reset")
        # print(
        #     "Action: start to set up the case, set the initial conditions and clean the action counter\n"
        # )
        self.clean(False)

        # Advance in episode
        self.episode_number += 1

        self.bool_restart_prevEP = bool_restart

        # Apply new time frame
        # TODO --> fix time interval detected in the time_interval.dat file
        #     it has to read and detect self.simulation_timeframe[1] auto
        self.simulation_timeframe: List[float] = simulation_params[
            "simulation_timeframe"
        ]
        t1: float = self.simulation_timeframe[0]
        if self.bool_restart_prevEP and self.episode_number > 1:
            t2: float = detect_last_timeinterval(
                os.path.join(
                    "alya_files",
                    f"{self.host}",
                    "1",
                    f"EP_{self.episode_number-1}",  # was "EP_*" # TODO: changed to episode_number-1 from episode_number - Pieter
                    "time_interval.dat",
                )
            )
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.reset: Detected last time interval: %f",
                self.ENV_ID,
                t2,
            )

            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.reset: POOOOOL PATH: %s",
                self.ENV_ID,
                os.path.join(
                    "alya_files",
                    f"{self.host}",
                    "1",
                    f"EP_{self.episode_number-1}",
                    "time_interval.dat",
                ),
            )
            # print(
            #     "POOOOOOOL PATH:",
            #     os.path.join(
            #         "alya_files",
            #         f"{self.host}",
            #         "1",
            #         f"EP_{self.episode_number}",
            #         "time_interval.dat",
            #     ),
            # )
        else:
            t2: float = self.simulation_timeframe[1]
        self.simulation_timeframe = [t1, t2]
        self.log(
            logging.INFO,
            "ENV_ID %s: Env3D.reset: New ALYA time interval: %f to %f",
            self.ENV_ID,
            t1,
            t2,
        )
        # print(f"EnvID: {self.ENV_ID} - The actual timeframe is between {t1} and {t2}: ")

        # Copy the baseline in the environment directory

        if self.action_count == 1 and self.ENV_ID[1] == 1:
            self.recover_start()

        if self.ENV_ID[1] == 1:
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.reset: Updating time_interval.dat file...",
                self.ENV_ID,
            )
            write_time_interval(
                os.path.join(
                    "alya_files",
                    f"{self.host}",
                    f"{self.ENV_ID[1]}",
                    f"EP_{self.episode_number}",
                ),
                t1,
                t2,
            )

        self.log(
            logging.INFO,
            "ENV_ID %s: Env3D.reset: Actual episode: %d",
            self.ENV_ID,
            self.episode_number,
        )
        # print(f"Actual episode: {self.episode_number}")
        # print("\n\Action: extract the probes")
        self.log(
            logging.INFO, "ENV_ID %s: Env3D.reset: Extracting probes...", self.ENV_ID
        )
        NWIT_TO_READ = 1  # Read n timesteps from witness file from behind, last instant

        # TODO: READ THE WITNESS OF EACH PSEUDOENV! - Pol
        # cp witness.dat to avoid IO problems in disk?
        # filename     = os.path.join('alya_files','%s'%self.host,'%s'%self.ENV_ID[1],'EP_%d'%self.episode_number,'%s.nsi.wit'%self.case)
        filename = os.path.join(
            "alya_files",
            f"{self.host}",
            "1",
            f"EP_{self.episode_number}",
            f"{self.case}.nsi.wit",
        )
        filepath_flag_sync_cp = os.path.join(
            "alya_files",
            f"{self.host}",
            "1",
            f"EP_{self.episode_number}",
            "flags_MARL",
        )

        action_end_flag_cp_path = os.path.join(
            filepath_flag_sync_cp, "action_end_flag_cp"
        )

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.reset: POOOOOOOL -> self.deterministics = %s",
            self.ENV_ID,
            self.deterministic,
        )
        # print("POOOOOOOL -> self.deterministic = ", self.deterministic)

        if not self.deterministic:
            while not os.path.exists(action_end_flag_cp_path):
                time.sleep(0.5)

        # Read witness file from behind, last instant (FROM THE INVARIANT [*,1])
        NWIT_TO_READ = 1
        filename = os.path.join(
            "alya_files",
            f"{self.host}",
            "1",
            f"EP_{self.episode_number}",
            f"{self.case}.nsi.wit",
        )

        # read witness file and extract the entire array list
        # This now outputs a dictionary of probe values for all probe types - Pieter
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.reset: Reading witness file...",
            self.ENV_ID,
        )
        self.probes_values_global_dict: Dict[str, np.ndarray] = read_last_wit(
            filename,
            output_params["probe_type"],
            self.norm_factors,
            NWIT_TO_READ,
        )

        # filter probes per jet (corresponding to the ENV.ID[])
        self.log(
            logging.DEBUG, "ENV_ID %s: Env3D.reset: Filtering probes...", self.ENV_ID
        )
        probes_values_2 = self.list_observation_updated()
        # print(
        #     "\n\n\nEnv3D_MARL_channel.Env3D.reset: probes_values_2 being returned to Tensorforce!!!\n\n\n"
        # )
        self.log(
            logging.INFO, "ENV_ID %s: Env3D.reset: Probes extracted!\n", self.ENV_ID
        )
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.reset: `reset` method complete!\n",
            self.ENV_ID,
        )
        return probes_values_2

    # -----------------------------------------------------------------------------------------------------
    # TODO: figure our where the actions in `execute` argument are coming from @pietero
    # TODO: figure out structure/type of actions in `execute` argument @pietero
    def execute(self, actions: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        # TODO: @pietero FINISH LOGGING IMPLEMENTATION STARTING HERE!!!!!!!!! - Pieter
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.execute: Starting `execute` method...",
            self.ENV_ID,
        )
        action: List[np.ndarray] = []
        # action = []
        if case == "cylinder":
            for i in range(self.actions_per_inv):
                action.append(self.optimization_params["norm_Q"] * actions[i])
                action.append(
                    -self.optimization_params["norm_Q"] * actions[i]
                )  # This is to ensure 0 mass flow rate in the jets
        if case == "channel":
            action.append(self.optimization_params["norm_Q"] * actions[0])

        # for i in range(self.actions_per_inv, self.actions_per_inv*2):
        # action.append(-self.optimization_params["norm_Q"]*actions[i-self.actions_per_inv])

        self.previous_action = self.action  # save the older one to smooth the change
        self.action = action  # update the new to reach at the end of smooth

        # Write the action
        self.save_this_action()

        if case == "cylinder":
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.execute: New action computed!\n",
                self.ENV_ID,
            )
        elif case == "airfoil":
            pass
        elif case == "channel":
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.execute: New action computed!\n",
                self.ENV_ID,
            )

        dir_name = os.path.join(
            "alya_files",
            f"{self.host}",
            "1",
            f"EP_{self.episode_number}",
            "flags_MARL",
        )
        run_subprocess(
            dir_name,
            "mkdir",
            f"{self.ENV_ID[1]}_inv_action_{self.action_count}_ready",
        )

        self.last_time = self.simulation_timeframe[1]
        t1 = round(self.last_time, 3)
        t2 = t1 + self.delta_t_smooth

        self.simulation_timeframe = [t1, t2]

        if self.ENV_ID[1] == 1:

            cr_start("ENV.actions_MASTER_thread1", 0)

            # wait until all the action from the others pseudoenvs are available
            all_actions_ready = False

            while not all_actions_ready:
                # loop over directory names
                for i in range(1, self.nb_inv_per_CFD):
                    dir_name = os.path.join(
                        "alya_files",
                        f"{self.host}",
                        "1",
                        f"EP_{self.episode_number}",
                        "flags_MARL",
                        f"{i}_inv_action_{self.action_count}_ready",
                    )
                    all_actions_ready = True
                    if not os.path.exists(dir_name):
                        all_actions_ready = False
                time.sleep(0.2)

            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.execute: \n\n!!!ALL ACTIONS ARE READY TO APPLY TO BOUNDARY CONDITIONS!!!\n",
                self.ENV_ID,
            )
            # print(
            #     "**************** ALL ACTIONS ARE READY TO UPDATE BC *****************"
            # )
            # run_subprocess(dir_name,'rm -rf ','*_inv_action_*')

            ## NOW READY TO MERGE ACTIONS:
            # append and reading file
            # open the file for reading

            for i in range(self.nb_inv_per_CFD):
                path_action_file = f"actions/{self.host}/{self.ENV_ID[0]}_{i+1}/ep_{self.episode_number}/output_actions.csv"
                with open(path_action_file, "r") as file:
                    # read the lines of the file into a list
                    lines = csv.reader(file, delimiter=";")
                    # skip the header row
                    next(lines)
                    # initialize a variable to store the last value
                    last_action = None
                    # read each row and extract the second value
                    for row in lines:
                        last_action = float(row[1].strip())

                    self.log(
                        logging.DEBUG,
                        "ENV_ID %s: Last action of %s_%d: %f",
                        self.ENV_ID,
                        self.ENV_ID[0],
                        i + 1,
                        last_action,
                    )
                    # print(
                    #     f"ENV_ID {self.ENV_ID}: Last action of {self.ENV_ID[0]}_{i+1}: {last_action}"
                    # )
                    self.previous_action_global[i] = self.action_global[i]
                    self.action_global[i] = last_action

            write_time_interval(
                os.path.join(
                    "alya_files",
                    f"{self.host}",
                    f"{self.ENV_ID[1]}",  # This is always 1
                    f"EP_{self.episode_number}",
                ),
                t1,
                t2,
            )

            simu_path = os.path.join(
                "alya_files",
                f"{self.host}",
                f"{self.ENV_ID[1]}",  # This is always 1
                f"EP_{self.episode_number}",
            )

            if self.case == "cylinder":

                for ijet, jet in enumerate(
                    self.Jets.values()
                ):  # Only need to loop on the values, i.e., Jet class
                    # Q_pre, Q_new, time_start, select smoothing law of the action
                    # print("POOOOOOL jet.update : ",self.previous_action_global,self.action_global,self.simulation_timeframe[0],self.smooth_func)
                    jet.update(
                        self.previous_action_global,
                        self.action_global,
                        self.simulation_timeframe[0],
                        self.smooth_func,
                        Qs_position_z=self.Qs_position_z,
                        delta_Q_z=self.delta_Q_z,
                    )
                    # Update the jet profile alya file
                    jet.update_file(simu_path)

            elif self.case == "airfoil":
                for ijet, jet in enumerate(
                    self.Jets.values()
                ):  # Only need to loop on the values, i.e., Jet class
                    # Update jets for the given epoch
                    if self.smooth_func == "parabolic":
                        self.slope_pre = jet.slope_new
                    else:
                        self.slope_pre = 0

                    # Q_pre, Q_new, time_start
                    jet.update(
                        self.previous_action[ijet],
                        self.action[ijet],
                        self.simulation_timeframe[0],
                    )
                    # Update the jet profile alya file
                    jet.update_file(simu_path)

            elif self.case == "channel":
                for ijet, jet in enumerate(self.Jets.values()):
                    jet.update(
                        self.previous_action_global,
                        self.action_global,
                        self.simulation_timeframe[0],
                        self.smooth_func,
                        Qs_position_z=self.Qs_position_z,
                        delta_Q_z=self.delta_Q_z,
                        Qs_position_x=self.Qs_position_x,
                        delta_Q_x=self.delta_Q_x,
                    )  # TODO: @pietero make sure this works for channel - Pieter

            if self.reward_function == "q_event_volume":
                self.log(
                    logging.INFO,
                    "ENV_ID %s: Env3D.execute: Starting reward calculation...",
                    self.ENV_ID,
                )
                ## Setting up for computing the rewards and save as .csv file
                # First need to identify and convert ALYA postprocessing files to VTK files
                directory_post = os.path.join(
                    "alya_files",
                    f"{self.host}",
                    f"{self.ENV_ID[1]}",  # this is always 1
                    f"EP_{self.episode_number}",
                )

                if self.output_params["probe_type"] == "velocity":
                    post_name = "VELOC"
                elif self.output_params["probe_type"] == "pressure":
                    post_name = "PRESS"
                else:
                    post_name = None
                    raise NotImplementedError(
                        f"{self.ENV_ID}: execute: post.mpio.bin associated with type {self.output_params['probe_type']} not implemented yet"
                    )

                self.log(
                    logging.DEBUG,
                    "ENV_ID %s: Env3D.execute: Identifying the file with the highest timestep...",
                    self.ENV_ID,
                )
                # Identify the file with the highest timestep
                last_post_file = find_highest_timestep_file(
                    directory_post, f"{self.case}", f"{post_name}"
                )

                # Copy the identified file to a specific directory for processing
                target_directory = os.path.join(directory_post, "final_post_of_action")

                copy_mpio2vtk_required_files(
                    self.case, directory_post, target_directory, last_post_file
                )

                # Convert the copied file to VTK format
                # Run subprocess that launches mpio2vtk to convert the file to VTK
                self.log(
                    logging.INFO,
                    "ENV_ID %s: Env3D.execute: Starting subprocess to convert ALYA postprocessing files to VTK...",
                    self.ENV_ID,
                )
                run_subprocess(
                    target_directory,
                    ALYA_VTK,
                    f"{self.case}",
                    nprocs=nb_proc,
                    mem_per_srun=mem_per_srun,
                    num_nodes_srun=num_nodes_srun,
                    host=self.nodelist,
                    slurm=USE_SLURM,
                )
                self.log(
                    logging.INFO,
                    "ENV_ID %s: Env3D.execute: VTK files created for episode %d action %d",
                    self.ENV_ID,
                    self.episode_number,
                    self.action_count,
                )

                # Second we set up for Q event identification and reward calculation
                directory_vtk = os.path.join(
                    target_directory,
                    "vtk",
                )
                averaged_data_path = os.path.join(
                    directory_post,
                    "calculated_data",
                )
                output_folder_path_reward = os.path.join(
                    directory_post,
                    "rewards",
                )
                qratio_output_file_name = (
                    f"qratios_{self.host}_EP_{self.episode_number}_action_{self.action_count}.csv"
                )
                qratio_output_file_path = os.path.join(
                    output_folder_path_reward, qratio_output_file_name
                )
                reward_output_file_name = (
                    f"rewards_{self.host}_EP_{self.episode_number}_action_{self.action_count}.csv"
                )
                reward_output_file_path = os.path.join(
                    output_folder_path_reward, reward_output_file_name
                )

                if not os.path.exists(directory_vtk):
                    raise ValueError(
                        f"{self.ENV_ID}: execute: Directory {directory_vtk} does not exist for action vtk files!!!"
                    )
                if not os.path.exists(averaged_data_path):
                    raise ValueError(
                        f"{self.ENV_ID}: execute: Directory {averaged_data_path} does not exist for pre-calculated data!!!"
                    )
                if not os.path.exists(output_folder_path_reward):
                    os.makedirs(output_folder_path_reward)

                # Launches a subprocess to calculate the reward
                # Must use a separate conda environment for compatibility
                runpath_vtk = "./"
                runbin_vtk = "python3 coco_calc_reward.py"

                runargs_vtk = (
                    f"--directory {directory_vtk} "
                    f"--Lx {reward_params['Lx']} "
                    f"--Lz {reward_params['Lz']} "
                    f"--H {reward_params['H']} "
                    f"--nx {reward_params['nx_Qs']} "
                    f"--nz {reward_params['nz_Qs']} "
                    f"--averaged_data_path {averaged_data_path} "
                    f"--output_qratio_file {qratio_output_file_path} "
                    f"--output_reward_file {reward_output_file_path} "
                )
                self.log(
                    logging.INFO,
                    "ENV_ID %s: Env3D.execute: Starting subprocess to activate `mesh_env` conda environment and calculate reward...",
                    self.ENV_ID,
                )
                run_subprocess(
                    runpath_vtk,
                    runbin_vtk,
                    runargs_vtk,
                    use_new_env=True,
                )

                self.log(
                    logging.INFO,
                    "ENV_ID %s: Env3D.execute: Reward calculation for EP_%d action %d complete!\n",
                    self.ENV_ID,
                    self.episode_number,
                    self.action_count,
                )

            cr_stop("ENV.actions_MASTER_thread1", 0)

        # Start an alya run
        t0 = time.time()

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.execute: Calling `run` method! ...",
            self.ENV_ID,
        )
        self.run(which="execute")
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.execute: `run` method complete!",
            self.ENV_ID,
        )
        self.log(
            logging.INFO,
            "ENV_ID %s: Env3D.execute: Time elapsed for `run`: %f\n",
            self.ENV_ID,
            time.time() - t0,
        )

        if self.case == "cylinder":
            # Get the new avg drag and lift --> LOCAL
            average_drag, average_lift = compute_avg_lift_drag(
                self.episode_number, cpuid=self.host, nb_inv=self.ENV_ID[1]
            )
            self.history_parameters["drag"].extend([average_drag])
            self.history_parameters["lift"].extend([average_lift])
            self.history_parameters["time"].extend([self.last_time])
            self.history_parameters["episode_number"].extend([self.episode_number])
            self.save_history_parameters_all(nb_actuations)

            # Get the new avg drag and lift --> GLOBAL
            average_drag_GLOBAL, average_lift_GLOBAL = compute_avg_lift_drag(
                self.episode_number,
                cpuid=self.host,
                nb_inv=self.nb_inv_per_CFD,
                global_rew=True,
            )
            self.history_parameters["drag_GLOBAL"].extend([average_drag_GLOBAL])
            self.history_parameters["lift_GLOBAL"].extend([average_lift_GLOBAL])

        elif self.case == "channel":
            # TODO: @pietero implement history parameters for channel if needed - Pieter
            pass

        # Compute the reward
        reward: float = self.compute_reward()
        self.save_reward(
            reward
        )  # TODO: @pietero Is this still needed? All rewards are saved by `coco_calc_reward.py`- Pieter
        self.log(
            logging.DEBUG, "ENV_ID %s: Env3D.execute: Reward: %f", self.ENV_ID, reward
        )
        # print(f"reward: {reward}")

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.execute: The actual action is %d of %d",
            self.ENV_ID,
            self.action_count,
            nb_actuations,
        )
        # print(f"The actual action is {self.action_count} of {nb_actuations}")

        self.action_count += 1

        if self.deterministic == False and self.action_count <= nb_actuations:
            terminal = False  # Episode is not done for training
        elif (
            self.deterministic == True
            and self.action_count <= nb_actuations_deterministic
        ):
            terminal = False  # Episode is not done for deterministic
        else:
            terminal = True  # Episode is done

            # write the last rewards at each episode to see the improvement
            self.save_final_reward(reward)
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.execute: Actual episode %d is finished and saved!",
                self.ENV_ID,
                self.episode_number,
            )
            # print(f"Actual episode: {self.episode_number} is finished and saved")
            # print(f"Results : \n\tAverage drag : {average_drag}\n\tAverage lift : {average_lift})

        # print("\n\nTask : extract the probes")
        self.log(
            logging.INFO, "ENV_ID %s: Env3D.execute: Extracting probes...", self.ENV_ID
        )
        # Read witness file from behind, last instant (FROM THE INVARIANT [*,1])
        NWIT_TO_READ = 1
        filename = os.path.join(
            "alya_files",
            f"{self.host}",
            "1",
            f"EP_{self.episode_number}",
            f"{self.case}.nsi.wit",
        )

        # read witness file and extract the entire array list
        # This now outputs a dictionary of probe values for all probe types - Pieter
        self.probes_values_global_dict: Dict[str, np.ndarray] = read_last_wit(
            filename,
            output_params["probe_type"],
            self.norm_factors,
            NWIT_TO_READ,
        )

        # filter probes per jet (corresponding to the ENV.ID[])
        probes_values_2 = self.list_observation_updated()
        self.log(
            logging.INFO,
            "ENV_ID %s: Env3D.execute: Probes extracted and filtered!\n",
            self.ENV_ID,
        )

        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.execute: `execute` method complete!\n",
            self.ENV_ID,
        )
        return probes_values_2, terminal, reward

    # -----------------------------------------------------------------------------------------------------

    def compute_reward(self) -> float:
        self.log(
            logging.DEBUG,
            "ENV_ID %s: Env3D.compute_reward: Starting `reward` method...",
        )
        # NOTE: reward should be computed over the whole number of iterations in each execute loop
        if (
            self.reward_function == "plain_drag"
        ):  # a bit dangerous, may be injecting some momentum
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: `plain_drag` reward function selected...",
            )
            values_drag_in_last_execute = self.history_parameters["drag"][-1:]
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.compute_reward: Reward computed!",
                self.ENV_ID,
            )
            return (
                np.mean(values_drag_in_last_execute) + 0.159
            )  # the 0.159 value is a proxy value corresponding to the mean drag when no control; may depend on the geometry

        elif (
            self.reward_function == "drag_plain_lift_2"
        ):  # a bit dangerous, may be injecting some momentum
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: `drag_plain_lift_2` reward function selected...",
            )
            avg_drag = np.mean(self.history_parameters["drag"])
            avg_lift = np.mean(self.history_parameters["lift"])
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.compute_reward: Reward computed!",
                self.ENV_ID,
            )
            return -avg_drag - 0.2 * abs(avg_lift)

        elif (
            self.reward_function == "drag"
        ):  # a bit dangerous, may be injecting some momentum
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: `drag` reward function selected...",
            )
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.compute_reward: Reward computed!",
                self.ENV_ID,
            )
            return self.history_parameters["drag"][-1] + 0.159

        elif (
            self.reward_function == "drag_plain_lift"
        ):  # a bit dangerous, may be injecting some momentum
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: `drag_plain_lift` reward function selected...",
            )
            ## get the last mean cd or cl value of the last Tk
            avg_drag_2 = np.mean(self.history_parameters["drag"][-1:])
            avg_lift_2 = np.mean(self.history_parameters["lift"][-1:])
            avg_drag_2_global = np.mean(self.history_parameters["drag_GLOBAL"][-1:])
            avg_lift_2_global = np.mean(self.history_parameters["lift_GLOBAL"][-1:])

            reward_local = (
                -avg_drag_2
                - self.optimization_params["penal_cl"] * abs(avg_lift_2)
                + self.optimization_params["offset_reward"]
            )
            reward_global = (
                -avg_drag_2_global
                - self.optimization_params["penal_cl"] * abs(avg_lift_2_global)
                + self.optimization_params["offset_reward"]
            )
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: POOOOOOOL ---> reward_local: %f",
                self.ENV_ID,
                reward_local,
            )
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: POOOOOOOL ---> reward_global: %f",
                self.ENV_ID,
                reward_global,
            )
            # print("POOOOOOOOL ---> reward_global: ", reward_global)
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: POOOOOOOL ---> cd_local: %f",
                self.ENV_ID,
                avg_drag_2,
            )
            # print("POOOOOOOOL ---> cd_local: ", avg_drag_2)
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: POOOOOOOL ---> cd_lift: %f",
                self.ENV_ID,
                avg_lift_2,
            )
            # print("POOOOOOOOL ---> cd_lift: ", avg_lift_2)
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: POOOOOOOL ---> cd_global: %f",
                self.ENV_ID,
                avg_drag_2_global,
            )
            # print("POOOOOOOOL ---> cd_local: ", avg_drag_2_global)
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: POOOOOOOL ---> cd_lift_global: %f",
                self.ENV_ID,
                avg_lift_2_global,
            )
            # print("POOOOOOOOL ---> cd_lift: ", avg_lift_2_global)

            alpha_rew = self.optimization_params["alpha_rew"]
            reward_total = self.optimization_params["norm_reward"] * (
                (alpha_rew) * reward_local + (1 - alpha_rew) * reward_global
            )
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.compute_reward: Reward computed!",
                self.ENV_ID,
            )
            ## le añadimos el offset de 3.21 para partir de reward nula y que solo vaya a (+)
            return reward_total

        elif (
            self.reward_function == "max_plain_drag"
        ):  # a bit dangerous, may be injecting some momentum
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: `max_plain_drag` reward function selected...",
            )
            values_drag_in_last_execute = self.history_parameters["drag"][-1:]
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.compute_reward: Reward computed!",
                self.ENV_ID,
            )
            return -(np.mean(values_drag_in_last_execute) + 0.159)

        elif (
            self.reward_function == "drag_avg_abs_lift"
        ):  # a bit dangerous, may be injecting some momentum
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: `drag_avg_abs_lift` reward function selected...",
            )
            avg_abs_lift = np.absolute(self.history_parameters["lift"][-1:])
            avg_drag = self.history_parameters["drag"][-1:]
            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.compute_reward: Reward computed!",
                self.ENV_ID,
            )
            return avg_drag + 0.159 - 0.2 * avg_abs_lift

        elif (
            self.reward_function == "lift_vs_drag"
        ):  # a bit dangerous, may be injecting some momentum
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: `lift_vs_drag` reward function selected...",
            )
            ## get the last mean cd or cl value of the last Tk
            avg_lift = np.mean(self.history_parameters["lift"][-1:])
            avg_drag = np.mean(self.history_parameters["drag"][-1:])

            self.log(
                logging.INFO,
                "ENV_ID %s: Env3D.compute_reward: Reward computed!",
                self.ENV_ID,
            )
            return self.optimization_params["norm_reward"] * (
                avg_lift / avg_drag + self.optimization_params["offset_reward"]
            )

        elif self.reward_function == "q_event_volume":
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: `q_event_volume` reward function selected...",
            )
            # TODO: @pietero implement q-event volume reward function - Pieter
            output_file_path = os.path.join(
                "alya_files",
                f"{self.host}",
                f"{1}",
                f"EP_{self.episode_number}",
                "rewards",
                f"rewards_{self.host}_EP_{self.episode_number}_action_{self.action_count}.csv",
            )
            self.log(
                logging.DEBUG,
                "ENV_ID %s: Env3D.compute_reward: Reading reward file at %s...",
                self.ENV_ID,
                output_file_path,
            )
            if not os.path.exists(output_file_path):
                raise ValueError(
                    f"No reward file found at {output_file_path} for ENV_ID {self.ENV_ID[1]}"
                )

            data = np.genfromtxt(output_file_path, delimiter=",", names=True)

            # Find the row where ENV_ID matches self.ENV_ID[1]
            matching_row = data[data["ENV_ID"] == self.ENV_ID[1]]

            if matching_row.size == 0:
                raise ValueError(
                    f"No matching row found for ENV_ID {self.ENV_ID[1]} in reward file at {output_file_path}"
                )

            reward_value: float = float(matching_row["reward"][0])
            primary_logger.info(
                "ENV_ID %s: Env3D.compute_reward: Reward value: %f\n",
                self.ENV_ID,
                reward_value,
            )
            return reward_value
