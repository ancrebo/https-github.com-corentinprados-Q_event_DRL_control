"""
env_utils.py
============
DEEP REINFORCEMENT LEARNING WITH ALYA
-------------------------------------

This module provides a collection of utility functions to assist with
environment setup, subprocess management, and file operations. The most
commonly used function is `run_subprocess`, which executes terminal commands
as subprocesses with optional parallel execution and logging.

Functions
---------
- run_subprocess(runpath: str, runbin: str, runargs: str, parallel: bool = False,
                 log: Optional[str] = None, check_return: bool = True,
                 host: Optional[str] = None, use_new_env: bool = False,
                 **kwargs) -> int
    Executes a terminal command as a subprocess with optional parallel execution
    and logging.

- detect_system(override: str = None) -> str
    Detects whether the script is running on a local machine or a SLURM cluster.

- _slurm_generate_node_list(outfile: str, num_servers: int, num_cores_server: int,
                            **kwargs) -> None
    Generates a list of nodes using SLURM for parallel execution.

- _localhost_generate_node_list(outfile: str, num_servers: int) -> None
    Generates a list of nodes for local execution.

- generate_node_list(override: str = None, outfile: str = NODELIST,
                     num_servers: int = 1, num_cores_server: int = 1) -> None
    Detects the system and generates the appropriate node list.

- read_node_list(file: str = NODELIST) -> List[str]
    Reads a list of nodes from a file.

- agent_index_2d_to_1d(i: int, j: int, nz_Qs: int) -> int
    Converts a 2D index to a 1D index for agents/jets/pseudo-parallel environments.

- agent_index_1d_to_2d(index: int, nz_Qs: int) -> List[int]
    Converts a 1D index to a 2D index for agents/jets/pseudo-parallel environments.

- find_highest_timestep_file(directory: str, case: str, parameter: str) -> str
    Finds the file with the highest timestep in the specified directory.

- copy_mpio2vtk_required_files(case: str, source_directory: str,
                               target_directory: str, identified_file: str) -> None
    Copies the identified post.mpio.bin file and additional files for mpio2vtk
    conversion to the target directory.

Authors
-------
- Pol Suarez
- Arnau Miro
- Francisco Alcantara
- Xavi Garcia
- Pieter Orlandini

Updated
-------
Major update: August 2024

Notes
-----
This module is typically imported and used within other scripts and is not
intended to be executed directly.
"""

from __future__ import print_function, division
from typing import Optional, List
import re
import shutil
import logging

import os, subprocess
from configuration import NODELIST, USE_SLURM, DEBUG

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s", __name__, logger.level)


def run_subprocess(
    runpath: str,
    runbin: str,
    runargs: str,
    parallel: bool = False,
    log: Optional[str] = None,
    check_return: bool = True,
    host: Optional[str] = None,
    use_new_env: bool = False,  # new parameter to activate separate conda envirnoment for process
    **kwargs,
) -> int:
    """
    Execute a terminal command as a subprocess.

    This function runs a specified terminal command in a subprocess, optionally
    with parallel execution using SLURM or mpirun. It supports logging,
    custom environment usage, and error checking.

    Parameters
    ----------
    runpath : str
        The working directory where the command will be executed.
    runbin : str
        The executable or script to run.
    runargs : str
        The arguments to pass to the executable or script.
    parallel : bool, optional
        Whether to run the command in parallel. Default is False.
    log : Optional[str], optional
        The file path to log the command's output. Default is None.
    check_return : bool, optional
        Whether to check the return code and raise an error if non-zero.
        Default is True.
    host : Optional[str], optional
        The hostname to run the command on. Default is None.
    use_new_env : bool, optional
        Whether to run the command in a separate conda environment. Default
        is False.
    **kwargs : dict, optional
        Additional parameters for parallel execution:

        - nprocs : int
            The number of processes for parallel execution.
        - mem_per_srun : int
            Memory per SLURM run in MB.
        - num_nodes_srun : int
            Number of nodes for SLURM.
        - slurm : bool
            Whether to use SLURM for parallel execution. If false, mpirun is used.
        - oversubscribe : bool
            Whether to allow oversubscription of resources.
        - nodelist : List[str]
            List of nodes for mpirun.

    Returns
    -------
    int
        The return code of the subprocess.

    Raises
    ------
    ValueError
        If the subprocess returns a non-zero code and `check_return` is True.

    Examples
    --------
    Run a simple command:
        >>> run_subprocess('/path/to/dir', 'echo', 'Hello World')
        # Executes the following command:
        # cd /path/to/dir && echo Hello World

    Run a parallel command using SLURM:
        >>> run_subprocess('/path/to/dir', 'my_program', '--arg1 --arg2',
        ...                parallel=True, nprocs=4, slurm=True)
        # Executes the following command:
        # cd /path/to/dir && srun --nodes=1 --ntasks=4 --overlap --mem=1 my_program --arg1 --arg2

    Run a command with custom logging folder and error checking:
        >>> run_subprocess('/path/to/dir', 'my_program', '--arg1 --arg2',
        ...                log='/path/to/logfile.log')
        # Executes the following command:
        # cd /path/to/dir && my_program --arg1 --arg2 > /path/to/logfile.log 2>&1

    Run a shell script in a separate conda environment:
        >>> run_subprocess('/path/to/dir', 'my_script.sh', '--arg1 --arg2',
        ...                use_new_env=True)
        # Executes the following command:
        # cd /path/to/dir && /scratch/pietero/andres_clone/DRL_POL/run_reward_in_new_env_NEW.sh my_script.sh --arg1 --arg2
        # where the shell script contains the conda environment activation command
    """
    logger.debug("run_subprocess: Starting run_subprocess...")

    # Auxilar function to build parallel command
    def _cmd_parallel(runbin: str, **kwargs) -> str:
        """
        Build the command to run in parallel.

        Parameters
        ----------
        runbin : str
            The executable or script to run.
        **kwargs : dict, optional
            Additional parameters for parallel execution:

            - nprocs : int
                The number of processes for parallel execution.
            - mem_per_srun : int
                Memory per SLURM run in MB.
            - num_nodes_srun : int
                Number of nodes for SLURM.
            - slurm : bool
                Whether to use SLURM for parallel execution. If false, mpirun is used.
            - oversubscribe : bool
                Whether to allow oversubscription of resources.
            - nodelist : List[str]
                List of nodes for mpirun.

        Returns
        -------
        str
            The command to run in parallel.
        """
        nprocs: int = kwargs.get("nprocs", 1)
        mem_per_srun: int = kwargs.get("mem_per_srun", 1)
        num_nodes_srun: int = kwargs.get("num_nodes_srun", 1)
        slurm: bool = kwargs.get("slurm", USE_SLURM)

        # Switch to run srun or mpirun
        arg_hosts = ""
        if slurm:
            # Using srun
            arg_nprocs = (
                f"--nodes={num_nodes_srun} --ntasks={nprocs} --overlap --mem={mem_per_srun}"
                if nprocs > 1
                else ""
            )
            # arg_nprocs = '--ntasks=%d --overlap' %(nprocs) if nprocs > 1 else ''
            arg_export_dardel = (
                "export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)"
            )
            launcher = f"{arg_export_dardel} && srun"
            # launcher   = 'srun'
            arg_ovsub = "--oversubscribe" if kwargs.get("oversubscribe", False) else ""
            if host != "localhost" and host is not None:
                arg_hosts += f"--nodelist={host}"
                # arg_hosts += ''
        else:
            # Using mpirun
            hostlist = kwargs.get("nodelist", [])
            arg_nprocs = f"-np {nprocs} --use-hwthread-cpus"
            arg_ovsub = "--oversubscribe" if kwargs.get("oversubscribe", False) else ""
            if host != "localhost" and host is not None:
                arg_hosts += f"-host {host}"
            launcher = "mpirun"

        # Return the command
        return f"{launcher} {arg_ovsub} {arg_nprocs} {arg_hosts} {runbin}"

    # Check for a parallel run
    nprocs = kwargs.get("nprocs", 1)
    if nprocs > 1:
        parallel = True  # Enforce parallel if nprocs > 1

    # Check for logs
    arg_log = f"> {log} 2>&1" if log is not None else ""

    # Build command to run
    if use_new_env:
        runbin = (
            "/scratch/pietero/andres_clone/DRL_POL/run_reward_in_new_env_NEW.sh "  # TODO: @pietero, @pol REMOVE HARDCODED PATH! - Pieter
            + runbin
        )  # Prepend the shell script
    cmd_bin = (
        _cmd_parallel(f"{runbin} {runargs}", **kwargs)
        if parallel
        else f"{runbin} {runargs}"
    )
    cmd = f"cd {runpath} && {cmd_bin} {arg_log}"  # TODO: DARDEL DEBUG ONGOING
    # print('POOOOOOOOOOOOOL --> cmd: %s' % cmd)

    # DEBUG
    logger.debug("run_subprocess: Running in PARALLEL: %s", parallel)
    logger.info("run_subprocess: Running command: %s", cmd)
    logger.debug("run_subprocess: Current working directory: %s", os.getcwd())

    # # Execute run
    # retval = subprocess.call(cmd, shell=True)  # old version - Pieter

    # Execute run (alternate, updated version introduced in Python 3.5 - Pieter)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Print the stdout and stderr from the shell script if length is longer than 0
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(line)
    if result.stderr:
        for line in result.stderr.splitlines():
            if "ERROR" in line or "Failed" in line:
                logger.error(line)
            else:
                logger.info(line)

    # Check return
    if check_return and result.returncode != 0:
        raise ValueError(f"Error running command <{cmd}>!\n{result.stderr}")
    logger.debug("run_subprocess: Finished run_subprocess.")
    # Return value
    return result.returncode


def detect_system(override: str = None) -> str:
    """
    Detect if the script is running on a local machine or a SLURM cluster.

    This function determines the system type by checking if the `srun` command
    (used in SLURM clusters) is available. It returns 'LOCAL' for local
    machines and 'SLURM' for SLURM clusters. An optional override allows for
    manual system selection.

    Parameters
    ----------
    override : str, optional
        Manually specify the system type. If provided, this value is returned
        without performing any detection. Default is None.

    Returns
    -------
    str
        The detected system type, either 'LOCAL' or 'SLURM'.

    Examples
    --------
    Detect the system type automatically:
        >>> detect_system()
        'LOCAL'

    Override the system detection:
        >>> detect_system(override='SLURM')
        'SLURM'
    """
    logger.debug("detect_system: Starting detect_system...")
    # Override detect system and manually select the system
    if override is not None:
        return override
    # Start assuming we are on the local machine
    out = "LOCAL"
    # 1. Test for SRUN, if so we are in a SLURM machine
    # and hence we should use SLURM to check the available nodes
    if run_subprocess("./", "which", "srun", check_return=False) == 0:
        out = "SLURM"
    # Return system value
    logger.debug("detect_system: %s system detected!" % out)
    logger.debug("detect_system: Finished detect_system.")
    return out


def _slurm_generate_node_list(
    outfile, num_servers: int, num_cores_server: int, **kwargs
) -> None:
    """
    Generate the list of nodes using SLURM.

    This function generates a list of nodes for a SLURM job based on the specified
    number of parallel runs (servers) and the number of cores per server. It retrieves
    the number of nodes and the number of cores per node from SLURM environment variables,
    performs sanity checks to ensure the allocation is consistent with the desired
    configuration, and writes the node list to an output file.

    Parameters
    ----------
    outfile : str
        The path to the output file where the node list will be written.
    num_servers : int
        The number of parallel runs (servers).
    num_cores_server : int
        The number of cores per server.
    **kwargs : dict, optional
        Additional arguments for node configuration:
        - num_nodes : int
            The number of nodes of the allocation (obtained from SLURM environment variable).
        - num_cores_node : int
            The number of cores per node (obtained from SLURM environment variable).

    Raises
    ------
    ValueError
        If the number of nodes in the hostlist is inconsistent with the expected number.
        If the allocation does not have enough nodes for the task.
    """
    logger.debug("_slurm_generate_node_list: Starting _slurm_generate_node_list...")
    # print("POOOOOL --> SLURM_NNODES: %s" %os.getenv('SLURM_NNODES'))
    # print("POOOOOL --> SLURM_JOB_CPUS_PER_NODE: %s" %os.getenv('SLURM_JOB_CPUS_PER_NODE'))

    num_nodes = kwargs.get("num_nodes", int(os.getenv("SLURM_NNODES")))
    num_cores_node = kwargs.get("num_cores_node", os.getenv("SLURM_JOB_CPUS_PER_NODE"))
    start = num_cores_node.find("(")
    num_cores_node = int(num_cores_node[:start])
    num_cores_node = 100
    logger.debug("POOOOOL --> SLURM_NNODES: %s" % num_cores_node)
    # print(f"POOOOOL --> SLURM_JOB_CPUS_PER_NODE: {num_cores_node}" % num_cores_node)

    # Query SLURM to print the nodes used for this job to a temporal file
    # read it and store it as a variable
    run_subprocess("./", "scontrol", "show hostnames", log="tmp.nodelist")
    hostlist = read_node_list(file="tmp.nodelist")
    # run_subprocess('./','rm','tmp.nodelist')

    # Perform a sanity check
    if len(hostlist) != num_nodes:
        raise ValueError(
            f"Inconsistent number of nodes <{num_nodes}> and hostlist <{len(hostlist)}>!"
        )  # Ensure that we have read the hostlist correctly
    if num_servers * num_cores_server > (num_nodes) * num_cores_node + 1:
        raise ValueError(
            "Inconsistent allocation and DRL settings!"
        )  # Always ensure we have enough nodes for the task

    # Write the proper hostlist
    file = open(outfile, "w")

    # Leave the first node for the DRL only
    file.write(f"{hostlist[0]}\n")
    # Write the rest of the nodes according to the allocation
    iserver = 0  # to debug in just 1 node be careful --- = 0
    total_cores = num_cores_server  # count the total servers to fill an entire node

    for ii in range(num_servers):
        # At least we will use one node
        total_cores += num_cores_server
        file.write(f"{hostlist[iserver]}")
        # Integer division
        # to put more enviroments per node

        if total_cores > num_cores_node:
            iserver += 1
            total_cores = num_cores_server
        # allocate more than one node. if num_cores_server < num_cores_node
        # then it will never get inside this for loop
        for jj in range(num_cores_server // (num_cores_node + 1)):
            file.write(f",{hostlist[iserver]}")
            iserver += 1
        # Finish and jump line
        file.write("\n")
    logger.debug("_slurm_generate_node_list: Finished _slurm_generate_node_list.")


def _localhost_generate_node_list(outfile, num_servers: int) -> None:
    """
    Generate the list of nodes for a local run.

    This function generates a list of 'localhost' entries for the specified
    number of servers and writes it to the specified output file. It prepares
    a node list for a local run, where all processes will run on the local
    machine.

    Parameters
    ----------
    outfile : str
        The path to the output file where the node list will be written.
    num_servers : int
        The number of parallel runs (servers).
    """
    logger.debug(
        "_localhost_generate_node_list: Starting _localhost_generate_node_list..."
    )
    hostlist = "localhost"
    for iserver in range(num_servers):
        hostlist += "\nlocalhost"
    # Basically write localhost as the list of nodes
    # Add n+1 nodes as required per the nodelist
    run_subprocess("./", "echo", f'"{hostlist}"', log=outfile)
    logger.debug(
        "_localhost_generate_node_list: Finished _localhost_generate_node_list."
    )


def generate_node_list(
    override: str = None,
    outfile=NODELIST,
    num_servers: int = 1,
    num_cores_server: int = 1,
) -> None:
    """
    Detect the system and generate the node list.

    This function detects whether the script is running on a local machine or
    a SLURM cluster and generates the appropriate node list. It writes the
    node list to the specified output file.

    Parameters
    ----------
    override : str, optional
        Manually specify the system type. If provided, this value is used
        without performing any detection. Default is None.
    outfile : str, optional
        The path to the output file where the node list will be written.
        Default is NODELIST.
    num_servers : int, optional
        The number of parallel runs (servers). Default is 1.
    num_cores_server : int, optional
        The number of cores per server. Default is 1.

    Examples
    --------
    Generate a node list automatically detecting the system:
        >>> generate_node_list()

    Generate a node list with manual system override:
        >>> generate_node_list(override='SLURM', num_servers=4, num_cores_server=16)
    """
    logger.debug("generate_node_list: Starting generate_node_list...")
    system = detect_system(override)
    if system == "LOCAL":
        _localhost_generate_node_list(outfile, num_servers)
    if system == "SLURM":
        _slurm_generate_node_list(outfile, num_servers, num_cores_server)
    logger.debug("generate_node_list: Finished generate_node_list.")


def read_node_list(file: str = NODELIST) -> List[str]:
    """
    Read the list of nodes from a file.

    This function reads the list of nodes from the specified file and returns
    it as a list of strings.

    Parameters
    ----------
    file : str, optional
        The path to the file containing the node list. Default is NODELIST.

    Returns
    -------
    List[str]
        A list of nodes read from the file.

    Examples
    --------
    Read the node list from the default file:
        >>> nodes = read_node_list()

    Read the node list from a specified file:
        >>> nodes = read_node_list('custom_nodelist.txt')
    """
    logger.debug("read_node_list: Starting read_node_list...")
    fp = open(file, "r")
    nodelist = [h.strip() for h in fp.readlines()]
    fp.close()
    logger.debug("read_node_list: Finished read_node_list.")
    return nodelist


def agent_index_2d_to_1d(i: int, j: int, nz_Qs: int) -> int:
    """
    Convert 2D index to 1D index for agents/jets/pseudo-parallel environments.

    This function converts a 2D index (i, j) to a 1D index in row-major order.
    The 1D index starts from 1 to align with `ENV_ID[1]` usage. Note that the
    2D index starts at zero.

    Parameters
    ----------
    i : int
        The row index (starting from 0).
    j : int
        The column index (starting from 0).
    nz_Qs : int
        The number of columns in the 2D grid.

    Returns
    -------
    int
        The corresponding 1D index.

    Examples
    --------
    Convert 2D index (2, 3) to 1D index with 5 columns:
        >>> agent_index_2d_to_1d(2, 3, 5)
        14
    """
    logger.debug("agent_index_2d_to_1d: Starting agent_index_2d_to_1d...")
    logger.debug("agent_index_2d_to_1d: Finished agent_index_2d_to_1d.")
    return i * nz_Qs + j + 1


def agent_index_1d_to_2d(index: int, nz_Qs: int) -> List[int]:
    """
    Convert 1D index to 2D index for agents/jets/pseudo-parallel environments.

    This function converts a 1D index to a 2D index (i, j) in row-major order.
    The 1D index starts from 1 to align with `ENV_ID[1]` usage. Note that the
    2D index starts at zero.

    Parameters
    ----------
    index : int
        The 1D index.
    nz_Qs : int
        The number of columns in the 2D grid.

    Returns
    -------
    List[int]
        The corresponding 2D index as a list [i, j].

    Examples
    --------
    Convert 1D index 14 to 2D index with 5 columns:
        >>> agent_index_1d_to_2d(14, 5)
        [2, 3]
    """
    logger.debug("agent_index_1d_to_2d: Starting agent_index_1d_to_2d...")
    index -= 1
    i = index // nz_Qs
    j = index % nz_Qs
    logger.debug("agent_index_1d_to_2d: Finished agent_index_1d_to_2d.")
    return [i, j]


def find_highest_timestep_file(directory, case, parameter):
    """
    Find the file with the highest timestep in the specified directory.

    This function searches for files in the specified directory that match
    the given case and parameter strings in their filenames, and identifies
    the file with the highest timestep.

    Parameters
    ----------
    directory : str
        The path to the directory containing the files.
    case : str
        The case string in the file name.
    parameter : str
        The parameter string in the file name.

    Returns
    -------
    str
        The path to the file with the highest timestep.

    Raises
    ------
    FileNotFoundError
        If no files matching the pattern are found.

    Examples
    --------
    Assume the directory contains the following files:

    - `case1-pressure-100.post.mpio.bin`
    - `case1-pressure-90.post.mpio.bin`
    - `case1-pressure-110.post.mpio.bin`

    The function will identify `case1-pressure-110.post.mpio.bin` as the file
    with the highest timestep.

    Find the file with the highest timestep:
        >>> find_highest_timestep_file('/path/to/files', 'case1', 'pressure')
        '/path/to/files/case1-pressure-110.post.mpio.bin'
    """
    logger.debug("find_highest_timestep_file: Starting find_highest_timestep_file...")
    pattern = re.compile(rf"{case}-{parameter}-(\d+)\.post\.mpio\.bin")
    highest_timestep = -1
    highest_file = None

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            timestep = int(match.group(1))
            if timestep > highest_timestep:
                highest_timestep = timestep
                highest_file = filename

    logger.debug("find_highest_timestep_file: Finished find_highest_timestep_file.")
    if highest_file:
        return os.path.join(directory, highest_file)
    else:
        raise FileNotFoundError(
            f"No files matching pattern {case}-{parameter}-*.post.mpio.bin found in {directory}"
        )


def copy_mpio2vtk_required_files(
    case, source_directory, target_directory, identified_file
):
    """
    Copy the identified post.mpio.bin file and additional files for mpio2vtk conversion to the target directory.

    This function copies the specified .mpio.bin file along with a set of
    additional required files from the source directory to the target directory
    for the mpio2vtk conversion process.

    Parameters
    ----------
    case : str
        The case prefix for the filenames.
    source_directory : str
        The directory containing the source files.
    target_directory : str
        The directory to copy the files to.
    identified_file : str
        The path to the identified .mpio.bin file of the last timestep to copy.

    Raises
    ------
    FileNotFoundError
        If any of the required files are not found in the source directory.

    Examples
    --------
    Copy the required files for mpio2vtk conversion:
        >>> copy_mpio2vtk_required_files(
        ...     'case1', '/path/to/source', '/path/to/target', '/path/to/source/case1-pressure-100.post.mpio.bin'
        ... )
    """
    logger.debug(
        "copy_mpio2vtk_required_files: Starting copy_mpio2vtk_required_files..."
    )
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Copy the identified file
    shutil.copy(identified_file, target_directory)

    # List of additional required files
    additional_files = [
        f"{case}-COORD.post.mpio.bin",
        f"{case}-LEINV.post.mpio.bin",
        f"{case}-LNINV.post.mpio.bin",
        f"{case}-LNODS.post.mpio.bin",
        f"{case}-LTYPE.post.mpio.bin",
        f"{case}.post.alyapar",
    ]

    # Copy each additional required file
    for filename in additional_files:
        source_path = os.path.join(source_directory, filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, target_directory)
        else:
            raise FileNotFoundError(
                f"Required file {filename} not found in {source_directory}"
            )
    logger.debug("copy_mpio2vtk_required_files: Finished copy_mpio2vtk_required_files.")


def printDebug(*args) -> None:
    """
    ... # TODO: @pol, @pietero REMOVE? - Pieter
    """
    logger.debug("printDebug: Starting printDebug...")
    if DEBUG:
        logger.debug(*args)
        print(*args)
    logger.debug("printDebug: Finished printDebug.")
