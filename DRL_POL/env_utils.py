#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Pol Suarez, Arnau Miro, Francisco Alcantara, Xavi Garcia
# 01/02/2023
from __future__ import print_function, division
from typing import Optional, List

import os, subprocess
from configuration import NODELIST, USE_SLURM, DEBUG


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
    Use python to call a terminal command
    """

    # Auxilar function to build parallel command
    def _cmd_parallel(runbin: str, **kwargs) -> str:
        """
        Build the command to run in parallel
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
        runbin = "run_reward_in_new_env.sh " + runbin  # Prepend the shell script
    cmd_bin = (
        _cmd_parallel(f"{runbin} {runargs}", **kwargs)
        if parallel
        else f"{runbin} {runargs}"
    )
    cmd = f"cd {runpath} && {cmd_bin} {arg_log}"  # TODO: DARDEL DEBUG ONGOING
    # print('POOOOOOOOOOOOOL --> cmd: %s' % cmd)

    # # Execute run
    # retval = subprocess.call(cmd, shell=True)  # old version

    # Execute run (alternate, updated version introduced in Python 3.5)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Print the stdout and stderr from the shell script
    print(result.stdout)
    print(result.stderr)

    # Check return
    if check_return and result.returncode != 0:
        raise ValueError(f"Error running command <{cmd}>!\n{result.stderr}")

    # Return value
    return result.returncode


def detect_system(override: str = None) -> str:
    """
    Test if we are in a cluster or on a local machine
    """
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
    return out


def _slurm_generate_node_list(
    outfile, num_servers: int, num_cores_server: int, **kwargs
) -> None:
    """
    Generate the list of nodes using slurm.
            > num_servers:      number of parallel runs
            > num_cores_server: number of cores per server using CFD
            > num_nodes:        number of nodes of the allocation (obtained from slurm environment variable)
            > num_cores_node:   number of cores per node (obtained from slurm environment variable)

    num_nodes and num_cores_node refer to the server configuration meanwhile
    num_servers (number of environments in parallel) and num_cores_server (processors per environment)
    refer to the DRL configuration.

    SLURM_JOB_NODELIST does not give the exact list of nodes as we would want
    """
    # print("POOOOOL --> SLURM_NNODES: %s" %os.getenv('SLURM_NNODES'))
    # print("POOOOOL --> SLURM_JOB_CPUS_PER_NODE: %s" %os.getenv('SLURM_JOB_CPUS_PER_NODE'))

    num_nodes = kwargs.get("num_nodes", int(os.getenv("SLURM_NNODES")))
    num_cores_node = kwargs.get("num_cores_node", os.getenv("SLURM_JOB_CPUS_PER_NODE"))
    start = num_cores_node.find("(")
    num_cores_node = int(num_cores_node[:start])
    num_cores_node = 100
    print(f"POOOOOL --> SLURM_JOB_CPUS_PER_NODE: {num_cores_node}" % num_cores_node)

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


def _localhost_generate_node_list(outfile, num_servers: int) -> None:
    """
    Generate the list of nodes for a local run
    """
    hostlist = "localhost"
    for iserver in range(num_servers):
        hostlist += "\nlocalhost"
    # Basically write localhost as the list of nodes
    # Add n+1 nodes as required per the nodelist
    run_subprocess("./", "echo", f'"{hostlist}"', log=outfile)


def generate_node_list(
    override: str = None,
    outfile=NODELIST,
    num_servers: int = 1,
    num_cores_server: int = 1,
) -> None:
    """
    Detect the system and generate the node list
    """
    system = detect_system(override)
    if system == "LOCAL":
        _localhost_generate_node_list(outfile, num_servers)
    if system == "SLURM":
        _slurm_generate_node_list(outfile, num_servers, num_cores_server)


def read_node_list(file: str = NODELIST) -> List[str]:
    """
    Read the list of nodes
    """
    fp = open(file, "r")
    nodelist = [h.strip() for h in fp.readlines()]
    fp.close()
    return nodelist


def agent_index_2d_to_1d(i: int, j: int, nz_Qs: int) -> int:
    """
    Convert 2D index to 1D index (for agents/jets/pseudo-parallel environments)
    Row-major order
    1D starts from 1 to align with `ENV_ID[1]` usage
    """
    return i * nz_Qs + j + 1


def agent_index_1d_to_2d(index: int, nz_Qs: int) -> List[int]:
    """
    Convert 1D index to 2D index (for agents/jets/pseudo-parallel environments)
    Row-major order
    1D starts from 1 to align with `ENV_ID[1]` usage
    """
    index -= 1
    i = index // nz_Qs
    j = index % nz_Qs
    return [i, j]


def printDebug(*args) -> None:
    """
    ...
    """
    if DEBUG:
        print(*args)
