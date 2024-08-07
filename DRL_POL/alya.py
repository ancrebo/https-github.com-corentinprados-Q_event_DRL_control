"""
alya.py

DEEP REINFORCEMENT LEARNING WITH ALYA

This module provides functions to generate and manage meshes, initial
conditions, and various configuration files for ALYA simulations. It
leverages pyAlya tools and other utility functions to streamline the
simulation setup process.

Many functions are currently unused and may be removed in future updates.

Functions
---------
- run_mesh(runpath: str, casename: str, ndim: int, ini_vel: Union[List[str], None] = None) -> None
    Generates the mesh and initial conditions for an ALYA simulation. (CURRENTLY UNUSED!)

- write_case_file(filepath: str, casename: str, simu_name: str) -> None
    Writes the [casename].dat file for an ALYA simulation.

- write_run_type(filepath: str, type: str, freq: int = 1) -> None
    Writes the run type file that is included in the [casename].dat file for ALYA.

- write_time_interval(filepath: str, start_time: float, end_time: float) -> None
    Writes the time interval file that is included in the [casename].dat file for ALYA.

- detect_last_timeinterval(filename: str) -> Union[float, None]
    Detects the last time interval value in a file.

- write_dom_file(filepath: str, casename: str, ncpus: int) -> None
    Writes the [casename].dom.dat file for an ALYA simulation.

- write_ker_file(filepath: str, casename: str, jetlist: List[str], steps: int,
                 postprocess: List[str] = []) -> None
    Writes the [casename].ker.dat file for an ALYA simulation.

- write_physical_properties(filepath: str, rho: float, mu: float) -> None
    Writes the physical_properties.dat file for an ALYA simulation.

- write_inflow_file(filepath: str, functions: List[str]) -> None
    Writes the inflow.dat file for an ALYA simulation.

- write_jet_file(filepath: str, name: str, functions: List[str]) -> None
    Writes a jet file for an ALYA simulation. Typically used to update the jets with actions
    from the DRL agent(s) before a new "action step" is started.

- write_nsi_file(filepath: str, casename: str, varlist=None, witlist=None) -> None
    Writes the [casename].nsi.dat file for an ALYA simulation.

Authors
-------
- Pol Suarez
- Arnau Miro
- Francisco Alcantara
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

import os

from typing import Union, List

from configuration import ALYA_GMSH, ALYA_INCON
from env_utils import run_subprocess

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s\n", __name__, logger.level)


def run_mesh(runpath, casename, ndim, ini_vel=None):
    """
    CURRENTLY UNUSED as of August 7, 2024 - Pieter

    Use pyAlya tools to generate the mesh from gmsh
    """
    logger.debug(
        "alya.run_mesh: Running mesh generation in %s with casename %s and ndim %d",
        runpath,
        casename,
        ndim,
    )
    # Build arguments string
    # Convert from GMSH to ALYA
    if ini_vel is None:
        ini_vel = ["0", "0", "0"]
    args = "-2 " if ndim == 2 else ""
    args += f"-c {casename} {casename}"
    run_subprocess(os.path.join(runpath, "mesh"), ALYA_GMSH, args)
    # TODO: generate periodicity if applicable
    # Generate initial condition
    args = f"--vx {ini_vel[0]} --vy {ini_vel[1]} "
    if len(ini_vel) > 2:
        args += f"--vz {ini_vel[2]} "
    args += f"{casename}"
    run_subprocess(os.path.join(runpath, "mesh"), ALYA_INCON, args)
    # Symbolic link the mesh to the case main folder
    run_subprocess(runpath, "ln", "-s mesh/*.mpio.bin .")
    logger.debug(
        "alya.run_mesh: Finished running mesh generation in %s with casename %s and ndim %d",
        runpath,
        casename,
        ndim,
    )


### Functions to write ALYA configuration files ###


def write_case_file(filepath: str, casename: str, simu_name: str) -> None:
    """
    Write the casename.dat file for ALYA.

    This function creates a .dat file containing the necessary configurations
    for running an ALYA simulation. The file includes information about run
    data, problem data, and MPI I/O settings. Everything is hardcoded except
    for the simulation name and case name.

    Parameters
    ----------
    filepath : str
        The directory path where the .dat file will be written.
    casename : str
        The name of the case for the simulation.
    simu_name : str
        The name of the simulation.

    Returns
    -------
    None
    """
    logger.debug(
        "alya.write_case_file: Writing .dat file to %s with casename %s and simu_name %s",
        filepath,
        casename,
        simu_name,
    )
    file = open(os.path.join(filepath, f"{casename}.dat"), "w")
    file.write(
        f"""$-------------------------------------------------------------------
RUN_DATA
  ALYA:                   {simu_name}
  INCLUDE                 run_type.dat
  LATEX_INFO_FILE:        YES
  LIVE_INFORMATION:       Screen
  TIMING:                 ON
END_RUN_DATA
$-------------------------------------------------------------------
PROBLEM_DATA
  TIME_COUPLING:          Global, from_critical
    INCLUDE               time_interval.dat
    NUMBER_OF_STEPS:      999999

  NASTIN_MODULE:          On
  END_NASTIN_MODULE

  PARALL_SERVICE          ON
    PARTITION_TYPE:       FACES
    POSTPROCESS:          MASTER
    PARTITIONING
    METHOD:               SFC
    EXECUTION_MODE:       PARALLEL
    END_PARTITIONING
  END_PARALL_SERVICE
END_PROBLEM_DATA
$-------------------------------------------------------------------
MPI_IO:        ON
  GEOMETRY:    ON
  RESTART:     ON
  POSTPROCESS: ON
END_MPI_IO
$-------------------------------------------------------------------"""
    )
    file.close()
    logger.debug(
        "alya.write_case_file: Finished writing .dat file to %s with casename %s and simu_name %s",
        filepath,
        casename,
        simu_name,
    )


def write_run_type(filepath: str, type: str, freq: int = 1) -> None:
    """
    Write the run type file included in the .dat file for ALYA.

    This function creates a run_type.dat file specifying the run type,
    preliminary setting, and frequency for the ALYA simulation. The file is
    referenced by the [casename].dat file.

    Parameters
    ----------
    filepath : str
        The directory path where the run_type.dat file will be written.
    type : str
        The type of run to be specified in the file. Used options are "CONTI" and "NONCONTI".
    freq : int, optional
        The frequency setting for the run. How many time steps between restart file creation. Default is 1.

    Returns
    -------
    None
    """
    logger.debug(
        "alya.write_run_type: Writing run type file to %s with type %s and freq %d",
        filepath,
        type,
        freq,
    )
    file = open(os.path.join(filepath, "run_type.dat"), "w")
    # Write file
    file.write(f"RUN_TYPE: {type}, PRELIMINARY, FREQUENCY={freq}\n")
    file.close()
    logger.debug(
        "alya.write_run_type: Finished writing run type file to %s with type %s and freq %d",
        filepath,
        type,
        freq,
    )


def write_time_interval(filepath: str, start_time: float, end_time: float) -> None:
    """
    Write the time interval file included in the [casename].dat file for ALYA.

    This function creates a time_interval.dat file specifying the start
    and end times for the ALYA simulation. Used during training to set a
    specified time interval in an "action step." The file is updated
    during DRL training.

    Parameters
    ----------
    filepath : str
        The directory path where the time_interval.dat file will be written.
    start_time : float
        The start time for the simulation.
    end_time : float
        The end time for the simulation.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.

    Examples
    --------
    Write a time interval file with start time 0.0 and end time 1.0:
        >>> write_time_interval('/path/to/dir', 0.0, 1.0)
    """
    logger.debug(
        "alya.write_time_interval: Writing time interval file to %s with interval %f, %f",
        filepath,
        start_time,
        end_time,
    )
    # Check that file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"alya.write_time_interval: File {filepath} not found")

    file = open(os.path.join(filepath, "time_interval.dat"), "w")
    # Write file
    file.write(f"TIME_INTERVAL: {start_time}, {end_time}\n")
    file.close()
    logger.debug(
        "alya.write_time_interval: Finished writing time interval file to %s with interval %f, %f",
        filepath,
        start_time,
        end_time,
    )


def detect_last_timeinterval(filename: str) -> Union[float, None]:
    """
    Detect the last time interval value in a file.

    This function reads a file and searches for the 'TIME_INTERVAL:' line.
    It returns the last value found in the 'TIME_INTERVAL:' entry. The file
    is typically `time_interval.dat` and is used to make sure the new "action
    step" continues from the previous time interval.

    Parameters
    ----------
    filename : str
        The path to the file to be read.

    Returns
    -------
    Union[float, None]
        The last time interval value found, or None if no interval is found.
    """
    logger.debug(
        "alya.detect_last_timeinterval: Detecting last time interval in %s", filename
    )
    # Open the file in read mode
    with open(filename, "r") as file:
        # Read all lines from the file
        lines = file.readlines()

        # Loop through each line
        for line in lines:
            # Check if the line contains 'TIME_INTERVAL'
            if "TIME_INTERVAL:" in line:
                # Find the position of 'TIME_INTERVAL:'
                time_interval_pos = line.find("TIME_INTERVAL:")
                # Extract the substring after 'TIME_INTERVAL:'
                substring = line[time_interval_pos + len("TIME_INTERVAL:") :]

                # Split the substring by comma to get individual values
                values = substring.split(",")

                # If there are values after 'TIME_INTERVAL:', return the LAST one
                if len(values) > 1:
                    logger.debug(
                        "alya.detect_last_timeinterval: Found time interval in %s, returning last value %s",
                        filename,
                        values[1].strip(),
                    )
                    return float(values[1].strip())

    # If no values are found after 'TIME_INTERVAL:', return None
    logger.debug(
        "alya.detect_last_timeinterval: No time interval found in %s", filename
    )
    return None


def write_dom_file(filepath: str, casename: str, ncpus: int) -> None:
    """
    Write the [casename].dom.dat file for an ALYA simulation.

    This function creates a [casename].dom.dat file containing configurations for
    dimensions, strategy, geometry, sets, boundary conditions, and fields
    for an ALYA simulation.

    Parameters
    ----------
    filepath : str
        The directory path where the [casename].dom.dat file will be written.
    casename : str
        The name of the case for the simulation.
    ncpus : int
        The number of CPUs to be specified in the strategy section.

    Returns
    -------
    None

    Examples
    --------
    Write a .dom.dat file for 'example_case' with 4 CPUs:
        >>> write_dom_file('/path/to/dir', 'example_case', 4)
    """
    logger.debug("alya.write_dom_file: Writing .dom.dat file to %s", filepath)
    file = open(os.path.join(filepath, "%s.dom.dat" % casename), "w")
    file.write(
        """$------------------------------------------------------------
DIMENSIONS
  INCLUDE	                  mesh/%s.dims.dat
  INCLUDE                   fields.dat
END_DIMENSIONS
$------------------------------------------------------------
STRATEGY
  GROUPS = %d, SEQUENTIAL_FRONTAL
  DOMAIN_INTEGRATION_POINTS:  0
  INTEGRATION_RULE:           OPEN
  BOUNDARY_ELEMENT:           OFF
  EXTRAPOLATE:                ON
  PERIODICITY:                MATRIX
END_STRATEGY
$-------------------------------------------------------------
GEOMETRY
END_GEOMETRY
$-------------------------------------------------------------
SETS
END_SETS
$-------------------------------------------------------------
BOUNDARY_CONDITIONS
END_BOUNDARY_CONDITIONS
$-------------------------------------------------------------
FIELDS
END_FIELDS
$-------------------------------------------------------------"""
        % (casename, ncpus)
    )
    file.close()
    logger.debug("alya.write_dom_file: Finished writing .dom.dat file to %s", filepath)


def write_ker_file(
    filepath: str,
    casename: str,
    jetlist: List[str],
    steps: int,
    postprocess: List[str] = [],
) -> None:
    """
    Write the [casename].ker.dat file for an ALYA simulation.

    This function creates a [casename].ker.dat file containing configurations for
    physical problems, numerical treatments, and output/post-process
    settings for an ALYA simulation. Postprocessing frequency and variables override
    any postprocessing set in the [casename].nsi.dat file.

    Parameters
    ----------
    filepath : str
        The directory path where the [casename].ker.dat file will be written.
    casename : str
        The name of the case for the simulation.
    jetlist : List[str]
        A list of jet files to be included.
    steps : int
        The number of steps between ALYA post-processing.
    postprocess : List[str], optional
        A list of variables for post-processing. Common variables include
        "VELOC", "PRESSURE", "AVVEL", "VELOX", "VELOY", and "VELOZ".
        Defaults to an empty list.

    Returns
    -------
    None

    Examples
    --------
    Write a .ker.dat file for 'example_case' with 100 steps and CODNO postprocess:
        >>> write_ker_file('/path/to/dir', 'example_case', ['jet1', 'jet2'], 100,
                           ['CODNO'])
    """
    logger.debug("alya.write_ker_file: Writing .ker.dat file to %s", filepath)
    # Create jet includes
    jet_includes = ""
    for jet in jetlist:
        jet_includes += f"    INCLUDE {jet}.dat\n"
    # Create variable postprocess
    var_includes = ""
    for var in postprocess:
        var_includes += f"  POSTPROCESS {var}\n"
    # Write file
    file = open(os.path.join(filepath, f"{casename}.ker.dat"), "w")
    file.write(
        f"""$------------------------------------------------------------
PHYSICAL_PROBLEM
  PROPERTIES
    INCLUDE       physical_properties.dat
  END_PROPERTIES
END_PHYSICAL_PROBLEM  
$------------------------------------------------------------
NUMERICAL_TREATMENT 
  MESH
    MASS:             CONSISTENT
    ELEMENTAL_TO_CSR: On
  END_MESH
  ELSEST
     STRATEGY: BIN
     NUMBER:   100,100,100
     DATAF:   LINKED_LIST
  END_ELSEST
  HESSIAN      OFF
  LAPLACIAN    OFF

  SPACE_&_TIME_FUNCTIONS
    INCLUDE inflow.dat
{jet_includes}
  END_SPACE_&_TIME_FUNCTIONS
END_NUMERICAL_TREATMENT  
$------------------------------------------------------------
OUTPUT_&_POST_PROCESS 
  $ Variable postprocess 
  STEPS={steps}
{var_includes}
  $ Witness points
  INCLUDE witness.dat
END_OUTPUT_&_POST_PROCESS  
$------------------------------------------------------------"""
    )
    file.close()
    logger.debug("alya.write_ker_file: Finished writing .ker.dat file to %s", filepath)


def write_physical_properties(filepath: str, rho: float, mu: float) -> None:
    """
    Write the physical_properties.dat file for an ALYA simulation.

    This function creates a physical_properties.dat file specifying the
    material properties, including density and viscosity, for an ALYA
    simulation. This file is included in the [casename].ker.dat file.

    Parameters
    ----------
    filepath : str
        The directory path where the physical_properties.dat file will be written.
    rho : float
        The density value for the material.
    mu : float
        The viscosity value for the material.

    Returns
    -------
    None

    Examples
    --------
    Write a physical properties file with density 1000 and viscosity 0.001:
        >>> write_physical_properties('/path/to/dir', 1000.0, 0.001)
    """
    logger.debug(
        "alya.write_physical_properties: Writing physical properties file to %s",
        filepath,
    )
    file = open(os.path.join(filepath, "physical_properties.dat"), "w")
    # Write file
    file.write("MATERIAL: 1\n")
    file.write(f"  DENSITY:   CONSTANT, VALUE={rho}\n")
    file.write(f"  VISCOSITY: CONSTANT, VALUE={mu}\n")
    file.write("END_MATERIAL\n")
    file.close()
    logger.debug(
        "alya.write_physical_properties: Finished writing physical properties file to %s",
        filepath,
    )


def write_inflow_file(filepath: str, functions: List[str]) -> None:
    """
    Write the inflow.dat file for an ALYA simulation.

    This function creates an inflow.dat file specifying the inflow functions
    to be included in the [casename].ker.dat file for an ALYA simulation.

    Parameters
    ----------
    filepath : str
        The directory path where the inflow.dat file will be written.
    functions : List[str]
        A list of inflow functions to be included.

    Returns
    -------
    None

    Examples
    --------
    Write an inflow file with two functions:
        >>> write_inflow_file('/path/to/dir', ['function1', 'function2'])
    """
    logger.debug("alya.write_inflow_file: Writing inflow file to %s", filepath)
    file = open(os.path.join(filepath, "inflow.dat"), "w")
    # Write file
    file.write(f"FUNCTION=INFLOW, DIMENSION={len(functions)}\n")
    for f in functions:
        file.write(f"  {f}\n")
    file.write("END_FUNCTION\n")
    file.close()
    logger.debug("alya.write_inflow_file: Finished writing inflow file to %s", filepath)


def write_jet_file(filepath: str, name: str, functions: List[str]) -> None:
    """
    Write a jet file for an ALYA simulation.

    This function creates a jet file specifying the jet functions to be
    included in the .ker.dat file for an ALYA simulation.

    Parameters
    ----------
    filepath : str
        The directory path where the jet file will be written.
    name : str
        The name of the jet.
    functions : List[str]
        A list of jet functions to be included.

    Returns
    -------
    None

    Examples
    --------
    Write a jet file named 'jet1' with two functions (2D not 3D):
        >>> write_jet_file('/path/to/dir', 'jet1', ['function1', 'function2'])
    """
    logger.debug("alya.write_jet_file: Writing jet file %s to %s", name, filepath)
    file = open(os.path.join(filepath, f"{name}.dat"), "w")
    # Write file
    file.write(f"FUNCTION={name.upper()}, DIMENSION={len(functions)}\n")
    for f in functions:
        file.write(f"  {f}\n")
    file.write("END_FUNCTION\n")
    file.close()
    logger.debug(
        "alya.write_jet_file: Finished writing jet file %s to %s", name, filepath
    )


def write_nsi_file(
    filepath: str,
    casename: str,
    varlist=None,
    witlist=None,
) -> None:
    """
    Write the [casename].nsi.dat file for an ALYA simulation.

    This function creates a [casename].nsi.dat file containing configurations for
    physical problems, numerical treatments, and output/post-process settings
    for an ALYA simulation.

    Parameters
    ----------
    filepath : str
        The directory path where the [casename].nsi.dat file will be written.
    casename : str
        The name of the case for the simulation.
    varlist : list of str, optional
        A list of variables for post-processing. Common variables include
        "VELOC", "PRESS", "AVVEL". Defaults to ["VELOC", "PRESS"].
    witlist : list of str, optional
        A list of variables for post-processing witness points. Variables are
        "VELOX", "VELOY", "VELOZ", "PRESS". Defaults to ["VELOX", "VELOY",
        "VELOZ", "PRESS"].

    Returns
    -------
    None

    Examples
    --------
    Write a .nsi.dat file for 'example_case':
        >>> write_nsi_file('/path/to/dir', 'example_case')
    """
    logger.debug(
        "alya.write_nsi_file: Writing .nsi.dat file to %s with casename %s",
        filepath,
        casename,
    )
    # Create variable postprocess
    if witlist is None:
        witlist = ["VELOX", "VELOY", "VELOZ", "PRESS"]

    if varlist is None:
        varlist = ["VELOC", "PRESS"]

    var_includes = ""
    for var in varlist:
        var_includes += f"  POSTPROCESS {var}\n"
    # Create jet includes
    wit_includes = ""
    for var in witlist:
        wit_includes += f"    {var}\n"
    # Write file
    file = open(os.path.join(filepath, f"{casename}.nsi.dat"), "w")
    file.write(
        f"""$------------------------------------------------------------
PHYSICAL_PROBLEM
  PROBLEM_DEFINITION       
    TEMPORAL_DERIVATIVES:	On  
    CONVECTIVE_TERM:	    EMAC
    VISCOUS_TERM:	        LAPLACIAN
  END_PROBLEM_DEFINITION  

  PROPERTIES
  END_PROPERTIES  
END_PHYSICAL_PROBLEM  
$------------------------------------------------------------
NUMERICAL_TREATMENT 
  TIME_STEP:            EIGENVALUE
  ELEMENT_LENGTH:       Minimum
  STABILIZATION:        OFF
  TIME_INTEGRATION:     RUNGE, ORDER:2
  SAFETY_FACTOR:        1.0
  STEADY_STATE_TOLER:   1e-10
  ASSEMBLY:             GPU2
  VECTOR:               ON
  NORM_OF_CONVERGENCE:  LAGGED_ALGEBRAIC_RESIDUAL
  MAXIMUM_NUMBER_OF_IT:	1
  GRAD_DIV:             ON
  DIRICHLET:            MATRIX

  ALGORITHM: SEMI_IMPLICIT
  END_ALGORITHM

  MOMENTUM
    ALGEBRAIC_SOLVER     
      SOLVER: EXPLICIT, LUMPED
    END_ALGEBRAIC_SOLVER        
  END_MOMENTUM

  CONTINUITY 
     ALGEBRAIC_SOLVER
       SOLVER:         DEFLATED_CG, COARSE: SPARSE
       CONVERGENCE:    ITERA=1000, TOLER=1.0e-10, ADAPTATIVE, RATIO=1e-2
       OUTPUT:         CONVERGENCE
       PRECONDITIONER: LINELET, NEVER_CHANGE
     END_ALGEBRAIC_SOLVER        
  END_CONTINUITY

  VISCOUS
    ALGEBRAIC_SOLVER
       SOLVER:         CG, COARSE: SPARSE, KRYLOV:10
       CONVERGENCE:    ITERA=500, TOLER=1.0e-10, ADAPTIVE, RATIO=1e-3
       OUTPUT:         CONVERGENCE
       PRECONDITIONER: DIAGONAL
    END_ALGEBRAIC_SOLVER    
  END_VISCOUS
END_NUMERICAL_TREATMENT  
$------------------------------------------------------------
OUTPUT_&_POST_PROCESS
  START_POSTPROCES_AT STEP  = 0
  $ Variables
{var_includes}  
  $ Forces at boundaries
  BOUNDARY_SET
	  FORCE
  END_BOUNDARY_SET
  $ Variables at witness points
  WITNESS_POINTS
{wit_includes}
  END_WITNESS
END_OUTPUT_&_POST_PROCESS  
$------------------------------------------------------------
BOUNDARY_CONDITIONS, NON_CONSTANT
  PARAMETERS
    INITIAL_CONDITIONS: VALUE_FUNCTION = 1 $ use field 1 for initial condition
    FIX_PRESSURE:       OFF
    VARIATION:          NON_CONSTANT
  END_PARAMETERS
  $ Boundary codes
  INCLUDE boundary_codes.dat
END_BOUNDARY_CONDITIONS  
$------------------------------------------------------------"""
    )
    file.close()
    logger.debug(
        "alya.write_nsi_file: Finished writing .nsi.dat file to %s with casename %s",
        filepath,
        casename,
    )
