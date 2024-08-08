"""
witness.py

DEEP REINFORCEMENT LEARNING WITH ALYA

This module focuses on witness file creation, reading, and extraction for ALYA
simulations. It provides functions to read and parse witness file headers, read
instants from witness files, and create new witness files. Additionally, it
includes a function to visualize the witness points, which is used during the
creation of witness.dat files.

The module is designed to be imported and called from other scripts, such as
`parameters.py`, where it can be used to generate new witness.dat files for
specific cases.

Functions
---------
- readWitnessHeader(file: TextIO) -> Dict[str, Union[int, Dict[str, int]]]:
    Reads the header from an ALYA witness file.

- readWitnessInstant(file: TextIO, nwit: int) -> Tuple[int, float, np.ndarray]:
    Reads an instant from an ALYA witness file.

- witnessReadNByFront(filename: str, n: int) -> Tuple[np.ndarray, np.ndarray,
  Dict[str, np.ndarray]]:
    Reads N instants from the top of an ALYA witness file.

- witnessReadNByBehind(filename: str, n: int) -> Tuple[np.ndarray, np.ndarray,
  Dict[str, np.ndarray]]:
    Reads N instants from the bottom of an ALYA witness file.

- read_last_wit(filename: str, probe_type: str, norm: Dict[str, float],
  n_to_read: int = 1) -> Dict[str, np.ndarray]:
    Reads and normalizes the last N instants from an ALYA witness file.

- calculate_channel_witness_coordinates(params: Dict[str, Any]) -> Dict[str,
  Any]:
    Calculates witness coordinates and indices for a channel pattern.

- write_witness_file(filepath: str, probes_positions: List[Tuple[float, float,
  float]]) -> None:
    Writes the witness.dat file that needs to be included in the .ker.dat file.

- write_witness_version_file(filepath: str, probes_location: int, probe_type:
  str, pattern: str, y_value_density: int, y_skipping: bool, y_skip_values:
  int, nx_Qs: int, nz_Qs: int) -> None:
    Writes a text file to indicate what version of the witness file is being
    used.

- write_witness_file_and_visualize(case_folder: str, output_params: Dict[str,
  Any], probes_location: int = 5, pattern: str = "X", y_value_density: int = 8,
  y_skipping: bool = False, y_skip_values: int = 1, nx_Qs: int = 1, nz_Qs: int =
  1) -> None:
    Creates the witness.dat file and visualizes the witness points, saving a
    plot to the case folder.

Usage
-----
This module is intended to be imported and used in other scripts. For example,
in `parameters.py`, you can create a new witness.dat file for a specific case
using the relevant functions from this module.

Dependencies
------------
- logging
- os
- numpy as np
- cr (for cr_start, cr_stop)
- logging_config (for configure_logger, DEFAULT_LOGGING_LEVEL)

Version History
---------------
- Major update in August 2024.

Authors
-------
- Pol Suarez
- Arnau Miro
- Francisco Alcantara
- Pieter Orlandini
"""

from __future__ import print_function, division

from typing import Tuple, TextIO, Dict, Union, List, Any

import logging

import os
import numpy as np

from cr import cr_start, cr_stop

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s", __name__, logger.level)


def readWitnessHeader(file: TextIO) -> Dict[str, Union[int, Dict[str, int]]]:
    """
    Reads a header line by line from an ALYA witness file. The file needs to be previously opened.

    This function parses the header of an ALYA witness file to extract key information
    such as the number of variables, number of witness points, and the mapping of
    variable names to their indices. It stops reading when the 'START' keyword is encountered.

    Parameters
    ----------
    file : TextIO
        The file object representing the ALYA witness file.

    Returns
    -------
    Dict[str, Union[int, Dict[str, int]]]
        A dictionary containing the header information with keys:
        - 'VARIABLES': A dictionary mapping variable names to their indices.
        - 'NVARS': The number of variables.
        - 'NWIT': The number of witness points.
        - 'NLINES': The number of lines read.

    Raises
    ------
    ValueError # TODO: @pietero ADD THIS TO THE FUNCTION - Pieter
        If the header contains invalid data or if expected keys are missing.
    """
    # Variables
    header = {"VARIABLES": {}, "NVARS": -1, "NWIT": -1, "NLINES": 0}
    do_skip = True
    # Start reading a file line by line
    for line in file:
        # Increase the number of lines read
        header["NLINES"] += 1
        # Stopping criteria when START is read
        if "START" in line:
            break
        # Stop skipping lines up to when HEADER is read
        if not "HEADER" in line and do_skip:
            continue
        if "HEADER" in line:
            do_skip = False
            continue
        # Parse the kind of line in the header
        if "NUMVARIABLES" in line:
            header["NVARS"] = int(line.split(":")[1])
            continue
        if "NUMSETS" in line:
            header["NWIT"] = int(line.split(":")[1])
            continue
        # Else we are heading variables
        linep = line.split()
        header["VARIABLES"][linep[1]] = (
            int(linep[5]) - 1
        )  # to account for python indexing
    # Function return
    return header


def readWitnessInstant(file: TextIO, nwit: int) -> Tuple[int, float, np.ndarray]:
    """
    Reads an instant from an ALYA witness file.

    This function reads a single time step (instant) from an ALYA witness file,
    extracting the iteration number, time, and data matrix for the specified number of witness points.
    The cursor should be positioned at the beginning of the time step data.

    Parameters
    ----------
    file : TextIO
        The file object representing the ALYA witness file.
    nwit : int
        The number of witness points.

    Returns
    -------
    Tuple[int, float, np.ndarray]
        A tuple containing:
        - int: The iteration number.
        - float: The time.
        - np.ndarray: The data matrix.

    Raises
    ------
    ValueError # TODO: @pietero ADD THIS TO THE FUNCTION - Pieter
        If the instant data is invalid or corrupted.
    """
    logger.debug("readWitnessInstant: Reading instant...")
    # Read iterations
    line = file.readline()
    split_line = line.split()
    it = int(line.split()[3])
    # Read time
    line = file.readline()
    time = float(line.split()[3])
    # Now read the data matrix
    data = np.genfromtxt(file, max_rows=nwit)
    # Function return
    logger.debug("readWitnessInstant: Finished reading instant!")
    return it, time, data


def witnessReadNByFront(
    filename: str, n: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Reads N instants from the top of an ALYA witness file.

    This function opens the specified ALYA witness file and reads the header
    to determine the structure of the data. It then reads the first N time
    steps (instants) from the file, extracting iteration numbers, times, and
    data arrays for each variable.

    (As of 2024-08-07, this function is not used in the codebase.)

    Parameters
    ----------
    filename : str
        The path to the witness file.
    n : int
        The number of instants to read.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]
        A tuple containing:
        - np.ndarray: An array of iteration numbers.
        - np.ndarray: An array of times.
        - Dict[str, np.ndarray]: A dictionary of data arrays for each variable.
    """
    # Open file for reading
    file = open(filename, "r")
    # Read witness file header
    header = readWitnessHeader(file)
    # Preallocate outputs
    iter = np.zeros((n,), dtype=np.double)
    time = np.zeros((n,), dtype=np.double)
    data = {}
    for v in header["VARIABLES"].keys():
        data[v] = np.zeros((n, header["NWIT"]), dtype=np.double)
    # Read a number of instants from the witness file starting
    # from the front of the file
    for ii in range(n):
        # Read file
        it, t, d = readWitnessInstant(file, header["NWIT"])
        # Set data
        iter[ii] = it
        time[ii] = t
        for v in header["VARIABLES"]:
            data[v][ii, :] = d[:, header["VARIABLES"][v]]
    # Close file
    file.close()
    # Function return
    return iter, time, data


def witnessReadNByBehind(
    filename: str, n: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Reads N instants (ALYA time steps) starting from the bottom of the witness file.

    This function reads the last N instants from an ALYA witness file. It calculates
    the offset required to position the cursor at the beginning of the last N time steps
    and reads the data from there. It extracts iteration numbers, times, and data arrays
    for each variable.

    Parameters
    ----------
    filename : str
        The path to the witness file.
    n : int
        The number of instants (ALYA time steps) to read.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]
        A tuple containing:
        - np.ndarray: An array of iteration numbers.
        - np.ndarray: An array of times.
        - Dict[str, np.ndarray]: A dictionary of data arrays for each variable.

    Raises
    ------
    ValueError
        If the file does not contain enough data for the requested number of timesteps.
    """
    logger.debug("witnessReadNByBehind: Reading %s instants from %s", n, filename)
    logger.debug("witnessReadNByBehind: Reading witness file...")
    # Open file for reading
    file = open(filename, "r")
    # Read witness file header
    header = readWitnessHeader(file)
    logger.debug(
        "witnessReadNByBehind: Finished reading witness file header: \n%s", header
    )
    logger.debug(
        "witnessReadNByBehind: Number of witness points from header: %s", header["NWIT"]
    )
    # Preallocate outputs
    iter = np.zeros((n,), dtype=np.double)
    time = np.zeros((n,), dtype=np.double)
    data = {}
    for v in header["VARIABLES"].keys():
        data[v] = np.zeros((n, header["NWIT"]), dtype=np.double)
    logger.debug("witnessReadNByBehind: Preallocated outputs")
    # Set the cursor from the bottom of the file up a certain
    # number of lines
    file.seek(0, 2)  # seek to end of file; f.seek(0, 2) is legal
    file_end_position = file.tell()
    logger.debug("witnessReadNByBehind: File end position: %d", file_end_position)

    # First we compute how much of an offset an instant is
    # a variable is 17 characters
    # the witness id is 10 characters
    # thus a line is 27 characters - we have nwit lines
    # We need to add the lines on iterations and time
    # Iterations are 37 characters # TODO: was this for 2D??? - Pieter
    # Iteration line for FOR 3D CHANNEL is 46 characters - Pieter
    # Time are 32 characters
    offset = (10 + 17 * (header["NVARS"] - 1) + 1) * header["NWIT"] + 46 + 1 + 32 + 1
    logger.debug("witnessReadNByBehind: Offset computed: %s", offset)
    # file.seek(0,2) moves the cursor to the end of the file
    # so the following should move the cursor from the bottom to
    # the beginning of the n instants that must be read

    # Ensure we have enough data in the file
    if file_end_position < n * offset:
        raise ValueError(
            "The file does not contain enough data for the requested number of timesteps."
        )

    file.seek(file.tell() - n * offset, os.SEEK_SET)  # go backwards n instants
    logger.debug(
        "witnessReadNByBehind: Cursor set to the beginning of the last %s instants", n
    )
    # Read a number of instants from the witness file starting
    # from the front of the file
    for ii in range(n):
        # Read file
        logger.debug(
            "witnessReadNByBehind: Reading instant %s of %s...",
            ii,
            n,
        )
        it, t, d = readWitnessInstant(file, header["NWIT"])
        # Set data
        iter[ii] = it
        time[ii] = t
        for v in header["VARIABLES"]:
            data[v][ii, :] = d[:, header["VARIABLES"][v]]
    # Close file
    file.close()
    logger.debug("witnessReadNByBehind: Closed file")
    # Function return
    return iter, time, data


def read_last_wit(
    filename: str, probe_type: str, norm: Dict[str, float], n_to_read: int = 1
) -> Dict[str, np.ndarray]:
    """
    Skips all the data from the entire time domain and reads the last time step from *.nsi.wit.

    Additionally, normalizes the data using the provided norm dictionary.

    This function reads the last N instants from an ALYA witness file, averages them if necessary,
    and normalizes the data using the provided normalization values. It supports two types of probes:
    'pressure' and 'velocity'.

    Parameters
    ----------
    filename : str
        The path to the witness file.
    probe_type : str
        The type of probe ('pressure' or 'velocity').
    norm : Dict[str, float]
        A dictionary containing normalization values for each variable.
    n_to_read : int, optional
        The number of instants to read. Default is 1.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the last values for each variable, normalized.

    Raises
    ------
    ValueError
        If `probe_type` is not 'pressure' or 'velocity'.
        If normalization value for a variable is not found in the norm dictionary.
    """
    cr_start("WIT.read_last_wit", 0)
    logger.info(
        "witness.read_last_wit: Starting to process the last witness file at %s ...",
        filename,
    )

    # Read witness file
    itw, timew, data = witnessReadNByBehind(filename, n_to_read)
    logger.debug(
        "witness.read_last_wit: Finished reading the last witness file at %s", filename
    )

    logger.debug("witness.read_last_wit: Length of data: %s", len(data))
    logger.debug("witness.read_last_wit: keys of data: %s", data.keys())

    # Initialize result dictionary
    result_data = {}

    # Select which variable to work with
    if probe_type == "pressure":
        result_data["pressure"] = data["PRESSURE"]
    if probe_type == "velocity":
        vel_components = ["VELOX", "VELOY", "VELOZ"]
        for comp in vel_components:
            result_data[comp] = data[comp]
    else:
        raise ValueError(
            "Witness.py: read_last_wit: Invalid `probe_type`: must be 'pressure' or 'velocity'"
        )
    logger.debug("witness.read_last_wit: Selected probe type: %s", probe_type)

    # If we have more than one instant, average them
    if n_to_read > 1:
        for key in result_data.keys():
            result_data[key] = (
                1.0 / (timew[-1] - timew[0]) * np.trapz(result_data[key], timew, axis=0)
            )
        logger.info("witness.read_last_wit: Averaged %s instants", n_to_read)

    # Normalize the data using the provided norm dictionary
    for key in result_data.keys():
        if key in norm:
            result_data[key] = result_data[key] / norm[key]
        else:
            raise ValueError(
                f"Witness.py: read_last_wit: Normalization value for {key} not found in norm dictionary"
            )
    logger.info("witness.read_last_wit: Data Normalized")

    # Ensure that data has the correct shape
    for key in result_data.keys():
        result_data[key] = result_data[key][0, :]  # (nprobes,)
    logger.debug("witness.read_last_wit: keys of result_data: %s", result_data.keys())

    logger.info("witness.read_last_wit: Finished witness file processing!")
    cr_stop("WIT.read_last_wit", 0)
    return result_data


def calculate_channel_witness_coordinates(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate witness coordinates and indices for a channel pattern.

    This function generates coordinates and indices for witness probes placed in a
    channel pattern. It supports two pattern types: 'X' and '+'. The function uses
    parameters such as the number of agents in each direction, lengths of the different directions,
    y-value density, and skipping options to compute the positions and indices of the probes.

    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing the parameters:
        - probe_type (str): Type of probes used, either 'velocity' or 'pressure'.
        - n (int): Number of sections in the x direction.
        - m (int): Number of sections in the z direction.
        - Lx (float): Length in the x direction.
        - Ly (float): Length in the y direction.
        - Lz (float): Length in the z direction.
        - y_value_density (int): Density of y values. Number of y values total.
        - pattern (str): Pattern type ('X' or '+').
        - y_skipping (bool): Whether to skip y values in between pattern layers.
        - y_skip_values (int): Number of layers between each full pattern layer if y_skipping is True.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the following keys:
        - locations (List[Tuple[float, float, float]]): Coordinates of the probes.
        - tag_probs (Dict[str, List[int]]): Tag ranges for different patterns.
        - probe_type (str): Type of probes used, either 'velocity' or 'pressure'.
        - indices2D (List[Tuple[int, int]]): 2D indices of the probes.
        - indices1D (List[int]): 1D indices of the probes.

    Raises
    ------
    ValueError
        If an invalid pattern is specified.
        If y-values are out of range.
    """
    logger.debug(
        "calculate_channel_witness_coordinates: Starting to calculate witness coordinates..."
    )

    probe_type = params["probe_type"]
    n = params["nx_Qs"]
    m = params["nz_Qs"]
    Lx = params["Lx"]
    Ly = params["Ly"]
    Lz = params["Lz"]
    y_value_density = params["y_value_density"]
    pattern = params["pattern"]
    y_skipping = params["y_skipping"]
    y_skip_values = params["y_skip_values"]

    # Create list of y values to place pattern - Exclude the first term (0) and last term (Ly)
    y_values: List[float] = np.linspace(0, Ly, y_value_density + 2).tolist()[1:-1]

    coordinates: List[Tuple[float, float, float]] = []
    indices2D: List[Tuple[int, int]] = []
    indices1D: List[int] = []

    step_x: float = Lx / n
    step_z: float = Lz / m

    for i in range(n):
        for j in range(m):
            center_x: float = (i + 0.5) * step_x
            center_z: float = (j + 0.5) * step_z

            if pattern == "X":
                end_points: List[Tuple[float, float]] = [
                    (center_x - 0.25 * step_x, center_z - 0.25 * step_z),
                    (center_x + 0.25 * step_x, center_z - 0.25 * step_z),
                    (center_x - 0.25 * step_x, center_z + 0.25 * step_z),
                    (center_x + 0.25 * step_x, center_z + 0.25 * step_z),
                ]
            elif pattern == "+":
                end_points: List[Tuple[float, float]] = [
                    (center_x - 0.25 * step_x, center_z),
                    (center_x + 0.25 * step_x, center_z),
                    (center_x, center_z - 0.25 * step_z),
                    (center_x, center_z + 0.25 * step_z),
                ]
            else:
                raise ValueError(
                    f"calculate_channel_witness_coordinates: Invalid pattern: {pattern}"
                )

            center_point: Tuple[float, float] = (center_x, center_z)

            for index, y in enumerate(y_values):
                if 0 <= y <= Ly:  # Ensure y-values are within the global y limit
                    if y_skipping and (index % y_skip_values != 0):
                        # Place only the center point
                        coordinates.append((center_point[0], y, center_point[1]))
                        indices2D.append((i, j))
                    else:
                        # Place the full pattern
                        for x, z in end_points:
                            coordinates.append((x, y, z))
                            indices2D.append((i, j))
                        # Also place the center point
                        coordinates.append((center_point[0], y, center_point[1]))
                        indices2D.append((i, j))
                else:
                    raise ValueError(
                        f"calculate_channel_witness_coordinates: Invalid y-value: {y}"
                    )

    # Create 1D index from 2D index using row-major format
    for index2D in indices2D:
        indices1D.append(index2D[0] * m + index2D[1])

    # Define tag ranges similar to the first script
    len_left_positions_probes = 0
    len_pattern_positions_probes = len(coordinates)
    pattern_range_min = len_left_positions_probes + 1
    pattern_range_max = len_left_positions_probes + len_pattern_positions_probes

    tag_probs: Dict[str, List[int]] = {
        "pattern": [pattern_range_min, pattern_range_max]
    }

    probe_dict = {
        "locations": coordinates,
        "tag_probs": tag_probs,
        "probe_type": probe_type,
        "indices2D": indices2D,
        "indices1D": indices1D,
    }
    logger.debug(
        "calculate_channel_witness_coordinates: Finished calculating witness coordinates!\n"
    )
    return probe_dict


def write_witness_file(
    filepath: str, probes_positions: List[Tuple[float, float, float]]
) -> None:
    """
    Writes the witness file that needs to be included in the .ker.dat file.

    This function creates a witness.dat file at the specified filepath. It writes the
    header and the positions of the probes in either 2D or 3D format. The function ensures
    that the directory exists and raises an error if the number of dimensions is unsupported.

    Parameters
    ----------
    filepath : str
        The path where the witness.dat file will be written.
    probes_positions : List[Tuple[float, float, float]]
        A list of probe positions.

    Raises
    ------
    ValueError
        If the number of dimensions of probe positions is unsupported.
    """
    logger.debug(
        "write_witness_file: Writing witness file to %s with %int witness points",
        filepath,
        len(probes_positions),
    )
    # Ensure the directory exists
    os.makedirs(filepath, exist_ok=True)

    nprobes = len(probes_positions)
    ndim = len(probes_positions[0]) if probes_positions else 0

    # Open file for writing
    with open(os.path.join(filepath, "witness.dat"), "w") as file:
        # Write header
        file.write(f"WITNESS_POINTS, NUMBER={nprobes}\n")

        # Write probes
        if ndim == 2:
            for pos in probes_positions:
                file.write(f"{pos[0]:.4f},{pos[1]:.4f}\n")
        elif ndim == 3:
            for pos in probes_positions:
                file.write(f"{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}\n")
        else:
            raise ValueError(
                f"write_witness_file: Unsupported number of dimensions: {ndim}"
            )

        # Write end
        file.write("END_WITNESS_POINTS\n")

    logger.debug(
        "write_witness_file: Finished writing witness file to %s with %int witness points",
        filepath,
        len(probes_positions),
    )


def write_witness_version_file(
    filepath: str,
    probes_location: int,
    probe_type: str,
    pattern: str,
    y_value_density: int,
    y_skipping: bool,
    y_skip_values: int,
    nx_Qs: int,
    nz_Qs: int,
) -> None:
    """
    Write a text file as a reference of the parameters of witness file being created.

    This function creates a witness_version.txt file at the specified filepath. The file
    includes information about the probe location type, probe type, pattern, y value density,
    y skipping configuration, and the number of sections in the x and z directions.

    Parameters
    ----------
    filepath : str
        The path where the witness version file will be written.
    probes_location : int
        Identifier for the probe location type.
    probe_type : str
        The type of probes to be used.
    pattern : str
        Pattern type ('X' or '+').
    y_value_density : int
        Number of y values total.
    y_skipping : bool
        Whether to skip full pattern placement on certain layers.
    y_skip_values : int
        Number of layers to skip if y_skipping is True.
    nx_Qs : int
        Number of sections in the x direction.
    nz_Qs : int
        Number of sections in the z direction.

    Returns
    -------
    None
    """
    logger.debug(
        "write_witness_version_file: Writing witness version file to %s", filepath
    )

    with open(os.path.join(filepath, "witness_version.txt"), "w") as file:
        file.write(f"Witness Probes Location Version: v{probes_location}\n")
        file.write(f"Probe Type: {probe_type}\n")
        file.write(f"Pattern: {pattern}\n")
        file.write(f"Y Value Density: {y_value_density}\n")
        file.write(f"Y Skipping: {y_skipping}\n")
        file.write(f"Y Skip Values: {y_skip_values}\n")
        file.write(f"nx_Qs: {nx_Qs}\n")
        file.write(f"nz_Qs: {nz_Qs}\n")

    logger.debug(
        "write_witness_version_file: Witness version file has been written to %s",
        filepath,
    )


def write_witness_file_and_visualize(
    case_folder: str,
    output_params: Dict[str, Any],
    probes_location: int = 5,
    pattern: str = "X",
    y_value_density: int = 8,
    y_skipping: bool = False,
    y_skip_values: int = 1,
    nx_Qs: int = 1,
    nz_Qs: int = 1,
) -> None:
    """
    Create the witness.dat file and visualize the witness points, saving a plot to the case folder.

    This function generates the witness.dat file with specified probe locations and
    writes a version file indicating the configuration used. It then visualizes the
    witness points and saves the resulting plot to the specified case folder.

    Parameters
    ----------
    case_folder : str
        The case folder path.
    output_params : Dict[str, Any]
        The output parameters containing witness point locations.
    probes_location : int, optional
        Identifier for the probe location type. Default is 5.
    pattern : str, optional
        Pattern type ('X' or '+'). Default is 'X'.
    y_value_density : int, optional
        Number of y values total. Default is 8.
    y_skipping : bool, optional
        Whether to skip full pattern placement on certain layers. Default is False.
    y_skip_values : int, optional
        Number of layers to skip if y_skipping is True. Default is 1.
    nx_Qs : int, optional
        Number of sections in the x direction. Default is 1.
    nz_Qs : int, optional
        Number of sections in the z direction. Default is 1.

    Returns
    -------
    None
    """
    write_witness_file(
        case_folder,
        output_params["locations"],
    )

    write_witness_version_file(
        case_folder,
        probes_location,
        output_params["probe_type"],
        pattern,
        y_value_density,
        y_skipping,
        y_skip_values,
        nx_Qs,
        nz_Qs,
    )

    from visualization import plot_witness_points

    plot_witness_points(
        output_params["locations"],
        filename=os.path.join(case_folder, f"witnessv{probes_location}_plot.png"),
        nx_Qs=nx_Qs,
        nz_Qs=nz_Qs,
        y_value_density=y_value_density,
        y_skip_values=y_skip_values,
    )

    logger.info(
        "write_witness_file_and_visualize: New witness.dat has been created in %s",
        case_folder,
    )
    logger.debug("write_witness_file_and_visualize: witness.dat creation parameters:")
    logger.debug(
        "write_witness_file_and_visualize: Witness Probes Location Type: %s",
        probes_location,
    )
    logger.debug(
        "write_witness_file_and_visualize: Probe Type: %s", output_params["probe_type"]
    )
    logger.debug("write_witness_file_and_visualize: Pattern: %s", pattern)
    logger.debug(
        "write_witness_file_and_visualize: Y Value Density: %s", y_value_density
    )
    logger.debug("write_witness_file_and_visualize: Y Skipping: %s", y_skipping)
    logger.debug("write_witness_file_and_visualize: Y Skip Values: %s", y_skip_values)
    logger.debug("write_witness_file_and_visualize: nx_Qs: %s", nx_Qs)
    logger.debug("write_witness_file_and_visualize: nz_Qs: %s", nz_Qs)
    logger.info(
        "write_witness_file_and_visualize: Witness point visualization has been saved at %s",
        os.path.join(case_folder, f"witnessv{probes_location}_plot.png"),
    )
