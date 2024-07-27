#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Witness file reading and extraction.
#
# Pol Suarez, Arnau Miro, Francisco Alcantara
# 07/07/2022
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
    Reads a header line by line from an ALYA witness file.
    File needs to be previouly opened.
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
    Cursor should be already positioned at the start of the instant
    """
    # Read iterations
    line = file.readline()
    it = int(line.split()[3])
    # Read time
    line = file.readline()
    time = float(line.split()[3])
    # Now read the data matrix
    data = np.genfromtxt(file, max_rows=nwit)
    # Function return
    return it, time, data


def witnessReadNByFront(
    filename: str, n: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    # TODO: this function is not used
    Reads N instants starting from the top of the witness file.
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
    Reads N instants starting from the bottom of the witness file.
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
    # Set the cursor from the bottom of the file up a certain
    # number of lines
    file.seek(0, 2)  # seek to end of file; f.seek(0, 2) is legal
    # First we compute how much of an offset an instant is
    # a variable is 17 characters
    # the witness id is 10 characters
    # thus a line is 27 characters - we have nwit lines
    # We need to add the lines on iterations and time
    # Iterations are 37 characters
    # Time are 32 characters
    offset = (10 + 17 * (header["NVARS"] - 1) + 1) * header["NWIT"] + 38 + 33
    # file.seek(0,2) moves the cursor to the end of the file
    # so the following should move the cursor from the bottom to
    # the beginning of the n instants that must be read
    file.seek(file.tell() - n * offset, os.SEEK_SET)  # go backwards 3 bytes
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


def read_last_wit(
    filename: str, probe_type: str, norm: Dict[str, float], n_to_read: int = 1
) -> Dict[str, np.ndarray]:
    """
    function that skips all the data from the entire time domain and gives the last value from nsi.wit
    expected increase the IO time extracting probes to send, restart are so much quicker
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


def calculate_channel_witness_coordinates(
    n: int,  # nx_Qs
    m: int,  # nz_Qs
    Lx: float,
    Ly: float,
    Lz: float,
    y_value_density: int,
    pattern: str = "X",
    y_skipping: bool = False,
    y_skipping_value: int = 3,
) -> Dict[str, Any]:
    """
    Calculate witness coordinates and indices for a channel pattern.

    Parameters:
        n (int): Number of sections in the x direction.
        m (int): Number of sections in the z direction.
        Lx (float): Length in the x direction.
        Ly (float): Length in the y direction.
        Lz (float): Length in the z direction.
        y_value_density (int): Density of y values.
        pattern (str): Pattern type ('X' or '+'). Default is 'X'.
        y_skipping (bool): Whether to skip y values in between pattern layers. Default is False.
        y_skipping_value (int): Number of layers with only the center point between each full
            pattern layer if y_skipping is True. Default is 3.

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            'locations' (List[Tuple[float, float, float]]): Coordinates of the probes.
            'indices2D' (List[Tuple[int, int]]): 2D indices of the probes.
            'indices1D' (List[int]): 1D indices of the probes.
            'tag_probs' (Dict[str, List[int]]): Tag ranges for different patterns.

    Raises:
        ValueError: If an invalid pattern is specified or if y-values are out of range.
    """
    # Create list of y values to place pattern - Exclude the first term (0)
    y_values: List[float] = np.linspace(0, Ly, y_value_density + 1).tolist()[1:]

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
                    if y_skipping and (index % y_skipping_value == 0):
                        # Place the full "X" pattern
                        for x, z in end_points:
                            coordinates.append((x / Lx, y / Ly, z / Lz))
                            indices2D.append((i, j))
                        # Also place the center point
                        coordinates.append(
                            (center_point[0] / Lx, y / Ly, center_point[1] / Lz)
                        )
                        indices2D.append((i, j))
                    else:
                        # Place only the center point
                        coordinates.append(
                            (center_point[0] / Lx, y / Ly, center_point[1] / Lz)
                        )
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
        "indices2D": indices2D,
        "indices1D": indices1D,
        "tag_probs": tag_probs,
    }

    return probe_dict


def write_witness_file(
    filepath: str, probes_positions: List[Tuple[float, float, float]]
) -> None:
    """
    UPDATED FUNCTION AS OF JULY 27, 2024 - @pietero

    Writes the witness file that needs to be included in the .ker.dat file.

    Parameters:
        filepath (str): The path where the witness.dat file will be written.
        probes_positions (np.ndarray): An array of probe positions.
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


def write_witness_file_and_visualize(
    case_folder: str,
    output_params: Dict[str, Any],
    probes_location: int = 5,
    pattern: str = "X",
    y_value_density: int = 8,
    y_skipping: bool = False,
    y_skip_values=None,
    nx_Qs: int = 1,
    nz_Qs: int = 1,
) -> None:
    """
    Create the witness.dat file and visualize the witness points, saving a plot to the case folder.

    Parameters:
        case_folder (str): The case folder path.
        output_params (Dict[str, Any]): The output parameters containing witness point locations.
        probes_location (int, optional): Identifier for the probe location type. Default is 5.
        pattern (str, optional): Pattern type ('X' or '+'). Default is 'X'.
        y_value_density (int, optional): Number of y values total. Default is 8.
        y_skipping (bool, optional): Whether to skip full pattern placement on certain layers. Default is False.
        y_skip_values (List[int], optional): Number of layers to skip if y_skipping is True. Default is None.
        nx_Qs (int, optional): Number of sections in the x direction. Default is 1.
        nz_Qs (int, optional): Number of sections in the z direction. Default is 1.
    """
    if y_skip_values is None:
        y_skip_values = []

    write_witness_file(
        case_folder,
        output_params["locations"],
    )
    from visualization import plot_witness_points

    plot_witness_points(
        output_params["locations"],
        filename=os.path.join(case_folder, f"witnessv{probes_location}_plot.png"),
        nx_Qs=nx_Qs,
        nz_Qs=nz_Qs,
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
