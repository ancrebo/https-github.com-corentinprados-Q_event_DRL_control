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

import os, numpy as np
from cr import cr_start, cr_stop


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
    # TODO: case name cannot be hardcoded as cylinder and should be passed as input - Pol
    # TODO: implement crashes - Pol
    cr_start("WIT.read_last_wit", 0)
    print(
        f"\n\n\nwitness.read_last_wit: Starting to read the last witness file at {filename}"
    )
    # Read witness file
    itw, timew, data = witnessReadNByBehind(filename, n_to_read)
    print(
        f"witness.read_last_wit: Finished reading the last witness file at {filename}"
    )
    print(f"witness.read_last_wit: Length of data: {len(data)}\n")
    print(f"witness.read_last_wit: keys of data: {data.keys()}\n")
    # Initialize result dictionary
    result_data = {}

    # Select which variable to work with
    if probe_type == "pressure":
        result_data["pressure"] = data["PRESSURE"]
    if probe_type == "velocity":
        vel_components = ["VELOX", "VELOY", "VELOZ"]
        for comp in vel_components:
            result_data[comp.lower()] = data[comp]
    else:
        raise ValueError(
            "Witness.py: read_last_wit: Invalid `probe_type`: must be 'pressure' or 'velocity'"
        )
    print(f"witness.read_last_wit: Selected probe type: {probe_type}")
    # If we have more than one instant, average them
    if n_to_read > 1:
        for key in result_data.keys():
            result_data[key] = (
                1.0 / (timew[-1] - timew[0]) * np.trapz(result_data[key], timew, axis=0)
            )
        print(f"witness.read_last_wit: Averaged {n_to_read} instants")

    # Normalize the data using the provided norm dictionary
    for key in result_data.keys():
        if key in norm:
            result_data[key] = result_data[key] / norm[key]
        else:
            raise ValueError(
                f"Witness.py: read_last_wit: Normalization value for {key} not found in norm dictionary"
            )
    print(f"witness.read_last_wit: Normalized data")
    # if var is None:
    #     raise ValueError("Invalid probe_type: must be 'pressure' or 'velocity'")
    #     # raise ValueError("Crash very hard!")  # TODO: Do crash very hard
    #
    # # If we have more than one instant, average them
    # if n_to_read > 1:
    #     data[var] = 1.0 / (timew[-1] - timew[0]) * np.trapz(data[var], timew, axis=0)

    # Ensure that data has the correct shape
    for key in result_data.keys():
        result_data[key] = result_data[key][0, :]  # (nprobes,)
    print(f"witness.read_last_wit: keys of result_data: {result_data.keys()}\n")
    print(
        f"witness.read_last_wit: Corrected shape if needed, about to return `result_data`"
    )
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
        y_skipping (bool): Whether to skip y values. Default is False.
        y_skipping_value (int): Interval for skipping y values if y_skipping is True. Default is 3.

    Returns:
        Dict[str, Any]: A dictionary containing 'locations' (probe coordinates),
                        'indices2D' (2D indices of probes), 'indices1D' (1D indices of probes),
                        and 'tag_probs' (tag ranges).
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

            center_point: Tuple[float, float] = (center_x, center_z)

            for index, y in enumerate(y_values):
                if 0 <= y <= Ly:  # Ensure y-values are within the global y limit
                    if y_skipping and (index % y_skipping_value != 0):
                        coordinates.append(
                            (center_point[0] / Lx, y / Ly, center_point[1] / Lz)
                        )
                        indices2D.append((i, j))
                    else:
                        for x, z in end_points:
                            coordinates.append((x / Lx, y / Ly, z / Lz))
                            indices2D.append((i, j))
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

    # Convert lists to numpy arrays
    coordinates_array: np.ndarray = np.array(coordinates)
    indices2D_array: np.ndarray = np.array(indices2D)
    indices1D_array: np.ndarray = np.array(indices1D)

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
