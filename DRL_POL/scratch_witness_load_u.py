"""
Script to extract and save witness data for a specific point over a given number of timesteps.

This script reads data from a specified .nsi.wit file and a corresponding witness.dat file,
extracts the 'u' velocity component for a specified witness point index over a given number 
of timesteps, and saves the data to a CSV file.

Usage
-----
Run the script with the following command:
    python scatch_witness_load_u.py --case_folder /path/to/case_folder --output_file /path/to/output.csv --point_index 0 --n_timesteps 10

Parameters
----------
--case_folder : str
    The path to the folder containing the .nsi.wit and witness.dat files.
--output_file : str
    The path to the output CSV file to save the witness data.
--point_index : int
    The index of the witness point to extract from the witness data.
--n_timesteps : int
    The number of timesteps to extract for the witness point.

Example
-------
An example of how to run the script:

    python script_name.py --case_folder ./case1 --output_file ./output.csv --point_index 5 --n_timesteps 50

Author
------
Pieter Orlandini, August 1st, 2024
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple
import numpy as np
import logging
from witness import witnessReadNByBehind

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_witness_data(
    case_folder: str, output_file: str, point_index: int, n_timesteps: int
) -> None:
    """
    Extract and save witness data for a specific point over a given number of timesteps.

    Parameters
    ----------
    case_folder : str
        The path to the folder containing the .nsi.wit and witness.dat files.
    output_file : str
        The path to the output CSV file to save the witness data.
    point_index : int
        The index of the witness point to extract from the witness data.
    n_timesteps : int
        The number of timesteps to extract for the witness point.
    """
    logging.info("Starting the extraction process...")
    case_folder = Path(case_folder)
    wit_file = case_folder.glob("*.nsi.wit")
    witness_dat_file = case_folder / "witness.dat"

    # Ensure the wit file exists
    try:
        wit_file = next(wit_file)
        logging.info(f"Found .nsi.wit file: {wit_file}")
    except StopIteration:
        logging.error("No .nsi.wit file found in the specified case folder.")
        raise FileNotFoundError("No .nsi.wit file found in the specified case folder.")

    # Read witness points from witness.dat to validate point index
    witness_points = read_witness_dat(witness_dat_file)
    logging.info(f"Number of witness points: {len(witness_points)}")
    if point_index < 0 or point_index >= len(witness_points):
        logging.error(
            f"Invalid point index: {point_index}. Must be between 0 and {len(witness_points)-1}."
        )
        raise ValueError(
            f"Invalid point index: {point_index}. Must be between 0 and {len(witness_points)-1}."
        )

    point_coords = witness_points[point_index]
    logging.info(f"Coordinates of the selected witness point: {point_coords}")
    coords_str = f"x{point_coords[0]}_y{point_coords[1]}_z{point_coords[2]}"
    output_file = f"{Path(output_file).stem}_{coords_str}.csv"

    # Read the last N timesteps from the .wit file
    try:
        _, time, data = witnessReadNByBehind(str(wit_file), n_timesteps)
        logging.info(f"Number of timesteps read: {len(time)}")
    except Exception as e:
        logging.error(f"Error reading .nsi.wit file: {e}")
        raise

    # Extract the u component for the specific witness point
    try:
        u_values = data["VELOX"][:, point_index]
        logging.info(f"Extracted u values for point index {point_index}")
    except KeyError as e:
        logging.error(f"Key error: {e}. Available keys: {data.keys()}")
        raise
    except IndexError as e:
        logging.error(f"Index error: {e}. Data shape: {data['VELOX'].shape}")
        raise

    # Save the extracted data to a CSV file
    save_to_csv(output_file, time, u_values)
    logging.info(f"Data successfully saved to {output_file}")


def read_witness_dat(file_path: Path) -> List[Tuple[float, float, float]]:
    """
    Read the witness.dat file to extract witness point coordinates.

    Parameters
    ----------
    file_path : Path
        Path to the witness.dat file.

    Returns
    -------
    List[Tuple[float, float, float]]
        List of tuples containing the coordinates of witness points.
    """
    logging.info(f"Reading witness.dat file from {file_path}")
    points = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("WITNESS_POINTS") or line.startswith(
                "END_WITNESS_POINTS"
            ):
                continue
            x, y, z = map(float, line.strip().split(","))
            points.append((x, y, z))
    logging.info(f"Read {len(points)} witness points")
    return points


def save_to_csv(output_file: str, times: np.ndarray, u_values: np.ndarray) -> None:
    """
    Save the extracted data to a CSV file.

    Parameters
    ----------
    output_file : str
        The path to the output CSV file.
    times : np.ndarray
        Array of timesteps.
    u_values : np.ndarray
        Array of u values for the specified witness point.
    """
    logging.info(f"Saving data to CSV file: {output_file}")
    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["timestep", "u_value"])
        for i in range(len(u_values)):
            csv_writer.writerow([times[i], u_values[i]])
    logging.info("Data successfully saved to CSV")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Load and save witness data for a specific witness point."
    )
    parser.add_argument(
        "--case_folder",
        type=str,
        required=True,
        help="The path to the folder containing the .nsi.wit and witness.dat files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The path to the output CSV file to save the witness data.",
    )
    parser.add_argument(
        "--point_index",
        type=int,
        required=True,
        help="The index of the witness point to extract from the witness data.",
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        required=True,
        help="The number of timesteps to extract for the witness point.",
    )

    args = parser.parse_args()

    try:
        extract_witness_data(
            args.case_folder, args.output_file, args.point_index, args.n_timesteps
        )
        logging.info("Data successfully extracted and saved.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
