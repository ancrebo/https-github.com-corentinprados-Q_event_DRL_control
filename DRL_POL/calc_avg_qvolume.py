import os
import argparse
import logging
import numpy as np
import xml.etree.ElementTree as ET  # For XML parsing
import pyvista as pv  # For reading and processing mesh data
import pandas as pd  # For data manipulation and DataFrame creation
from typing import Tuple, List, Dict
from pathlib import Path
import gc
from env_utils import agent_index_2d_to_1d

from coco_calc_reward import (
    normalize_all_single,
    process_velocity_data_single,
    detect_Q_events_single,
)

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger("calc_avg_qvolume", default_level=DEFAULT_LOGGING_LEVEL)

logger.info("calc_avg_qvolume.py: Logging level set to %s", logger.level)


def load_data_for_timestep(directory: str, file_name: str) -> pd.DataFrame:
    """
    Load data from a specific PVTU file and convert it into a Pandas DataFrame.

    Parameters:
    - directory (str): The path to the directory containing the PVD and PVTU files.
    - file_name (str): The name of the PVTU file to be loaded.

    Returns:
    - df (pd.DataFrame): DataFrame with columns for spatial coordinates (x, y, z) and velocity components (u, v, w).
    """
    logger.info(f"Loading data from PVTU file: {file_name}")
    path = os.path.join(directory, file_name)
    mesh = pv.read(path)  # Read the mesh data from the PVTU file

    # Extract the spatial coordinates and velocity components from the mesh
    points = mesh.points  # x, y, z coordinates
    u, v, w = mesh["VELOC"].T  # Transpose to separate the velocity components (u, v, w)

    # Create a DataFrame with the extracted data
    df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "u": u,
            "v": v,
            "w": w,
        }
    )

    logger.info(f"Data from {file_name} loaded into DataFrame.")
    return df


def create_timestep_file_map(pvd_path: str) -> Dict[str, float]:
    """
    Create a mapping of timesteps to their corresponding PVTU files.

    Parameters:
    - pvd_path (str): Path to the PVD file.

    Returns:
    - timestep_file_map (dict): Dictionary mapping PVTU file names to timesteps.
    """
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    timestep_file_map = {
        dataset.attrib["file"]: float(dataset.attrib["timestep"])
        for dataset in root.find("Collection")
    }
    return timestep_file_map


def calculate_avg_qevent_ratio(
    directory: str, averaged_data_path: str, H: float, num_timesteps: int
) -> None:
    pvd_path = os.path.join(directory, "channel.pvd")
    timestep_file_map = create_timestep_file_map(pvd_path)

    # Ensure we process only the specified number of timesteps from the end
    timesteps = list(timestep_file_map.items())[-num_timesteps:]

    precalc_values = pd.read_csv(
        os.path.join(averaged_data_path, "calculated_values.csv")
    )
    u_tau = precalc_values.iloc[0, 1]
    delta_tau = precalc_values.iloc[1, 1]
    averaged_data = pd.read_csv(os.path.join(averaged_data_path, "averaged_data.csv"))

    q_event_ratios = []

    for file_name, timestep in timesteps:
        df = load_data_for_timestep(directory, file_name)
        normalized_df = normalize_all_single((timestep, df), u_tau, delta_tau)[1]
        processed_df = process_velocity_data_single(
            (timestep, normalized_df), averaged_data
        )[1]
        q_event_df = detect_Q_events_single((timestep, processed_df), averaged_data, H)[
            1
        ]

        # Calculate the global Q event volume ratio
        total_points = len(q_event_df)
        q_event_count = q_event_df["Q"].sum()
        q_event_ratio = q_event_count / total_points if total_points > 0 else 0

        q_event_ratios.append({"timestep": timestep, "q_event_ratio": q_event_ratio})

        # Clear memory
        del df, normalized_df, processed_df, q_event_df
        gc.collect()

    q_event_ratios_df = pd.DataFrame(q_event_ratios)
    average_q_event_ratio = q_event_ratios_df["q_event_ratio"].mean()
    std_dev_q_event_ratio = q_event_ratios_df["q_event_ratio"].std()

    # Save the results
    q_event_ratios_df.to_csv("q_event_ratios.csv", index=False)
    summary_df = pd.DataFrame(
        {
            "average_q_event_ratio": [average_q_event_ratio],
            "std_dev_q_event_ratio": [std_dev_q_event_ratio],
        }
    )
    summary_df.to_csv("q_event_summary.csv", index=False)

    logger.info(f"Average Q Event Volume Ratio: {average_q_event_ratio}")
    logger.info(f"Standard Deviation of Q Event Volume Ratio: {std_dev_q_event_ratio}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Calculate the average Q event volume ratio for the last N timesteps."
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to the directory containing the PVD and PVTU files.",
    )
    parser.add_argument(
        "--averaged_data_path",
        type=str,
        required=True,
        help="Path to the directory containing the averaged data.",
    )
    parser.add_argument(
        "--H",
        type=float,
        required=True,
        help="Threshold value for Q event detection.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        required=True,
        help="Number of timesteps to process from the end.",
    )
    args = parser.parse_args()

    directory = args.directory
    averaged_data_path = args.averaged_data_path
    H = args.H
    num_timesteps = args.num_timesteps

    # HARDCODED PATHS FOR TESTING
    directory = "/scratch/pietero/baseline/prados_recent/re180_min_channel_900_calculating_Ubar/vtk_for_average/vtk"
    averaged_data_path = "/scratch/pietero/andres_clone/DRL_POL/alya_files/baseline/calculated_data/"
    H = 3.0
    num_timesteps = 10

    calculate_avg_qevent_ratio(directory, averaged_data_path, H, num_timesteps)
