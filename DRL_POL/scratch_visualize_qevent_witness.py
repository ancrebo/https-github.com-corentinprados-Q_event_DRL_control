import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET  # For XML parsing
import pyvista as pv
from typing import Tuple, List, Dict

from coco_calc_reward import (
    normalize_all_single,
    process_velocity_data_single,
    detect_Q_events_single,
)

from calc_avg_qvolume import load_data_for_timestep

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(
    "visualize_qevent_witness", default_level=DEFAULT_LOGGING_LEVEL
)

logger.info("visualize_qevent_witness.py: Logging level set to %s\n", logger.level)

if __name__ == "__main__":
    ####################################################################################################
    H: float = 3.0  # Threshold for Q event detection

    averaged_data_path: str = (
        "/scratch/pietero/andres_clone/DRL_POL/alya_files/case_channel_3D_MARL/calculated_data"
    )

    precalc_value_filename = "calculated_values.csv"
    q_event_summary_filename = "q_event_ratio_summary.csv"
    averaged_data_filename = "averaged_data.csv"

    directory_base: str = (
        "/scratch/pietero/DRL_episode_analysis/testrun_witnessv5_np18_initial_longRun"
    )

    episode: str = "EP_1"

    pvdname: str = "channel.pvd"

    directory: str = os.path.join(directory_base, episode, "vtk")

    save_directory: str = "/scratch/pietero/DRL_visualizations"
    ####################################################################################################

    def load_last_timestep(directory: str, pvdname: str) -> Tuple[float, pd.DataFrame]:
        """
        Loads the last timestep data from PVTU files and converts it into a Pandas DataFrame.

        Parameters:
        - directory (str): The path to the directory containing the PVD and PVTU files.
        - pvdname (str): The name of the PVD file.

        Returns:
        - Tuple containing:
          * The last timestep (float)
          * A DataFrame with columns for spatial coordinates (x, y, z) and velocity components (U, V, W)
        """

        # Parse the PVD file to extract mappings of timesteps to their corresponding PVTU files
        pvd_path = os.path.join(directory, pvdname)
        tree = ET.parse(pvd_path)
        root = tree.getroot()
        timestep_file_map = {
            dataset.attrib["file"]: float(dataset.attrib["timestep"])
            for dataset in root.find("Collection")
        }

        # Get the last timestep and corresponding file
        last_file = max(timestep_file_map, key=timestep_file_map.get)
        last_timestep = timestep_file_map[last_file]
        path = os.path.join(directory, last_file)

        logger.info(f"Loading data from {last_file} at timestep {last_timestep}")

        # Read the mesh data from the PVTU file
        mesh = pv.read(path)
        print(mesh)

        # Extract the spatial coordinates and velocity components from the mesh
        points = mesh.points  # x, y, z coordinates

        # Extract cells and cell types
        cells = mesh.cells
        celltypes = mesh.celltypes

        print("Number of Cell Types: ", len(celltypes))

        # Print the first few points, cells, and cell types
        print("First 5 points: \n", points[:5])
        print("First 5 cells: \n", cells[:5])
        print("First 5 cell types: \n", celltypes[:5])

        U, V, W = mesh[
            "VELOC"
        ].T  # Transpose to separate the velocity components (U, V, W)

        # Create a DataFrame with the extracted data
        df = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "u": U,
                "v": V,
                "w": W,
            }
        )

        return last_timestep, df

    logger.info("Starting `load_last_timestep` function...")
    ## Extract the spatial coordinates and velocity components from the mesh
    data: Tuple[float, pd.DataFrame] = load_last_timestep(directory, pvdname)
    logger.info("Finished `load_last_timestep` function.\n")
    ####################################################################################################
    ## Normalize the velocity data
    # Load pre-calculated values
    logger.info("Loading pre-calculated values...")
    precalc_value_filepath = os.path.join(averaged_data_path, precalc_value_filename)
    precalc_values = pd.read_csv(precalc_value_filepath)
    u_tau = precalc_values.iloc[0, 1]
    delta_tau = precalc_values.iloc[1, 1]

    averaged_data_filepath = os.path.join(averaged_data_path, averaged_data_filename)
    averaged_data = pd.read_csv(averaged_data_filepath)
    logger.info("Finished loading pre-calculated values.\n")

    # Load global Q event average ratio and standard deviation
    # q_event_summary_filepath = os.path.join(averaged_data_path, q_event_summary_filename)
    # q_event_summary = pd.read_csv(q_event_summary_filepath)
    # avg_qratio = q_event_summary["average_q_event_ratio"].values[0]
    # std_dev_qratio = q_event_summary["std_dev_q_event_ratio"].values[0]

    logger.info("Starting `normalize_all_single` function...")
    data_normalized: Tuple[float, pd.DataFrame] = normalize_all_single(
        data, u_tau, delta_tau
    )
    logger.info("Finished `normalize_all_single` function.\n")

    logger.info("Starting `process_velocity_data_single` function...")
    processed_data: Tuple[float, pd.DataFrame] = process_velocity_data_single(
        data_normalized, averaged_data
    )
    logger.info("Finished `process_velocity_data_single` function.\n")

    logger.info("Starting `detect_Q_events_single` function...")
    Q_event_frames: Tuple[float, pd.DataFrame] = detect_Q_events_single(
        processed_data, averaged_data, H
    )
    logger.info("Finished `detect_Q_events_single` function.\n")
    ####################################################################################################
    # Use timestep and the pvd file to identify the correct VTK file to load and read the mesh
    timestep_to_load, Q_event_df = Q_event_frames
    pvd_path = os.path.join(directory, pvdname)
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    timestep_file_map = {
        dataset.attrib["file"]: float(dataset.attrib["timestep"])
        for dataset in root.find("Collection")
    }
    # Find the file for the given timestep
    file_name = [
        key for key, value in timestep_file_map.items() if value == timestep_to_load
    ][0]
    path = os.path.join(directory, file_name)
    mesh = pv.read(path)

    # Extract the points and verify the order (optional, for sanity check)
    points = mesh.points
    # Optionally, you can verify the order by comparing with your dataframe coordinates
    # df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]})
    # assert (df[['x', 'y', 'z']] == your_existing_dataframe[['x', 'y', 'z']]).all().all()

    # Assuming you have a dataframe with Q values
    # Example dataframe structure
    # your_existing_dataframe = pd.DataFrame({'x': [...], 'y': [...], 'z': [...], 'Q': [...]})
    Q_values = Q_event_df["Q"].to_numpy()

    # Add Q values as a new scalar field to the mesh
    mesh.point_data["Q"] = Q_values

    # Save the modified mesh to a new VTK file
    modified_path = os.path.join(
        save_directory, f"EP_{episode}_timestep_{timestep_to_load}_Q_event.vtu"
    )

    # Verify the addition by printing the fields available in the modified mesh
    print(mesh.point_data)
