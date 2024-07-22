import os
import argparse
import logging
import xml.etree.ElementTree as ET  # For XML parsing
import pyvista as pv  # For reading and processing mesh data
import pandas as pd  # For data manipulation and DataFrame creation
from typing import Tuple, List
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data_and_convert_to_dataframe(directory, file_name):
    """
    This function loads CFD simulation data from PVTU files and converts them into Pandas DataFrames.
    Each DataFrame is stored along with its respective timestep, facilitating time-series analysis.

    Parameters:
    - directory (str): The path to the directory containing the PVD and PVTU files.

    Returns:
    - data_frames (list of tuples): A list where each element is a tuple containing:
      * A timestep (float)
      * A DataFrame with columns for spatial coordinates (x, y, z) and velocity components (u, v, w)
    """
    logger.info("Loading data from PVTU files...")
    # Parse the PVD file to extract mappings of timesteps to their corresponding PVTU files
    pvd_path = os.path.join(directory, file_name)
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    timestep_file_map = {
        dataset.attrib["file"]: float(dataset.attrib["timestep"])
        for dataset in root.find("Collection")
    }

    # List to store data tuples of timestep and DataFrame
    data_frames = []

    # Process each PVTU file according to its mapped timestep, wrapped in tqdm:
    for file, timestep in tqdm(
        timestep_file_map.items(), desc="Loading data", unit="file"
    ):
        path = os.path.join(directory, str(file))
        mesh = pv.read(path)  # Read the mesh data from the PVTU file

        # Extract the spatial coordinates and velocity components from the mesh
        points = mesh.points  # x, y, z coordinates
        u, v, w = mesh[
            "VELOC"
        ].T  # Transpose to separate the velocity components (u, v, w)

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

        # Append the timestep and DataFrame as a tuple to the list
        data_frames.append((timestep, df))
        logger.debug(f"Data from {file} at timestep {timestep} loaded into DataFrame.")

    logger.info(f"Total data sets loaded: {len(data_frames)}")
    return data_frames


def normalize_all(data, u_tau, delta_tau):
    """
    Processes a list of tuples, each containing a timestep and a DataFrame, normalizes the velocity components
    and spatial coordinates in each DataFrame using the provided friction velocity and characteristic length scale,
    and returns a new list with the updated DataFrames.

    Parameters:
    - data (list of tuples): List where each tuple contains a timestep and a DataFrame with the original velocity components and spatial coordinates.
    - u_tau (float): The friction velocity used for normalizing the velocity components.
    - delta_tau (float): The characteristic length scale used for normalizing the spatial coordinates.

    Returns:
    - normalized_data (list of tuples): New list where each tuple contains a timestep and the updated DataFrame with normalized velocities and spatial coordinates.
    """
    normalized_data = []

    logger.info("Normalizing data...")

    for timestep, df in tqdm(data):
        # Copy the DataFrame to preserve original data
        df_copy = df.copy()

        # Normalize the velocity components
        df_copy["u"] = df_copy["u"] / u_tau
        df_copy["v"] = df_copy["v"] / u_tau
        df_copy["w"] = df_copy["w"] / u_tau

        # Normalize the spatial coordinates
        df_copy["x"] = df_copy["x"] / delta_tau
        df_copy["y"] = df_copy["y"] / delta_tau
        df_copy["z"] = df_copy["z"] / delta_tau

        # Append the timestep and updated DataFrame to the new list
        normalized_data.append((timestep, df_copy))

    logger.info("Data normalization complete.")
    return normalized_data


def process_velocity_data(data, N):
    """
    Processes a list of tuples containing CFD simulation data to calculate averaged and
    fluctuating components of velocity fields over the last N entries. It computes these metrics for
    the horizontal (u), vertical (v), and lateral (w) velocity components.

    Parameters:
    - data (list of tuples): Each tuple contains a timestep and a DataFrame with spatial coordinates (x, y, z)
      and velocity components (u, v, w).
    - N (int): Number of most recent timesteps to include in the averaging process.

    Returns:
    - averaged_data (DataFrame): Contains averaged velocities ($\overline{U}(y)$, $\overline{V}(y)$, $\overline{W}(y)$)
      and rms of velocity fluctuations ($u'(y)$, $v'(y)$, $w'(y)$) as columns, indexed by the y-coordinate.
    - data_process (list of tuples): Each tuple contains a timestep and a DataFrame with original and fluctuating
      velocity components (U, V, W, u, v, w).
    """
    processed_data = []
    recent_data = pd.DataFrame()

    logger.info(f"Processing velocity data for the last {N} timesteps.")

    # Aggregate data from the last N timesteps to compute averages and fluctuations
    for timestep, df in tqdm(data[-N:]):
        df["timestep"] = timestep  # Temporarily add timestep to differentiate data
        recent_data = pd.concat([recent_data, df], ignore_index=True)
        logger.debug(
            "Data from timestep {} added to data to be averaged.".format(timestep)
        )

    # Calculate mean and standard deviation for u, v, w across the recent data
    averaged_data = (
        recent_data.groupby("y")
        .agg({"u": ["mean", "std"], "v": ["mean", "std"], "w": ["mean", "std"]})
        .rename(columns={"mean": "bar", "std": "prime"}, level=1)
    )
    averaged_data.columns = [
        "U_bar",
        "u_prime",
        "V_bar",
        "v_prime",
        "W_bar",
        "w_prime",
    ]  # Clear column names

    # Process each individual dataset for detailed fluctuation analysis
    for timestep, df in tqdm(data):
        y_means = averaged_data.loc[df["y"]]
        df_processed = df.copy()
        df_processed["U"] = df["u"]
        df_processed["V"] = df["v"]
        df_processed["W"] = df["w"]
        df_processed["u"] = df["u"] - y_means["U_bar"].values
        df_processed["v"] = df["v"] - y_means["V_bar"].values
        df_processed["w"] = df["w"] - y_means["W_bar"].values

        # Ensure no 'timestep' column remains in the output data
        df_processed.drop(columns="timestep", inplace=True, errors="ignore")
        processed_data.append((timestep, df_processed))
        logger.debug("Data from timestep {} processed.".format(timestep))

    # Prepare averaged data for output
    averaged_data = averaged_data.reset_index()

    logger.info("Velocity data processing complete.")

    return averaged_data, processed_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process CFD simulation data and save averaged data."
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing the PVD and PVTU files.",
    )
    parser.add_argument(
        "--u_tau",
        type=float,
        required=True,
        help="Friction velocity used for normalizing the velocity components. PRECALCULATED FOR SPECIFIC CASE!!!",
    )
    parser.add_argument(
        "--delta_tau",
        type=float,
        required=True,
        help="Characteristic length scale used for normalizing the spatial coordinates. PRECALCULATED FOR SPECIFIC CASE!!!",
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Number of most recent timesteps to include in the averaging process.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output .csv file.",
    )
    parser.add_argument(
        "--loglvl",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    args = parser.parse_args()

    # Set logging level
    logger.setLevel(args.loglvl)

    # Last directory
    directory_path: str = args.directory
    file_name: str = "channel.pvd"

    # Load the data and store it in `data`
    data: List[Tuple[float, pd.DataFrame]] = load_data_and_convert_to_dataframe(
        directory_path, file_name
    )

    u_tau: float = args.u_tau
    delta_tau: float = args.delta_tau

    # Normalize the data
    normalized_data: List[Tuple[float, pd.DataFrame]] = normalize_all(
        data, u_tau, delta_tau
    )

    del data

    N: int = args.N  # Averaging over the last N timesteps
    averaged_data, _ = process_velocity_data(normalized_data, N)

    del normalized_data

    # Save the averaged data to a CSV file
    output_path: str = args.output_path
    output_filename = "averaged_data.csv"
    output_filepath = os.path.join(output_path, output_filename)
    averaged_data.to_csv(output_filepath, index=False)
    logger.info(f"Averaged data saved to {output_filepath}")

    # Clean up
    del averaged_data
