import os
import argparse
import logging
import numpy as np
import xml.etree.ElementTree as ET  # For XML parsing
import pyvista as pv  # For reading and processing mesh data
import pandas as pd  # For data manipulation and DataFrame creation
from typing import Tuple, List
from pathlib import Path
import gc
from env_utils import agent_index_2d_to_1d

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger("coco_calc_reward", default_level=DEFAULT_LOGGING_LEVEL)

logger.info("coco_calc_reward.py: Logging level set to %s", logger.level)


def load_data_and_convert_to_dataframe_single(
    directory: str, file_name: str
) -> Tuple[float, pd.DataFrame]:
    """
    This function loads CFD simulation data from a single PVTU file and converts it into a Pandas DataFrame.
    The DataFrame is stored along with its respective timestep.

    Parameters:
    - directory (str): The path to the directory containing the PVD and PVTU files.

    Returns:
    - data_frame (tuple): A tuple containing:
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

    # Assuming we are only interested in a single timestep
    file, timestep = next(iter(timestep_file_map.items()))
    path = os.path.join(directory, file)
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

    data_frames: Tuple[float, pd.DataFrame] = (timestep, df)

    logger.info(f"Data from {file} at timestep {timestep} loaded into DataFrame.")
    return data_frames


def normalize_all_single(
    timestep_df: Tuple[float, pd.DataFrame], u_tau: float, delta_tau: float
) -> Tuple[float, pd.DataFrame]:
    """
    Processes a tuple containing a timestep and a DataFrame, normalizes the velocity components
    and spatial coordinates in the DataFrame using the provided friction velocity and characteristic length scale,
    and returns a tuple with the updated DataFrame.

    Parameters:
    - timestep_df (tuple): A tuple containing a timestep (float) and a DataFrame with the original velocity components and spatial coordinates.
    - u_tau (float): The friction velocity used for normalizing the velocity components.
    - delta_tau (float): The characteristic length scale used for normalizing the spatial coordinates.

    Returns:
    - normalized_data (tuple): A tuple containing the timestep and the updated DataFrame with normalized velocities and spatial coordinates.
    """
    timestep, df = timestep_df
    logger.info("Normalizing data...")

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

    # Log NaN counts
    nan_u_count = df_copy["u"].isna().sum()
    nan_v_count = df_copy["v"].isna().sum()
    logger.debug(
        "normalize_all_single: %s: Number of NaNs in 'u' after normalization: %d",
        timestep,
        nan_u_count,
    )
    logger.debug(
        "normalize_all_single: %s: Number of NaNs in 'v' after normalization: %d",
        timestep,
        nan_v_count,
    )

    logger.info("Data normalization complete.")
    return timestep, df_copy


def process_velocity_data_single(
    timestep_df: Tuple[float, pd.DataFrame],
    averaged_data: pd.DataFrame,
    precision: int = 3,
) -> Tuple[float, pd.DataFrame]:
    """
    Processes a tuple containing CFD simulation data to calculate fluctuating components of velocity fields.
    It computes these metrics for the horizontal (u), vertical (v), and lateral (w) velocity components.

    Parameters:
    - timestep_df (tuple): A tuple containing a timestep and a DataFrame with spatial coordinates (x, y, z)
      and velocity components (u, v, w). (NORMALIZED!)
    - averaged_data (DataFrame): Contains averaged velocities ($\overline{U}(y)$, $\overline{V}(y)$, $\overline{W}(y)$)
      and rms of velocity fluctuations ($u'(y)$, $v'(y)$, $w'(y)$) as columns, indexed by the y-coordinate. (NORMALIZED!)
    - precision (int): The number of decimal places to round the 'y' values to. Ensures they are recognized as equal!

    Returns:
    - processed_data (tuple): A tuple containing a timestep and a DataFrame with original and fluctuating
      velocity components (U, V, W, u, v, w).
    """
    timestep, df = timestep_df
    logger.info("Processing velocity data using loaded averaged data...")

    precision: int = 3

    # Print 'y' values before rounding
    logger.debug(
        "Original 'y' values in main DataFrame before rounding:\n%s", df["y"].unique()
    )
    logger.debug(
        "Original 'y' values in averaged data before rounding:\n%s",
        averaged_data["y"].unique(),
    )

    # iterate through each row of main dataframe and manually assign the rounded value to the 'y' column
    for index, row in df.iterrows():
        df.at[index, "y"] = np.round(row["y"], precision)
    for index, row in averaged_data.iterrows():
        averaged_data.at[index, "y"] = np.round(row["y"], precision)

    # # Apply rounding to 'y' values using pandas.DataFrame.round
    # df_altered_copy = df.copy()
    # df_altered_copy["y"] = df["y"].round(precision)
    # averaged_data_copy = averaged_data.copy()
    # averaged_data_copy["y"] = averaged_data["y"].round(precision)

    # Print 'y' values after rounding
    logger.debug(
        "Rounded 'y' values in main DataFrame after rounding:\n%s", df["y"].unique()
    )
    logger.debug(
        "Rounded 'y' values in averaged data after rounding:\n%s",
        averaged_data["y"].unique(),
    )

    # Check if altered 'y' values are equal to the original 'y' values
    logger.debug(
        "Are 'y' values in main DataFrame equal to altered 'y' values? %s",
        df["y"].equals(df_altered_copy["y"]),
    )
    logger.debug(
        "Are 'y' values in averaged data equal to altered 'y' values? %s",
        averaged_data["y"].equals(averaged_data_copy["y"]),
    )

    # Ensure y values are rounded to the same precision in both DataFrames
    # df["y"] = np.round(df["y"], precision)
    # averaged_data["y"] = np.round(averaged_data["y"], precision)
    #
    # df["y"] = df.round({"y": precision})
    # averaged_data["y"] = averaged_data.round({"y": precision})

    # Convert 'y' values to float64 to ensure consistency
    df["y"] = df["y"].astype("float64")
    averaged_data["y"] = averaged_data["y"].astype("float64")

    # Verify the rounding operation
    logger.debug("Rounded 'y' values in main DataFrame: \n%s\n", df["y"].unique())
    logger.debug(
        "Rounded 'y' values in averaged data: \n%s\n", averaged_data["y"].unique()
    )

    # Check and log data types of 'y' columns
    logger.debug(
        "Data type of 'y' in main DataFrame AFTER converting: %s", df["y"].dtype
    )
    logger.debug(
        "Data type of 'y' in averaged data AFTER converting: %s",
        averaged_data["y"].dtype,
    )

    # Check for NaN values in the initial dataframe
    nan_u_initial = df["u"].isna().sum()
    nan_v_initial = df["v"].isna().sum()
    logger.debug(
        "process_velocity_data_single: %s: Initial NaNs in 'u': %d",
        timestep,
        nan_u_initial,
    )
    logger.debug(
        "process_velocity_data_single: %s: Initial NaNs in 'v': %d",
        timestep,
        nan_v_initial,
    )

    # Check the structure of averaged_data
    logger.debug(
        "process_velocity_data_single: averaged_data columns: \n%s\n",
        averaged_data.columns,
    )
    logger.debug(
        "process_velocity_data_single: averaged_data sample: \n%s\n",
        averaged_data.head(),
    )

    # Log unique 'y' values and their counts in the main DataFrame
    unique_y_main = df["y"].value_counts().sort_index()
    logger.debug(
        "%s: Unique 'y' values in main DataFrame:\n %s\n",
        timestep,
        unique_y_main.head(20),
    )

    # Log unique 'y' values and their counts in the averaged data
    unique_y_averaged = averaged_data["y"].value_counts().sort_index()
    logger.debug(
        "%s: Unique 'y' values in averaged data:\n %s\n",
        timestep,
        unique_y_averaged.head(20),
    )

    # Compare and log differences in 'y' values
    y_in_main_not_in_averaged = set(unique_y_main.index) - set(unique_y_averaged.index)
    y_in_averaged_not_in_main = set(unique_y_averaged.index) - set(unique_y_main.index)
    logger.debug(
        "Y values in main DataFrame not in averaged data: %s", y_in_main_not_in_averaged
    )
    logger.debug(
        "Y values in averaged data not in main DataFrame: %s", y_in_averaged_not_in_main
    )

    # Select 5 rows with a specific y value before the merge
    sample_y_value = unique_y_main.index[2]  # Taking the first y value as sample
    df_sample_before_merge = df[df["y"] == sample_y_value].head(10)
    logger.debug("Sample rows before merge:\n%s", df_sample_before_merge)

    # Process the dataset for detailed fluctuation analysis
    df_merged = pd.merge(df, averaged_data, on="y", how="left")

    # Select the same 5 rows after the merge
    df_sample_after_merge = df_merged[df_merged["y"] == sample_y_value].head(10)
    logger.debug("Sample rows after merge:\n%s", df_sample_after_merge)

    # Log the result of the merge
    nan_U_bar = df_merged["U_bar"].isna().sum()
    nan_V_bar = df_merged["V_bar"].isna().sum()
    nan_W_bar = df_merged["W_bar"].isna().sum()
    logger.debug(
        "process_velocity_data_single: %s: NaNs in 'U_bar' after merge: %d",
        timestep,
        nan_U_bar,
    )
    logger.debug(
        "process_velocity_data_single: %s: NaNs in 'V_bar' after merge: %d",
        timestep,
        nan_V_bar,
    )
    logger.debug(
        "process_velocity_data_single: %s: NaNs in 'W_bar' after merge: %d",
        timestep,
        nan_W_bar,
    )

    df_processed = df.copy()

    # logger.debug(f"df_processed columns: {df_processed.columns.tolist()}")
    # logger.debug(f"df_processed index: {df_processed.index}")
    # logger.debug(f"df_processed y values: {df_processed['y'].values}")
    df_processed["U"] = df["u"]
    df_processed["V"] = df["v"]
    df_processed["W"] = df["w"]
    df_processed["u"] = df["u"] - df_merged["U_bar"]
    df_processed["v"] = df["v"] - df_merged["V_bar"]
    df_processed["w"] = df["w"] - df_merged["W_bar"]

    logger.debug(f"NEW df_processed columns: {df_processed.columns.tolist()}")
    logger.debug(f"NEW df_processed index: {df_processed.index}")
    logger.debug(f"NEW df_processed y values: {df_processed['y'].values}")

    # Ensure no 'timestep' column remains in the output data
    df_processed.drop(columns="timestep", inplace=True, errors="ignore")

    # Log NaN counts
    nan_u_count = df_processed["u"].isna().sum()
    nan_v_count = df_processed["v"].isna().sum()
    nan_U_count = df_processed["U"].isna().sum()
    nan_V_count = df_processed["V"].isna().sum()
    logger.debug(
        "process_velocity_data_single: %s: Number of NaNs in 'u' after processing: %d",
        timestep,
        nan_u_count,
    )
    logger.debug(
        "process_velocity_data_single: %s: Number of NaNs in 'v' after processing: %d",
        timestep,
        nan_v_count,
    )
    logger.debug(
        "process_velocity_data_single: %s: Number of NaNs in 'U' after processing: %d",
        timestep,
        nan_U_count,
    )
    logger.debug(
        "process_velocity_data_single: %s: Number of NaNs in 'V' after processing: %d",
        timestep,
        nan_V_count,
    )

    logger.info("Velocity data processing complete.")

    return timestep, df_processed


def detect_Q_events_single(
    timestep_df: Tuple[float, pd.DataFrame], averaged_data: pd.DataFrame, H: float
) -> Tuple[float, pd.DataFrame]:
    """
    Detects Q events in the fluid dynamics data based on the specified condition.

    Parameters:
    - timestep_df (tuple): Data processed by `process_velocity_data_single`, containing:
      * A timestep (float)
      * A DataFrame with spatial coordinates (x, y, z) and velocity components u, v, w, U, V, W (NORMALIZED!)
    - averaged_data (DataFrame): Data containing the rms values for velocity components u and v for each y coordinate. ** u' and v' **
    - H (float): The sensitivity threshold for identifying Q events.

    Returns:
    - q_event_data (tuple): A tuple containing:
      * A timestep (float)
      * A DataFrame with columns ['x', 'y', 'z', 'Q'], where 'Q' is a boolean indicating whether a Q event is detected.
    """
    logger.info("Detecting Q events...")
    timestep, df = timestep_df

    # Check for NaN values in u and v columns
    nan_u_count = df["u"].isna().sum()
    nan_v_count = df["v"].isna().sum()
    logger.debug("%s: Number of NaNs in 'u': %d", timestep, nan_u_count)
    logger.debug("%s: Number of NaNs in 'v': %d", timestep, nan_v_count)

    # Fetch the rms values for 'u' and 'v' based on y-coordinate
    rms_values = (
        averaged_data.set_index("y")[["u_prime", "v_prime"]].reindex(df["y"]).values
    )
    logger.debug("%s: Number of RMS values: %d", timestep, len(rms_values))

    # Calculate the product of fluctuating components u and v
    uv_product = np.abs(df["u"] * df["v"])
    logger.debug("%s: Number of uv products: %d", timestep, len(uv_product))
    logger.debug("%s: uv_product: %s", timestep, uv_product)

    # Calculate the threshold product of rms values u' and v'
    threshold = H * rms_values[:, 0] * rms_values[:, 1]
    logger.debug("%s: Number of threshold values: %d", timestep, len(threshold))
    logger.debug("%s: Threshold: %s", timestep, threshold)

    # Determine where the Q event condition is met
    q_events = uv_product > threshold  # to avoid detection on 0

    # Create DataFrame with Q event boolean flag
    q_df = pd.DataFrame({"x": df["x"], "y": df["y"], "z": df["z"], "Q": q_events})
    logger.debug("%s: number of Q events detected: %d", timestep, q_events.sum())

    logger.info("Q event detection complete.")

    return timestep, q_df


def calculate_local_Q_ratios(
    df: pd.DataFrame, nx: int, nz: int, Lx_norm: float, Lz_norm: float
) -> pd.DataFrame:
    """
    Calculate the ratio of Q-events in local volumes for a single timestep.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns ['x', 'y', 'z', 'Q'].
    - nx (int): Number of sections in the x direction.
    - nz (int): Number of sections in the z direction.
    - Lx_norm (float): Length in the x direction. - NORMALIZED BY DIVIDING BY DELTA_TAU
    - Lz_norm (float): Length in the z direction. - NORMALIZED BY DIVIDING BY DELTA_TAU

    Returns:
    - pd.DataFrame: DataFrame with columns ['x_index', 'z_index', 'Q_event_count', 'total_points', 'Q_ratio'].
    """
    logger.info("Calculating local Q ratios...")
    step_x = Lx_norm / nx
    step_z = Lz_norm / nz
    results = []

    for i in range(nz):
        for j in range(nz):
            x_min = i * step_x
            x_max = (i + 1) * step_x
            z_min = j * step_z
            z_max = (j + 1) * step_z

            local_points = df[
                (df["x"] >= x_min)
                & (df["x"] < x_max)
                & (df["z"] >= z_min)
                & (df["z"] < z_max)
            ]
            total_points = len(local_points)
            Q_event_count = local_points["Q"].sum()
            Q_ratio = Q_event_count / total_points if total_points > 0 else 0

            results.append(
                {
                    "x_index": i,
                    "z_index": j,
                    "Q_event_count": Q_event_count,
                    "total_points": total_points,
                    "Q_ratio": Q_ratio,
                }
            )

    result_df = pd.DataFrame(results)

    logger.info("Local Q ratio calculation complete.")

    return result_df


def calculate_reward(df: pd.DataFrame, nx: int, nz: int) -> float:
    """
    Calculate the reward based on the Q ratio in the local volume.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns ['x_index', 'z_index', 'Q_ratio'].

    Returns:
    - float: The calculated reward value.
    """
    logger.debug(f"Calculating reward for a environment [{nx}, {nz}] ...")

    q_ratio = df[(df["x_index"] == nx) & (df["z_index"] == nz)]["Q_ratio"].values[0]
    reward = 1 - q_ratio
    logger.debug("Reward calculation complete.")
    return reward


def calculate_reward_full(
    directory: str,
    Lx: float,
    Lz: float,
    H: float,
    nx: int,
    nz: int,
    averaged_data_path: str,
    output_file: str,
) -> None:
    """
    Calculate the rewards based on Q events for a single timestep, saving results to a CSV file.

    Parameters:
    - directory (str): Directory containing the PVD and PVTU files.
    - Lx (float): Length in the x direction.
    - Lz (float): Length in the z direction.
    - H (float): Sensitivity threshold for identifying Q events.
    - nx (int): Number of sections in the x direction.
    - nz (int): Number of sections in the z direction.
    - averaged_data_path (str): Path to the CSV file with averaged data.
    - output_file (str): Path to the output CSV file for rewards.
    """
    logging.info("Starting to calculate rewards based on Q events...")

    # Load and process the data
    filename = "channel.pvd"
    data = load_data_and_convert_to_dataframe_single(directory, filename)

    precalc_value_filename = "calculated_values.csv"
    precalc_value_filepath = os.path.join(averaged_data_path, precalc_value_filename)
    precalc_values = pd.read_csv(precalc_value_filepath)

    # List all precalculated values
    logger.debug(f"Pre-calculated values: {precalc_values}")
    # logger.debug(
    #     f"u_tau: {precalc_values['u_tau'].values[0]}, delta_tau: {precalc_values['delta_tau'].values[0]}"
    # )

    # List all pre-calculated values
    logger.debug(f"Pre-calculated values DataFrame:\n{precalc_values}")
    logger.debug(
        f"Columns in pre-calculated values DataFrame: {precalc_values.columns.tolist()}"
    )

    # Print first few rows of the pre-calculated values DataFrame
    logger.debug(
        f"First few rows of pre-calculated values DataFrame:\n{precalc_values.head()}"
    )

    u_tau = precalc_values.iloc[
        0, 1
    ]  # HARDCODED TO u_tau location in file saved by `coco_calc_avg_vel.py` - Pieter
    delta_tau = precalc_values.iloc[
        1, 1
    ]  # HARDCODED TO delta_tau location in file saved by `coco_calc_avg_vel.py` - Pieter

    data_normalized = normalize_all_single(data, u_tau, delta_tau)

    averaged_data_filename = "averaged_data.csv"
    averaged_data_filepath = os.path.join(averaged_data_path, averaged_data_filename)
    averaged_data = pd.read_csv(averaged_data_filepath)

    processed_data = process_velocity_data_single(data_normalized, averaged_data)

    Q_event_frames = detect_Q_events_single(processed_data, averaged_data, H)

    Lx_norm = Lx / delta_tau
    Lz_norm = Lz / delta_tau

    all_results = []
    timestep, df = Q_event_frames  # Since there is only one timestep

    result_df = calculate_local_Q_ratios(df, nx, nz, Lx_norm, Lz_norm)
    result_df["timestep"] = timestep
    all_results.append(result_df)

    final_result_df = pd.concat(all_results, ignore_index=True)

    rewards = []
    for i in range(nx):
        for j in range(nz):
            reward = calculate_reward(final_result_df, i, j)
            env_id = agent_index_2d_to_1d(i, j, nz)  # Converted 2D index to 1D
            rewards.append({"ENV_ID": env_id, "reward": reward})

    reward_df = pd.DataFrame(rewards)

    reward_df.to_csv(output_file, index=False)
    logger.info(f"Rewards saved to {output_file}!")

    # Clean up
    del (
        data,
        data_normalized,
        averaged_data,
        processed_data,
        Q_event_frames,
        final_result_df,
        rewards,
    )
    gc.collect()
    logger.debug("Memory cleaned up.")


"""
Example usage:
python calc_reward.py --directory path/to/data --Lx 1.0 --Lz 1.0 --H 3.0 --nx 4 --nz 3 --averaged_data_path path/to/averaged_data --output_file rewards.csv
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate rewards based on Q events.")
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing the PVD and PVTU files. MUST INCLUDE channel.pvd!!!",
    )
    parser.add_argument(
        "--Lx",
        type=float,
        required=True,
        help="Length in the x direction.",
    )
    parser.add_argument(
        "--Lz",
        type=float,
        required=True,
        help="Length in the z direction.",
    )
    parser.add_argument(
        "--H",
        type=float,
        required=True,
        help="Sensitivity threshold for identifying Q events.",
    )
    parser.add_argument(
        "--nx", type=int, required=True, help="Number of sections in the x direction."
    )
    parser.add_argument(
        "--nz", type=int, required=True, help="Number of sections in the z direction."
    )
    parser.add_argument(
        "--averaged_data_path",
        type=str,
        required=True,
        help="Path to DIRECTORY containing csv file with averaged data AND csv file with pre-calculated values for u_tau and delta_tau.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output CSV file for rewards.",
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

    calculate_reward_full(
        args.directory,
        args.Lx,
        args.Lz,
        args.H,
        args.nx,
        args.nz,
        args.averaged_data_path,
        args.output_file,
    )
