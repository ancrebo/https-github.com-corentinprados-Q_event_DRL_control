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

logger.info("coco_calc_reward.py: Logging level set to %s\n", logger.level)


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
    logger.debug("load_data_and_convert_to_dataframe: Loading data from PVTU files...")
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

    logger.debug(
        "load_data_and_convert_dataframe_single: Data from %s at timestep %s loaded into DataFrame.",
        file,
        timestep,
    )
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
    logger.debug("normalize_all_single: Normalizing data...")

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

    logger.debug("noramlize_all_single: Data normalization complete!")
    return timestep, df_copy


def process_velocity_data_single(
    timestep_df: Tuple[float, pd.DataFrame],
    averaged_data: pd.DataFrame,
    tolerance: float = 1e-4,
) -> Tuple[float, pd.DataFrame]:
    """
    Processes a tuple containing CFD simulation data to calculate fluctuating components of velocity fields.
    It computes these metrics for the horizontal (u), vertical (v), and lateral (w) velocity components.

    Parameters:
    - timestep_df (tuple): A tuple containing a timestep and a DataFrame with spatial coordinates (x, y, z)
      and velocity components (u, v, w). (NORMALIZED!)
    - averaged_data (DataFrame): Contains averaged velocities ($\overline{U}(y)$, $\overline{V}(y)$, $\overline{W}(y)$)
      and rms of velocity fluctuations ($u'(y)$, $v'(y)$, $w'(y)$) as columns, indexed by the y-coordinate. (NORMALIZED!)
    - tolerance (float): The tolerance for matching 'y' values between the main and averaged data. ENSURES THEY ARE THE SAME!

    Returns:
    - processed_data (tuple): A tuple containing a timestep and a DataFrame with original and fluctuating
      velocity components (U, V, W, u, v, w).
    """
    timestep, df = timestep_df
    logger.debug(
        "process_velocity_data_single: Processing velocity data using loaded averaged data..."
    )

    precision: int = 3
    tolerance: float = 1e-3

    # Adjust 'y' values in averaged data to match main data if within tolerance
    main_y_unique = df["y"].unique()
    averaged_y_unique = averaged_data["y"].unique()

    for main_y in main_y_unique:
        for i, avg_y in enumerate(averaged_y_unique):
            if np.abs(main_y - avg_y) <= tolerance:
                logger.debug(
                    "process_velocity_data_single: Overwriting averaged y value %s with main y value %s",
                    avg_y,
                    main_y,
                )
                averaged_data.loc[averaged_data["y"] == avg_y, "y"] = main_y
                averaged_y_unique[i] = (
                    main_y  # Update the unique values list to reflect the change
                )
                break

    # Process the dataset for detailed fluctuation analysis
    df_merged = pd.merge(df, averaged_data, on="y", how="left")

    df_processed = df.copy()

    df_processed["U"] = df["u"]
    df_processed["V"] = df["v"]
    df_processed["W"] = df["w"]
    df_processed["u"] = df["u"] - df_merged["U_bar"]
    df_processed["v"] = df["v"] - df_merged["V_bar"]
    df_processed["w"] = df["w"] - df_merged["W_bar"]

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

    logger.debug("process_velocity_data_single: Velocity data processing complete!")

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
    logger.debug("detect_Q_events_single: Detecting Q events...")
    timestep, df = timestep_df

    # Fetch the rms values for 'u' and 'v' based on y-coordinate
    rms_values = (
        averaged_data.set_index("y")[["u_prime", "v_prime"]].reindex(df["y"]).values
    )

    # Calculate the product of fluctuating components u and v
    uv_product = np.abs(df["u"] * df["v"])

    # Calculate the threshold product of rms values u' and v'
    threshold = H * rms_values[:, 0] * rms_values[:, 1]

    # Determine where the Q event condition is met
    q_events = uv_product > threshold  # to avoid detection on 0

    # Create DataFrame with Q event boolean flag
    q_df = pd.DataFrame({"x": df["x"], "y": df["y"], "z": df["z"], "Q": q_events})
    logger.debug(
        "detect_Q_events_single: %s: number of Q events detected: %d",
        timestep,
        q_events.sum(),
    )

    logger.debug("detect_Q_events_single: Q event detection complete!")

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
    logger.debug("calculate_local_Q_ratios: Calculating local Q ratios...")
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

    logger.debug("calculate_local_Q_ratios: Local Q ratio calculation complete!")

    return result_df


def calculate_reward(df: pd.DataFrame, avg_qratio: float, nx: int, nz: int) -> float:
    """
    Calculate the reward based on the Q ratio in the local volume.

    The reward is scaled to range from -1 to 1, where a lower local Q ratio compared
    to the global average results in a positive reward and a higher local Q ratio
    results in a negative reward. The goal is to minimize the Q ratio.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the columns ['x_index', 'z_index', 'Q_ratio'] which
        represents the Q ratios at different grid points.
    avg_qratio : float
        The average global Q event volume ratio used for scaling the reward.
    nx : int
        The x-index of the local volume.
    nz : int
        The z-index of the local volume.

    Returns:
    -------
    float
        The calculated reward value. A value of 1 indicates no Q events in the
        local volume, a value of 0 indicates the local Q ratio equals the global
        average, and negative values indicate the local Q ratio exceeds the global average.
    """
    logger.debug(
        "calculate_reward: Calculating reward for a environment [%d, %d] ...", nx, nz
    )

    q_ratio = df[(df["x_index"] == nx) & (df["z_index"] == nz)]["Q_ratio"].values[0]
    # reward = 1 - q_ratio
    reward = 1 - (q_ratio / avg_qratio)

    # clip the reward to [-1, 1] # TODO: Is this necessary??? - Pieter
    reward = np.clip(reward, -1, 1)

    logger.debug("calculate_reward: Reward calculation complete!")
    return reward


def calculate_reward_full(
    directory: str,
    Lx: float,
    Lz: float,
    H: float,
    nx: int,
    nz: int,
    averaged_data_path: str,
    output_qratio_file: str,
    output_reward_file: str,
) -> None:
    """
    Calculate the rewards based on Q events for a single timestep, saving results to a CSV file.

    Parameters:
    ----------
    directory : str
        Directory containing the PVD and PVTU files.
    Lx : float
        Length in the x direction.
    Lz : float
        Length in the z direction.
    H : float
        Sensitivity threshold for identifying Q events.
    nx : int
        Number of sections in the x direction.
    nz : int
        Number of sections in the z direction.
    averaged_data_path : str
        Path to the directory with averaged data CSV files.
    output_qratio_file : str
        Path to the output CSV file for saving calculated local Q event volume ratios.
    output_reward_file : str
        Path to the output CSV file for saving calculated rewards.
    """
    logging.info(
        "calculate_reward_full: Starting to calculate rewards based on Q events..."
    )

    # Load and process the data
    filename = "channel.pvd"
    data = load_data_and_convert_to_dataframe_single(directory, filename)

    # Load pre-calculated values
    precalc_value_filename = "calculated_values.csv"
    precalc_value_filepath = os.path.join(averaged_data_path, precalc_value_filename)
    precalc_values = pd.read_csv(precalc_value_filepath)
    u_tau = precalc_values.iloc[0, 1]
    delta_tau = precalc_values.iloc[1, 1]

    # Load global Q event average ratio and standard deviation
    q_event_summary_filename = "q_event_ratio_summary.csv"
    q_event_summary_filepath = os.path.join(
        averaged_data_path, q_event_summary_filename
    )
    q_event_summary = pd.read_csv(q_event_summary_filepath)
    avg_qratio = q_event_summary["average_q_event_ratio"].values[0]
    std_dev_qratio = q_event_summary["std_dev_q_event_ratio"].values[0]

    logger.debug("calculate_reward_full: Pre-calculated values: \n%s", precalc_values)
    logger.debug(
        "calculate_reward_full: Loaded Global Q event average ratio: %f", avg_qratio
    )
    logger.debug(
        "calculate_reward_full: Loaded Global Q event standard deviation: %f",
        std_dev_qratio,
    )

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

    # Add ENV_ID column to result_df
    result_df["ENV_ID"] = result_df.apply(
        lambda row: int(agent_index_2d_to_1d(row["x_index"], row["z_index"], nz)),
        axis=1,
    )

    all_results.append(result_df)
    logger.debug("calculate_reward_full: all_results: \n%s\n", all_results)

    final_result_df = pd.concat(all_results, ignore_index=True)
    logger.debug("calculate_reward_full: final_result_df: \n%s\n", final_result_df)

    # Save the local Q event ratios to a CSV file
    final_result_df.to_csv(output_qratio_file, index=False)
    logger.info(
        "calculate_reward_full: Local Q event ratios saved to %s!", output_qratio_file
    )

    # Calculate and save rewards
    final_result_df["reward"] = final_result_df.apply(
        lambda row: calculate_reward(
            final_result_df, avg_qratio, row["x_index"], row["z_index"]
        ),
        axis=1,
    )

    reward_df = final_result_df[["ENV_ID", "reward"]]

    reward_df.to_csv(output_reward_file, index=False)
    logger.info("calculate_reward_full: Rewards saved to %s!", output_reward_file)

    # Clean up
    del (
        data,
        data_normalized,
        averaged_data,
        processed_data,
        Q_event_frames,
        final_result_df,
    )
    gc.collect()
    logger.debug("calculate_reward_full: Memory cleaned up.")


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
        "--output_qratio_file",
        type=str,
        required=True,
        help="Path to the output CSV file for saving calculated local Q event volume ratio.",
    )
    parser.add_argument(
        "--output_reward_file",
        type=str,
        required=True,
        help="Path to the output CSV file for saving calculated rewards.",
    )

    args = parser.parse_args()

    calculate_reward_full(
        args.directory,
        args.Lx,
        args.Lz,
        args.H,
        args.nx,
        args.nz,
        args.averaged_data_path,
        args.output_qratio_file,
        args.output_reward_file,
    )
