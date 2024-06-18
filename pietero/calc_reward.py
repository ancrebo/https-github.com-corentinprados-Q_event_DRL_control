import os
import pandas as pd
import pyvista as pv
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import gc
from typing import List, Tuple

# TODO: Remove code that deals with multiple timesteps


def load_data_and_convert_to_dataframe(
    directory: str,
) -> List[Tuple[float, pd.DataFrame]]:
    """
    This function loads CFD simulation data from PVTU files and converts them into Pandas DataFrames.
    Each DataFrame is stored along with its respective timestep, facilitating time-series analysis.

    Parameters:
    - directory (str): The path to the directory containing the PVD and PVTU files.

    Returns:
    - data_frames (list of tuples): A list where each element is a tuple containing:
      * A timestep (float)
      * A DataFrame with columns for spatial coordinates (x, y, z) and velocity components (U, V, W)
    """
    pvd_path = os.path.join(directory, "channel.pvd")
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    timestep_file_map = {
        dataset.attrib["file"]: float(dataset.attrib["timestep"])
        for dataset in root.find("Collection")
    }

    data_frames = []

    for file, timestep in timestep_file_map.items():
        path = os.path.join(directory, file)
        mesh = pv.read(path)
        points = mesh.points
        U, V, W = mesh["VELOC"].T

        df = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "U": U,
                "V": V,
                "W": W,
            }
        )

        data_frames.append((timestep, df))
        print(f"Data from {file} at timestep {timestep} loaded into DataFrame.")

    print(f"Total data sets loaded: {len(data_frames)}")
    return data_frames


def normalize_data_frame(
    df: pd.DataFrame, Lx: float, Ly: float, Lz: float
) -> pd.DataFrame:
    """
    Normalizes the spatial coordinates to the range [0, 1] based on their maximum values
    and scales the velocity components by the corresponding channel lengths.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns for spatial coordinates (x, y, z) and velocity components (U, V, W).
    - Lx (float): Length in the x direction.
    - Ly (float): Length in the y direction.
    - Lz (float): Length in the z direction.

    Returns:
    - pd.DataFrame: Scaled/Normalized DataFrame.
    """
    df_copy = df.copy()

    df_copy["x"] = df_copy["x"] / df_copy["x"].max()
    df_copy["y"] = df_copy["y"] / df_copy["y"].max()
    df_copy["z"] = df_copy["z"] / df_copy["z"].max()
    df_copy["U"] = df_copy["U"] / df_copy["x"].max()
    df_copy["V"] = df_copy["V"] / df_copy["y"].max()
    df_copy["W"] = df_copy["W"] / df_copy["z"].max()

    df_copy["x"] = df_copy["x"] * Lx
    df_copy["y"] = df_copy["y"] * Ly
    df_copy["z"] = df_copy["z"] * Lz
    df_copy["U"] = df_copy["U"] * Lx
    df_copy["V"] = df_copy["V"] * Ly
    df_copy["W"] = df_copy["W"] * Lz

    return df_copy


def load_averaged_data(file_path: str) -> pd.DataFrame:
    """
    Load the averaged data from a Parquet file.

    Parameters:
    - file_path (str): The path to the Parquet file.

    Returns:
    - pd.DataFrame: The loaded averaged data.
    """
    return pd.read_parquet(file_path)


def process_velocity_data(
    data: List[Tuple[float, pd.DataFrame]], averaged_data: pd.DataFrame
) -> List[Tuple[float, pd.DataFrame]]:
    """
    Processes a list of tuples containing CFD simulation data using pre-calculated averaged data to calculate
    fluctuating components of velocity fields. It computes these metrics for the horizontal (U), vertical (V),
    and lateral (W) velocity components.

    Parameters:
    - data (list of tuples): Each tuple contains a timestep and a DataFrame with spatial coordinates (x, y, z)
      and velocity components (U, V, W).
    - averaged_data (DataFrame): Pre-calculated averaged data containing velocities ($\overline{U}(y)$, $\overline{V}(y)$, $\overline{W}(y)$)
      and rms of velocity fluctuations ($u'(y)$, $v'(y)$, $w'(y)$) as columns, indexed by the y-coordinate.

    Returns:
    - processed_data (list of tuples): Each tuple contains a timestep and a DataFrame with original and fluctuating
      velocity components (U, V, W, u, v, w).
    """
    processed_data = []

    # Process each individual dataset for detailed fluctuation analysis
    for timestep, df in data:
        y_means = averaged_data.loc[df["y"]]
        df_processed = df.copy()
        df_processed["U"] = df["U"]
        df_processed["V"] = df["V"]
        df_processed["W"] = df["W"]
        df_processed["u"] = df["U"] - y_means["U_bar"].values
        df_processed["v"] = df["V"] - y_means["V_bar"].values
        df_processed["w"] = df["W"] - y_means["W_bar"].values

        # Ensure no 'timestep' column remains in the output data
        df_processed.drop(columns="timestep", inplace=True, errors="ignore")
        processed_data.append((timestep, df_processed))

    return processed_data


def detect_Q_events(
    processed_data: List[Tuple[float, pd.DataFrame]],
    averaged_data: pd.DataFrame,
    H: float,
) -> List[Tuple[float, pd.DataFrame]]:
    """
    Detects Q events in the fluid dynamics data based on the specified condition.

    Parameters:
    - processed_data (list of tuples): Data processed by `process_velocity_data`, containing:
      * A timestep (float)
      * A DataFrame with spatial coordinates (x, y, z) and velocity components U, V, W, u, v, w
    - averaged_data (DataFrame): Data containing the rms values for velocity components u and v for each y coordinate. **(u' and v')**
    - H (float): The sensitivity threshold for identifying Q events.

    Returns:
    - data_frames (list of tuples): Each tuple contains:
      * A timestep (float)
      * A DataFrame with columns ['x', 'y', 'z', 'Q'], where 'Q' is a boolean indicating whether a Q event is detected.
    """
    q_event_data = []

    for timestep, df in processed_data:
        rms_values = (
            averaged_data.set_index("y")[["u_prime", "v_prime"]].reindex(df["y"]).values
        )
        uv_product = np.abs(df["u"] * df["v"])
        threshold = H * rms_values[:, 0] * rms_values[:, 1]
        q_events = uv_product > threshold
        q_df = pd.DataFrame({"x": df["x"], "y": df["y"], "z": df["z"], "Q": q_events})
        q_event_data.append((timestep, q_df))

    return q_event_data


def calculate_local_Q_ratios(
    df: pd.DataFrame, n: int, m: int, Lx: float, Lz: float
) -> pd.DataFrame:
    """
    Calculate the ratio of Q-events in local volumes for a single timestep.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns ['x', 'y', 'z', 'Q'].
    - n (int): Number of sections in the x direction.
    - m (int): Number of sections in the z direction.
    - Lx (float): Length in the x direction.
    - Lz (float): Length in the z direction.

    Returns:
    - pd.DataFrame: DataFrame with columns ['x_index', 'z_index', 'Q_event_count', 'total_points', 'Q_ratio'].
    """
    step_x = Lx / n
    step_z = Lz / m
    results = []

    for i in range(n):
        for j in range(m):
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
    return result_df


def calculate_reward(df: pd.DataFrame, n: int, m: int) -> float:
    """
    Calculate the reward based on the Q ratio in the local volume.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns ['x_index', 'z_index', 'Q_ratio'].

    Returns:
    - float: The calculated reward value.
    """
    q_ratio = df[(df["x_index"] == n) & (df["z_index"] == m)]["Q_ratio"].values[0]
    reward = 1 - q_ratio
    return reward


def calculate_reward_full(
    directory: str,
    Lx: float,
    Ly: float,
    Lz: float,
    H: float,
    n: int,
    m: int,
    averaged_data_path: str,
    output_file: str,
) -> None:
    """
    Calculate the rewards based on Q events, saving results to a CSV file.

    Parameters:
    - directory (str): Directory containing the PVD and PVTU files.
    - Lx (float): Length in the x direction.
    - Ly (float): Length in the y direction.
    - Lz (float): Length in the z direction.
    - H (float): Sensitivity threshold for identifying Q events.
    - n (int): Number of sections in the x direction.
    - m (int): Number of sections in the z direction.
    - averaged_data_path (str): Path to the Parquet file with averaged data.
    - output_file (str): Path to the output Parquet file for rewards.
    """
    data = load_data_and_convert_to_dataframe(directory)
    data_normalized = [
        (timestep, normalize_data_frame(df, Lx, Ly, Lz)) for timestep, df in data
    ]
    averaged_data = load_averaged_data(averaged_data_path)

    processed_data = process_velocity_data(data_normalized, averaged_data)

    Q_event_frames = detect_Q_events(processed_data, averaged_data, H)

    all_results = []

    for timestep, df in Q_event_frames:
        result_df = calculate_local_Q_ratios(df, n, m, Lx, Lz)
        result_df["timestep"] = timestep
        all_results.append(result_df)

    final_result_df = pd.concat(all_results, ignore_index=True)

    rewards = []
    for i in range(n):
        for j in range(m):
            reward = calculate_reward(final_result_df, i, j)
            rewards.append({"x_index": i, "z_index": j, "reward": reward})

    reward_df = pd.DataFrame(rewards)
    reward_df.to_parquet(output_file, index=False)
    print(f"Rewards saved to {output_file}")

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
    print("Memory cleaned up.")


# Example usage:
"""
python step2_calculate_rewards.py --directory path/to/data --Lx 1.0 --Ly 1.0 --Lz 1.0 --H 3.0 --n 4 --m 3 --averaged_data_path averaged_data.parquet --output_file rewards.csv
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate rewards based on Q events.")
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing the PVD and PVTU files.",
    )
    parser.add_argument(
        "--Lx", type=float, required=True, help="Length in the x direction."
    )
    parser.add_argument(
        "--Ly", type=float, required=True, help="Length in the y direction."
    )
    parser.add_argument(
        "--Lz", type=float, required=True, help="Length in the z direction."
    )
    parser.add_argument(
        "--H",
        type=float,
        required=True,
        help="Sensitivity threshold for identifying Q events.",
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of sections in the x direction."
    )
    parser.add_argument(
        "--m", type=int, required=True, help="Number of sections in the z direction."
    )
    parser.add_argument(
        "--averaged_data_path",
        type=str,
        required=True,
        help="Path to the Parquet file with averaged data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output CSV file for rewards.",
    )

    args = parser.parse_args()
    calculate_reward_full(
        args.directory,
        args.Lx,
        args.Ly,
        args.Lz,
        args.H,
        args.n,
        args.m,
        args.averaged_data_path,
        args.output_file,
    )
