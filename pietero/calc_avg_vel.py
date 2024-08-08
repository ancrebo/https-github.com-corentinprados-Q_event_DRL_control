import os
import pandas as pd
import pyvista as pv
import xml.etree.ElementTree as ET
import argparse
import gc
from typing import List, Tuple


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


def process_velocity_data_multiple_timesteps(
    data: List[Tuple[float, pd.DataFrame]], N: int
) -> Tuple[pd.DataFrame, List[Tuple[float, pd.DataFrame]]]:
    """
    Processes a list of tuples containing CFD simulation data to calculate averaged and
    fluctuating components of velocity fields over the last N entries. It computes these metrics for
    the horizontal (U), vertical (V), and lateral (W) velocity components.

    Parameters:
    - data (list of tuples): Each tuple contains a timestep and a DataFrame with spatial coordinates (x, y, z)
      and velocity components (U, V, W).
    - N (int): Number of most recent timesteps to include in the averaging process.

    Returns:
    - averaged_data (pd.DataFrame): Contains averaged velocities ($\overline{U}(y)$, $\overline{V}(y)$, $\overline{W}(y)$)
      and rms of velocity fluctuations ($u'(y)$, $v'(y)$, $w'(y)$) as columns, indexed by the y-coordinate.
    - data_process (list of tuples): Each tuple contains a timestep and a DataFrame with original and fluctuating
      velocity components (U, V, W, u, v, w).
    """
    processed_data = []
    recent_data = pd.DataFrame()

    for timestep, df in data[-N:]:
        df["timestep"] = timestep
        recent_data = pd.concat([recent_data, df], ignore_index=True)

    averaged_data = (
        recent_data.groupby("y")
        .agg({"U": ["mean", "std"], "V": ["mean", "std"], "W": ["mean", "std"]})
        .rename(columns={"mean": "bar", "std": "prime"}, level=1)
    )
    averaged_data.columns = [
        "U_bar",
        "u_prime",
        "V_bar",
        "v_prime",
        "W_bar",
        "w_prime",
    ]

    for timestep, df in data:
        y_means = averaged_data.loc[df["y"]]
        df_processed = df.copy()
        df_processed["U"] = df["U"]
        df_processed["V"] = df["V"]
        df_processed["W"] = df["W"]
        df_processed["u"] = df["U"] - y_means["U_bar"].values
        df_processed["v"] = df["V"] - y_means["V_bar"].values
        df_processed["w"] = df["W"] - y_means["W_bar"].values

        df_processed.drop(columns="timestep", inplace=True, errors="ignore")
        processed_data.append((timestep, df_processed))

    averaged_data = averaged_data.reset_index()

    return averaged_data, processed_data


def save_dataframe_to_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a Parquet file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - file_path (str): The path to the file where the DataFrame should be saved.
    """
    df.to_parquet(file_path, index=None)


def calculate_and_save_averaged_velocity_full(
    directory: str, Lx: float, Ly: float, Lz: float, N: int, output_file: str
) -> None:
    """
    Calculate and save the averaged velocity components for CFD simulation data.

    Parameters:
    - directory (str): Directory containing the PVD and PVTU files.
    - Lx (float): Length in the x direction.
    - Ly (float): Length in the y direction.
    - Lz (float): Length in the z direction.
    - N (int): Number of most recent timesteps to include in the averaging process.
    - output_file (str): Path to the output Parquet file.
    """
    data = load_data_and_convert_to_dataframe(directory)

    data_normalized = [
        (timestep, normalize_data_frame(df, Lx, Ly, Lz)) for timestep, df in data
    ]

    averaged_data, _ = process_velocity_data_multiple_timesteps(data_normalized, N)

    save_dataframe_to_parquet(averaged_data, output_file)
    print(f"Averaged data saved to {output_file}")

    # Clean up
    del data, data_normalized, averaged_data
    gc.collect()
    print("Memory cleaned up.")


def main(directory: str, Lx: float, Ly: float, Lz: float, N: int, output_file: str):
    data = load_data_and_convert_to_dataframe(directory)

    data_normalized = [
        (timestep, normalize_data_frame(df, Lx, Ly, Lz)) for timestep, df in data
    ]

    averaged_data, _ = process_velocity_data_multiple_timesteps(data_normalized, N)

    save_dataframe_to_parquet(averaged_data, output_file)
    print(f"Averaged data saved to {output_file}")

    # Clean up
    del data, data_normalized, averaged_data
    gc.collect()
    print("Memory cleaned up.")


# TODO: use config file with default values?

"""
EXAMPLE USAGE:
python calc_avg_vel --directory path/to/data --Lx 1.0 --Ly 1.0 --Lz 1.0 --N 5 --output_file averaged_data.parquet
"""

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
        "--Lx", type=float, required=True, help="Length in the x direction."
    )
    parser.add_argument(
        "--Ly", type=float, required=True, help="Length in the y direction."
    )
    parser.add_argument(
        "--Lz", type=float, required=True, help="Length in the z direction."
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Number of most recent timesteps to include in the averaging process.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output Parquet file.",
    )

    args = parser.parse_args()
    calculate_and_save_averaged_velocity_full(
        args.directory, args.Lx, args.Ly, args.Lz, args.N, args.output_file
    )
