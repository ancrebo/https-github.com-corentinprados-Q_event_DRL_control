import os
import numpy as np
import pyvista as pv
import xml.etree.ElementTree as ET
import gc
from typing import List, Tuple


def load_data_and_convert_to_numpy(
    directory: str,
) -> List[Tuple[float, np.ndarray]]:
    """
    This function loads CFD simulation data from PVTU files and converts them into NumPy arrays.
    Each array is stored along with its respective timestep, facilitating time-series analysis.

    Parameters:
    - directory (str): The path to the directory containing the PVD and PVTU files - including channel.pvd.

    Returns:
    - data_arrays (list of tuples): A list where each element is a tuple containing:
      * A timestep (float)
      * An array with columns for spatial coordinates (x, y, z) and velocity components (U, V, W)
    """
    pvd_path = os.path.join(directory, "channel.pvd")
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    timestep_file_map = {
        dataset.attrib["file"]: float(dataset.attrib["timestep"])
        for dataset in root.find("Collection")
    }

    data_arrays = []

    for file, timestep in timestep_file_map.items():
        path = os.path.join(directory, file)
        mesh = pv.read(path)
        points = mesh.points

        # Print available data arrays
        print(f"Available data arrays in {file}: {mesh.array_names}")

        if "VELOC" not in mesh.array_names:
            raise KeyError(f"Data array 'VELOC' not found in {file}.")
        U, V, W = mesh["VELOC"].T

        data_array = np.column_stack((points, U, V, W))

        data_arrays.append((timestep, data_array))
        print(f"Data from {file} at timestep {timestep} loaded into NumPy array.")

    print(f"Total data sets loaded: {len(data_arrays)}")
    return data_arrays


def normalize_data_array(
    data_array: np.ndarray, Lx: float, Ly: float, Lz: float
) -> np.ndarray:
    """
    Normalizes the spatial coordinates to the range [0, 1] based on their maximum values
    and scales the velocity components by the corresponding channel lengths.

    Parameters:
    - data_array (np.ndarray): Input array with columns for spatial coordinates (x, y, z) and velocity components (U, V, W).
    - Lx (float): Length in the x direction.
    - Ly (float): Length in the y direction.
    - Lz (float): Length in the z direction.

    Returns:
    - np.ndarray: Scaled/Normalized array.
    """
    data_copy = data_array.copy()

    data_copy[:, 0] /= data_copy[:, 0].max()
    data_copy[:, 1] /= data_copy[:, 1].max()
    data_copy[:, 2] /= data_copy[:, 2].max()
    data_copy[:, 3] /= data_copy[:, 0].max()
    data_copy[:, 4] /= data_copy[:, 1].max()
    data_copy[:, 5] /= data_copy[:, 2].max()

    data_copy[:, 0] *= Lx
    data_copy[:, 1] *= Ly
    data_copy[:, 2] *= Lz
    data_copy[:, 3] *= Lx
    data_copy[:, 4] *= Ly
    data_copy[:, 5] *= Lz

    return data_copy


def load_averaged_data_csv(file_path: str) -> np.ndarray:
    """
    Load the averaged data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - np.ndarray: The loaded averaged data.
    """
    return np.loadtxt(file_path, delimiter=",", skiprows=1)  # Do I want to skip rows???


def process_velocity_data(
    data: List[Tuple[float, np.ndarray]], averaged_data: np.ndarray
) -> List[Tuple[float, np.ndarray]]:
    """
    Processes a list of tuples containing CFD simulation data using pre-calculated averaged data to calculate
    fluctuating components of velocity fields. It computes these metrics for the horizontal (U), vertical (V),
    and lateral (W) velocity components.

    Parameters:
    - data (list of tuples): Each tuple contains a timestep and a NumPy array with spatial coordinates (x, y, z)
      and velocity components (U, V, W).
    - averaged_data (np.ndarray): Pre-calculated averaged data containing velocities ($\overline{U}(y)$, $\overline{V}(y)$, $\overline{W}(y)$)
      and rms of velocity fluctuations ($u'(y)$, $v'(y)$, $w'(y)$) as columns, indexed by the y-coordinate.

    Returns:
    - processed_data (list of tuples): Each tuple contains a timestep and an array with original and fluctuating
      velocity components (U, V, W, u, v, w).
    """
    processed_data = []

    y_values = averaged_data[:, 0]
    U_bar = averaged_data[:, 1]
    V_bar = averaged_data[:, 3]
    W_bar = averaged_data[:, 5]

    for timestep, data_array in data:
        y_data = data_array[:, 1]

        U_interp = np.interp(y_data, y_values, U_bar)
        V_interp = np.interp(y_data, y_values, V_bar)
        W_interp = np.interp(y_data, y_values, W_bar)

        data_processed = np.zeros((data_array.shape[0], data_array.shape[1] + 3))
        data_processed[:, :6] = data_array

        data_processed[:, 6] = data_array[:, 3] - U_interp
        data_processed[:, 7] = data_array[:, 4] - V_interp
        data_processed[:, 8] = data_array[:, 5] - W_interp

        processed_data.append((timestep, data_processed))

    return processed_data


def detect_Q_events(
    processed_data: List[Tuple[float, np.ndarray]],
    averaged_data: np.ndarray,
    H: float,
) -> List[Tuple[float, np.ndarray]]:
    """
    Detects Q events in the fluid dynamics data based on the specified condition.

    Parameters:
    - processed_data (list of tuples): Data processed by `process_velocity_data`, containing:
      * A timestep (float)
      * An array with spatial coordinates (x, y, z) and velocity components U, V, W, u, v, w
    - averaged_data (np.ndarray): Data containing the rms values for velocity components u and v for each y coordinate. **(u' and v')**
    - H (float): The sensitivity threshold for identifying Q events.

    Returns:
    - data_arrays (list of tuples): Each tuple contains:
      * A timestep (float)
      * An array with columns ['x', 'y', 'z', 'Q'], where 'Q' is a boolean indicating whether a Q event is detected.
    """
    q_event_data = []

    y_values = averaged_data[:, 0]
    u_prime = averaged_data[:, 2]
    v_prime = averaged_data[:, 4]

    for timestep, data_array in processed_data:
        y_data = data_array[:, 1]

        u_prime_interp = np.interp(y_data, y_values, u_prime)
        v_prime_interp = np.interp(y_data, y_values, v_prime)

        uv_product = np.abs(data_array[:, 6] * data_array[:, 7])
        threshold = H * u_prime_interp * v_prime_interp
        q_events = uv_product > threshold
        q_array = np.column_stack((data_array[:, :3], q_events))
        q_event_data.append((timestep, q_array))

    return q_event_data


def calculate_local_Q_ratios(
    data_array: np.ndarray, n: int, m: int, Lx: float, Lz: float
) -> np.ndarray:
    """
    Calculate the ratio of Q-events in local volumes for a single timestep.

    Parameters:
    - data_array (np.ndarray): Input array with columns ['x', 'y', 'z', 'Q'].
    - n (int): Number of sections in the x direction.
    - m (int): Number of sections in the z direction.
    - Lx (float): Length in the x direction.
    - Lz (float): Length in the z direction.

    Returns:
    - np.ndarray: Array with columns ['x_index', 'z_index', 'Q_event_count', 'total_points', 'Q_ratio'].
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

            local_points = data_array[
                (data_array[:, 0] >= x_min)
                & (data_array[:, 0] < x_max)
                & (data_array[:, 2] >= z_min)
                & (data_array[:, 2] < z_max)
            ]
            total_points = len(local_points)
            Q_event_count = local_points[:, 3].sum()
            Q_ratio = Q_event_count / total_points if total_points > 0 else 0

            results.append([i, j, Q_event_count, total_points, Q_ratio])

    result_array = np.array(results)
    return result_array


def calculate_reward(data_array: np.ndarray, n: int, m: int) -> float:
    """
    Calculate the reward based on the Q ratio in the local volume.

    Parameters:
    - data_array (np.ndarray): Array with columns ['x_index', 'z_index', 'Q_ratio'].

    Returns:
    - float: The calculated reward value.
    """
    q_ratio = data_array[(data_array[:, 0] == n) & (data_array[:, 1] == m), 4]
    reward = 1 - q_ratio[0] if q_ratio.size > 0 else 1
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
    Calculate the rewards based on Q events, saving results to a file.

    Parameters:
    - directory (str): Directory containing the PVD and PVTU files.
    - Lx (float): Length in the x direction.
    - Ly (float): Length in the y direction.
    - Lz (float): Length in the z direction.
    - H (float): Sensitivity threshold for identifying Q events.
    - n (int): Number of sections in the x direction.
    - m (int): Number of sections in the z direction.
    - averaged_data_path (str): Path to the file with averaged data.
    - output_file (str): Path to the output file for rewards.
    """
    data = load_data_and_convert_to_numpy(directory)
    data_normalized = [
        (timestep, normalize_data_array(data_array, Lx, Ly, Lz))
        for timestep, data_array in data
    ]
    averaged_data = load_averaged_data_csv(averaged_data_path)

    processed_data = process_velocity_data(data_normalized, averaged_data)

    Q_event_arrays = detect_Q_events(processed_data, averaged_data, H)

    all_results = []

    for timestep, data_array in Q_event_arrays:
        result_array = calculate_local_Q_ratios(data_array, n, m, Lx, Lz)
        result_array = np.insert(result_array, 0, timestep, axis=1)
        all_results.append(result_array)

    final_result_array = np.vstack(all_results)

    rewards = []
    for i in range(n):
        for j in range(m):
            reward = calculate_reward(final_result_array, i, j)
            rewards.append([i, j, reward])

    reward_array = np.array(rewards)
    np.savetxt(
        output_file,
        reward_array,
        delimiter=",",
        header="x_index,z_index,reward",
        comments="",
    )
    print(f"Rewards saved to {output_file}")

    # Clean up
    del (
        data,
        data_normalized,
        averaged_data,
        processed_data,
        Q_event_arrays,
        final_result_array,
        rewards,
    )
    gc.collect()
    print("Memory cleaned up.")


if __name__ == "__main__":
    import argparse

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
        help="Path to the file with averaged data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file for rewards.",
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
