import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET  # For XML parsing
import pyvista as pv  # For reading and processing mesh data
import pandas as pd  # For data manipulation and DataFrame creation
import plotly.graph_objs as go  # For 3D interactive plot
from plotly.subplots import make_subplots  # For 3D interactive plot
from scipy.ndimage import label  # Function to count the number of clusters
import plotly.express as px  # Package to color each clusters different Path # To have an absolute path
from pathlib import Path
from tqdm import tqdm  # For progress bar

# Argument parsing
parser = argparse.ArgumentParser(
    description="Calculate average and RMS velocity profiles of given VTK files."
)
parser.add_argument(
    "--loglvl",
    type=str,
    default="INFO",
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
parser.add_argument(
    "--logfilelvl",
    type=str,
    default="INFO",
    help="Logging level for the file handler (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)

args = parser.parse_args()
logging_level = getattr(logging, args.loglvl.upper(), None)
if not isinstance(logging_level, int):
    raise ValueError("Invalid log level: %s" % args.loglvl)
logging_level_file = getattr(logging, args.logfilelvl.upper(), None)
if not isinstance(logging_level_file, int):
    raise ValueError("Invalid log level: %s" % args.logfilelvl)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Create console handler and set level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging_level)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

# Create file handler and set level
file_handler = logging.FileHandler(logs_dir / "logfile.log")
file_handler.setLevel(logging_level_file)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Logger started. Logging level: %s", args.loglvl)
logger.info("File handler logging level: %s", args.logfilelvl)

logger.info("Libraries imported successfully.")


def load_data_and_give_average_velocity(directory, file_name):
    logger.info("Loading data and calculating average velocity...")
    # Parse the PVD file to extract mappings of timesteps to their corresponding PVTU files
    pvd_path = os.path.join(
        directory, file_name
    )  # Not the full simulation, only the points after the transition state
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    file_list = [
        os.path.join(directory, dataset.attrib["file"])
        for dataset in root.find("Collection")
    ]

    # Initialize arrays for accumulating data
    sum_u, sum_v, sum_w = 0, 0, 0
    count = 0

    # Wrap in tqdm for progress bar
    # Process each PVTU file
    for path in tqdm(file_list, desc="Loading data", unit="file"):
        mesh = pv.read(path)  # Read the mesh data from the PVTU file
        # Accumulate the velocity components
        u, v, w = mesh["VELOC"].T
        sum_u += u
        sum_v += v
        sum_w += w

        count += 1
        filename = os.path.basename(path)
        tqdm.write("Data from %s loaded and velocity components added." % filename)
        logger.debug("Data from %s loaded and velocity components added.", filename)

    # Calculate the average of the velocity components
    avg_u = sum_u / count
    avg_v = sum_v / count
    avg_w = sum_w / count

    # Get the spatial coordinates (assuming they are the same for all files)
    points = mesh.points  # x, y, z coordinates

    # Create a DataFrame with the averaged data
    data_t_avg = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "U_t_avg": avg_u,
            "V_t_avg": avg_v,
            "W_t_avg": avg_w,
        }
    )

    logger.info("Total data sets processed for average velocity calculation: %d", count)

    # Calculate mean over x and z for u, v, w
    data_avg = (
        data_t_avg.groupby("y")
        .agg({"U_t_avg": ["mean"], "V_t_avg": ["mean"], "W_t_avg": ["mean"]})
        .rename(columns={"mean": "bar"}, level=1)
    )
    data_avg.columns = ["U_bar", "V_bar", "W_bar"]  # Clear column names
    logger.info("Average velocity calculated successfully.")
    return data_avg


def load_data_and_give_RMS_velocity(directory, file_name, mean_velocities):
    """
    This function loads PVTU files listed in a PVD file located in the given directory,
    calculates the RMS velocity components (u', v', w') across all timesteps,
    and returns a DataFrame with the RMS velocity components averaged over x and z coordinates.

    Parameters:
    directory (str): Path to the directory containing the PVD and PVTU files.
    avg_velocities : Averaged velocity from load_data_and_give_average_velocity(directory)

    Returns:
    pd.DataFrame: DataFrame with RMS velocity components for each y-coordinate.
    """
    logger.info("Loading data and calculating RMS velocity fluctuations...")
    # Parse the PVD file to extract mappings of timesteps to their corresponding PVTU files
    pvd_path = os.path.join(
        directory, file_name
    )  # Not the full simulation, only the points after the transition state
    tree = ET.parse(pvd_path)
    root = tree.getroot()

    # Extract PVTU file paths from the PVD file
    file_list = [
        os.path.join(directory, dataset.attrib["file"])
        for dataset in root.find("Collection")
    ]

    # Initialize arrays for accumulating squared velocity fluctuations
    sum_u2, sum_v2, sum_w2 = 0, 0, 0
    count = 0

    # Process each PVTU file
    for path in tqdm(file_list, desc="Loading data", unit="file"):
        mesh = pv.read(path)  # Read the mesh data from the PVTU file

        # Extract velocity components
        u, v, w = mesh["VELOC"].T

        # Create a DataFrame with the instantaneous velocities and coordinates
        df = pd.DataFrame(
            {
                "x": mesh.points[:, 0],
                "y": mesh.points[:, 1],
                "z": mesh.points[:, 2],
                "u": u,
                "v": v,
                "w": w,
            }
        )

        # Merge with data_avg to get the mean velocity for each y
        df = df.merge(mean_velocities, on="y")

        # Calculate the velocity fluctuations
        df["u_fluc"] = df["u"] - df["U_bar"]
        df["v_fluc"] = df["v"] - df["V_bar"]
        df["w_fluc"] = df["w"] - df["W_bar"]

        # Accumulate the squared velocity fluctuations
        sum_u2 += df["u_fluc"] ** 2
        sum_v2 += df["v_fluc"] ** 2
        sum_w2 += df["w_fluc"] ** 2

        count += 1
        filename = os.path.basename(path)
        logger.debug("Data from %s loaded and velocity components added.", filename)

    # Calculate the RMS of the velocity fluctuations
    rms_u = (sum_u2 / count) ** 0.5
    rms_v = (sum_v2 / count) ** 0.5
    rms_w = (sum_w2 / count) ** 0.5

    # Create a DataFrame with the RMS data
    data_t_rms = pd.DataFrame(
        {
            "x": mesh.points[:, 0],
            "y": mesh.points[:, 1],
            "z": mesh.points[:, 2],
            "u_prime_t_rms": rms_u,
            "v_prime_t_rms": rms_v,
            "w_prime_t_rms": rms_w,
        }
    )

    logger.info("Total data sets processed for RMS velocity calculation: %d", count)

    # Calculate mean over x and z for u', v', w'
    rms_velocities = data_t_rms.groupby("y").agg(
        {"u_prime_t_rms": "mean", "v_prime_t_rms": "mean", "w_prime_t_rms": "mean"}
    )

    # Rename columns for clarity
    rms_velocities.columns = ["u_prime", "v_prime", "w_prime"]
    logger.info("RMS velocity fluctuations calculated successfully.")
    return rms_velocities


def plot_velocity_profiles(directory, mean_velocities, rms_velocities):

    # Plot U_bar
    plt.figure(figsize=(10, 6))
    plt.plot(mean_velocities.index, mean_velocities["U_bar"], label="U_bar", marker="o")
    plt.xlabel("y-coordinate")
    plt.ylabel("Mean Velocity (U_bar)")
    plt.title("Mean Velocity Profile (U_bar)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot u_prime
    plt.figure(figsize=(10, 6))
    plt.plot(
        rms_velocities.index, rms_velocities["u_prime"], label="u_prime", marker="o"
    )
    plt.xlabel("y-coordinate")
    plt.ylabel("RMS Velocity Fluctuation (u_prime)")
    plt.title("RMS Velocity Fluctuation Profile (u_prime)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot v_prime
    plt.figure(figsize=(10, 6))
    plt.plot(
        rms_velocities.index, rms_velocities["v_prime"], label="v_prime"
    )  # , marker='o')
    plt.xlabel("y-coordinate")
    plt.ylabel("RMS Velocity Fluctuation (v_prime)")
    plt.title("RMS Velocity Fluctuation Profile (v_prime)")
    plt.legend()
    plt.grid(True)
    plt.show()


def normalize_data(mean_velocities, rms_velocities, nu):
    """
    Normalizes the velocity components and the y-coordinate in the mean_velocities and rms_velocities DataFrames using:
    u_tau = (Omega_bar * nu) ** 0.5 and y^+ = y * u_tau / nu

    Parameters:
    - mean_velocities (DataFrame): DataFrame with columns for y and
      time-averaged velocity components (U_bar, V_bar, W_bar).
    - rms_velocities (DataFrame): DataFrame with columns for y and
      RMS velocity components (u_prime, v_prime, w_prime).
    - nu (float): Kinematic viscosity of the fluid.

    Returns:
    - mean_velocities_normalized (DataFrame): DataFrame with the same columns as mean_velocities,
      where the velocity components and y-coordinate have been normalized.
    - rms_velocities_normalized (DataFrame): DataFrame with the same columns as rms_velocities,
      where the RMS velocity components and y-coordinate have been normalized.
    """
    logger.info("Normalizing the velocity profiles...")
    # Calculate u_tau based on the mean velocities
    y = mean_velocities.index.values
    u_y = mean_velocities["U_bar"]

    # Compute delta_u and delta_y at the minimum y (wall)
    delta_u_min = u_y.iloc[1] - u_y.iloc[0]
    delta_y_min = y[1] - y[0]
    Omega_bar_min = delta_u_min / delta_y_min

    # Compute delta_u and delta_y at the maximum y (wall)
    delta_u_max = u_y.iloc[-2] - u_y.iloc[-1]
    delta_y_max = y[-1] - y[-2]
    Omega_bar_max = delta_u_max / delta_y_max

    # Calculate u_tau at each wall and average them
    u_tau_min = (Omega_bar_min * nu) ** 0.5
    u_tau_max = (Omega_bar_max * nu) ** 0.5
    u_tau = (u_tau_min + u_tau_max) / 2
    # u_tau = 0.57231059E-01 # Test

    logger.info("Calculated u_tau: %f", u_tau)

    # Normalize the mean velocity components
    mean_velocities_normalized = mean_velocities.copy()
    mean_velocities_normalized["U_bar"] /= u_tau
    mean_velocities_normalized["V_bar"] /= u_tau
    mean_velocities_normalized["W_bar"] /= u_tau

    # Normalize the RMS velocity components
    rms_velocities_normalized = rms_velocities.copy()
    rms_velocities_normalized["u_prime"] /= u_tau
    rms_velocities_normalized["v_prime"] /= u_tau
    rms_velocities_normalized["w_prime"] /= u_tau

    # Normalize the y-coordinate in both DataFrames
    delta_tau = nu / u_tau
    # delta_tau = 0.005376316864804261# Test

    logger.info("Calculated delta_tau: %f", delta_tau)

    mean_velocities_normalized["y"] = mean_velocities_normalized.index / delta_tau
    rms_velocities_normalized["y"] = rms_velocities_normalized.index / delta_tau

    mean_velocities_normalized.set_index("y", inplace=True)
    rms_velocities_normalized.set_index("y", inplace=True)
    logger.info("Velocity profiles normalized successfully.")
    return mean_velocities_normalized, rms_velocities_normalized, u_tau, delta_tau


def plot_velocity_normalise_profiles(directory, mean_velocities, rms_velocities):

    # Plot U_bar
    plt.figure(figsize=(10, 6))
    plt.plot(mean_velocities.index, mean_velocities["U_bar"], label="U_bar", marker="o")
    plt.xlabel("y-coordinate")
    plt.ylabel("Mean Velocity (U_bar)")
    plt.title("Mean Velocity Profile (U_bar)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot u_prime
    plt.figure(figsize=(10, 6))
    plt.plot(
        rms_velocities.index, rms_velocities["u_prime"], label="u_prime", marker="o"
    )
    plt.xlabel("y-coordinate")
    plt.ylabel("RMS Velocity Fluctuation (u_prime)")
    plt.title("RMS Velocity Fluctuation Profile (u_prime)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot v_prime
    plt.figure(figsize=(10, 6))
    plt.plot(
        rms_velocities.index, rms_velocities["v_prime"], label="v_prime"
    )  # , marker='o')
    plt.xlabel("y-coordinate")
    plt.ylabel("RMS Velocity Fluctuation (v_prime)")
    plt.title("RMS Velocity Fluctuation Profile (v_prime)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocity_normalize_profiles(mean_velocities, rms_velocities):
    """
    Plots the normalized mean and RMS velocity profiles and saves the plots as images in the specified folder.

    Parameters:
    - mean_velocities (DataFrame): DataFrame containing the normalized mean velocities (U_bar, V_bar, W_bar) indexed by y-coordinate.
    - rms_velocities (DataFrame): DataFrame containing the normalized RMS velocities (u_prime, v_prime, w_prime) indexed by y-coordinate.
    """
    # Create the directory if it doesn't exist
    output_dir = "Plots_for_the_report"
    os.makedirs(output_dir, exist_ok=True)

    # Common settings for LaTeX-compatible plots
    fig_width = 8  # inches, can be adjusted to fit \textwidth or \columnwidth
    fig_height = 4.5  # inches
    font_size = 20  # points, increased for better readability
    label_size = 20  # points, increased for better readability
    tick_size = 18  # points, for axis tick labels
    line_width = 3  # points, increased line thickness

    # Plot U_bar
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(
        mean_velocities.index,
        mean_velocities["U_bar"],
        label="$\overline{U}^+$",
        linestyle="-",
        linewidth=line_width,
    )
    plt.xlabel("$y^+$", fontsize=label_size)
    plt.ylabel("$\overline{U}^+$", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_velocity_profile.png"))
    plt.show()

    # Plot u_prime and v_prime together
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(
        rms_velocities.index,
        rms_velocities["u_prime"],
        label="$u'^+$",
        linestyle="-",
        linewidth=line_width,
    )
    plt.plot(
        rms_velocities.index,
        rms_velocities["v_prime"],
        label="$v'^+$",
        linestyle="--",
        linewidth=line_width,
    )
    plt.xlabel("$y^+$", fontsize=label_size)
    plt.ylabel("RMS Velocity Fluctuations", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rms_velocity_fluctuation_profiles.png"))
    plt.show()


if __name__ == "__main__":
    # First check if output file already exists:
    output_file_path = logs_dir / "calculated_values.csv"
    if output_file_path.exists():
        logger.info(
            f"Output file {output_file_path} already exists. Please back it up or delete it before running the script."
        )
        exit()

    # Old directory
    # directory_path = Path('/Users/corentinprados/Documents/Stage_M2/testALYA.nosync/Ancien/long_run/vtk_for_average_vtk')

    # Last directory
    directory_path = Path(
        "/scratch/pietero/baseline/prados_recent/re180_min_channel_900_RESTART_WITNESS_18nproc/vtk_for_average"
    )
    file_name = "channel.pvd"

    # Small
    # directory_path = Path('/Users/corentinprados/Documents/Stage_M2/Q_event_DRL_control/ALYA/longer_run_vtk')

    mean_velocities = load_data_and_give_average_velocity(directory_path, file_name)

    rms_velocities = load_data_and_give_RMS_velocity(
        directory_path, file_name, mean_velocities
    )

    # plot_velocity_profiles(directory_path, mean_velocities, rms_velocities)

    # Need to calculate normalized profiles before plotting them!!!
    # Normalisation

    nu = 0.0003546986585888876
    mean_velocities_normalized, rms_velocities_normalized, u_tau, delta_tau = (
        normalize_data(mean_velocities, rms_velocities, nu)
    )

    # Re_tau
    h = 1
    Re_tau = u_tau / nu * h
    logger.info("Calculated Re_tau: %f", Re_tau)

    # t_tau
    t_tau = delta_tau / u_tau
    logger.info("Calculated t_tau: %f", t_tau)

    t_i = 891.744
    t_i_plus = t_i / t_tau
    logger.info("Calculated t_i_plus: %f", t_i_plus)

    t_f = 1376.61
    t_f_plus = t_f / t_tau
    logger.info("Calculated t_f_plus: %f", t_f_plus)

    # U_b
    U_b = mean_velocities["U_bar"].mean()
    logger.info("Calculated U_b: %f", U_b)

    # t_w
    L_x = 2.67
    t_w = L_x / U_b
    t_w_plus = t_w / t_tau
    logger.info("Calculated t_w: %f", t_w)
    logger.info("Calculated t_w_plus: %f", t_w_plus)

    ## Save calculated values to a csv file
    calculated_values = {
        "u_tau": u_tau,
        "delta_tau": delta_tau,
        "Re_tau": Re_tau,
        "t_tau": t_tau,
        "t_i": t_i,
        "t_i_plus": t_i_plus,
        "t_f": t_f,
        "t_f_plus": t_f_plus,
        "U_b": U_b,
        "t_w": t_w,
        "t_w_plus": t_w_plus,
    }

    # Convert dictionary to DataFrame
    df_calculated_values = pd.DataFrame(
        list(calculated_values.items()), columns=["Parameter", "Value"]
    )

    # Save DataFrame to CSV
    output_file_path = logs_dir / "calculated_values.csv"
    df_calculated_values.to_csv(output_file_path, index=False)

    logger.info(f"Final calculated values saved to {output_file_path}")

    # plot_velocity_normalize_profiles(mean_velocities_normalized, rms_velocities_normalized)
