import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET  # For XML parsing
import pyvista as pv
from scipy.interpolate import griddata
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

directory: str = os.path.join(directory_base, episode, "VTK")

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

    # Extract the spatial coordinates and velocity components from the mesh
    points = mesh.points  # x, y, z coordinates
    U, V, W = mesh["VELOC"].T  # Transpose to separate the velocity components (U, V, W)

    # Create a DataFrame with the extracted data
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
data_normalized = normalize_all_single(data, u_tau, delta_tau)
logger.info("Finished `normalize_all_single` function.\n")

logger.info("Starting `process_velocity_data_single` function...")
processed_data = process_velocity_data_single(data_normalized, averaged_data)
logger.info("Finished `process_velocity_data_single` function.\n")

logger.info("Starting `detect_Q_events_single` function...")
Q_event_frames = detect_Q_events_single(processed_data, averaged_data, H)
logger.info("Finished `detect_Q_events_single` function.\n")
####################################################################################################
## Create Structured 3D Grid
# Extract relevant data
df_last_timestep = Q_event_frames[1]
points = df_last_timestep[["x", "y", "z"]].values
Q_values = df_last_timestep["Q"].values

# Define grid resolution
grid_resolution = 100  # Higher resolution for better quality, can be adjusted

logger.info("Creating structured 3D grid...")
# Create a grid of points
x = np.linspace(
    df_last_timestep["x"].min(), df_last_timestep["x"].max(), grid_resolution
)
y = np.linspace(
    df_last_timestep["y"].min(), df_last_timestep["y"].max(), grid_resolution
)
z = np.linspace(
    df_last_timestep["z"].min(), df_last_timestep["z"].max(), grid_resolution
)
X, Y, Z = np.meshgrid(x, y, z)
logger.info("Finished creating structured 3D grid.\n")


# Interpolate the Q values onto the grid
logger.info("Interpolating Q values onto the grid GLOBALLY...")
Q_grid = griddata(points, Q_values, (X, Y, Z), method="linear", fill_value=0)
logger.info("Finished interpolating Q values onto the grid.\n")

## Apply Marching Cubes Algorithm
# Convert the grid to a PyVista structured grid
logger.info("Creating PyVista structured grid...")
structured_grid = pv.StructuredGrid(X, Y, Z)
structured_grid["Q"] = Q_grid.ravel(order="F")
logger.info("Finished creating PyVista structured grid.\n")

# Apply the marching cubes algorithm to extract isosurfaces
logger.info("Applying marching cubes algorithm...")
contour = structured_grid.contour(isosurfaces=[0.5], scalars="Q")
logger.info("Finished applying marching cubes algorithm.\n")

## PLOTTING
# Initialize a PyVista plotter
logger.info("Initializing PyVista plotter...")
plotter = pv.Plotter(off_screen=True)

# Add the global surface
logger.info("Adding global surface to plot...")
plotter.add_mesh(contour, color="red", opacity=0.5)

# Highlight the local volume
# Define local volume boundaries based on your explanation
n, m = 2, 2  # Example values
Lx, Ly, Lz = 2.67, 1.0, 0.8  # Example global dimensions
local_x_index, local_z_index = 1, 1  # Example local volume indices

step_x = Lx / n
step_z = Lz / m
local_x_min = local_x_index * step_x
local_x_max = (local_x_index + 1) * step_x
local_z_min = local_z_index * step_z
local_z_max = (local_z_index + 1) * step_z

# Filter points within this local volume
logger.info("Filtering points within the local volume...")
local_points = df_last_timestep[
    (df_last_timestep["x"] >= local_x_min)
    & (df_last_timestep["x"] <= local_x_max)
    & (df_last_timestep["z"] >= local_z_min)
    & (df_last_timestep["z"] <= local_z_max)
]

# Interpolate Q values for the local volume
logger.info("Interpolating Q values for the local volume...")
Q_local_grid = griddata(
    points=(local_points["x"], local_points["y"], local_points["z"]),
    values=local_points["Q"],
    xi=(X, Y, Z),
    method="linear",
    fill_value=0,
)

# Convert the local grid to a PyVista grid and apply marching cubes
logger.info("Creating PyVista structured grid for the local volume...")
local_structured_grid = pv.StructuredGrid(X, Y, Z)
local_structured_grid["Q"] = Q_local_grid.ravel(order="F")

logger.info("Applying marching cubes algorithm to the local volume...")
local_contour = local_structured_grid.contour(isosurfaces=[0.5], scalars="Q")

# Add the local surface with a different color
logger.info("Adding local surface to plot...")
plotter.add_mesh(local_contour, color="yellow", opacity=0.8)

# Add a bounding box for the local volume
logger.info("Adding bounding box for the local volume...")
plotter.add_mesh(
    pv.Box(bounds=(local_x_min, local_x_max, 0, Ly, local_z_min, local_z_max)),
    color="blue",
    style="wireframe",
)

# Set plotter properties for high-quality output
logger.info("Setting plotter properties...")
plotter.view_xy()
plotter.show_grid()

# Save the figure
logger.info("Saving the figure...")
save_dir = '/path/to/save/directory'  # Update this to your desired directory
png_path = os.path.join(save_dir, 'q_events_surface_plot.png')
svg_path = os.path.join(save_dir, 'q_events_surface_plot.svg')

plotter.screenshot(png_path)
logger.info("PNG file saved successfully at %s", png_path)
plotter.save_graphic(svg_path)  # For vector format
logger.info("SVG file saved successfully at %s\n", svg_path)
logger.info("Visualization complete!!!\n")
