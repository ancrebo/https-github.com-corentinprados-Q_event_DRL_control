import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET  # For XML parsing
import pyvista as pv
from scipy.interpolate import griddata
from typing import Tuple, List, Dict
import dask

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
    data_normalized = normalize_all_single(data, u_tau, delta_tau)
    logger.info("Finished `normalize_all_single` function.\n")

    logger.info("Starting `process_velocity_data_single` function...")
    processed_data = process_velocity_data_single(data_normalized, averaged_data)
    logger.info("Finished `process_velocity_data_single` function.\n")

    logger.info("Starting `detect_Q_events_single` function...")
    Q_event_frames = detect_Q_events_single(processed_data, averaged_data, H)
    logger.info("Finished `detect_Q_events_single` function.\n")

    ####################################################################################################
    df_last_timestep = Q_event_frames[1]

    # Extract points and scalar values
    points = df_last_timestep[["x", "y", "z"]].values
    scalars = df_last_timestep["Q"].values

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

    logger.info(f"Loading mesh from {last_file} at timestep {last_timestep}")

    # Read the mesh data from the PVTU file
    mesh = pv.read(path)
    # Define the cell connectivity
    # Assuming the cells array is already in the format required by PyVista
    # The cell array starts with the number of points per cell followed by the point indices
    # For a VTK_HEXAHEDRON, the first number will always be 8, followed by the 8 point indices

    cells = mesh.cells
    celltypes = mesh.celltypes

    # Create the unstructured grid
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    # Add scalar data to the points
    grid.point_data["Q"] = scalars

    # Apply the marching cubes algorithm
    contour = grid.contour(isosurfaces=[0.5])  # Adjust the isovalue as needed

    # Plot the result
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(contour, color="red")
    plotter.add_mesh(grid, style="wireframe", color="black")
    plotter.show()

    # Save the result
    save_dir = "/path/to/save/directory"  # Update this to your desired directory
    png_path = os.path.join(save_dir, "q_events_surface_plot.png")
    svg_path = os.path.join(save_dir, "q_events_surface_plot.svg")

    plotter.screenshot(png_path)
    plotter.save_graphic(svg_path)
    plotter.close()  # Close the plotter to free resources

    # def interpolate_with_logging(points, values, X, Y, Z, chunk_size=10):
    #     """
    #     Interpolate values onto a grid in chunks and log progress.
    #
    #     Parameters:
    #     - points (np.ndarray): Array of input points (N, 3).
    #     - values (np.ndarray): Array of values to interpolate (N,).
    #     - X, Y, Z (np.ndarray): Meshgrid arrays.
    #     - chunk_size (int): Number of slices to process at a time.
    #
    #     Returns:
    #     - Q_grid (np.ndarray): Interpolated grid values.
    #     """
    #     nx, ny, nz = X.shape
    #     Q_grid = np.zeros_like(X, dtype=np.float64)
    #
    #     total_chunks = (
    #         (nx // chunk_size + 1) * (ny // chunk_size + 1) * (nz // chunk_size + 1)
    #     )
    #     logger.info("Total number of chunks: %d", total_chunks)
    #     chunk_count = 0
    #
    #     for i in range(0, nx, chunk_size):
    #         for j in range(0, ny, chunk_size):
    #             for k in range(0, nz, chunk_size):
    #                 # Start timer
    #
    #                 chunk_count += 1
    #
    #                 x_slice = slice(i, min(i + chunk_size, nx))
    #                 y_slice = slice(j, min(j + chunk_size, ny))
    #                 z_slice = slice(k, min(k + chunk_size, nz))
    #
    #                 grid_points = (
    #                     X[x_slice, y_slice, z_slice],
    #                     Y[x_slice, y_slice, z_slice],
    #                     Z[x_slice, y_slice, z_slice],
    #                 )
    #
    #                 grid_points_flat = np.array(
    #                     [
    #                         grid_points[0].flatten(),
    #                         grid_points[1].flatten(),
    #                         grid_points[2].flatten(),
    #                     ]
    #                 ).T
    #
    #                 Q_slice = griddata(
    #                     points, values, grid_points_flat, method="linear", fill_value=0
    #                 )
    #                 Q_grid[x_slice, y_slice, z_slice] = Q_slice.reshape(
    #                     grid_points[0].shape
    #                 )
    #
    #                 logger.info(
    #                     f"Interpolated chunk {chunk_count} of {total_chunks}: X[{x_slice}], Y[{y_slice}], Z[{z_slice}]"
    #                 )
    #
    #     return Q_grid
    #
    # import dask.array as da
    # from dask import delayed
    #
    # def interpolate_chunk(points, values, grid_points_flat):
    #     """
    #     Interpolate a chunk using scipy's griddata function.
    #     This function is delayed to be used with Dask for parallel processing.
    #     """
    #     return griddata(points, values, grid_points_flat, method="linear", fill_value=0)
    #
    # def interpolate_with_dask(points, values, X, Y, Z, chunk_size=10):
    #     """
    #     Interpolate values onto a grid in parallel using Dask.
    #
    #     Parameters:
    #     - points (np.ndarray): Array of input points (N, 3).
    #     - values (np.ndarray): Array of values to interpolate (N,).
    #     - X, Y, Z (np.ndarray): Meshgrid arrays.
    #     - chunk_size (int): Number of slices to process at a time.
    #
    #     Returns:
    #     - Q_grid (np.ndarray): Interpolated grid values.
    #     """
    #     nx, ny, nz = X.shape
    #     Q_grid = da.zeros_like(X, dtype=np.float64)
    #
    #     tasks = []
    #     # Calculate the total number of chunks using numpy's ceil
    #     total_chunks = int(
    #         np.ceil(nx / chunk_size)
    #         * np.ceil(ny / chunk_size)
    #         * np.ceil(nz / chunk_size)
    #     )
    #     chunk_count = 0
    #
    #     for i in range(0, nx, chunk_size):
    #         for j in range(0, ny, chunk_size):
    #             for k in range(0, nz, chunk_size):
    #                 chunk_count += 1
    #
    #                 x_slice = slice(i, min(i + chunk_size, nx))
    #                 y_slice = slice(j, min(j + chunk_size, ny))
    #                 z_slice = slice(k, min(k + chunk_size, nz))
    #
    #                 grid_points = (
    #                     X[x_slice, y_slice, z_slice],
    #                     Y[x_slice, y_slice, z_slice],
    #                     Z[x_slice, y_slice, z_slice],
    #                 )
    #
    #                 grid_points_flat = np.array(
    #                     [
    #                         grid_points[0].flatten(),
    #                         grid_points[1].flatten(),
    #                         grid_points[2].flatten(),
    #                     ]
    #                 ).T
    #
    #                 # Use delayed to create a task for this chunk
    #                 task = delayed(interpolate_chunk)(points, values, grid_points_flat)
    #                 tasks.append(task)
    #
    #                 logger.info(
    #                     f"Queued chunk {chunk_count} out of {total_chunks}: X[{x_slice}], Y[{y_slice}], Z[{z_slice}]"
    #                 )
    #
    #     # Compute all tasks in parallel
    #     logger.info("Computing interpolated chunks in parallel...")
    #     Q_chunks = dask.compute(*tasks)
    #     logger.info("Finished computing interpolated chunks!!!\n")
    #
    #     # Combine results into Q_grid
    #     chunk_count = 0
    #     for i in range(0, nx, chunk_size):
    #         for j in range(0, ny, chunk_size):
    #             for k in range(0, nz, chunk_size):
    #                 x_slice = slice(i, min(i + chunk_size, nx))
    #                 y_slice = slice(j, min(j + chunk_size, ny))
    #                 z_slice = slice(k, min(k + chunk_size, nz))
    #
    #                 Q_grid[x_slice, y_slice, z_slice] = Q_chunks[chunk_count].reshape(
    #                     grid_points[0].shape
    #                 )
    #                 chunk_count += 1
    #
    #                 # Log progress
    #                 progress_percentage = (chunk_count / total_chunks) * 100
    #                 logger.info(
    #                     f"Processed chunk {chunk_count} out of {total_chunks} ({progress_percentage:.2f}% complete)"
    #                 )
    #
    #     return Q_grid.compute()
    #
    # from dask.distributed import Client
    #
    # client = Client()  # This will use all available cores by default
    # logger.info(client)
    #
    # ## Create Structured 3D Grid
    # # Extract relevant data
    # df_last_timestep = Q_event_frames[1]
    # points = df_last_timestep[["x", "y", "z"]].values
    # Q_values = df_last_timestep["Q"].values
    #
    # # Define grid resolution
    # grid_resolution = 30  # Higher resolution for better quality, can be adjusted
    # chunk_size = 5  # Number of slices to process at a time
    #
    # logger.info("Creating structured 3D grid...")
    # # Create a grid of points
    # x = np.linspace(
    #     df_last_timestep["x"].min(), df_last_timestep["x"].max(), grid_resolution
    # )
    # y = np.linspace(
    #     df_last_timestep["y"].min(), df_last_timestep["y"].max(), grid_resolution
    # )
    # z = np.linspace(
    #     df_last_timestep["z"].min(), df_last_timestep["z"].max(), grid_resolution
    # )
    # X, Y, Z = np.meshgrid(x, y, z)
    # logger.info("Finished creating structured 3D grid.\n")
    #
    # # Interpolate the Q values onto the grid
    # logger.info("Interpolating Q values onto the grid GLOBALLY...")
    # Q_grid = interpolate_with_logging(points, Q_values, X, Y, Z, chunk_size=chunk_size)
    # logger.info("Finished interpolating Q values onto the grid.\n")
    #
    # ## Apply Marching Cubes Algorithm
    # # Convert the grid to a PyVista structured grid
    # logger.info("Creating PyVista structured grid...")
    # structured_grid = pv.StructuredGrid(X, Y, Z)
    # structured_grid["Q"] = Q_grid.ravel(order="F")
    # logger.info("Finished creating PyVista structured grid.\n")
    #
    # # Apply the marching cubes algorithm to extract isosurfaces
    # logger.info("Applying marching cubes algorithm...")
    # contour = structured_grid.contour(isosurfaces=[0.5], scalars="Q")
    # logger.info("Finished applying marching cubes algorithm.\n")
    #
    # ## PLOTTING
    # # Initialize a PyVista plotter
    # logger.info("Initializing PyVista plotter...")
    # plotter = pv.Plotter(off_screen=True)
    #
    # # Add the global surface
    # logger.info("Adding global surface to plot...")
    # plotter.add_mesh(contour, color="red", opacity=0.5)
    #
    # # Highlight the local volume
    # # Define local volume boundaries based on your explanation
    # n, m = 2, 2  # Example values
    # Lx, Ly, Lz = 2.67, 1.0, 0.8  # Example global dimensions
    # local_x_index, local_z_index = 1, 1  # Example local volume indices
    #
    # step_x = Lx / n
    # step_z = Lz / m
    # local_x_min = local_x_index * step_x
    # local_x_max = (local_x_index + 1) * step_x
    # local_z_min = local_z_index * step_z
    # local_z_max = (local_z_index + 1) * step_z
    #
    # # Filter points within this local volume
    # logger.info("Filtering points within the local volume...")
    # local_points = df_last_timestep[
    #     (df_last_timestep["x"] >= local_x_min)
    #     & (df_last_timestep["x"] <= local_x_max)
    #     & (df_last_timestep["z"] >= local_z_min)
    #     & (df_last_timestep["z"] <= local_z_max)
    # ]
    #
    # # Interpolate Q values for the local volume
    # logger.info("Interpolating Q values for the local volume...")
    # Q_local_grid = interpolate_with_logging(
    #     points=local_points[["x", "y", "z"]].values,
    #     values=local_points["Q"].values,
    #     X=X,
    #     Y=Y,
    #     Z=Z,
    #     chunk_size=chunk_size,
    # )
    #
    # # Convert the local grid to a PyVista grid and apply marching cubes
    # logger.info("Creating PyVista structured grid for the local volume...")
    # local_structured_grid = pv.StructuredGrid(X, Y, Z)
    # local_structured_grid["Q"] = Q_local_grid.ravel(order="F")
    #
    # logger.info("Applying marching cubes algorithm to the local volume...")
    # local_contour = local_structured_grid.contour(isosurfaces=[0.5], scalars="Q")
    #
    # # Add the local surface with a different color
    # logger.info("Adding local surface to plot...")
    # plotter.add_mesh(local_contour, color="yellow", opacity=0.8)
    #
    # # Add a bounding box for the local volume
    # logger.info("Adding bounding box for the local volume...")
    # plotter.add_mesh(
    #     pv.Box(bounds=(local_x_min, local_x_max, 0, Ly, local_z_min, local_z_max)),
    #     color="blue",
    #     style="wireframe",
    # )
    #
    # # Set plotter properties for high-quality output
    # logger.info("Setting plotter properties...")
    # plotter.view_xy()
    # plotter.show_grid()
    #
    # # Save the figure
    # logger.info("Saving the figure...")
    # save_dir = "/path/to/save/directory"  # Update this to your desired directory
    # png_path = os.path.join(save_dir, "q_events_surface_plot.png")
    # svg_path = os.path.join(save_dir, "q_events_surface_plot.svg")
    #
    # plotter.screenshot(png_path)
    # logger.info("PNG file saved successfully at %s", png_path)
    # plotter.save_graphic(svg_path)  # For vector format
    # logger.info("SVG file saved successfully at %s\n", svg_path)
    # logger.info("Visualization complete!!!\n")
