import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List, Tuple
from matplotlib.ticker import MultipleLocator

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s\n", __name__, logger.level)


def plot_witness_points_broken(
    coordinates: List[Tuple[float, float, float]],
    filename: str,
    nx_Qs: int,
    nz_Qs: int,
    y_value_density: int,
    y_skip_values: int,
) -> None:
    """
    Plot the witness points in 3D and save the plot as an image file.

    Parameters:
        coordinates (List[Tuple[float, float, float]]): List of witness point coordinates.
        filename (str): Filename to save the plot.
        nx_Qs (int): Number of agents in the x direction.
        nz_Qs (int): Number of agents in the z direction.
        y_value_density (int): Number of y layers in witness points.
        y_skip_values (int): Number of layers to skip for displaying ticks on the y-axis.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Create a color map for different local volumes
    colors = plt.cm.get_cmap("tab20", nx_Qs * nz_Qs)

    # Adjust orientation so that X-Z plane is horizontal and Y is vertical
    x_vals = np.array([coord[0] * nx_Qs for coord in coordinates])
    y_vals = np.array([coord[1] * y_value_density for coord in coordinates])
    z_vals = np.array([coord[2] * nz_Qs for coord in coordinates])

    # Plot points for each local volume separately
    for i in range(nx_Qs):
        for j in range(nz_Qs):
            # Define the local volume boundaries
            x_min = i
            x_max = i + 1
            z_min = j
            z_max = j + 1

            # Filter points within this local volume
            volume_filter = (
                (x_vals >= x_min)
                & (x_vals < x_max)
                & (z_vals >= z_min)
                & (z_vals < z_max)
            )
            ax.scatter(
                x_vals[volume_filter],
                z_vals[volume_filter],
                y_vals[volume_filter],
                color=colors(i * nz_Qs + j),
                marker="o",
                label=f"Volume ({i}, {j})",
            )

    ax.set_xlabel("X Agent Index")
    ax.set_ylabel("Z Agent Index")
    ax.set_zlabel("Y Layer Index")
    ax.set_title("3D Plot of Witness Points")

    # Set limits to ensure the ticks are correctly displayed
    ax.set_xlim(0, nx_Qs)
    ax.set_ylim(0, nz_Qs)
    ax.set_zlim(0, y_value_density)

    # Setting the tick values based on the number of agents but labeled in between grid lines
    ax.set_xticks([i + 0.5 for i in range(nx_Qs)])
    ax.set_xticklabels([str(i) for i in range(nx_Qs)])
    ax.set_yticks([i + 0.5 for i in range(nz_Qs)])
    ax.set_yticklabels([str(i) for i in range(nz_Qs)])

    # Show "z" ticks based on y_value_density, but only show every y_skip_values
    z_tick_indices = [i for i in range(1, y_value_density + 1, y_skip_values)]
    ax.set_zticks([i for i in z_tick_indices])
    ax.set_zticklabels([str(i) for i in z_tick_indices])

    # Draw custom grid lines for X and Z axes
    for i in range(1, nx_Qs):
        ax.plot([i, i], [0, nz_Qs], [0, 0], color="grey", linestyle="--", linewidth=0.5)
        ax.plot(
            [i, i],
            [0, 0],
            [0, y_value_density],
            color="grey",
            linestyle="--",
            linewidth=0.5,
        )
    for j in range(1, nz_Qs):
        ax.plot([0, nx_Qs], [j, j], [0, 0], color="grey", linestyle="--", linewidth=0.5)
        ax.plot(
            [0, 0],
            [j, j],
            [0, y_value_density],
            color="grey",
            linestyle="--",
            linewidth=0.5,
        )

    ax.grid(False)  # Disable default grid to avoid overlap

    ax.legend(loc="upper right")

    plt.savefig(filename)
    plt.close(fig)
    print(f"3D plot saved as {filename}")


def plot_witness_points(
    coordinates: List[Tuple[float, float, float]],
    filename: str,
    nx_Qs: int,
    nz_Qs: int,
    y_value_density: int,
    y_skip_values: int,
) -> None:
    """
    Plot the witness points in the first local volume (0, 0) in 3D and save the plot as an image file.

    Parameters:
        coordinates (List[Tuple[float, float, float]]): List of witness point coordinates.
        filename (str): Filename to save the plot.
        nx_Qs (int): Number of agents in the x direction.
        nz_Qs (int): Number of agents in the z direction.
        y_value_density (int): Number of y layers in witness points.
        y_skip_values (int): Number of layers to skip for displaying ticks on the y-axis.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x_vals = np.array([coord[0] for coord in coordinates])
    y_vals = np.array([coord[1] for coord in coordinates])
    z_vals = np.array([coord[2] for coord in coordinates])

    # Filter points within the first local volume (0, 0)
    volume_filter = (x_vals < 1 / nx_Qs) & (z_vals < 1 / nz_Qs)
    x_vals = x_vals[volume_filter] * nx_Qs
    y_vals = y_vals[volume_filter] * y_value_density
    z_vals = z_vals[volume_filter] * nz_Qs

    # Plot filtered points
    ax.scatter(x_vals, z_vals, y_vals, c="r", marker="o")

    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Z")
    ax.set_zlabel("Y Layer Index")
    ax.set_title("3D Plot of Witness Points in Local Volume (0, 0)")

    # Set tick values
    ax.set_xticks(np.linspace(0, 1, nx_Qs + 1))
    ax.set_xticklabels([f"{i / nx_Qs:.2f}" for i in range(nx_Qs + 1)])
    ax.set_yticks(np.linspace(0, 1, nz_Qs + 1))
    ax.set_yticklabels([f"{i / nz_Qs:.2f}" for i in range(nz_Qs + 1)])

    # Show "z" ticks based on y_value_density, but only show every y_skip_values
    z_tick_indices = [i for i in range(1, y_value_density + 1, y_skip_values)]
    ax.set_zticks(z_tick_indices)
    ax.set_zticklabels([str(i) for i in z_tick_indices])

    # Set limits to ensure the ticks are correctly displayed
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, y_value_density)

    ax.grid(True)
    ax.legend(loc="upper right")

    plt.savefig(filename)
    plt.close(fig)
    print(f"3D plot saved as {filename}")
