import matplotlib.pyplot as plt
from typing import List, Tuple

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s\n", __name__, logger.level)


def plot_witness_points(
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

    # Adjust orientation so that X-Z plane is horizontal and Y is vertical
    x_vals = [coord[0] * nx_Qs for coord in coordinates]
    y_vals = [coord[1] for coord in coordinates]
    z_vals = [coord[2] * nz_Qs for coord in coordinates]

    # FOR VISUALIZATION ONLY, Y AND Z ARE SWAPPED - Pieter
    ax.scatter(x_vals, z_vals, y_vals, c="r", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title("3D Plot of Witness Points")

    # Setting the grid based on the number of agents
    ax.set_xticks([i for i in range(nx_Qs + 1)])
    ax.set_yticks([i for i in range(nz_Qs + 1)])

    # Show "z" ticks based on y_value_density, but only show every y_skip_values
    z_tick_indices = [i for i in range(1, y_value_density + 1, y_skip_values)]
    ax.set_zticks([i / y_value_density for i in z_tick_indices])
    ax.set_zticklabels([str(i) for i in z_tick_indices])

    # Highlighting the (0, 0) indexed volume
    step_x = 1 / nx_Qs
    step_z = 1 / nz_Qs
    volume_outline = [
        # Bottom face
        [0, step_x, step_x, 0, 0, 0],
        [0, 0, step_z, step_z, 0, 0],
        [0, 0, 0, 0, 0, 0],  # y values for bottom face
        # Top face
        [0, step_x, step_x, 0, 0, 0],
        [0, 0, step_z, step_z, 0, 0],
        [1, 1, 1, 1, 1, 1],  # y values for top face
        # Vertical lines
        [0, 0, step_x, step_x, step_x, step_x, 0, 0],
        [0, 0, 0, 0, step_z, step_z, step_z, step_z],
        [0, 1, 0, 1, 0, 1, 0, 1],  # y values for vertical lines
    ]

    for i in range(0, len(volume_outline[0]), 2):
        ax.plot(
            volume_outline[0][i : i + 2],
            volume_outline[1][i : i + 2],
            volume_outline[2][i : i + 2],
            color="blue",
            linestyle="dotted",
        )

    for i in range(4, len(volume_outline[0]), 2):
        ax.plot(
            volume_outline[0][i : i + 2],
            volume_outline[1][i : i + 2],
            volume_outline[2][i : i + 2],
            color="blue",
            linestyle="dotted",
        )

    ax.grid(True)

    plt.savefig(filename)
    plt.close(fig)
    print(f"3D plot saved as {filename}")
