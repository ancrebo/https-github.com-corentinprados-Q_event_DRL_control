import matplotlib.pyplot as plt
from typing import List, Tuple

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s\n", __name__, logger.level)


def plot_witness_points(
    coordinates: List[Tuple[float, float, float]], filename: str, nx_Qs: int, nz_Qs: int
) -> None:
    """
    Plot the witness points in 3D and save the plot as an image file.

    Parameters:
        coordinates (List[Tuple[float, float, float]]): List of witness point coordinates.
        filename (str): Filename to save the plot.
        nx_Qs (int): Number of agents in the x direction.
        nz_Qs (int): Number of agents in the z direction.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Adjust orientation so that X-Z plane is horizontal and Y is vertical
    x_vals = [coord[0] for coord in coordinates]
    y_vals = [coord[1] for coord in coordinates]
    z_vals = [coord[2] for coord in coordinates]

    ax.scatter(x_vals, z_vals, y_vals, c="r", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title("3D Plot of Witness Points")

    # Setting the grid based on the number of agents
    ax.set_xticks([i / nx_Qs for i in range(nx_Qs + 1)])
    ax.set_yticks([i / nz_Qs for i in range(nz_Qs + 1)])
    ax.set_zticks(
        [i * 0.2 for i in range(6)]
    )  # Assuming y-axis (height) is normalized 0-1

    ax.grid(True)

    plt.savefig(filename)
    plt.close(fig)
    print(f"3D plot saved as {filename}")
