import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s\n", __name__, logger.level)


def plot_witness_points(
    coordinates: List[Tuple[float, float, float]], filename: str
) -> None:
    """
    Plot the witness points in 3D and save the plot as an image file.

    Parameters:
        coordinates (List[Tuple[float, float, float]]): List of witness point coordinates.
        filename (str): Filename to save the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x_vals = [coord[0] for coord in coordinates]
    y_vals = [coord[1] for coord in coordinates]
    z_vals = [coord[2] for coord in coordinates]

    ax.scatter(x_vals, y_vals, z_vals, c="r", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Plot of Witness Points")

    plt.savefig(filename)
    logger.info("plot_witness_points: Witness points plotted and saved as %s", filename)
