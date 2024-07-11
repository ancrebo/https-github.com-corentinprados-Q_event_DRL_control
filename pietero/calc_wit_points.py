from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

Lx: float = 1.0  # Length in the x direction
Ly: float = 1.0  # Length in the y direction
Lz: float = 1.0  # Length in the z direction

n: int = 2  # Number of sections in the x direction
m: int = 2  # Number of sections in the z direction

pattern: str = 'X'  # Pattern type ('X' or '+')
y_value_density: int = 8  # Number of y values total
y_skipping: bool = True  # Skip y values in the list
y_skip_values: int = 3  # Number of y values to skip between full pattern layers



def calculate_y_values(Ly: float, y_value_density: int) -> List[float]:
    """
    Calculate the y-values for placing the pattern, excluding 0.

    Parameters:
    Ly (float): Length in the y direction.
    y_value_density (int): Number of y values total.

    Returns:
    List[float]: List of y-values for placing the pattern, excluding 0.
    """
    y_values: List[float] = np.linspace(0, Ly, y_value_density + 1).tolist()[
                            1:
                            ]  # Exclude the first term (0)
    return y_values


# TODO: might need to change order of coordinates in list
# TODO: add skipping of y values in list as argument
# TODO: change y_value list to density instead of specific values


def calculate_witness_coordinates(
        n: int,
        m: int,
        Lx: float,
        Ly: float,
        Lz: float,
        y_values: List[float],
        pattern: str = "X",
        y_skipping: bool = False,
        y_skipping_value: int = 3,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]:
    """
    Calculate witness point coordinates in a 3D grid with specified pattern and their local volume indices.

    Parameters:
    n (int): Number of sections in the x direction.
    m (int): Number of sections in the z direction.
    Lx (float): Length in the x direction.
    Ly (float): Length in the y direction.
    Lz (float): Length in the z direction.
    y_values (List[float]): List of y-values for placing the pattern.
    pattern (str): Pattern type ('X' or '+').
    y_skipping (bool): Whether to skip full pattern placement on certain layers.
    y_skipping_value (int): The interval for skipping layers if y_skipping is True.

    Returns:
    Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]: List of normalized witness point coordinates and their local volume indices.
    """
    coordinates: List[Tuple[float, float, float]] = []
    indices: List[Tuple[int, int]] = []

    # Step sizes for local volumes
    step_x: float = Lx / n
    step_z: float = Lz / m

    # Calculate the coordinates for each local volume
    for i in range(n):
        for j in range(m):
            center_x: float = (i + 0.5) * step_x
            center_z: float = (j + 0.5) * step_z

            if pattern == "X":
                # End points of the X in the local volume, 1/4 length from the sides
                end_points: List[Tuple[float, float]] = [
                    (center_x - 0.25 * step_x, center_z - 0.25 * step_z),
                    (center_x + 0.25 * step_x, center_z - 0.25 * step_z),
                    (center_x - 0.25 * step_x, center_z + 0.25 * step_z),
                    (center_x + 0.25 * step_x, center_z + 0.25 * step_z),
                ]
            elif pattern == "+":
                # End points of the + in the local volume, 1/4 length from the sides
                end_points: List[Tuple[float, float]] = [
                    (center_x - 0.25 * step_x, center_z),
                    (center_x + 0.25 * step_x, center_z),
                    (center_x, center_z - 0.25 * step_z),
                    (center_x, center_z + 0.25 * step_z),
                ]

            # Center point of the local volume
            center_point: Tuple[float, float] = (center_x, center_z)

            # Add points for each specified y-value
            for index, y in enumerate(y_values):
                if 0 <= y <= Ly:  # Ensure y-values are within the global y limit
                    if y_skipping and (index % y_skipping_value != 0):
                        # Add only the center point for skipped layers
                        coordinates.append(
                            (center_point[0] / Lx, y / Ly, center_point[1] / Lz)
                        )
                        indices.append((i, j))
                    else:
                        # Add full pattern for non-skipped layers
                        for x, z in end_points:
                            coordinates.append((x / Lx, y / Ly, z / Lz))
                            indices.append((i, j))
                        coordinates.append(
                            (center_point[0] / Lx, y / Ly, center_point[1] / Lz)
                        )
                        indices.append((i, j))

    return coordinates, indices


def format_witness_points(coordinates: List[Tuple[float, float, float]]) -> str:
    """
    Format witness point coordinates into the required string format.

    Parameters:
    coordinates (List[Tuple[float, float, float]]): List of normalized witness point coordinates.

    Returns:
    str: Formatted string of witness points.
    """
    header: str = f"WITNESS_POINTS, NUMBER={len(coordinates)}\n"
    points_str: str = "\n".join(f"{x:.6f},{y:.6f},{z:.6f}" for x, y, z in coordinates)
    footer: str = "\nEND_WITNESS_POINTS"
    return header + points_str + footer


def save_witness_points_to_file(formatted_points: str, filename: str) -> None:
    """
    Save the formatted witness points to a file.

    Parameters:
    formatted_points (str): Formatted witness points as a string.
    filename (str): Name of the file to save the witness points.
    """
    with open(filename, "w") as file:
        file.write(formatted_points)


def format_witness_index(indices: List[Tuple[int, int]]) -> str:
    """
    Format witness point indices into the required string format.

    Parameters:
    indices (List[Tuple[int, int]]): List of witness point local volume indices.

    Returns:
    str: Formatted string of witness point indices.
    """
    header: str = f"WITNESS_INDEX, NUMBER={len(indices)}\n"
    indices_str: str = "\n".join(f"{i},{j}" for i, j in indices)
    footer: str = "\nEND_WITNESS_INDEX"
    return header + indices_str + footer


def save_witness_index_to_file(formatted_index: str, filename: str) -> None:
    """
    Save the formatted witness index to a file.

    Parameters:
    formatted_index (str): Formatted witness index as a string.
    filename (str): Name of the file to save the witness index.
    """
    with open(filename, "w") as file:
        file.write(formatted_index)


# TODO: based on ALYA witness output, concatenate U, V, W with x, y, z coordinates and split into local volumes based on index for Agents


# Example Usage
y_values = calculate_y_values(Ly, y_value_density)

coordinates, indices = calculate_witness_coordinates(
    n,
    m,
    Lx,
    Ly,
    Lz,
    y_values,
    pattern=pattern,
    y_skipping=y_skipping,
    y_skipping_value=y_skip_values,
)

formatted_points: str = format_witness_points(coordinates)
formatted_index: str = format_witness_index(indices)

# Save to files
save_witness_points_to_file(formatted_points, "witness.dat")
save_witness_index_to_file(formatted_index, "witness_index.dat")