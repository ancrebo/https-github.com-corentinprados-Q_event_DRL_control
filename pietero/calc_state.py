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


def parse_output_file(file_path: str) -> pd.DataFrame:
    """
    Parse the output file from ALYA Witness set results.

    Parameters:
    - file_path (str): Path to the output file.

    Returns:
    - pd.DataFrame: DataFrame containing the parsed data with columns ['U', 'V', 'W'].
    """
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        start = False
        for line in lines:
            if start:
                values = line.split()
                data.append(
                    [
                        float(values[1]),
                        float(values[2]),
                        float(values[3]),
                    ]
                )
            if line.startswith("# Time"):
                start = True

    df = pd.DataFrame(data, columns=["U", "V", "W"])
    return df


def read_witness_data(witness_file: str) -> pd.DataFrame:
    """
    Read the witness data file.

    Parameters:
    - witness_file (str): Path to the witness file.

    Returns:
    - pd.DataFrame: DataFrame containing the witness data.
    """
    with open(witness_file, "r") as file:
        lines = file.readlines()

    start = False
    data = []
    for line in lines:
        if line.strip() == "END_WITNESS_POINTS":
            break
        if start:
            values = line.split(",")
            data.append([float(v) for v in values])
        if line.startswith("WITNESS_POINTS"):
            start = True

    df = pd.DataFrame(data, columns=["x", "y", "z"])
    return df


def read_witness_index(index_file: str) -> pd.DataFrame:
    """
    Read the witness index file.

    Parameters:
    - index_file (str): Path to the witness index file.

    Returns:
    - pd.DataFrame: DataFrame containing the witness index data.
    """
    with open(index_file, "r") as file:
        lines = file.readlines()

    start = False
    data = []
    for line in lines:
        if line.strip() == "END_WITNESS_INDEX":
            break
        if start:
            values = line.split(",")
            data.append([int(v) for v in values])
        if line.startswith("WITNESS_INDEX"):
            start = True

    df = pd.DataFrame(data, columns=["local_x_index", "local_z_index"])
    return df


def create_combined_dataframe(
        output_df: pd.DataFrame, witness_df: pd.DataFrame, index_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine output data with witness data and indices.

    Parameters:
    - output_df (pd.DataFrame): DataFrame containing the parsed output data.
    - witness_df (pd.DataFrame): DataFrame containing the witness data.
    - index_df (pd.DataFrame): DataFrame containing the local volume indices.

    Returns:
    - pd.DataFrame: Combined DataFrame with coordinates and indices.
    """
    combined_df = pd.concat([output_df, witness_df, index_df], axis=1)
    return combined_df


def split_state_into_local_volumes(combined_df: pd.DataFrame) -> List[Tuple[Tuple[int, int], pd.DataFrame]]:
    """
    Split the combined DataFrame into separate DataFrames for each local volume.

    Parameters:
    - combined_df (pd.DataFrame): The combined DataFrame containing columns 'local_x_index' and 'local_z_index'.

    Returns:
    - List[Tuple[Tuple[int, int], pd.DataFrame]]: A list of tuples, where each tuple contains:
        * A tuple (local_x_index, local_z_index)
        * A DataFrame with the data for that local volume
    """
    local_volume_dataframes = []

    # Group the combined_df by 'local_x_index' and 'local_z_index'
    grouped = combined_df.groupby(['local_x_index', 'local_z_index'])

    # Iterate over each group and create a DataFrame for each local volume
    for (local_x, local_z), group in grouped:
        # Append a tuple of ((local_x, local_z), group) to the list
        local_volume_dataframes.append(((local_x, local_z), group.reset_index(drop=True)))

    return local_volume_dataframes








# Example usage
witness_file_path = "witness.dat"
index_file_path = "witness_index.dat"
ALYA_witness_file_path = "channel.nsi.wit"

output_df = parse_output_file(ALYA_witness_file_path)
# print(output_df.head())

witness_df = read_witness_data(witness_file_path)
# print(f"first 5 rows of witness_df:\n{witness_df.head()}")
index_df = read_witness_index(index_file_path)
# print(f"first 5 rows of index_df:\n{index_df.head()}")
combined_df = create_combined_dataframe(index_df, witness_df, output_df)

# remove rows after the last output line
combined_df = combined_df.iloc[: len(output_df)]

# Example usage
local_volume_dataframes = split_state_into_local_volumes(combined_df)

# TODO: code to send state data to each agent
