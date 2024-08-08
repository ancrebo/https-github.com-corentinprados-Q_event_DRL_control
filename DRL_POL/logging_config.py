"""
logging_config.py
=================

DEEP REINFORCEMENT LEARNING WITH ALYA
-------------------------------------

This module sets up the default logging configuration for the entire project.
It allows for overriding logging levels for individual modules and provides
functions to configure loggers with specified settings.

The script includes:
1. Default logging configuration.
2. Dictionary to set logging levels for each module if default values are to be overridden.
3. Function to clear old log files in the specified directory.
4. Functions to configure loggers for general use and specific module cases.

Usage
-----
To set up logging in a new module, add the following lines:

```python
from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s", __name__, logger.level)
```

You must add the new module's name to logging_config_dict in logging_config.py to
be able to specify custom logging levels if needed.

Functions
---------
configure_logger(module_name: str, default_level: str = "INFO",
                 log_dir: str = LOG_DIR) -> logging.Logger
    Configure a logger with specified settings or defaults.

configure_env_logger(log_dir: str = LOG_DIR) ->
Tuple[logging.Logger, logging.Logger]
    Configure loggers specifically for the Environment class.

clear_old_logs(log_dir: str) -> None
    Clear old log files in the specified directory.

See Also
--------
PARALLEL_TRAINING_3D_CHANNEL_MARL.py : Sets the custom log directory which can
be used to override the default log directory via an argument.

Authors
-------
- Pieter Orlandini

Version History
---------------
- Initial implementation in August 2024.
"""

import os
from typing import Dict, Any
import logging

## This file sets the logging levels for the project
# IN GENERAL: logging messages should use %s formatting, not f-strings
# IN GENERAL: "DEBUG" and "INFO" are the most commonly used logging levels in the project

# Possible logging levels (for reference).
possible_logging_levels = {
    "DEBUG": 10,  # Detailed information, typically of interest only when diagnosing problems.
    "INFO": 20,  # Confirmation that things are working as expected.
    "WARNING": 30,  # An indication that something unexpected happened, or indicative of some problem in the near future.
    "ERROR": 40,  # Due to a more serious problem, the software has not been able to perform some function.
    "CRITICAL": 50,  # A very serious error, indicating that the program itself may be unable to continue running.
}

# Set the DEFAULT logging level for each module
DEFAULT_LOGGING_LEVEL: str = "INFO"

# Set the DEFAULT log directory for each module
DEFAULT_LOG_DIR: str = "logsPYTHON"

CUSTOM_LOG_DIR: str = None  # Set this to override the default log directory

# Set the CUSTOM log directory if specified
LOG_DIR: str = CUSTOM_LOG_DIR if CUSTOM_LOG_DIR is not None else DEFAULT_LOG_DIR

# Default Config for a Module
DEFAULT_CONFIG: Dict[str, Any] = {
    "console_level": DEFAULT_LOGGING_LEVEL,
    "file_level": DEFAULT_LOGGING_LEVEL,
    "override": False,
}

# Dictionary to set logging levels and override for default values for each module
logging_config_dict: Dict[str, Dict[str, Any]] = {
    "PARALLEL_TRAINING_3D_CHANNEL_MARL": {
        "console_level": "INFO",
        "file_level": "DEBUG",
        "override": True,
    },
    "Env3D_MARL_channel": {
        "console_level": "INFO",
        "file_level": "DEBUG",
        "override": True,
    },
    "parameters": {
        "console_level": "INFO",
        "file_level": "DEBUG",
        "override": True,
    },
    "alya": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "calc_avg_qvolume": {
        "console_level": "INFO",
        "file_level": "DEBUG",
        "override": True,
    },
    "coco_calc_avg_vel": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "coco_calc_reward": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "cr": {
        "console_level": "WARNING",
        "file_level": "WARNING",
        "override": True,
    },
    "env_utils": {
        "console_level": "WARNING",
        "file_level": "INFO",
        "override": True,
    },
    "extract_forces": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "jets": {
        "console_level": "INFO",
        "file_level": "DEBUG",
        "override": True,
    },
    "visualization": {
        "console_level": "INFO",
        "file_level": "DEBUG",
        "override": True,
    },
    "witness": {
        "console_level": "WARNING",
        "file_level": "INFO",
        "override": True,
    },
}


def clear_old_logs(log_dir: str) -> None:
    """
    Clear old log files in the specified directory.

    Parameters
    ----------
    log_dir : str
        The directory where log files are stored.
    """
    if os.path.exists(log_dir):
        for file in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)


# TODO: @pietero use `funcName` to automatically add the function name to the log - Pieter
def configure_logger(
    module_name: str,
    default_level: str = "INFO",
    log_dir: str = LOG_DIR,
) -> logging.Logger:
    """
    Configure a logger with specified settings or defaults.

    Parameters
    ----------
    module_name : str
        The name of the module for the logger.
    default_level : str, optional
        The default logging level (default is "INFO").
    log_dir : str, optional
        The directory where log files are stored (default is "logsPYTHON").

    Returns
    -------
    logging.Logger
        Configured logger.

    Raises
    ------
    ValueError
        If the module_name is not found in logging_config_dict.
        If the 'override' key is missing in the configuration.

    Examples
    --------
    Setting up logging for a new module:

    >>> from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL
    >>> logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)
    >>> logger.info("%s.py: Logging level set to %s", __name__, logger.level)

    Console log format:
    module_name - LEVEL - message

    Example:
    Env3D_MARL_channel - INFO - Environment initialized

    File log format:
    YYYY-MM-DD HH:MM:SS,mmm - module_name - LEVEL - message

    Example:
    2024-08-07 10:00:00,123 - Env3D_MARL_channel - INFO - Environment initialized
    """
    logger = logging.getLogger(module_name)

    if module_name not in logging_config_dict:
        raise ValueError(
            f"Logger configuration for module '{module_name}' not found in logging_config_dict"
        )

    config = logging_config_dict.get(module_name, DEFAULT_CONFIG)

    if "override" not in config:
        raise ValueError(
            f"Missing 'override' key in logger configuration for module '{module_name}'"
        )

    if config["override"]:
        console_level = config["console_level"].upper()
        file_level = config["file_level"].upper()
    else:
        console_level = DEFAULT_CONFIG["console_level"].upper()
        file_level = DEFAULT_CONFIG["file_level"].upper()

    if config is None:
        raise ValueError(f"Module {module_name} not found in logging_config_dict")

    # Clear existing handlers if override is specified
    if config["override"]:
        logger.handlers = []

    if not logger.hasHandlers():
        logger.setLevel(
            min(
                possible_logging_levels[console_level],
                possible_logging_levels[file_level],
            )
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        formatter_ch = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter_ch)
        logger.addHandler(ch)

        # File handler
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        module_log_path = f"{log_dir}/{module_name}.log"
        fh = logging.FileHandler(module_log_path)
        fh.setLevel(file_level)
        formatter_fh = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)

        # Global file handler for all logs
        global_log_path = f"{log_dir}/all.log"
        fh_all = logging.FileHandler(global_log_path)
        fh_all.setLevel(file_level)
        fh_all.setFormatter(formatter_fh)
        logger.addHandler(fh_all)

    # Disable propagation to prevent duplicate logs
    logger.propagate = False

    return logger


# Function to configure loggers specifically for the Environment class
def configure_env_logger(
    log_dir: str = LOG_DIR,
) -> (logging.Logger, logging.Logger):
    """
    Configure loggers specifically for the Environment class.

    One logger (primary_logger) logs to both console and file, while the other
    (file_only_logger) logs to file only. This is to separate the console output
    for multiple instances of the Environment class, only logging to the console
    instances with ENV_ID [*, 1] and logging to file for all instances.

    Parameters
    ----------
    log_dir : str, optional
        The directory where log files are stored (default is "logsPYTHON").

    Returns
    -------
    tuple
        A tuple containing the primary logger and file-only logger.

    Examples
    --------
    Setting up logging for the Environment class:

    >>> primary_logger, file_only_logger = configure_env_logger()
    >>> primary_logger.info("Primary logger set up for Env3D_MARL_channel")
    >>> file_only_logger.info("File-only logger set up for Env3D_MARL_channel")

    Console log format (primary_logger):
    Env3D_MARL_channel - LEVEL - message

    Example:
    Env3D_MARL_channel - INFO - Environment initialized

    File log format (primary_logger and file_only_logger):
    YYYY-MM-DD HH:MM:SS,mmm - Env3D_MARL_channel - LEVEL - message

    Example:
    2024-08-07 10:00:00,123 - Env3D_MARL_channel - INFO - Environment initialized

    See Also
    --------
    `Environment.log`: Method to log messages in the Environment class.
    (`DRL_POL/Env3D_MARL_channel.py`)
    """
    # Define the logging levels for the Environment class
    if logging_config_dict["Env3D_MARL_channel"]["override"]:
        console_level = logging_config_dict["Env3D_MARL_channel"][
            "console_level"
        ].upper()
        file_level = logging_config_dict["Env3D_MARL_channel"]["file_level"].upper()
    else:
        console_level = DEFAULT_LOGGING_LEVEL
        file_level = DEFAULT_LOGGING_LEVEL

    # Create primary logger that logs to both console and file
    primary_logger = logging.getLogger("primary_logger")
    primary_logger.setLevel(console_level)

    # Create file handler
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    env_log_path = f"{log_dir}/Env3D_MARL_channel.log"
    primary_fh = logging.FileHandler(env_log_path)
    primary_fh.setLevel(file_level)
    primary_formatter_fh = logging.Formatter(
        "%(asctime)s - Env3D_MARL_channel - %(levelname)s - %(message)s"
    )
    primary_fh.setFormatter(primary_formatter_fh)
    primary_logger.addHandler(primary_fh)

    # Create console handler
    primary_ch = logging.StreamHandler()
    primary_ch.setLevel(console_level)
    primary_formatter_ch = logging.Formatter(
        "Env3D_MARL_channel - %(levelname)s - %(message)s"
    )
    primary_ch.setFormatter(primary_formatter_ch)
    primary_logger.addHandler(primary_ch)

    # Create global file handler
    global_env_log_path = f"{log_dir}/all.log"
    global_fh = logging.FileHandler(global_env_log_path)
    global_fh.setLevel(file_level)  # Set to the lowest level to capture all messages
    global_fh.setFormatter(primary_formatter_fh)
    primary_logger.addHandler(global_fh)

    # Create file-only logger
    file_only_logger = logging.getLogger("file_only_logger")
    file_only_logger.setLevel(file_level)
    file_only_logger.addHandler(primary_fh)
    file_only_logger.addHandler(global_fh)

    primary_logger.propagate = False
    file_only_logger.propagate = False

    return primary_logger, file_only_logger
