import os
from typing import Dict, Any
import logging

## This file sets the logging levels for the project
# IN GENERAL: logging messages should use %s formatting, not f-strings

# Possible logging levels (for reference)
possible_logging_levels = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

# Set the DEFAULT logging level, passed to individual modules
DEFAULT_LOGGING_LEVEL = "INFO"

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
        "override": True,
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
    "witness": {
        "console_level": "WARNING",
        "file_level": "INFO",
        "override": True,
    },
}


# TODO: @pietero use `funcName` to automatically add the function name to the log - Pieter
def configure_logger(module_name: str, default_level: str = "INFO") -> logging.Logger:
    """
    Configure a logger with specified settings or defaults.

    Args:
        module_name (str): The name of the module for the logger.
        default_level (str): The default logging level (default is "INFO").

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(module_name)

    if module_name in logging_config_dict:
        config = logging_config_dict[module_name]
        if config.get("override", False):
            console_level = config["console_level"].upper()
            file_level = config["file_level"].upper()
        else:
            console_level = default_level.upper()
            file_level = default_level.upper()
    else:
        config = None
        console_level = default_level.upper()
        file_level = default_level.upper()

    if config is None:
        raise ValueError(f"Module {module_name} not found in logging_config_dict")

    # Clear existing handlers if override is specified
    if config.get("override", False):
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
        if not os.path.exists("logsPYTHON"):
            os.makedirs("logsPYTHON")
        fh = logging.FileHandler(f"logsPYTHON/{module_name}.log")
        fh.setLevel(file_level)
        formatter_fh = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)

        # Global file handler for all logs
        global_log_path = "logsPYTHON/all.log"
        fh_all = logging.FileHandler(global_log_path)
        fh_all.setLevel(file_level)
        fh_all.setFormatter(formatter_fh)
        logger.addHandler(fh_all)

    # Disable propagation to prevent duplicate logs
    logger.propagate = False

    return logger


def configure_env_logger() -> (logging.Logger, logging.Logger):
    """
    Configure loggers specifically for the Environment class.

    One logger (primary_logger) logs to both console and file, while the other (file_only_logger) logs to file only.
    This is to separate the console output for multiple instances of the Environment class, only logging to the console
    instances with ENV_ID [*, 1] and logging to file for all instances.

    Returns:
        tuple: A tuple containing the primary logger and file-only logger.
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
    if not os.path.exists("logsPYTHON"):
        os.makedirs("logsPYTHON")
    primary_fh = logging.FileHandler("logsPYTHON/Env3D_MARL_channel.log")
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
    global_fh = logging.FileHandler("logsPYTHON/all.log")
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
