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

# Dictionary to define logging levels and bool for each module

logging_config_dict: Dict[str, Dict[str, Any]] = {
    "PARALLEL_TRAINING_3D_CHANNEL_MARL": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "Env3D_MARL_channel": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "parameters": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "alya": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
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
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "extract_forces": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "jets": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "witness": {
        "console_level": "WARNING",
        "file_level": "WARNING",
        "override": True,
    },
}


# TODO: @pietero use `funcName` to automatically add the function name to the log - Pieter
def configure_logger(module_name: str, default_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(module_name)
    if module_name in logging_config_dict:
        config = logging_config_dict[module_name]
        console_level = config.get("console_level", default_level).upper()
        file_level = config.get("file_level", default_level).upper()
        override = config.get("override", False)
    else:
        console_level = default_level.upper()
        file_level = default_level.upper()
        override = False

    # Check if logger already has handlers
    if not logger.hasHandlers() or override:
        # Clear existing loggers if override is true
        if override:
            logger.handlers = []

            # Set the logging level to the lowest level needed
            logger.setLevel(
                min(
                    possible_logging_levels[console_level],
                    possible_logging_levels[file_level],
                )
            )
        else:
            # Set the logging level to the default passed via argument
            logger.setLevel(possible_logging_levels[default_level])

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        formatter_ch_funcname = logging.Formatter(
            "%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s"
        )
        formatter_ch = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter_ch)
        logger.addHandler(ch)

        # File handler
        # check log folder exists
        if not os.path.exists("pythonlogs"):
            os.makedirs("pythonlogs")
        fh = logging.FileHandler(f"pythonlogs/{module_name}.log")
        fh.setLevel(file_level)
        formatter_fh_funcname = logging.Formatter(
            "%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s"
        )
        formatter_fh = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)

    return logger
