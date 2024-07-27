#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Maxence Deferrez, Pol Suarez, Arnau Miro
# 07/07/2022
from __future__ import print_function, division

import os, numpy as np
from abc import ABC, abstractmethod
from typing import List, Type, Any, Dict, Optional

from alya import write_jet_file

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s\n", __name__, logger.level)


# Function to build and return the jets
# See jets_definition in parameters.py for the use of this function
def build_jets(
    jet_class: Type[Any],
    jets_definition: Dict[str, Dict[str, Any]],
    delta_t_smooth: float,
) -> Dict[str, Any]:
    """
    This helper function is used to build and return the dictionary that
    contains information on the jets.
    For that one has to give the kind of jet class (directly the class object)
    as well as the jet names and jet geometric parameters.
    """
    logger.info("build_jets: Initializing jets of class %s...", jet_class.__name__)
    names = list(jets_definition.keys())
    # Build jets dictionary
    jets = {}
    for name in names:
        jets[name] = jet_class(name, jets_definition[name], T_smoo=delta_t_smooth)
    logger.info(
        "build_jets: %d %s class instance(s) created.\n",
        len(names),
        jet_class.__name__,
    )
    return jets


# Function to write atan2 as a string
def atan2_str(X: str, Y: str) -> str:
    return f"2*atan({Y}/({X} + sqrt({X}^2+{Y}^2)))"


# Smoothing functions
#


def Q_smooth_linear(
    Qnew_single: float, Qpre_single: float, timestart: float, Tsmooth: float
) -> str:
    """
    Linear smoothing law:
        Q(t) = (Qn - Qs)*(t - ts)/Tsmooth + Qs
    """
    logger.debug(
        "Q_smooth_linear: Creating linear smoothing function over time interval [%f, %f]",
        timestart,
        timestart + Tsmooth,
    )
    deltaQ_single = Qnew_single - Qpre_single
    return f"({deltaQ_single}/{Tsmooth}*(t-{timestart}) + ({Qpre_single}))"


def Q_smooth_exp(ts: float, Tsmooth: float) -> str:
    """
    Exponential smoothing law: from (https://en.wikipedia.org/wiki/Non-analytic_smooth_function#Smooth_transition_functions)

    f(x) = e^(-1/x) if x > 0
         = 0        if x <= 0

    Between two points:

    'x' => (x-a)/(b-a)

    g(x) = f(x)/(f(x) + f(1-x))

    """
    logger.debug(
        "Q_smooth_exp: Creating exponential smoothing function over time interval [%f, %f]",
        ts,
        ts + Tsmooth,
    )
    t1 = ts
    t2 = ts + Tsmooth

    xp = f"(pos((t-{t1:.2f})/{t2 - t1:.2f}))"
    f1 = f"exp(-1/{xp})"
    f2 = f"exp(-1/pos(1-{xp}))"
    h = f"{f1}/({f1}+{f2})"

    # return '((%f) + ((%s)*(%f)))' % (Q1,h,Q2-Q1)
    logger.debug("Q_smooth_exp: Exponential smoothing function: %s", h)
    return h


def heav_func(position: float, delta: float) -> str:
    """
    Define the heaviside function in spanwise to change the Q in diferent locations at an axis
    takes de position and activates the Q inside range [position-delta,position+delta]
    """
    logger.debug(
        "heav_func: Creating heaviside function for position %f and delta %f",
        position,
        delta,
    )
    return f"heav((z-{position - delta * 0.5:.3f})*({position + delta * 0.5:.3f}-z))"


def heav_func_channel(
    position_x: float, delta_x: float, position_z: float, delta_z: float
) -> str:
    """
    Define the heaviside function xz-grid to change the Q in diferent locations
    takes the x and z positions and activates the Q inside range [x-delta,x+delta],[z-delta,z+delta] -Chriss
    """
    logger.debug(
        "heav_func_channel: Creating heaviside function for position_x %f, delta_x %f, position_z %f, delta_z %f",
        position_x,
        delta_x,
        position_z,
        delta_z,
    )
    return f"heav((x-{position_x - delta_x * 0.5:.3f})*({position_x + delta_x * 0.5:.3f}-x)) * heav((z-{position_z - delta_z * 0.5:.3f})*({position_z + delta_z * 0.5:.3f}-z))"


class Jet(ABC):
    """
    Parent class to implement jets on the DRL.

    Implements a generic class constructor which calls specialized functions from
    children classes in order to set up the jet.

    It also implements the following generic methods:
        -
    """

    def __init__(
        self,
        name: str,
        params: Dict[str, Dict[str, Any]],
        Q_pre: List[float] = None,
        Q_new: List[float] = None,
        time_start: float = 0.0,
        dimension: int = 2,
        T_smoo: float = 0.2,
        smooth_func: str = "",
    ) -> None:
        """
        Class initializer, generic.
        Sets up the basic parameters and starts the class.
        After creating the class we should initialize the geometry manually for
        each of the specialized jets.
        """
        from parameters import (
            dimension,
            short_spacetime_func,
            nb_inv_per_CFD,
        )

        if Q_new is None:
            Q_new = [0.0]
        if Q_pre is None:
            Q_pre = [0.0]

        logger.debug("Jet: Base class super init of jet %s...", name)

        # Basic jet variables
        self.name: str = name
        self.T_smoo: float = T_smoo
        self.smooth_func = smooth_func
        self.dimension: int = dimension
        self.theta: float = 0  # to be updated during DRL
        # Jet velocity functions (to be updated during DRL)
        self.Vx: str = ""
        self.Vy: str = ""
        self.Vz: str = ""
        # Jet Action List
        self.Q_pre: List[float] = Q_pre
        self.Q_new: List[float] = Q_new
        # Call specialized method to set up the jet geometry
        self.set_geometry(params)
        # TO BE DELETED!!! This is pulled from parameters in JetChannel instead -Chriss juli 2024
        # Update to this current timestep
        # self.Qs_position_x: float = self.Qs_position_x
        # self.Qs_position_z: float = self.Qs_position_z
        # self.delta_Q_x: float = self.delta_Q_x
        # self.delta_Q_z: float = self.delta_Q_z
        self.time_start = time_start
        self.short_spacetime_func: bool = short_spacetime_func
        self.nb_inv_per_CFD: int = nb_inv_per_CFD
        # self.update(Q_pre, Q_new, time_start, smooth_func)

    def update_file(self, filepath: str) -> None:
        """
        Replaces the jets path file for a new one, generic.
        The name of the file must be the same of that of the jet.
        """
        logger.info("Jet: Updating jet file %s...", filepath)
        functions = (
            [self.Vx, self.Vy] if self.dimension == 2 else [self.Vx, self.Vy, self.Vz]
        )
        write_jet_file(filepath, self.name, functions)

    @abstractmethod
    def update(
        self,
        Q_pre: List[float],
        Q_new: List[float],
        time_start: float,
        smooth_func: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates a jet for a given epoch of the DRL, generic.
        To be implemented by child classes
        """
        raise NotImplementedError(
            "Jet.update: Must specialize the `update` method for each specific jet kind"
        )

    @abstractmethod
    def set_geometry(self, geometry_params: Dict[str, Any]) -> Any:
        """
        Placeholder for specialized function that sets the jet geometry
        per each of the independent cases.
        """
        raise NotImplementedError(
            "Jet.set_geometry: Must specialize the `set_geometry` method for each specific jet kind"
        )

    @abstractmethod
    def create_smooth_funcs(
        self,
        Q_new: List[float],
        Q_pre: List[float],
        time_start: float,
        T_smoo: float,
        smooth_func: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Placeholder for specialized function that sets the jet geometry
        per each of the independent cases.
        """
        raise NotImplementedError(
            "Jet.create_smooth_funcs: Must specialize the `create_smooth_funcs` method for each jet class"
        )


class JetCylinder(Jet):
    """
    Specialized jet class to deal with jets specified in cylindrical coordinates.
    """

    def __init__(
        self,
        name: str,
        params: Dict[str, Any],
        Q_pre: List[float] = None,
        Q_new: List[float] = None,
        time_start: float = 0.0,
        dimension: int = 2,
        T_smoo: float = 0.2,
        smooth_func: str = "",
    ) -> None:
        """
        Initialize the JetCylinder class.
        """
        logger.info("JetCylinder.init: Initializing jet %s...", name)

        super().__init__(
            name, params, Q_pre, Q_new, time_start, dimension, T_smoo, smooth_func
        )
        from parameters import (
            Qs_position_z,
            delta_Q_z,
        )

        logger.info("JetCylinder.init: Jet %s initialized.", name)
        self.update(
            self.Q_pre,
            self.Q_new,
            self.time_start,
            self.smooth_func,
            Qs_position_z=self.Qs_position_z,
            delta_Q_z=self.delta_Q_z,
        )
        logger.debug("JetCylinder.init: Jet %s intial update complete.\n", name)

    def set_geometry(self, params: Dict[str, Any]) -> None:
        """
        Specialized method that sets up the geometry of the jet
        """
        logger.debug(
            "JetCylinder.set_geometry: Importing geometry params for jet %s ...",
            self.name,
        )
        from parameters import (
            cylinder_coordinates,
            Qs_position_z,
            delta_Q_z,
        )

        logger.debug(
            "JetCylinder.set_geometry: Setting up jet geometry for jet %s ...",
            self.name,
        )
        # Sanity check
        # TODO: asserts are dangerous... we need a function that stops everything!!
        if params["width"] <= 0.0:
            raise ValueError(f"Invalid jet width={params['width']}")
        if params["radius"] <= 0.0:
            raise ValueError(f"Invalid jet radius={params['radius']}")
        if params["positions_angle"] <= 0.0:
            raise ValueError(f"Invalid jet angle={params['positions_angle']}")
        # Recover parameters from dictionary
        self.radius: float = params["radius"]
        self.width: float = params["width"]
        self.theta0: float = self.normalize_angle(np.deg2rad(params["positions_angle"]))
        self.theta: str = self.get_theta(cylinder_coordinates)

        self.Qs_position_z = Qs_position_z
        self.delta_Q_z = delta_Q_z
        logger.debug(
            "JetCylinder.set_geometry: Jet geometry set up for jet %s.", self.name
        )

    def update(
        self,
        Q_pre: List[float],
        Q_new: List[float],
        time_start: float,
        smooth_func: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates a jet for a given epoch of the DRL, generic.
        Calls the specialized method smoothfunc to set up the jet geometry
        per each of the child classes.
        """
        logger.info("JetCylinder.update: Updating jet %s...", self.name)

        Qs_position_z: List[float] = kwargs.get("Qs_position_z")
        delta_Q_z: float = kwargs.get("delta_Q_z")

        if Qs_position_z is None:
            raise ValueError(
                "JetCylinder.update: Missing required keyword argument: 'Qs_position_z'"
            )
        if delta_Q_z is None:
            raise ValueError(
                "JetCylinder.update: Missing required keyword argument: 'delta_Q_z'"
            )

        # Up
        self.Q_pre: List[float] = Q_pre
        self.Q_new: List[float] = Q_new
        self.time_start: float = time_start
        self.smooth_func: str = smooth_func
        self.Qs_position_z: List[float] = (
            Qs_position_z  # Updating positions just in case (could be removed, already assigned during init) - Pieter
        )
        self.delta_Q_z: float = (
            delta_Q_z  # Updating delta just in case (could be removed, already assigned during init) - Pieter
        )
        logger.debug("JetCylinder.update: creating new smoothing functions...")
        # Call the specialized method that creates a smoothing function for the current time
        smooth_fun: str = self.create_smooth_funcs(
            self.Q_new,
            self.Q_pre,
            self.time_start,
            self.T_smoo,
            self.smooth_func,
            Qs_position_z=self.Qs_position_z,
            delta_Q_z=self.delta_Q_z,
        )
        logger.debug("JetCylinder.update: assigning new velocity functions...")
        # Create the velocities (function?) using the smoothing functions,
        if self.dimension == 2:
            # For 2D jets set Vx and Vy
            self.Vx = f"{smooth_fun}*cos({self.theta})"
            self.Vy = f"{smooth_fun}*sin({self.theta})"
        else:
            # For 3D jets raise an error
            self.Vx = f"{smooth_fun}*cos({self.theta})"
            self.Vy = f"{smooth_fun}*abs(sin({self.theta}))"  # TODO: temporal fix for component y (not opposite? check update_jet)
            self.Vz = "0"
        logger.info("JetCylinder.update: Jet %s updated.\n", self.name)

    def create_smooth_funcs(
        self,
        Q_new: List[float],
        Q_pre: List[float],
        time_start: float,
        T_smoo: float,
        smooth_func: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Specialized method that creates the smooth functions for cylinder cases
        """
        logger.debug("JetCylinder.create_smooth_funcs: Creating smooth functions...")
        Qs_position_z: List[float] = kwargs.get("Qs_position_z")
        delta_Q_z: float = kwargs.get("delta_Q_z")

        if Qs_position_z is None:
            raise ValueError(
                "JetCylinder.create_smooth_funcs: Missing required keyword argument: 'Qs_position_z'"
            )
        if delta_Q_z is None:
            raise ValueError(
                "JetCylinder.create_smooth_funcs: Missing required keyword argument: 'delta_Q_z'"
            )

        w = self.width * (np.pi / 180)  # deg2rad
        scale = np.pi / (2.0 * w * self.radius)  #### FIX: NOT R**2 --> D

        string_all_Q_pre = "0"
        string_all_Q_new = "0"
        string_heav = ""
        # print(f"\nJetCylinder: create_smooth_funcs: Q_new: {Q_new}\n")
        # print(f"JetCylinder: create_smooth_funcs: Q_pre: {Q_pre}\n")
        # print(f"JetCylinder: create_smooth_funcs: self.smooth_func: {self.smooth_func}\n")
        if self.smooth_func == "EXPONENTIAL":
            logger.debug(
                "JetCylinder.create_smooth_funcs: Exponential smoothing selected..."
            )
            ## Q_pre and Q_new --> list! with nz_Qs dimensions
            string_h = Q_smooth_exp(time_start, T_smoo)

            # create the new Q string
            string_heav = heav_func(Qs_position_z[0], delta_Q_z)
            string_all_Q_pre = f"{string_heav}*({Q_pre[0]:.4f})"
            string_all_Q_new = f"{string_heav}*({Q_new[0]:.4f})"

            for i in range(1, self.nb_inv_per_CFD):
                string_heav = heav_func(Qs_position_z[i], delta_Q_z)
                string_all_Q_pre += f"+ {string_heav}*({Q_pre[i]:.4f})"
                string_all_Q_new += f"+ {string_heav}*({Q_new[i]:.4f})"
            string_Q = f"(({string_all_Q_pre}) + ({string_h})*(({string_all_Q_new})-({string_all_Q_pre})))"

        elif self.smooth_func == "LINEAR":  # Same as "" currently
            logger.debug(
                "JetCylinder.create_smooth_funcs: Linear smoothing selected..."
            )
            string_Q = Q_smooth_linear(Q_new[0], Q_pre[0], time_start, T_smoo)

        elif self.smooth_func == "":
            logger.debug(
                "JetCylinder.create_smooth_funcs: No smoothing selected, defaulting to Linear..."
            )
            string_Q = Q_smooth_linear(Q_new[0], Q_pre[0], time_start, T_smoo)

        else:
            raise ValueError(
                f"JetCylinder.create_smooth_funcs: `smooth_func` arg: Invalid smoothing function type:{self.smooth_func}"
            )

        if self.short_spacetime_func == True:
            # just with Qnorm*Qi -- no projection or smoothing in time/space
            logger.debug(
                "JetCylinder.create_smooth_funcs: Short spacetime function option selected..."
            )
            return f"({scale:.1f})({string_all_Q_new})"
        else:
            logger.debug(
                "JetCylinder.create_smooth_funcs: Long spacetime function option selected..."
            )  # NEED BETTER logger.debug message - Pieter
            string_C = f"cos({np.pi:.3f}/{w:.3f}*({self.theta}-({self.theta0:.3f})))"
            return f"({scale:.1f})*({string_Q})*({string_C})"

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize angle between [-pi,pi]
        """
        logger.debug("JetCylinder.normalize_angle: Normalizing angle %f...", angle)
        # TODO: check this... not very clear to me
        if angle > np.pi:
            angle -= 2 * np.pi
        if angle < 2.0 * np.pi:
            angle = -((2.0 * np.pi) - angle)
        logger.debug("JetCylinder.normalize_angle: Normalized angle: %f", angle)
        return angle

    @staticmethod
    def get_theta(cylinder_coordinates: List[float]) -> str:
        """
        TODO: documentation!
        """
        logger.debug("JetCylinder.get_theta: Getting theta...")
        X: str = f"(x-{cylinder_coordinates[0]})"
        Y: str = f"(y-{cylinder_coordinates[1]})"
        result: str = atan2_str(X, Y)
        logger.debug("JetCylinder.get_theta: Theta string: %s", result)
        return result


# TODO: finish updating JetAirfoil for new base class code @pietero
class JetAirfoil(Jet):
    """
    NOT IMPLEMENTED YET

    Specialized jet class to deal with jets specified in cartesian coordinates.
    """

    def __init__(
        self,
        name: str,
        params: Dict[str, Any],
        Q_pre: List[float] = None,
        Q_new: List[float] = None,
        time_start: float = 0.0,
        dimension: int = 2,
        T_smoo: float = 0.2,
        smooth_func: str = "",
    ) -> None:
        """
        Initialize the JetAirfoil class.
        """
        super().__init__(
            name, params, Q_pre, Q_new, time_start, dimension, T_smoo, smooth_func
        )

        # NO UPDATE METHOD DEFINED YET - Pieter July 2024
        self.update(
            Q_pre,
            Q_new,
            time_start,
            smooth_func,
        )

    def set_geometry(self, params):
        """
        Specialized method that sets up the geometry of the jet
        """
        from parameters import (
            rotate_airfoil,
            aoa,
        )

        # Get jet positions
        self.x1 = params["x1"]
        self.x2 = params["x2"]
        self.y1 = params["y1"]
        self.y2 = params["y2"]

        if rotate_airfoil:
            self.x1 = self.x1 * np.cos(np.deg2rad(aoa)) + self.y1 * np.sin(
                np.deg2rad(aoa)
            )
            self.y1 = self.y1 * np.cos(np.deg2rad(aoa)) - self.x1 * np.sin(
                np.deg2rad(aoa)
            )
            self.x2 = self.x2 * np.cos(np.deg2rad(aoa)) + self.y2 * np.sin(
                np.deg2rad(aoa)
            )
            self.y2 = self.y2 * np.cos(np.deg2rad(aoa)) - self.x2 * np.sin(
                np.deg2rad(aoa)
            )

        # Get the angle of the slope normal to the surface
        self.theta = self.get_slope(self)

    def create_smooth_funcs(
        self,
        Q_new: List[float],
        Q_pre: List[float],
        time_start: float,
        T_smoo: float,
        smooth_func: str,
        **kwargs: Any,
    ) -> str:
        """
        Specialized method that creates the smooth functions
        """
        w = np.sqrt((self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2)
        scale = np.pi / (2 * w)

        if smooth_func == "EXPONENTIAL":
            string_Q = Q_smooth_exp(Q_new[0], Q_pre[0], time_start, T_smoo)

        elif smooth_func == "LINEAR":
            string_Q = Q_smooth_linear(Q_new[0], Q_pre[0], time_start, T_smoo)
        else:
            raise ValueError(
                f"JetAirfoil.create_smooth_funcs: `smooth_func` arg: Invalid smoothing function type {smooth_func}"
            )

        # delta_Q  = Q_new - Q_pre
        # string_Q = '{}*({}/{}*(t-{}) + ({}))'.format(scale, delta_Q, T_smoo, time_start, Q_pre) # Change this for Xavi's approach
        string_S = "sin({}*(x-{})/({}-{}))".format(np.pi, self.x1, self.x2, self.x1)
        return "({})*({})".format(string_Q, string_S)

    @staticmethod
    def get_slope(self):
        """
        We are actually getting the angle of the slope
        """
        X = "({}-({}))".format(self.y2, self.y1)
        Y = "({}-({}))".format(self.x1, self.x2)
        return atan2_str(X, Y)


class JetChannel(Jet):
    """
    Specialized jet class to deal with jets in a channel.
    """

    def __init__(
        self,
        name: str,
        params: Dict[str, Any],
        Q_pre: List[float] = None,
        Q_new: List[float] = None,
        time_start: float = 0.0,
        dimension: int = 2,
        T_smoo: float = 0.2,
        smooth_func: str = "",
    ) -> None:
        """
        Initialize the JetChannel class.
        """
        logger.info("JetChannel.init: Initializing jet %s...", name)

        super().__init__(
            name, params, Q_pre, Q_new, time_start, dimension, T_smoo, smooth_func
        )

        logger.info("JetChannel.init: Jet %s initialized.\n", name)
        self.update(
            self.Q_pre,
            self.Q_new,
            self.time_start,
            self.smooth_func,
            Qs_position_x=self.Qs_position_x,
            delta_Q_x=self.delta_Q_x,
            Qs_position_z=self.Qs_position_z,
            delta_Q_z=self.delta_Q_z,
        )
        logger.debug("JetChannel.init: Jet %s intial update complete.", name)

    def set_geometry(self, params: Dict[str, Any]) -> None:
        """
        Specialized method that sets up the geometry of the jet, including importing Qs_position_x, Qs_position_z, delta_Q_z and delta_Q_z
        """
        logger.debug(
            "JetChannel.set_geometry: Importing geometry params for jet %s ...",
            self.name,
        )
        from parameters import (
            Qs_position_x,
            delta_Q_x,
            Qs_position_z,
            delta_Q_z,
        )

        logger.debug(
            "JetChannel.set_geometry: Setting up jet geometry for jet %s ...",
            self.name,
        )

        self.Qs_position_x: List[float] = Qs_position_x
        self.delta_Q_x: float = delta_Q_x
        self.Qs_position_z: List[float] = Qs_position_z
        self.delta_Q_z: float = delta_Q_z

        logger.debug(
            "JetChannel.set_geometry: Jet geometry set up for jet %s.", self.name
        )

    def update(
        self,
        Q_pre: List[float],
        Q_new: List[float],
        time_start: float,
        smooth_func: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        TO BE IMPLEMENTED FOR CHANNEL CASE
        """
        logger.info("JetChannel.update: Updating jet %s...", self.name)

        Qs_position_x: List[float] = kwargs.get("Qs_position_x")
        delta_Q_x: float = kwargs.get("delta_Q_x")
        Qs_position_z: List[float] = kwargs.get("Qs_position_z")
        delta_Q_z: float = kwargs.get("delta_Q_z")

        if Qs_position_x is None:
            raise ValueError(
                "JetChannel.update: Missing required keyword argument: 'Qs_position_x'"
            )
        if delta_Q_x is None:
            raise ValueError(
                "JetChannel.update: Missing required keyword argument: 'delta_Q_z'"
            )
        if Qs_position_z is None:
            raise ValueError(
                "JetChannel.update: Missing required keyword argument: 'Qs_position_z'"
            )
        if delta_Q_z is None:
            raise ValueError(
                "JetChannel.update: Missing required keyword argument: 'delta_Q_z'"
            )

        # Up
        self.Q_pre: List[float] = Q_pre
        self.Q_new: List[float] = Q_new
        self.time_start: float = time_start
        self.smooth_func: str = smooth_func
        self.Qs_position_z: List[float] = (
            Qs_position_z  # Updating positions just in case (could be removed, already assigned during init) - Pieter
        )
        self.delta_Q_z: float = (
            delta_Q_z  # Updating delta just in case (could be removed, already assigned during init) - Pieter
        )
        logger.debug("JetChannel.update: creating new smoothing functions...")
        # Call the specialized method that creates a smoothing function for the current time
        smooth_fun: str = self.create_smooth_funcs(
            self.Q_new,
            self.Q_pre,
            self.time_start,
            self.T_smoo,
            self.smooth_func,
            Qs_position_z=self.Qs_position_z,
            delta_Q_z=self.delta_Q_z,
            Qs_position_x=self.Qs_position_x,
            delta_Q_x=self.delta_Q_x,
        )
        logger.debug("JetChannel.update: assigning new velocity functions...")
        # Create the velocities (function?) using the smoothing functions,
        if self.dimension == 2:
            # For 2D jets set Vx and Vy
            self.Vx = f"{smooth_fun}"
            self.Vy = f"{smooth_fun}"
        else:
            self.Vx = "0"
            self.Vy = f"{smooth_fun}"
            self.Vz = "0"
        logger.info("JetChannel.update: Jet %s updated.\n", self.name)

    # TODO: Update this function for channel

    def create_smooth_funcs(
        self,
        Q_new: List[float],
        Q_pre: List[float],
        time_start: float,
        T_smoo: float,
        smooth_func: str,
        **kwargs: Any,
    ) -> str:
        """
        Specialized method that creates the smooth functions in 2D
        """
        logger.debug("JetChannel.create_smooth_funcs: Creating smooth functions...")

        Qs_position_x: List[float] = kwargs.get("Qs_position_x")
        delta_Q_x: float = kwargs.get("delta_Q_x")
        Qs_position_z: List[float] = kwargs.get("Qs_position_z")
        delta_Q_z: float = kwargs.get("delta_Q_z")

        # scale = ? for channel
        # w = 1.0  # NOT channel width but width of jet. Leftover from cylinder case
        # w = self.width * (np.pi / 180)  # deg2rad
        scale = 1.0  # TODO: fix this with correct scaling value. Can be assigned as needed - Chriss

        string_all_Q_pre = "0"
        string_all_Q_new = "0"
        string_heav = ""

        # TODO: implement smoothing in space for channel case - Chriss
        if smooth_func == "EXPONENTIAL":
            logger.debug(
                "JetChannel.create_smooth_funcs: Exponential smoothing selected..."
            )
            ## Q_pre and Q_new --> list! with nz_Qs dimensions
            # Exponential smoothing law. Can be applied in time or space
            string_h = Q_smooth_exp(time_start, T_smoo)

            # create the new Q string
            string_heav = heav_func_channel(
                Qs_position_x[0], delta_Q_x, Qs_position_z[0], delta_Q_z
            )
            string_all_Q_pre = f"{string_heav}*({Q_pre[0]:.4f})"
            string_all_Q_new = f"{string_heav}*({Q_new[0]:.4f})"

            for i in range(1, self.nb_inv_per_CFD):
                string_heav = heav_func_channel(
                    Qs_position_x[0], delta_Q_x, Qs_position_z[0], delta_Q_z
                )
                string_all_Q_pre += f"+ {string_heav}*({Q_pre[i]:.4f})"
                string_all_Q_new += f"+ {string_heav}*({Q_new[i]:.4f})"
            string_Q = f"(({string_all_Q_pre}) + ({string_h})*(({string_all_Q_new})-({string_all_Q_pre})))"

        elif smooth_func == "LINEAR":
            logger.debug("JetChannel.create_smooth_funcs: Linear smoothing selected...")
            string_Q = Q_smooth_linear(Q_new[0], Q_pre[0], time_start, T_smoo)
        elif smooth_func == "":
            logger.debug(
                "JetChannel.create_smooth_funcs: No smoothing selected, defaulting to Linear..."
            )
            string_Q = Q_smooth_linear(Q_new[0], Q_pre[0], time_start, T_smoo)
        else:
            raise ValueError(
                f"JetChannel.fcreate_smooth_funcs: `smooth_func` arg: Invalid smoothing function type: {smooth_func}"
            )

        if self.short_spacetime_func:
            logger.debug(
                "JetChannel.create_smooth_funcs: Short spacetime function option selected..."
            )
            # just with Qnorm*Qi -- no projection or smoothing in time/space
            return f"({scale:.1f})({string_all_Q_new})"
        else:
            logger.debug(
                "JetChannel.create_smooth_funcs: Long spacetime function option selected..."
            )
            # Here we only had cos to show the projection; comes from how the jets were on the cylinder surface before
            # string_C is smoothing in space. This will be added at a later time -Chriss
            #           string_C = f"cos({np.pi:.3f}/{w:.3f}*({self.theta}-({self.theta0:.3f})))"
            #           return f"({scale:.1f})*({string_Q})*({string_C})"
            return f"({scale:.1f})*({string_Q}))"
