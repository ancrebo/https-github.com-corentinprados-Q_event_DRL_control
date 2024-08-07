"""
jets.py

DEEP REINFORCEMENT LEARNING WITH ALYA

This module provides classes and functions to manage jet implementations in ALYA
simulations. It includes the `Jet` base class and specialized classes such as
`JetCylinder` and `JetChannel`.

These classes handle the initialization, geometry setup, and action updates for
jets in different configurations. Additionally, the module offers utility
functions for creating smooth transitions for jet actions. The module is designed
to be imported and called from other scripts, such as `parameters.py`, where it is
used to create and manage jet instances. The `build_jets` function is particularly
important for initializing jet instances based on the definitions provided in
`parameters.py`.

Classes
-------
- Jet(ABC):
    Abstract base class for jets. Provides generic methods for jet setup and updates.

- JetCylinder(Jet):
    Specialized class for handling jets in cylindrical coordinates.

- JetChannel(Jet):
    Specialized class for handling jets in channel configurations.

Functions
---------
- build_jets(jet_class: Type[Any], jets_definition: Dict[str, Dict[str, Any]], delta_t_smooth: float) -> Dict[str, Any]:
    Helper function to build and return a dictionary of jet instances.

- atan2_str(X: str, Y: str) -> str:
    Utility function to write atan2, as a string.

- Q_smooth_linear(Qnew_single: float, Qpre_single: float, timestart: float, Tsmooth: float) -> str:
    Creates a linear smoothing function over time, as a string.

- Q_smooth_exp(ts: float, Tsmooth: float) -> str:
    Creates an exponential smoothing function over time, as a string.

- heav_func(position: float, delta: float) -> str:
    Defines a Heaviside function for spanwise changes, as a string.

- heav_func_channel(position_x: float, delta_x: float, position_z: float, delta_z: float) -> str:
    Defines a Heaviside function for xz-grid changes, as a string.

Usage
-----
This module is intended to be imported and used in other scripts. For example,
in `parameters.py`, you can create new jet instances and in `Env3D_MARL_channel.py` you manage their actions
using the relevant classes and functions from this module like the `update` and `update_file` methods.

Dependencies
------------
- logging
- numpy as np
- typing (for Type, Any, Dict, List)
- env_utils (for agent_index_1d_to_2d)
- alya (for write_jet_file)
- logging_config (for configure_logger, DEFAULT_LOGGING_LEVEL)

Examples
--------
Example usage of `build_jets` function:

>>> from jets import build_jets, JetCylinder
>>> jets_definition = {
...     "JET_TOP": {"width": 10, "radius": 5, "angle": 45, "positions_angle": 90, "positions": [0, 5], "remesh": False, "Qs_position_z": 1.0, "delta_Q_z": 0.1},
...     "JET_BOTTOM": {"width": 10, "radius": 5, "angle": 45, "positions_angle": 270, "positions": [0, -5], "remesh": False, "Qs_position_z": 1.0, "delta_Q_z": 0.1}
... }
>>> jets = build_jets(JetCylinder, jets_definition, delta_t_smooth=0.2)

Example usage of updating a jet for ALYA simulation:
>>> jets['JET_TOP'].update(Q_pre=[0.0], Q_new=[1.0], time_start=0.0, smooth_func='LINEAR')
>>> jets['JET_TOP'].update_file('JET_TOP.dat')

Version History
---------------
- Major update in August 2024.

Authors
-------
- Maxence Deferrez
- Pol Suarez
- Arnau Miro
- Pieter Orlandini
- Christine Anne Nordquist
"""

from __future__ import print_function, division

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Type, Any, Dict, Optional

from env_utils import agent_index_1d_to_2d
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
    logger.info("build_jets: Initializing jets of class %s...\n", jet_class.__name__)
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
    """
    Generate a string representation of the atan2 function.

    This function returns a string representation of the atan2 function for
    the given variables X and Y.

    Parameters
    ----------
    X : str
        The x-coordinate variable as a string.
    Y : str
        The y-coordinate variable as a string.

    Returns
    -------
    str
        A string representing the atan2 function of X and Y.

    Examples
    --------
    Generate the atan2 string for 'a' and 'b':
        >>> atan2_str('a', 'b')
        '2*atan(b/(a + sqrt(a^2+b^2)))'
    """
    return f"2*atan({Y}/({X} + sqrt({X}^2+{Y}^2)))"


# Smoothing functions
#


def Q_smooth_linear(
    Qnew_single: float, Qpre_single: float, timestart: float, Tsmooth: float
) -> str:
    """
    Generate a linear smoothing function for Q over a time interval.

    This function creates a string representation of a linear smoothing
    function for a quantity Q over a specified time interval.

    Parameters
    ----------
    Qnew_single : float
        The new value of Q.
    Qpre_single : float
        The previous value of Q.
    timestart : float
        The start time of the smoothing interval.
    Tsmooth : float
        The duration of the smoothing interval.

    Returns
    -------
    str
        A string representing the linear smoothing function.

    Notes
    -----
    The linear smoothing law is defined as:

    .. math::
        Q(t) = \left( Q_{new} - Q_{pre} \right) \frac{t - t_{start}}{T_{smooth}} + Q_{pre}

    Examples
    --------
    Generate a linear smoothing function:
        >>> Q_smooth_linear(10.0, 5.0, 0.0, 2.0)
        '(5.0/2.0*(t-0.0) + (5.0))'
    """
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
    Generate an exponential smoothing function for Q over a time interval.

    This function creates a string representation of an exponential smoothing
    function for a quantity Q over a specified time interval.

    Parameters
    ----------
    ts : float
        The start time of the smoothing interval.
    Tsmooth : float
        The duration of the smoothing interval.

    Returns
    -------
    str
        A string representing the exponential smoothing function.

    Notes
    -----
    The exponential smoothing law is defined as:

    .. math::
        f(x) =
        \begin{cases}
        e^{-1/x} & \text{if } x > 0 \\
        0        & \text{if } x \le 0
        \end{cases}

    The function is transitioned between two points:

    .. math::
        x \to \frac{x-a}{b-a}

    And the final smoothing function is:

    .. math::
        g(x) = \frac{f(x)}{f(x) + f(1-x)}

    For more details, refer to the Wikipedia page on
    `Non-analytic smooth functions <https://en.wikipedia.org/wiki/Non-analytic_smooth_function#Smooth_transition_functions>`_.

    Examples
    --------
    Generate an exponential smoothing function:
        >>> Q_smooth_exp(0.0, 2.0)
        'exp(-1/pos((t-0.00)/2.00))/(exp(-1/pos((t-0.00)/2.00))+exp(-1/pos(1-pos((t-0.00)/2.00))))'
    """
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
    Generate a Heaviside function for a spanwise range.

    This function creates a string representation of a Heaviside function
    that activates the quantity Q within the range
    [position - delta, position + delta].

    Parameters
    ----------
    position : float
        The central position of the range.
    delta : float
        The width of the range.

    Returns
    -------
    str
        A string representing the Heaviside function for the specified range.

    Notes
    -----
    The Heaviside function is defined as:

    .. math::
        H(z) = \text{heav}((z - (position - \delta/2)) * ((position + \delta/2) - z))

    Examples
    --------
    Generate a Heaviside function for position 5.0 and delta 2.0:
        >>> heav_func(5.0, 2.0)
        'heav((z-4.000)*(6.000-z))'
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
    Generate a Heaviside function for an x-z grid.

    This function creates a string representation of a Heaviside function
    that activates the quantity Q within the ranges
    [x-delta_x, x+delta_x] and [z-delta_z, z+delta_z].

    Parameters
    ----------
    position_x : float
        The central position of the range in the x direction.
    delta_x : float
        The width of the range in the x direction.
    position_z : float
        The central position of the range in the z direction.
    delta_z : float
        The width of the range in the z direction.

    Returns
    -------
    str
        A string representing the Heaviside function for the specified ranges.

    Notes
    -----
    The Heaviside function for the x-z grid is defined as:

    .. math::
        H(x, z) = \text{heav}((x - (position_x - \delta_x/2)) * ((position_x + \delta_x/2) - x))
                  * \text{heav}((z - (position_z - \delta_z/2)) * ((position_z + \delta_z/2) - z))

    Examples
    --------
    Generate a Heaviside function for x position 5.0, delta_x 2.0, z position
    10.0, and delta_z 3.0:
        >>> heav_func_channel(5.0, 2.0, 10.0, 3.0)
        'heav((x-4.000)*(6.000-x)) * heav((z-8.500)*(11.500-z))'
    """
    logger.debug(
        "heav_func_channel: Creating heaviside function for position_x %f, delta_x %f, position_z %f, delta_z %f",
        position_x,
        delta_x,
        position_z,
        delta_z,
    )
    heav_str: str = (
        f"heav((x-{position_x - delta_x * 0.5:.3f})*({position_x + delta_x * 0.5:.3f}-x)) * heav((z-{position_z - delta_z * 0.5:.3f})*({position_z + delta_z * 0.5:.3f}-z))"
    )
    logger.debug("heav_func_channel: Heaviside function: \n%s", heav_str)
    return heav_str


class Jet(ABC):
    """
    Abstract base class for jets in the Deep Reinforcement Learning (DRL) framework.

    This class defines the generic interface and basic parameters for jets,
    which are then specialized by child classes for specific jet geometries
    and behaviors.

    Attributes
    ----------
    name : str
        The name of the jet.
    T_smoo : float
        The smoothing time parameter.
    smooth_func : str
        The smoothing function type.
    dimension : int
        The dimensionality of the jet (2 or 3).
    theta : float
        The angle parameter for the jet.
    Vx : str
        The velocity function in the x direction.
    Vy : str
        The velocity function in the y direction.
    Vz : str
        The velocity function in the z direction (only for 3D jets).
    Q_pre : List[float]
        The previous jet action values.
    Q_new : List[float]
        The new jet action values.
    time_start : float
        The start time of the jet action.
    short_spacetime_func : bool
        Flag to indicate if short spacetime functions are used.
    nb_inv_per_CFD : int
        Number of intervals per CFD.

    Methods
    -------
    update_file(filepath: str) -> None
        Update the jet file with the current velocity functions.
    update(Q_pre: List[float], Q_new: List[float], time_start: float, smooth_func: str, *args: Any, **kwargs: Any) -> None
        Abstract method to update the jet for a given action of the DRL.
    set_geometry(geometry_params: Dict[str, Any]) -> Any
        Abstract method to set the jet geometry.
    create_smooth_funcs(Q_new: List[float], Q_pre: List[float], time_start: float, T_smoo: float, smooth_func: str, *args: Any, **kwargs: Any) -> str
        Abstract method to create the smooth functions for the jet.

    Notes
    -----
    This class is designed to be extended by specific jet classes that implement
    the abstract methods for their specific geometries and behaviors.
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
        Initialize the Jet class with the given parameters.

        This initializer sets up the basic parameters for the jet and calls the
        specialized `set_geometry` method to set up the jet geometry.

        Parameters
        ----------
        name : str
            The name of the jet.
        params : Dict[str, Dict[str, Any]]
            The parameters for the jet.
        Q_pre : List[float], optional
            The previous jet action values (default is [0.0]).
        Q_new : List[float], optional
            The new jet action values (default is [0.0]).
        time_start : float, optional
            The start time of the jet action (default is 0.0).
        dimension : int, optional
            The dimensionality of the jet (default is 2).
        T_smoo : float, optional
            The smoothing time parameter (default is 0.2).
        smooth_func : str, optional
            The smoothing function type (default is "").
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
        Update the jet file with the current velocity functions.

        Parameters
        ----------
        filepath : str
            The path to the file to be updated.

        Notes
        -----
        This method writes the current velocity functions to the jet file.
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
        Update the jet for a given action of the DRL.

        This method must be implemented by child classes to update the jet
        according to their specific behaviors.

        Parameters
        ----------
        Q_pre : List[float]
            The previous jet action values.
        Q_new : List[float]
            The new jet action values.
        time_start : float
            The start time of the jet action.
        smooth_func : str
            The smoothing function type.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a child class.
        """
        raise NotImplementedError(
            "Jet.update: Must specialize the `update` method for each specific jet kind"
        )

    @abstractmethod
    def set_geometry(self, geometry_params: Dict[str, Any]) -> Any:
        """
        Set the jet geometry.

        This method must be implemented by child classes to set the jet
        geometry according to their specific requirements.

        Parameters
        ----------
        geometry_params : Dict[str, Any]
            The parameters for the jet geometry.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a child class.
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
        Create the smooth functions for the jet.

        This method must be implemented by child classes to create the smooth
        functions according to their specific requirements.

        Parameters
        ----------
        Q_new : List[float]
            The new jet action values.
        Q_pre : List[float]
            The previous jet action values.
        time_start : float
            The start time of the jet action.
        T_smoo : float
            The smoothing time parameter.
        smooth_func : str
            The smoothing function type.

        Returns
        -------
        str
            The smooth function string.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a child class.
        """
        raise NotImplementedError(
            "Jet.create_smooth_funcs: Must specialize the `create_smooth_funcs` method for each jet class"
        )


class JetCylinder(Jet):
    """
    Specialized jet class to handle jets in cylindrical coordinates.

    This class extends the `Jet` class and provides specific implementations for
    handling jets in cylindrical coordinates, including setting up geometry and
    creating smoothing functions.

    Attributes
    ----------
    radius : float
        The radius of the cylinder.
    width : float
        The width of the jet.
    theta0 : float
        The initial angle position of the jet.
    theta : str
        The angle function for the jet.
    Qs_position_z : List[float]
        Positions along the z-axis for the jet action.
    delta_Q_z : float
        Delta value for the z-axis position range.

    Methods
    -------
    __init__(name: str, params: Dict[str, Any], Q_pre: List[float] = None, Q_new: List[float] = None, time_start: float = 0.0, dimension: int = 2, T_smoo: float = 0.2, smooth_func: str = "") -> None
        Initialize the JetCylinder class with the given parameters.
    set_geometry(params: Dict[str, Any]) -> None
        Set the geometry of the jet.
    update(Q_pre: List[float], Q_new: List[float], time_start: float, smooth_func: str, *args: Any, **kwargs: Any) -> None
        Update the jet for a given action of the DRL.
    create_smooth_funcs(Q_new: List[float], Q_pre: List[float], time_start: float, T_smoo: float, smooth_func: str, *args: Any, **kwargs: Any) -> str
        Create the smooth functions for the jet in cylindrical coordinates.
    normalize_angle(angle: float) -> float
        Normalize an angle to the range [-pi, pi].
    get_theta(cylinder_coordinates: List[float]) -> str
        Get the theta function based on the cylinder coordinates.
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

        Parameters
        ----------
        name : str
            The name of the jet.
        params : Dict[str, Any]
            The parameters for the jet.
        Q_pre : List[float], optional
            The previous jet action values (default is [0.0]).
        Q_new : List[float], optional
            The new jet action values (default is [0.0]).
        time_start : float, optional
            The start time of the jet action (default is 0.0).
        dimension : int, optional
            The dimensionality of the jet (default is 2).
        T_smoo : float, optional
            The smoothing time parameter (default is 0.2).
        smooth_func : str, optional
            The smoothing function type (default is "").
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
        Set the geometry of the jet in cylindrical coordinates.

        # TODO: @pietero add documentation about specific geometry parameters imported from parameters.py - Pieter

        Parameters
        ----------
        params : Dict[str, Any]
            The parameters for the jet geometry.

        Raises
        ------
        ValueError
            If any of the geometric parameters are invalid.
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
        Update the jet for a given action of the DRL.

        This method updates the jet's velocity functions and other properties
        based on the provided action values and smoothing function.

        Parameters
        ----------
        Q_pre : List[float]
            The previous jet action values.
        Q_new : List[float]
            The new jet action values.
        time_start : float
            The start time of the jet action.
        smooth_func : str
            The smoothing function type.
        Qs_position_z : List[float]
            Positions along the z-axis for the jet action.
        delta_Q_z : float
            Delta value for the z-axis position range.

        Raises
        ------
        ValueError
            If required keyword arguments 'Qs_position_z' or 'delta_Q_z' are missing.
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
        Create the smooth functions for the jet in cylindrical coordinates.

        Parameters
        ----------
        Q_new : List[float]
            The new jet action values.
        Q_pre : List[float]
            The previous jet action values.
        time_start : float
            The start time of the jet action.
        T_smoo : float
            The smoothing time parameter.
        smooth_func : str
            The smoothing function type.
        Qs_position_z : List[float]
            Positions along the z-axis for the jet action.
        delta_Q_z : float
            Delta value for the z-axis position range.

        Returns
        -------
        str
            The smooth function string.

        Raises
        ------
        ValueError
            If required keyword arguments 'Qs_position_z' or 'delta_Q_z' are missing.

        Notes
        -----
        The smoothing function can be either 'EXPONENTIAL' or 'LINEAR'. If no
        smoothing function is specified, 'LINEAR' is used by default.
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
        Normalize an angle to the range [-pi, pi].

        Parameters
        ----------
        angle : float
            The angle to be normalized.

        Returns
        -------
        float
            The normalized angle.
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
        Get the theta function based on the cylinder coordinates.

        Parameters
        ----------
        cylinder_coordinates : List[float]
            The coordinates of the cylinder.

        Returns
        -------
        str
            The theta function string.
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
    Specialized jet class to handle jets in a channel.

    This class extends the `Jet` class and provides specific implementations
    for handling jets in channel geometries, including setting up geometry and
    creating smoothing functions.

    Attributes
    ----------
    Qs_position_x : List[float]
        Positions along the x-axis for the jet action.
    delta_Q_x : float
        Delta value for the x-axis position range.
    Qs_position_z : List[float]
        Positions along the z-axis for the jet action.
    delta_Q_z : float
        Delta value for the z-axis position range.

    Methods
    -------
    __init__(name: str, params: Dict[str, Any], Q_pre: List[float] = None, Q_new: List[float] = None, time_start: float = 0.0, dimension: int = 2, T_smoo: float = 0.2, smooth_func: str = "") -> None
        Initialize the JetChannel class with the given parameters.
    set_geometry(params: Dict[str, Any]) -> None
        Set the geometry of the jet.
    update(Q_pre: List[float], Q_new: List[float], time_start: float, smooth_func: str, *args: Any, **kwargs: Any) -> None
        Update the jet for a given action of the DRL.
    create_smooth_funcs(Q_new: List[float], Q_pre: List[float], time_start: float, T_smoo: float, smooth_func: str, **kwargs: Any) -> str
        Create the smooth functions for the jet in channel coordinates.
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

        Parameters
        ----------
        name : str
            The name of the jet.
        params : Dict[str, Any]
            The parameters for the jet.
        Q_pre : List[float], optional
            The previous jet action values (default is [0.0]).
        Q_new : List[float], optional
            The new jet action values (default is [0.0]).
        time_start : float, optional
            The start time of the jet action (default is 0.0).
        dimension : int, optional
            The dimensionality of the jet (default is 2).
        T_smoo : float, optional
            The smoothing time parameter (default is 0.2).
        smooth_func : str, optional
            The smoothing function type (default is "").
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
        Set the geometry of the jet in channel coordinates.

        Parameters
        ----------
        params : Dict[str, Any]
            The parameters for the jet geometry.

        Raises
        ------
        ValueError
            If any of the geometric parameters are invalid.
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
        Update the jet for a given action of the DRL.

        This method updates the jet's velocity functions and other properties
        based on the provided action values and smoothing function.

        Parameters
        ----------
        Q_pre : List[float]
            The previous jet action values.
        Q_new : List[float]
            The new jet action values.
        time_start : float
            The start time of the jet action.
        smooth_func : str
            The smoothing function type.
        Qs_position_x : List[float]
            Positions along the x-axis for the jet action.
        delta_Q_x : float
            Delta value for the x-axis position range.
        Qs_position_z : List[float]
            Positions along the z-axis for the jet action.
        delta_Q_z : float
            Delta value for the z-axis position range.

        Raises
        ------
        ValueError
            If required keyword arguments 'Qs_position_x', 'delta_Q_x',
            'Qs_position_z', or 'delta_Q_z' are missing.
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
            self.Vx = "0"
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
        Create the smooth functions for the jet in channel coordinates.

        Parameters
        ----------
        Q_new : List[float]
            The new jet action values.
        Q_pre : List[float]
            The previous jet action values.
        time_start : float
            The start time of the jet action.
        T_smoo : float
            The smoothing time parameter.
        smooth_func : str
            The smoothing function type.
        Qs_position_x : List[float]
            Positions along the x-axis for the jet action.
        delta_Q_x : float
            Delta value for the x-axis position range.
        Qs_position_z : List[float]
            Positions along the z-axis for the jet action.
        delta_Q_z : float
            Delta value for the z-axis position range.

        Returns
        -------
        str
            The smooth function string.

        Raises
        ------
        ValueError
            If required keyword arguments 'Qs_position_x', 'delta_Q_x',
            'Qs_position_z', or 'delta_Q_z' are missing.

        Notes
        -----
        The smoothing function can be either 'EXPONENTIAL' or 'LINEAR'. If no
        smoothing function is specified, 'LINEAR' is used by default.
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

            nz_Qs: int = len(Qs_position_x)

            # create the new Q string
            for i in range(1, self.nb_inv_per_CFD + 1):
                x_index, z_index = agent_index_1d_to_2d(i, nz_Qs)

                string_heav = heav_func_channel(
                    Qs_position_x[x_index], delta_Q_x, Qs_position_z[z_index], delta_Q_z
                )

                if i == 1:
                    string_all_Q_pre = f"{string_heav}*({Q_pre[i-1]:.4f})"
                    string_all_Q_new = f"{string_heav}*({Q_new[i-1]:.4f})"
                else:
                    string_all_Q_pre += f"+ {string_heav}*({Q_pre[i-1]:.4f})"
                    string_all_Q_new += f"+ {string_heav}*({Q_new[i-1]:.4f})"

            string_Q = f"(({string_all_Q_pre}) + ({string_h})*(({string_all_Q_new})-({string_all_Q_pre})))"
            # string_heav = heav_func_channel(
            #     Qs_position_x[0], delta_Q_x, Qs_position_z[0], delta_Q_z
            # )
            # string_all_Q_pre = f"{string_heav}*({Q_pre[0]:.4f})"
            # string_all_Q_new = f"{string_heav}*({Q_new[0]:.4f})"
            #
            # for i in range(1, self.nb_inv_per_CFD):
            #     string_heav = heav_func_channel(
            #         Qs_position_x[i], delta_Q_x, Qs_position_z[i], delta_Q_z
            #     )
            #     string_all_Q_pre += f"+ {string_heav}*({Q_pre[i]:.4f})"
            #     string_all_Q_new += f"+ {string_heav}*({Q_new[i]:.4f})"

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
            return f"({scale:.1f})*({string_Q})"
