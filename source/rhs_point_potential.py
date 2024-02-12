"""
System right hand side calculation for satellite equation of motion in point potential
"""
import numpy as np
from .exceptions import InvalidInputException


def rhs_point_potential(t, state_vector, grav_param):
    """
    Calculation of rhs for satellite equation of motion in point potential
        Args:
            t (int, float, np.int64): time
            state_vector (np.array[float]): state vector
            grav_param (int, float, np.int64): gravitational parameter

        Returns:
            (np.array[float]): vector of rhs

        Raises:
            InvalidInputException: invalid input data
    """

    if not isinstance(state_vector, np.ndarray) and len(state_vector) != 6:
        raise InvalidInputException("Input data error in state vector rhs!")

    if not isinstance(grav_param, (int, float, np.int64)):
        raise InvalidInputException("Input data error in grav_param rhs!")

    r = np.linalg.norm(state_vector[:3])**3
    x_dot = state_vector[3:]
    force = -grav_param / r * state_vector[:3]

    return np.concatenate((x_dot, force))

