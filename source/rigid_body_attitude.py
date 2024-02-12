"""
System right hand side calculation for satellite attitude motion
"""
import numpy as np
from .exceptions import InvalidInputException
from gravity_gradient import gravity_gradient_calculation
import math_functions


def rhs_rigid_body_motion(t, state_vector, position, inertia_matrix_principal_axis, grav_param):
    """
        Calculation of rhs for satellite attitude equation
            Args:
                t (int, float, np.int64): time
                state_vector (np.array[float]): state vector [q0, q1, q2, q3, wx, wy, wz, hx, hy, hz]
                position (np.array[float]): position vector
                inertia_matrix_principal_axis (np.array[float]): diagonal matrix 3x3
                grav_param (int, float, np.int64): gravitational parameter

            Returns:
                (np.array[float]): vector of rhs

            Raises:
                InvalidInputException: invalid input data
    """

    if not isinstance(state_vector, np.ndarray) and len(state_vector) != 10:
        raise InvalidInputException("Input data error in state vector rhs!")

    if not isinstance(position, np.ndarray) and len(position) != 3:
        raise InvalidInputException("Input data error in state vector rhs!")

    if not isinstance(inertia_matrix_principal_axis, np.ndarray) and inertia_matrix_principal_axis.shape != (3, 3):
        raise InvalidInputException("Input data error in inertia_matrix_principal_axis rhs!")

    quat = state_vector[:4] / np.linalg.norm(state_vector[:4])
    omega = state_vector[4:7]
    h = state_vector[7:]
    gravity_gradient = gravity_gradient_calculation(position_vec=position, attitude_quat=quat,
                                                    inertia_matrix_principal_axis=inertia_matrix_principal_axis,
                                                    grav_param=grav_param)

    x_dot = np.zeros(10)
    x_dot[0] = -0.5 * quat[1:].dot(omega)
    x_dot[1:4] = 0.5 * (quat[0] * omega + math_functions.cross_product(quat[1:], omega))
    x_dot[4:7] = np.linalg.inv(inertia_matrix_principal_axis).dot(
        gravity_gradient - math_functions.cross_product(omega, inertia_matrix_principal_axis.dot(omega) + h))
    x_dot[7:] = -gravity_gradient

    return x_dot
