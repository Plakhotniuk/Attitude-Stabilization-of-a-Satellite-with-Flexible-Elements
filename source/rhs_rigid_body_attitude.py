"""
System right hand side calculation for satellite attitude motion
"""
import numpy as np
from .exceptions import InvalidInputException
from .math_functions import cross_product


def rhs_rigid_body_motion(t, state_vector, inertia_matrix_principal_axis, external_moment=np.zeros(3),
                          control_torque=np.zeros(3)):
    """
        Calculation of rhs for satellite attitude equation
            Args:
                t (int, float, np.int64): time
                state_vector (np.array[float]): state vector [q0, q1, q2, q3, wx, wy, wz, hx, hy, hz]
                inertia_matrix_principal_axis (np.array[float]): diagonal matrix 3x3
                external_moment (np.array[float]): external moment
                control_torque (np.array[float]): control_torque

            Returns:
                (np.array[float]): vector of rhs

            Raises:
                InvalidInputException: invalid input data
    """

    if not isinstance(state_vector, np.ndarray) or len(state_vector) != 10:
        raise InvalidInputException("Input data error in state vector rhs!")

    if not isinstance(external_moment, np.ndarray) or len(external_moment) != 3:
        raise InvalidInputException("Input data error in external moment rhs!")

    if not isinstance(inertia_matrix_principal_axis, np.ndarray) or inertia_matrix_principal_axis.shape != (3, 3):
        raise InvalidInputException("Input data error in inertia_matrix_principal_axis rhs!")

    quat = state_vector[:4] / np.linalg.norm(state_vector[:4])
    omega = state_vector[4:7]
    h = state_vector[7:]

    x_dot = np.zeros(10)
    x_dot[0] = -0.5 * quat[1:].dot(omega)
    x_dot[1:4] = 0.5 * (quat[0] * omega + cross_product(quat[1:], omega))
    x_dot[4:7] = np.linalg.inv(inertia_matrix_principal_axis).dot(
        external_moment + control_torque - cross_product(omega, inertia_matrix_principal_axis.dot(omega) + h))
    x_dot[7:] = -control_torque

    return x_dot
