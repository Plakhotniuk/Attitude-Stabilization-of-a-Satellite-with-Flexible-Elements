from scipy.spatial.transform import Rotation
import numpy as np
from .exceptions import InvalidInputException


def gravity_gradient(position_vec, inertia_matrix, grav_param):
    """
    Calculates the gravity gradient of the given position and attitude quaternion
    Args:
        position_vec: body center of mass position in inertial frame
        inertia_matrix:
        grav_param:

    Returns:

    """

    if not isinstance(position_vec, np.ndarray) or len(position_vec) != 3:
        raise InvalidInputException("Input data error in state vector !")

    if not isinstance(inertia_matrix, np.ndarray) or inertia_matrix.shape != (3, 3):
        raise InvalidInputException("Input data error in inertia_matrix_principal_axis !")

    if not isinstance(grav_param, (int, float, np.int64)):
        raise InvalidInputException("Input data error in grav_param !")

    R_0 = np.linalg.norm(position_vec)

    return 3 * grav_param / (R_0 ** 5) * np.cross(position_vec, inertia_matrix.dot(position_vec))
