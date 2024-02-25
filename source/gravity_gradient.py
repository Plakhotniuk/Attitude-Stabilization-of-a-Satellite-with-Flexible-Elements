from scipy.spatial.transform import Rotation
import numpy as np
from .exceptions import InvalidInputException


def gravity_gradient(position_vec, attitude_quat, inertia_matrix_principal_axis, grav_param):
    """
    Calculates the gravity gradient of the given position and attitude quaternion
    Args:
        position_vec: body center of mass position in inertial frame
        attitude_quat: attitude quaternion of rigid body in inertial frame
        inertia_matrix_principal_axis:
        grav_param:

    Returns:

    """

    if not isinstance(position_vec, np.ndarray) or len(position_vec) != 3:
        raise InvalidInputException("Input data error in state vector !")

    if not isinstance(attitude_quat, np.ndarray) or len(attitude_quat) != 4:
        raise InvalidInputException("Input data error in state attitude quaternion !")

    if not isinstance(inertia_matrix_principal_axis, np.ndarray) or inertia_matrix_principal_axis.shape != (3, 3):
        raise InvalidInputException("Input data error in inertia_matrix_principal_axis !")

    if not isinstance(grav_param, (int, float, np.int64)):
        raise InvalidInputException("Input data error in grav_param !")

    rotation = Rotation.from_quat(attitude_quat)

    rotation_matrix_to_body_frame = rotation.as_matrix()
    rotation_matrix = np.linalg.inv(rotation_matrix_to_body_frame)
    inertia_matrix_inertial_axis = rotation_matrix.transpose().dot(inertia_matrix_principal_axis.dot(rotation_matrix))

    R_0 = np.linalg.norm(position_vec)

    return 3 * grav_param / (R_0 ** 5) * np.cross(position_vec, inertia_matrix_inertial_axis.dot(position_vec))
