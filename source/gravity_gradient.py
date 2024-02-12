from scipy.spatial.transform import Rotation
import numpy as np
from .exceptions import InvalidInputException


def gravity_gradient_calculation(position_vec, attitude_quat, inertia_matrix_principal_axis, grav_param):

    if not isinstance(position_vec, np.ndarray) and len(position_vec) != 3:
        raise InvalidInputException("Input data error in state vector !")

    if not isinstance(attitude_quat, np.ndarray) and len(attitude_quat) != 4:
        raise InvalidInputException("Input data error in state attitude quaternion !")

    if not isinstance(inertia_matrix_principal_axis, np.ndarray) and inertia_matrix_principal_axis.shape != (3, 3):
        raise InvalidInputException("Input data error in inertia_matrix_principal_axis !")

    if not isinstance(grav_param, (int, float, np.int64)):
        raise InvalidInputException("Input data error in grav_param !")

    rotation = Rotation.from_quat(attitude_quat)
    rotation = rotation.as_euler('xyz', degrees=False)
    ksi = rotation[0]
    etta = rotation[1]
    phi = rotation[2]

    R_0 = np.linalg.norm(position_vec)
    Ix = inertia_matrix_principal_axis[0, 0]
    Iy = inertia_matrix_principal_axis[1, 1]
    Iz = inertia_matrix_principal_axis[2, 2]

    Gx = (Iz - Iy) * np.sin(2 * phi) * np.cos(etta)**2
    Gy = (Iz - Ix) * np.sin(2 * etta) * np.cos(phi)
    Gz = (Ix - Iy) * np.sin(2 * etta) * np.sin(phi)

    return np.array([Gx, Gy, Gz]) * 3 * grav_param / (2 * R_0)

