import numpy as np

def normalize(obj):
    return obj / np.linalg.norm(obj)


def cross_product(a, b):
    def check_dimensions(vec, string):

        if vec.ndim != 1:
            raise Exception("The {} input is not a vector".format(string))
        if len(vec) != 3:
            raise Exception("Wrong number of coordinates in the {0} vector: {1}, should be 3".format(string, len(vec)))

    check_dimensions(a, 'first')
    check_dimensions(b, 'second')

    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def quat_product(q1, q2):
    def check_dimensions(q, string):

        if q.ndim != 1:
            raise Exception("The {} input is not a quaternion".format(string))
        if len(q) != 4:
            raise Exception(
                "Wrong number of coordinates in the {0} quaternion: {1}, should be 4".format(string, len(q)))

    check_dimensions(q1, 'first')
    check_dimensions(q2, 'second')

    q = np.zeros(4)
    q[0] = q1[0] * q2[0] - q1[1:].dot(q2[1:])
    q[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + cross_product(q1[1:], q2[1:])

    return q


def rotate_vec_with_quat(q, vec):
    def check_dimensions(obj, is_quat):

        if obj.ndim != 1:
            raise Exception("Not a {}".format('quaternion' * is_quat + 'vector' * (1 - is_quat)))
        if len(obj) != (3 + 1 * is_quat):
            raise Exception("Wrong number of coordinates in the {0}: {1}, should be {2}"
                            .format('quaternion' * is_quat + 'vector' * (1 - is_quat), len(obj), 3 + 1 * is_quat))

    check_dimensions(q, True)
    check_dimensions(vec, False)

    q = quat_conjugate(q)

    qxvec = cross_product(q[1:], vec)

    return q[1:].dot(vec) * q[1:] + q[0] ** 2. * vec + 2. * q[0] * qxvec + cross_product(q[1:], qxvec)


def quat2rpy(q0, q1, q2, q3):
    roll = np.arctan2(2. * (q0 * q1 + q2 * q3), 1. - 2. * (q1 ** 2 + q2 ** 2))
    pitch = np.arcsin(2. * (q0 * q2 - q1 * q3))
    yaw = np.arctan2(2. * (q0 * q3 + q1 * q2), 1. - 2. * (q2 ** 2 + q3 ** 2))

    return [roll, pitch, yaw]


def quat2rpy_deg(q0, q1, q2, q3):
    norm_q = np.linalg.norm([q0, q1, q2, q3])
    q0, q1, q2, q3 = q0 / norm_q, q1 / norm_q, q2 / norm_q, q3 / norm_q

    roll = np.arctan2(2. * (q0 * q1 + q2 * q3), 1. - 2. * (q1 ** 2 + q2 ** 2)) * 180 / np.pi
    pitch = np.arcsin(2. * (q0 * q2 - q1 * q3)) * 180 / np.pi
    yaw = np.arctan2(2. * (q0 * q3 + q1 * q2), 1. - 2. * (q2 ** 2 + q3 ** 2)) * 180 / np.pi

    return [roll, pitch, yaw]


def quat_conjugate(q):
    q_new = np.copy(q)
    q_new[1:] *= -1.

    return q_new
