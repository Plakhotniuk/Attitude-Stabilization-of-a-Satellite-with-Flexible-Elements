import numpy as np
from source.exceptions import InvalidInputException
from scipy.optimize import newton


def rv_to_keplerMean(rv, grav_param):
    if isinstance(rv, np.ndarray) and len(rv) != 6:
        raise InvalidInputException("Input data error!")

    position = rv[:3]
    velocity = rv[3:]

    # Вектор орбитального момента L
    L = np.cross(position, velocity)

    # Наклонение орбиты
    inclination = np.arctan2(np.sqrt(L[0] * L[0] + L[1] * L[1]), L[2])

    z = np.array([0, 0, 1])
    x = np.array([1, 0, 0])
    z_cross_L = np.cross(z, L)
    lNorm = np.linalg.norm(z_cross_L)

    # Направление на восходящий узел
    N = z_cross_L / lNorm if lNorm > 0 else x

    # Долгота восходящего узла
    ascending_node_longitude = np.arctan2(N[1], N[0])

    # Векторный эксцентриситет
    eccentricity_vec = ((np.dot(velocity, velocity) - grav_param / np.linalg.norm(position)) * position - np.dot(
        position, velocity) * velocity) / grav_param

    # Эксцентриситет
    eccentricity = np.linalg.norm(eccentricity_vec)

    # Для исключения вырожение вводится вектор е_1
    e_1 = eccentricity_vec if np.linalg.norm(eccentricity_vec) > 0 else N

    # Вводим базис в плоскости орбиты d_1, d_2
    d_1 = N
    d_2 = np.cross(L, N) / np.linalg.norm(L)

    # Аргумент перицентра
    periapsis_argument = np.arctan2(np.dot(d_2, e_1), np.dot(d_1, e_1))

    # Главная полуось орбиты a из интеграла энергии
    semi_major_axis = grav_param / (2 * grav_param / np.linalg.norm(position) - np.dot(velocity, velocity))

    e_2 = np.cross(L, e_1) / np.linalg.norm(L)

    # Истинная аномалия
    true_anomaly = np.arctan2(np.dot(e_2, position), np.dot(e_1, position))

    # Эксцентрическая аномалия
    E = np.arctan2(np.sin(true_anomaly) * np.sqrt(1. - eccentricity * eccentricity), eccentricity + np.cos(
        true_anomaly))

    # Средняя аномалия
    mean_anomaly = E - eccentricity * np.sin(E)

    return np.array(
        [inclination, ascending_node_longitude, periapsis_argument, semi_major_axis, eccentricity, mean_anomaly])


def keplerMean_to_rv(kepler_mean_params, grav_param, tolerance=1.e-15):
    """
        Conversion kepler mean to rv
            Args:
                kepler_mean_params (np.array[float]): [inclination, ascendingNodeLongitude, periapsisArgument,
                                                        semiMajorAxis, eccentricity, meanAnomaly]
                grav_param (int, float, np.int64): gravitational parameter

                tolerance (int, float, np.int64): calculation tolerance


            Returns:
                (np.array[float]): vector rv

            Raises:
                InvalidInputException: invalid input data
        """
    if isinstance(kepler_mean_params, np.ndarray) and len(kepler_mean_params) != 6:
        raise InvalidInputException("Input data error!")

    # kepler mean -> kepler true -> rv
    E = kepler_mean_params[5] - kepler_mean_params[4]

    calc_func = lambda x: (x - kepler_mean_params[4] * np.sin(x) - kepler_mean_params[5]) / (
            1 - kepler_mean_params[4] * np.cos(x))

    E = newton(func=calc_func, x0=E, tol=tolerance)

    true_anomaly = np.arctan2(np.sin(E) * np.sqrt(1 - kepler_mean_params[4] * kepler_mean_params[4]),
                              np.cos(E) - kepler_mean_params[4])

    cosTrueAnomaly = np.cos(true_anomaly)
    sinTrueAnomaly = np.sin(true_anomaly)
    cosLongitOfAscendingNode = np.cos(kepler_mean_params[1])
    sinLongitOfAscendingNode = np.sin(kepler_mean_params[1])
    cosPericenterArg = np.cos(kepler_mean_params[2])
    sinPericenterArg = np.sin(kepler_mean_params[2])
    cosInclination = np.cos(kepler_mean_params[0])
    sinInclination = np.sin(kepler_mean_params[0])
    orbParam = kepler_mean_params[3] * (1. - kepler_mean_params[4] * kepler_mean_params[4])

    # Some auxiliary multipliers
    multiplier1 = orbParam / (1. + kepler_mean_params[4] * cosTrueAnomaly)
    multiplier2 = np.sqrt(grav_param / orbParam)

    positionTmp = np.array([multiplier1 * cosTrueAnomaly, multiplier1 * sinTrueAnomaly, 0.])
    velocityTmp = np.array([-multiplier2 * sinTrueAnomaly, multiplier2 * (kepler_mean_params[4] + cosTrueAnomaly), 0.])

    rotationMatrix = np.array(
        [[cosLongitOfAscendingNode * cosPericenterArg - sinLongitOfAscendingNode * sinPericenterArg * cosInclination,
          -cosLongitOfAscendingNode * sinPericenterArg - sinLongitOfAscendingNode * cosPericenterArg * cosInclination,
          sinLongitOfAscendingNode * sinInclination],
         [sinLongitOfAscendingNode * cosPericenterArg + cosLongitOfAscendingNode * sinPericenterArg * cosInclination,
          -sinLongitOfAscendingNode * sinPericenterArg + cosLongitOfAscendingNode * cosPericenterArg * cosInclination,
          -cosLongitOfAscendingNode * sinInclination],
         [sinPericenterArg * sinInclination,
          cosPericenterArg * sinInclination,
          cosInclination]])

    return np.concatenate((np.dot(rotationMatrix, positionTmp), np.dot(rotationMatrix, velocityTmp)))
