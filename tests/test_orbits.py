import numpy as np
import pytest
from source.orbits.orbits_conversion import rv_to_keplerMean, keplerMean_to_rv


def compare_angles(ang1, ang2, tolerance=1.e-12):
    temp = np.abs(np.abs(ang1 - ang2) - 2 * np.pi)
    return np.abs(ang1 - ang2) < tolerance or np.abs(np.abs(ang1 - ang2) - 2 * np.pi) < tolerance


def test_rv2kepler_mean():
    tolerance = 1.e-12
    # mu,x,y,z,vx,vy,vz,i,RAAN,a,e,argPer,trueAnom,meanAnom
    parameters_array = np.array(
        [398600441580000, 6800000, 0, 0, -0, 7656.220477302023, 0, 0, 6.283185307179586, 6799999.999999999,
         1.271045959162934e-16, 3.141592653589793, 3.141592653589793, 3.141592653589793
         ])
    rv = parameters_array[1:7]
    grav_param = parameters_array[0]
    kepler_mean_calculated = rv_to_keplerMean(rv, grav_param)
    keplaer_mean = np.array(
        [parameters_array[7], parameters_array[8], parameters_array[11], parameters_array[9], parameters_array[10],
         parameters_array[13]])

    assert compare_angles(kepler_mean_calculated[0], keplaer_mean[0])
    assert compare_angles(kepler_mean_calculated[1], keplaer_mean[1])


def test_kepler_mean2rv():
    tolerance = 1.e-8
    # mu,x,y,z,vx,vy,vz,i,RAAN,a,e,argPer,trueAnom,meanAnom
    parameters_array = np.array(
        [398600441580000, 6800000, 0, 0, -0, 7656.220477302023, 0, 0, 6.283185307179586, 6799999.999999999,
         1.271045959162934e-16, 3.141592653589793, 3.141592653589793, 3.141592653589793
         ])
    rv = parameters_array[1:7]
    grav_param = parameters_array[0]
    keplaer_mean = np.array(
        [parameters_array[7], parameters_array[8], parameters_array[11], parameters_array[9], parameters_array[10],
         parameters_array[13]])

    rv_calculated = keplerMean_to_rv(keplaer_mean, grav_param)

    assert np.allclose(rv, rv_calculated, atol=tolerance)
