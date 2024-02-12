import numpy as np
from source.orbits.orbits_conversion import rv_to_keplerMean, keplerMean_to_rv
from unittest import TestCase


def compare_angles(ang1, ang2, tolerance=1.e-12):
    return np.abs(ang1 - ang2) < tolerance or np.abs(np.abs(ang1 - ang2) - 2 * np.pi) < tolerance


class TestOrbitsConvertion(TestCase):

    def setUp(self):
        self.tolerance = 1.e-12
        # mu,x,y,z,vx,vy,vz,i,RAAN,a,e,argPer,trueAnom,meanAnom
        self.parameters_array = np.array(
            [398600441580000, 6800000, 0, 0, -0, 7656.220477302023, 0, 0, 6.283185307179586, 6799999.999999999,
             1.271045959162934e-16, 3.141592653589793, 3.141592653589793, 3.141592653589793
             ])
        self.rv = self.parameters_array[1:7]
        self.grav_param = self.parameters_array[0]

    def test_rv2kepler_mean(self):
        kepler_mean_calculated = rv_to_keplerMean(self.rv, self.grav_param)
        keplaer_mean = np.array(
            [self.parameters_array[7], self.parameters_array[8], self.parameters_array[11], self.parameters_array[9],
             self.parameters_array[10],
             self.parameters_array[13]])

        assert compare_angles(kepler_mean_calculated[0], keplaer_mean[0])
        assert compare_angles(kepler_mean_calculated[1], keplaer_mean[1])

    def test_kepler_mean2rv(self):
        self.tolerance = 1.e-8

        kepler_mean = np.array(
            [self.parameters_array[7], self.parameters_array[8], self.parameters_array[11], self.parameters_array[9], self.parameters_array[10],
             self.parameters_array[13]])

        rv_calculated = keplerMean_to_rv(kepler_mean, self.grav_param)

        assert np.allclose(self.rv, rv_calculated, atol=self.tolerance)
