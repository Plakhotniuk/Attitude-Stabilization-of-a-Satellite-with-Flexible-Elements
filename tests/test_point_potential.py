import numpy as np
from source.rhs_point_potential import rhs_point_potential
from source.orbits.orbits_conversion import keplerMean_to_rv
from scipy.integrate import solve_ivp
from unittest import TestCase


class TestPointPotentialMotion(TestCase):

    def setUp(self):
        self.time_step = 100.
        self.time_start = 0.
        self.time_end = 40000.
        self.times = np.arange(self.time_start, self.time_end, self.time_step)

        self.G = 6.6742e-11
        self.m_earth = 5.974e24
        self.m_satellite = 1.
        self.mu = self.G * (self.m_satellite + self.m_earth)
        self.semimajor = 7000e3
        self.ecc = 0
        self.incl = 0.1
        self.raan = 0.1
        self.argPerig = 0.1
        self.mean_anomaly = 0.1
        self.kepler_mean_params = np.array(
            [self.incl, self.raan, self.argPerig, self.semimajor, self.ecc, self.mean_anomaly])
        self.rv_0 = keplerMean_to_rv(self.kepler_mean_params, self.mu)
        sol = solve_ivp(fun=lambda t, x: rhs_point_potential(t, x, self.mu), t_span=(self.time_start, self.time_end),
                        y0=self.rv_0,
                        t_eval=self.times, rtol=1e-10,
                        atol=1e-10, method='DOP853')
        self.x_sol = sol.y.T
        self.r = self.x_sol[:, :3]
        self.r_dot = self.x_sol[:, 3:6]

    def test_angular_momentum(self):
        angular_momentum = np.cross(self.r, self.r_dot)
        assert np.all(np.isclose(angular_momentum, angular_momentum[0], rtol=1e-9, atol=1e-9))
        np.savetxt("saved_data/point_potential/point_potential_angular_momentum.txt", angular_momentum, delimiter=" ")

    def test_energy(self):
        # Energy
        r_norms = np.sqrt(np.sum(self.r**2, axis=1))
        energy = np.sum(self.r_dot ** 2, axis=1) - 2 * self.mu / r_norms
        assert np.all(np.isclose(energy, energy[0], rtol=1e-8, atol=1e-10))
        np.savetxt("saved_data/point_potential/point_potential_traj_energy.txt", energy, delimiter=" ")

    def laplace_runge_lenz_vector(self, r, v, mu):
        h = np.cross(r, v)  # Specific angular momentum vector
        r_magnitude = np.linalg.norm(r)
        A = (np.cross(v, h) / mu) - (r / r_magnitude)
        return A

    def test_laplace_vector(self):
        # Laplace integral
        laplace_vec = [self.laplace_runge_lenz_vector(self.r[i], self.r_dot[i], self.mu) for i in range(len(self.times))]
        assert np.all(np.isclose(laplace_vec, laplace_vec[0], rtol=1e-9, atol=1e-9))
        np.savetxt("saved_data/point_potential/point_potential_traj_laplace.txt", laplace_vec, delimiter=" ")

    def test_save_trajectory(self):
        trajectory = np.c_[self.times, self.x_sol]
        np.savetxt("saved_data/point_potential/point_potential_traj.txt", trajectory, delimiter=" ")
