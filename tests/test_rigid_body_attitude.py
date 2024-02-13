import numpy as np
from unittest import TestCase
from source.rigid_body_attitude import rhs_rigid_body_motion
from scipy.integrate import solve_ivp
from source.math_functions import rotate_vec_with_quat
from source.gravity_gradient import gravity_gradient_calculation


class TestRigidBodyAttitude(TestCase):
    def setUp(self):
        self.times = np.arange(self.time_start, self.time_end, self.time_step)
        self.state_vector = np.zeros(10)
        self.basis_vector_x = np.array([1., 0., 0.])
        self.state_vector[:4] = np.array([1., 0., 0., 0])  # attitude quaternion
        self.state_vector[4:7] = np.array([0., 0., 1.])  # angular velocity vector
        self.inertia_matrix = np.array([[1., 0., 0.],
                                        [0., 1., 0.],
                                        [0., 0., 1.]])

        self.G = 6.6742e-11
        self.m_earth = 5.974e24
        self.m_satellite = 1.
        self.grav_param = self.G * (self.m_satellite + self.m_earth)
        self.external_moment = np.array([0., 0., 0.])
        data_traj = np.loadtxt("../tests/saved_data/point_potential/point_potential_traj.txt")
        self.times = data_traj[:, 0]
        self.traj = data_traj[:, 1:4]
        self.time_step = 1.e-3
        self.time_start = 0.
        self.time_end = 10.

    def test_rotation_no_external_moments(self):
        sol_rotation = solve_ivp(fun=lambda t, x: rhs_rigid_body_motion(t, x, self.inertia_matrix, external_moment=gravity_gradient_calculation(x)),
                        t_span=(self.time_start, self.time_end),
                        y0=self.state_vector,
                        t_eval=self.times, rtol=1e-8,
                        atol=1e-8, method='RK45')
        self.x_sol = sol_rotation.y.T

        quaternions = self.x_sol[:, 0:4]

        rotated_basis_vector_x = [rotate_vec_with_quat(q, self.basis_vector_x) for q in quaternions]

        assert np.all(np.isclose(np.linalg.norm(quaternions, axis=1), 1., rtol=1e-6, atol=1e-6))

        np.savetxt("saved_data/rotations/quaternion_norms.txt", np.linalg.norm(quaternions, axis=1), delimiter=" ")

        np.savetxt("saved_data/rotations/basis_vector_rotation.txt", rotated_basis_vector_x, delimiter=" ")

