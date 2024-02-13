import numpy as np
from unittest import TestCase
from source.rigid_body_attitude import rhs_rigid_body_motion
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from source.math_functions import rotate_vec_with_quat
from source.gravity_gradient import gravity_gradient_calculation


class TestRigidBodyAttitude(TestCase):
    def setUp(self):
        self.time_start = 0.
        self.time_end = 40.
        self.time_step = 1e-3

        self.times = np.arange(self.time_start, self.time_end, self.time_step)
        self.state_vector = np.zeros(10)
        self.basis_vector_x = np.array([1., 0., 0.])
        self.basis_vector_y = np.array([0., 1., 0.])
        self.basis_vector_z = np.array([0., 0., 1.])
        self.basis_vectors = np.array([[1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., 1.]])
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
        self.times_traj = data_traj[:, 0]
        self.position_traj = data_traj[:, 1:4]
        self.interpolator_traj = interp1d(x=self.times_traj, y=self.position_traj, kind='cubic', axis=0)

    def test_rotation_no_external_moments(self):
        sol_rotation = solve_ivp(fun=lambda t, x: rhs_rigid_body_motion(t=t, state_vector=x,
                                                                        inertia_matrix_principal_axis=self.inertia_matrix,
                                                                        external_moment=np.zeros(3)),
                                 t_span=(self.time_start, self.time_end),
                                 y0=self.state_vector,
                                 t_eval=self.times, rtol=1e-8,
                                 atol=1e-8, method='RK45')
        self.x_sol = sol_rotation.y.T

        quaternions = self.x_sol[:, 0:4]

        rotated_basis_vector_x = [rotate_vec_with_quat(q, self.basis_vector_x) for q in quaternions]
        rotated_basis_vector_y = [rotate_vec_with_quat(q, self.basis_vector_y) for q in quaternions]
        rotated_basis_vector_z = [rotate_vec_with_quat(q, self.basis_vector_z) for q in quaternions]

        assert np.all(np.isclose(np.linalg.norm(quaternions, axis=1), 1., rtol=1e-6, atol=1e-6))

        np.savetxt("saved_data/rotations/quaternion_norms.txt", np.linalg.norm(quaternions, axis=1), delimiter=" ")

        np.savetxt("saved_data/rotations/basis_vector_rotation.txt",
                   np.c_[rotated_basis_vector_x, rotated_basis_vector_y, rotated_basis_vector_z], delimiter=" ")

    def test_interpolation(self):

        for i in range(len(self.times_traj)):
            assert np.all(np.isclose(
                self.interpolator_traj(self.times_traj[i]),
                self.position_traj[i],
                rtol=1e-10, atol=1e-10))

    def test_rotation_gravity_moment(self):
        sol_rotation = solve_ivp(fun=lambda t, x: rhs_rigid_body_motion(t=t, state_vector=x,
                                                                        inertia_matrix_principal_axis=self.inertia_matrix,
                                                                        external_moment=gravity_gradient_calculation(
                                                                            position_vec=self.interpolator_traj(t),
                                                                            attitude_quat=x[:4],
                                                                            inertia_matrix_principal_axis=self.inertia_matrix,
                                                                            grav_param=self.grav_param)),
                                 t_span=(self.time_start, self.time_end),
                                 y0=self.state_vector,
                                 t_eval=self.times, rtol=1e-8,
                                 atol=1e-8, method='RK45')

        self.x_sol = sol_rotation.y.T

        quaternions = self.x_sol[:, 0:4]

        rotated_basis_vector_x = [rotate_vec_with_quat(q, self.basis_vector_x) for q in quaternions]
        rotated_basis_vector_y = [rotate_vec_with_quat(q, self.basis_vector_y) for q in quaternions]
        rotated_basis_vector_z = [rotate_vec_with_quat(q, self.basis_vector_z) for q in quaternions]

        assert np.all(np.isclose(np.linalg.norm(quaternions, axis=1), 1., rtol=1e-6, atol=1e-6))

        np.savetxt("saved_data/rotations/attitude_quaternions.txt", quaternions, delimiter=" ")

        np.savetxt("saved_data/rotations/basis_vector_rotation_gravity_moment.txt",
                   np.c_[rotated_basis_vector_x, rotated_basis_vector_y, rotated_basis_vector_z], delimiter=" ")

    def test_external_moment_conservation(self):
        attitude_quats = np.loadtxt("saved_data/rotations/attitude_quaternions.txt")
        gravity_moment = []
        for i in range(len(self.times)):
            gravity_moment.append(gravity_gradient_calculation(
                position_vec=self.interpolator_traj(self.times[i]),
                attitude_quat=attitude_quats[i],
                inertia_matrix_principal_axis=self.inertia_matrix,
                grav_param=self.grav_param))

        np.savetxt("saved_data/rotations/gravity_moment.txt", gravity_moment, delimiter=" ")
