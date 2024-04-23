import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib as mpl

import rdiffusion as rd
from rdiffusion import quaternions as qops
from rdiffusion import plotting as rdplot


class TestQuaternionOperations(unittest.TestCase):

    def setUp(self):
        self.rotations = R.random(10)
        self.rotmats = self.rotations.as_matrix()
        self.quats = self.rotations.as_quat()[..., (-1, 0, 1, 2)]
        self.quats_red = self.rotations.as_quat(True)[..., (-1, 0, 1, 2)]
        self.quats_inv = self.rotations.inv().as_quat()[..., (-1, 0, 1, 2)]

    def test_invert_quats(self):
        quats_inv_2test = qops.invert_quat(self.quats)
        self.assertTrue(np.allclose(self.quats_inv, quats_inv_2test))

    def test_rotmat2quat(self):
        quats_2test = qops.rotmat2quat(self.rotmats)
        self.assertTrue(np.allclose(self.quats_red, quats_2test))

    def test_quat2rotmat(self):
        rotmats_2test = qops.quat2rotmat(self.quats)
        self.assertTrue(np.allclose(self.rotmats, rotmats_2test))

    def test_reduce_quat_angle(self):
        quat_2test = np.array([-1, 2, 3, 4])
        quat_red_2test = qops.reduce_quat_angle(quat_2test)
        quats_red_2test = qops.reduce_quat_angle(self.quats)
        self.assertTrue(np.allclose(-quat_2test, quat_red_2test))
        self.assertTrue(np.allclose(self.quats_red, quats_red_2test))

    def test_multiply_quats(self):
        # Test 4 products of quaternion arrays containing 1 or N elements.
        prod_1_1_ref = self.rotations[0] * self.rotations[0]
        prod_1_N_ref = self.rotations[0] * self.rotations
        prod_N_1_ref = self.rotations * self.rotations[0]
        prod_N_N_ref = self.rotations * self.rotations

        prod_1_1_ref = prod_1_1_ref.as_quat()[..., (-1, 0, 1, 2)]
        prod_1_N_ref = prod_1_N_ref.as_quat()[..., (-1, 0, 1, 2)]
        prod_N_1_ref = prod_N_1_ref.as_quat()[..., (-1, 0, 1, 2)]
        prod_N_N_ref = prod_N_N_ref.as_quat()[..., (-1, 0, 1, 2)]

        prod_1_1_2test = qops.multiply_quats(self.quats[0], self.quats[0])
        prod_1_N_2test = qops.multiply_quats(self.quats[0], self.quats)
        prod_N_1_2test = qops.multiply_quats(self.quats, self.quats[0])
        prod_N_N_2test = qops.multiply_quats(self.quats, self.quats)

        self.assertTrue(np.allclose(prod_1_1_ref, prod_1_1_2test))
        self.assertTrue(np.allclose(prod_1_N_ref, prod_1_N_2test))
        self.assertTrue(np.allclose(prod_N_1_ref, prod_N_1_2test))
        self.assertTrue(np.allclose(prod_N_N_ref, prod_N_N_2test))

        # Test that multiplying arrays with N not equal to M elements fails.
        self.assertRaises(ValueError, qops.multiply_quats,
                          self.quats, self.quats[:-2])


class TestDiffusionFunctions(unittest.TestCase):

    def setUp(self):
        self.t = np.array([2e-2])
        self.D_diag = np.array([5, 10, 20])

        # Define PAF and apply PAF convention.
        self.PAF = R.random().as_matrix()
        self.PAF[np.diag(self.PAF) < 0] *= -1 # Elements 11 and 22 positive.
        if np.linalg.det(self.PAF) < 0:
            self.PAF[0] *= -1 # Make right-handed.

        Q_ref_diag = self.favros_Q_from_D(self.D_diag, self.t[0])
        self.Q_ref = np.diag(Q_ref_diag)[np.newaxis, ...]
        self.Q_rot_ref = np.einsum('im,tmn,nj->tij',
                                   self.PAF.T, self.Q_ref, self.PAF)

    def favros_Q_from_D(self, D_diag, t):
        # Helper method.
        D1, D2, D3 = D_diag
        D = np.average(D_diag)
        Q11 = 0.25 * (1 + np.exp(-3 * D * t) * (
                    np.exp(D1 * t) - np.exp(D2 * t) - np.exp(D3 * t)))
        Q22 = 0.25 * (1 + np.exp(-3 * D * t) * (
                    np.exp(D2 * t) - np.exp(D1 * t) - np.exp(D3 * t)))
        Q33 = 0.25 * (1 + np.exp(-3 * D * t) * (
                    np.exp(D3 * t) - np.exp(D1 * t) - np.exp(D2 * t)))
        return np.array([Q11, Q22, Q33])

    def test_arange_lag_times(self):
        nsteps, timestep = 10, 0.7
        lag_times = rd.arange_lag_times(np.zeros((nsteps, 1, 1)), timestep)
        self.assertEqual(lag_times.size, 10)
        self.assertEqual(lag_times[0], timestep)
        self.assertTrue(np.allclose(lag_times / timestep,
                                    np.arange(1, nsteps+1)))

    def test_construct_Q_model(self):
        # Q: Time-dependent quaternion covariance matrix.
        # Compare Favro's equations against own implementation.
        Q_2test = rd.construct_Q_model(self.t, self.D_diag)
        self.assertEqual(Q_2test.shape, (1, 3, 3))
        self.assertTrue(np.allclose(self.Q_ref, Q_2test))

        # Compare in different, random PAF.
        Q_rot_2test = rd.construct_Q_model(self.t, self.D_diag, self.PAF)
        self.assertEqual(Q_rot_2test.shape, (1, 3, 3))
        self.assertTrue(np.allclose(self.Q_rot_ref, Q_rot_2test))

    def test_apply_PAF_convention(self):
        # Test single PAF.
        PAF_ref = np.array([[1, -2, -3], [-4, 5, -6], [7, 8, 9]])
        PAF_2test = rd.apply_PAF_convention(-PAF_ref)
        self.assertTrue(np.allclose(PAF_ref, PAF_2test))

        # Test array of PAFs.
        PAFs_ref = np.array([PAF_ref])
        PAFs_2test = rd.apply_PAF_convention(-PAFs_ref)
        self.assertTrue(np.allclose(PAFs_ref, PAFs_2test))

    def test_instantaneous_tensor(self):
        D_inst, PAF_inst = rd.instantaneous_tensors(self.t, self.Q_rot_ref)
        self.assertTrue(np.allclose(D_inst[0], self.D_diag))
        self.assertTrue(np.allclose(PAF_inst[0], self.PAF))

    def test_convert2D_and_PAF(self):
        D, PAF = rd.convert2D_and_PAF([1, 1, 0, 0, 0])
        self.assertEqual(D, [10, 10, 10])
        self.assertTrue(np.allclose(PAF, np.eye(3)))
        pass

    def test_least_squares_fit(self):
        # Test anisotropic fit for self.D and PAF = np.eye(3).
        fit_aniso_EYE = rd.least_squares_fit(self.t, self.Q_ref)
        self.assertTrue(fit_aniso_EYE.success)
        self.assertEqual(fit_aniso_EYE.model, 'anisotropic')
        self.assertEqual(fit_aniso_EYE.shape, 'triaxial')
        self.assertTrue(np.allclose(fit_aniso_EYE.D, self.D_diag))
        self.assertTrue(np.allclose(fit_aniso_EYE.rotation_axes, np.eye(3)))

        # Test anisotropic fit in random PAF.
        fit_aniso = rd.least_squares_fit(self.t, self.Q_rot_ref)
        self.assertTrue(fit_aniso.success)
        self.assertEqual(fit_aniso.model, 'anisotropic')
        self.assertTrue(np.allclose(fit_aniso.D, self.D_diag))
        self.assertEqual(fit_aniso.rotation_axes.shape, self.PAF.shape)
        self.assertTrue(np.allclose(fit_aniso.rotation_axes, self.PAF))

        # Test semi-isotropic fit in random PAF.
        fit_semi = rd.least_squares_fit(self.t, self.Q_rot_ref,
                                        model='semi-isotropic')
        D_ref = [7.5, 7.5, 19.9375253]
        self.assertTrue(fit_semi.success)
        self.assertEqual(fit_semi.model, 'semi-isotropic')
        self.assertEqual(fit_semi.shape, 'prolate')
        self.assertTrue(np.allclose(fit_semi.D, D_ref))
        self.assertTrue(np.allclose(fit_semi.rotation_axes, self.PAF[2]),
                        [fit_semi.rotation_axes, self.PAF[2]])

        # Test semi-isotropic fit for oblate diffusion tensor.
        D_oblate = [20, 20, 5]
        Q_ref_diag = self.favros_Q_from_D(D_oblate, self.t[0])
        Q_ref = np.diag(Q_ref_diag)[np.newaxis, ...]
        fit_semi_oblate = rd.least_squares_fit(self.t, Q_ref,
                                               model='semi-isotropic')
        self.assertTrue(fit_semi_oblate.success)
        self.assertEqual(fit_semi_oblate.model, 'semi-isotropic')
        self.assertEqual(fit_semi_oblate.shape, 'oblate')
        self.assertTrue(np.allclose(fit_semi_oblate.D, D_oblate))
        self.assertTrue(np.allclose(fit_semi_oblate.rotation_axes, [0, 0, 1]))

        # Test isotropic fit in random PAF.
        fit_iso = rd.least_squares_fit(self.t, self.Q_rot_ref,
                                       model='isotropic')
        self.assertTrue(fit_iso.success)
        self.assertEqual(fit_iso.model, 'isotropic')
        self.assertEqual(fit_iso.shape, 'spherical')
        self.assertEqual(fit_iso.rotation_axes, None)
        self.assertTrue(np.allclose(fit_iso.D, 11.469530730369259))

        # Test KeyError is raised if the specified model is invalid.
        self.assertRaises(KeyError, rd.least_squares_fit,
                          self.t, self.Q_rot_ref, model='something wrong')


class TestPlottingFunctions(unittest.TestCase):

    def test_create_plot_for_Q(self):
        fig, axs = rdplot.create_plot_for_Q()
        self.assertEqual(axs.shape, (2, 3))
        self.assertIsInstance(fig, mpl.figure.Figure)


if __name__ == '__main__':
    unittest.main()
