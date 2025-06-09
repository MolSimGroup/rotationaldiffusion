"""
Theoretical models for rotational diffusion analysis.

This module implements
"""
import numpy as np


def quaternion_covariance_matrix(lag_times, diffusion_coeffs, principal_axes=np.eye(3)):
    """
    :math:`3\times 3`(sub-)quaternion covariance matrix of an ideal
    Brownian rotor with diffusion tensor :math:`D`.

    Utilizes Equations 2-4 from Favro (1960) as described by Linke et al. (2017).

    Parameters
    ----------
    lag_times: (N,) ndarray°
        Discrete lag times at which to compute the quaternion covariance
        matrix.
    diffusion_coeffs: (3,) ndarray
        Diffusion coefficients.

    Returns
    -------
    Q: (N, 3, 3) ndarray
        Quaternion covariance matrix of D.
    """
    Q = np.zeros((3, 3, len(lag_times)))
    for i, j, k in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        Q[i, i] = 1 / 4 * (1
           + np.exp(-(diffusion_coeffs[j] + diffusion_coeffs[k]) * lag_times)
           - np.exp(-(diffusion_coeffs[i] + diffusion_coeffs[j]) * lag_times)
           - np.exp(-(diffusion_coeffs[i] + diffusion_coeffs[k]) * lag_times))
    if not np.allclose(principal_axes - np.eye(3), np.zeros((3, 3))):
        Q = np.tensordot(np.tensordot(principal_axes, principal_axes, 0), Q,
                         axes=((0, 2), (0, 1)))
    return np.moveaxis(Q, -1, 0)


def variance_matrix(lag_times, diffusion_coeffs, principal_axes=np.eye(3),
                    precomputed_Q_model=None):
    """
    Variances of quaternion component products of an ideal Brownian
    rotor with diffusion tensor :math:`D`.

    Utilizes Equations 2-4, 6.3, and 6.4 from Favro (1960) as described by Linke et al. (2017).

    Parameters
    ----------
    lag_times: (N,) ndarray
        Discrete lag times at which to compute the variance matrix.
    diffusion_coeffs: (3,) ndarray
        Trace of a diffusion tensor in its diagonal form / principal axis frame.

    Returns
    -------
    V: (N, 3, 3) ndarray
        Analytical variances of the quaternion products :math:`q_i q_j`.
    """
    # Use Var(q_i * q_j) = <q_i^2 * q_j^2> - <q_i * q_j>^2 =: qi2_qj2 - qij2.
    # Helpers:
    D_av = np.mean(diffusion_coeffs)
    Dx, Dy, Dz = diffusion_coeffs
    delta = np.sqrt(Dx**2 + Dy**2 + Dz**2 - Dx*Dy - Dx*Dz - Dy*Dz)
    delta = 0 if np.isnan(delta) else delta

    # Start with <q_i^2 * q_j^2> =: qi2_qj2
    qi2_qj2 = np.zeros((3, 3, len(lag_times)))
    for ndx in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        i, j, k = ndx
        Dx, Dy, Dz = diffusion_coeffs[ndx]

        # Diagonal elements qi2_qi2 using Eq. 6.4 from Favro (1960).
        qi2_qj2[i, i] += 1 / 8 * (
                1 + 3 / 2 * np.exp(-3 * D_av * lag_times) * (
                np.exp(Dx * lag_times)
                - np.exp(Dy * lag_times)
                - np.exp(Dz * lag_times))
                + 1 / 2 * np.exp(-3 * D_av * lag_times) * (
                np.exp(-3 * Dx * lag_times)
                - np.exp(-3 * Dy * lag_times)
                - np.exp(-3 * Dz * lag_times))
                + np.exp(-6 * D_av * lag_times) * np.cosh(2 * lag_times * delta))

        # Off-diagonal elements qi2_qj2 using Eq. 6.3 from Favro (1960).
        # Catch zero-division in isotropic case (delta=0) by l'Hospital.
        if np.isclose(delta, 1e-12):
            lHospital = 2 * lag_times * (Dz - D_av)
        else:
            lHospital = (1 / delta * (Dz - D_av)
                         * np.sinh(2 * lag_times * delta))

        qi2_qj2[i, j] = qi2_qj2[j, i] = 1 / 8 * (
                1 / 3 - 1 / 2 * np.exp(-3 * D_av * lag_times) * (
                np.exp(Dz * lag_times) - np.exp(-3 * Dz * lag_times)
                )
                + np.exp(-6 * D_av * lag_times) * (
                lHospital - 1 / 3 * np.cosh(2 * lag_times * delta)
                )
            )

    # Rotate qi2_qj2 into reference frame.
    if not np.allclose(principal_axes - np.eye(3), np.zeros((3, 3))):
        PAF_2nd_power_shifted = principal_axes * principal_axes[(1, 2, 0),]
        PAF_4th_power_shifted = np.multiply(
            PAF_2nd_power_shifted[:, np.newaxis, :],
            PAF_2nd_power_shifted[:, :, np.newaxis]).T

        qi2_qj2_rotated = np.tensordot(
            np.tensordot(principal_axes ** 2, principal_axes ** 2, axes=0),
            qi2_qj2, axes=((0, 2), (0, 1)))
        qi2_qj2_rotated += 4 * np.matmul(PAF_4th_power_shifted,
                                         qi2_qj2[(0, 1, 0), (1, 2, 2)])
    else:
        qi2_qj2_rotated = qi2_qj2

    if precomputed_Q_model is not None:
        var = np.moveaxis(qi2_qj2_rotated, -1, 0) - precomputed_Q_model ** 2
    else:
        var = (np.moveaxis(qi2_qj2_rotated, -1, 0)
               - quaternion_covariance_matrix(lag_times, diffusion_coeffs, principal_axes) ** 2)
    return var
