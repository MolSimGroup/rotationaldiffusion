"""
This module provides a function for the time-dependent analysis of
rotational diffusion parameters.
"""
import numpy as np
from .utils import apply_PCS_convention


def td_parameters(lag_times, Q_data):
    """Extract time-dependent rotational diffusion parameters from
    rotational correlation functions.

    Parameters
    ----------
    lag_times : (N,) array_like
        Discrete lag times at which the rotational correlation functions
        were computed.
    Q_data : (..., N, 3, 3) array_like
        Symmetric quaternion covariance matrices containing six
        rotational correlation functions. The matrix elements are stored
        along the last two dimensions.

    Returns
    -------
    D_time_dep : (..., N, 3) ndarray
        The time-dependent diffusion coefficients in the principal
        coordinate system (PCS) in increasing order. The units are
        inverse time, matching the unit of the input ``lag_times``.
    PCS_time_dep : (..., N, 3, 3) ndarray
        The corresponding time-dependent principal coordinate system.
        The row vectors of ``PCS_time_dep`` are the time-dependent
        principal axes. By convention, the PCS is right-handed and
        the :math:`xx` and :math:`yy` coordinates are positive.

    Notes
    -----
    **Theoretical Background**

    First, the time-dependent principal component system is found by
    solving the matrix-eigenvalue equation of ``Q_data`` at each lag
    time. Then, the eigenvalues are used to compute the time-dependent
    diffusion coefficients.
    """
    # Diagonalise Q_data at every lag_time.
    Q_diag, PCS_time_dep = np.linalg.eigh(Q_data)
    PCS_time_dep = apply_PCS_convention(np.swapaxes(PCS_time_dep, -2, -1))

    # Use Favro's equations to extract D_time_dep from Q_diag (1960).
    exponentials = np.array([
        1 - 2 * Q_diag[..., 1] - 2 * Q_diag[..., 2],
        1 - 2 * Q_diag[..., 2] - 2 * Q_diag[..., 0],
        1 - 2 * Q_diag[..., 0] - 2 * Q_diag[..., 1]])
    D_time_dep = np.zeros(exponentials.shape)
    for i, j, k in zip((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        D_time_dep[i] = - np.log(exponentials[j] * exponentials[k]
                                 / exponentials[i]) / (2 * lag_times)
    return np.moveaxis(D_time_dep, 0, -1), PCS_time_dep
