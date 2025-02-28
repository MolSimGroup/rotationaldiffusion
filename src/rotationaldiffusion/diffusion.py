#! /usr/bin/env python

import numpy as np

try:
    from pydiffusion import quaternionsimulation as qsim
except ImportError:
    qsim = None

"""
Quaternions describe rotations in an axis-angle representation. A
quaternion consists of one real part and three imaginary parts, so
q = w + x*i + y*j + z*k. The rotation angle is given by 
Re(q) = w = sin(phi/2), whereas the rotation axis is given by
Im(q) = x*i + y*j + z*k = cos(phi/2) * (u1*i + u2*j + u3*j), where
"""

# TODO: Reimplement option to use commandline.


def arange_lag_times(Q, timestep):
    return np.arange(1, Q.shape[-3] + 1, 1) * timestep


def apply_PAF_convention(PAFs):
    assert PAFs.shape[-2:] == (3, 3)
    PAFs = np.copy(PAFs)

    # Enforce positive elements 11 and 22.
    PAFs[np.diagonal(PAFs, axis1=-2, axis2=-1) < 0] *= -1

    # Make PAFs right-handed (by inverting x-axis).
    if PAFs.ndim > 2:
        PAFs[np.linalg.det(PAFs) < 0, 0] *= -1
    elif np.linalg.det(PAFs) < 0:
        PAFs[0] *= -1
    return PAFs


def instantaneous_tensors(lag_times, Q_data):
    # Diagonalise Q_data at every lag_time.
    Q_diag, PAFs = np.linalg.eigh(Q_data)
    PAFs = apply_PAF_convention(np.swapaxes(PAFs, -2, -1))

    # Use Favro's equations to extract D_diag_inst from Q_diag (1960).
    exponentials = np.array([
        1 - 2 * Q_diag[..., 1] - 2 * Q_diag[..., 2],
        1 - 2 * Q_diag[..., 2] - 2 * Q_diag[..., 0],
        1 - 2 * Q_diag[..., 0] - 2 * Q_diag[..., 1]])
    D_diag_inst = np.zeros(exponentials.shape)
    for i, j, k in zip((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        D_diag_inst[i] = - np.log(exponentials[j] * exponentials[k]
                                 / exponentials[i]) / (2 * lag_times)
    return np.moveaxis(D_diag_inst, 0, -1), PAFs


