"""
This module provides auxiliary functions for operations on unit
quaternions. Each quaternion is represented by a numpy array with four
elements, :math:`(w, x, y, z)`, where :math:`w` is the scalar (real)
part and :math:`(x, y, z)` is the vector (imaginary) part. All functions
can operate on arrays of quaternions.

This is an auxiliary module. Hence, the functions are not imported into
the main name space of RotationalDiffusion. To access them, we recommend
importing this module as follows:

>>> from RotationalDiffusion import quaternions as qops
"""
import numpy as np


def conjugate(quats):
    """Complex conjugation of quaternions.

    For unit quaternions :math:`q`, the complex conjugate is also the
    inverse :math:`q^{-1}`.

    Parameters
    ----------
    quats : (..., 4) array_like
        Quaternions to conjugate.

    Returns
    -------
    quats_conjugated : (..., 4) ndarray
        Conjugated quaternions.
    """
    return np.array(quats) * [1, -1, -1, -1]


def rotmat2quat(rotmats):
    """Convert 3D rotation matrices to their quaternion representation.

    Implementation of the Bar-Itzhack
    algorithm,\ :footcite:ps:`bar-itzhack2000,wiki-bar-itzhack` which
    determines the optimal unit quaternion corresponding to the
    orthogonal rotation matrix closest to the input matrix.

    Parameters
    ----------
    rotmats : (..., 3, 3) array_like
        Rotation matrices to be converted to quaternions.

    Returns
    -------
    quats : (..., 4) ndarray
        Quaternions.

    References
    ----------
    .. footbibliography::
    """
    # Initialize matrix K3 (see [1, 2]).
    # Matrix indices are row major here, but column major on Wikipedia.
    Q = np.moveaxis(rotmats, [-2, -1], [0, 1])

    Qxx, Qxy, Qxz = Q[0]
    Qyx, Qyy, Qyz = Q[1]
    Qzx, Qzy, Qzz = Q[2]

    K3 = 1/3 * np.array([
        [Qxx - Qyy - Qzz, Qxy + Qyx, Qxz + Qzx, Qzy - Qyz],
        [Qxy + Qyx, Qyy - Qxx - Qzz, Qyz + Qzy, Qxz - Qzx],
        [Qxz + Qzx, Qyz + Qzy, Qzz - Qxx - Qyy, Qyx - Qxy],
        [Qzy - Qyz, Qxz - Qzx, Qyx - Qxy, Qxx + Qyy + Qzz]
    ])

    # Solve eigenvalue problem.
    K3 = np.moveaxis(K3, [0, 1], [-2, -1])
    eigvals, eigvecs = np.linalg.eigh(K3)

    # Select eigenvector with largest eigenvalue (close to 1).
    quats = eigvecs[..., -1]

    # Swap order from (x, y, z, w) to (w, x, y, z).
    quats = quats[..., [3, 0, 1, 2]]

    # Prefer quaternions with rotation angles <= pi.
    quats = limit_angle(quats)
    return quats


def quat2rotmat(quats):
    """
    Convert unit quaternions to their rotational matrix representation.

    Parameters
    ----------
    quats : (..., 4) array_like
        Quaternions to be converted to rotational matrices.

    Returns
    -------
    rotmats : (..., 3, 3) ndarray
        Rotational matrices.
    """
    quats = np.moveaxis(quats, -1, 0)
    w, x, y, z = quats

    rotmats = np.moveaxis(np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ]), [0, 1], [-2, -1])
    return rotmats


def limit_angle(quats):
    """
    Limit the rotation angle to the interval :math:`[0, \pi]`.

    Describing rotations with quaternions is ambiguous, because two
    quaternions :math:`q` and :math:`-q` describe the same rotation in
    clockwise and counter-clockwise direction, respectively. To remove
    this ambiguity, the rotation angle is limited by convention to the
    interval :math:`[0, \pi]`. In other words, if the angle of an input
    quaternion :math:`q` is beyond this interval, i.e., it is is in
    :math:`(\pi, 2\pi)`, then :math:`-q` is returned. Otherwise, if
    the angle is already in :math:`[0, \pi]`, then :math:`q` is returned
    without modification.

    Parameters
    ----------
    quats : (..., 4) ndarray
        Quaternions.

    Returns
    -------
    quats_limited : (..., 4) ndarray
        Quaternions with rotation angles limited to :math:`[0, \pi]`.
    """
    quats_limited = np.copy(quats)
    quats_limited[quats_limited[..., 0] < 0] *= -1
    return quats_limited


def multiply(q1, q2):
    """
    Compute the Hamilton product (of arrays) of quaternions :math:`q_1`
    and :math:`q_2`.

    The Hamilton product\ :footcite:`wiki-hamilton-product`
    of two quaternions
    :math:`q_1 = w_1 + x_1 \cdot i + y_1 \cdot j + z_1 \cdot k` and
    :math:`q_2 = w_2 + x_2 \cdot i + y_2 \cdot j + z_2 \cdot k` is

    .. math::

        q_1 \cdot q_2 \\quad = \\quad
              &(w_1 \cdot w_2 - x_1 \cdot x_2 - y_1 \cdot y_2 - z_1 \cdot z_2) \\\\
            + &(w_1 \cdot x_2 + x_1 \cdot w_2 + y_1 \cdot z_2 - z_1 \cdot y_2) \cdot i \\\\
            + &(w_1 \cdot y_2 - x_1 \cdot z_2 + y_1 \cdot w_2 + z_1 \cdot x_2) \cdot j \\\\
            + &(w_1 \cdot z_2 + x_1 \cdot y_2 - y_1 \cdot x_2 + z_1 \cdot w_2) \cdot k .

    Parameters
    ----------
    q1, q2 : (..., 4) ndarray
        Input arrays to be multiplied, must be broadcastable to a common
        shape.

    Returns
    -------
    q_prod : (..., 4) ndarray
        The pairwise Hamilton products of quaternions in :math:`q_1` and
        :math:`q_2`.

    References
    ----------
    .. footbibliography::
    """
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + w2*x1 + y1*z2 - z1*y2
    y = w1*y2 + w2*y1 + z1*x2 - z2*x1
    z = w1*z2 + w2*z1 + x1*y2 - x2*y1

    q_prod = np.moveaxis([w, x, y, z], 0, -1)
    return q_prod
