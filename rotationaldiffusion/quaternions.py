import numpy as np


def invert_quat(quats):
    """Invert quaternions q to q^{-1} by complex conjugation."""
    return quats * [1, -1, -1, -1]


def rotmat2quat(rotmats):
    """
    Convert 3D rotation matrices to their quaternion representation.

    Implementation of the Bar-Itzhack algorithm [1, 2], which determines
    the optimal unit quaternion corresponding to the orthogonal
    rotation matrix closest to the input matrix.

    Parameters
    ----------
    rotmats : (..., 3, 3) array_like
        Rotation matrices to be converted to quaternions.

    Returns
    -------
    quats : (..., 4) ndarray
        Quaternions in order (w, x, y, z).

    References
    ----------
    [1] Bar-Itzhack, 2000, J. Guid. Control. Dyn., "New Method for
    Extracting the Quaternion from a Rotation Matrix",
    doi: 10.2514/2.4654 .
    [2] https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    (17th October 2023).
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
    quats = reduce_quat_angle(quats)
    return quats


def quat2rotmat(quats):
    """
    Convert unit quaternions to their rotational matrix representation.

    Parameters
    ----------
    quats: (..., 4) array_like
        Quaternions in order (q, x, y, z)

    Returns
    -------
    rotmats: (..., 3, 3) ndarray
        Rotational matrices.
    """
    quats = np.moveaxis(quats, -1, 0)
    w, x, y, z = quats

    rotmats = np.moveaxis([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], [0, 1], [-2, -1])
    return rotmats


def reduce_quat_angle(quats):
    """
    Reduce rotation angles in (pi, 2pi] to [0, pi).

    Two quaternions q and -q describe the same rotation. If q describes
    the rotation by alpha around some axis, -q describes the counter
    rotation by 2pi-alpha. If alpha is in (pi, 2pi], then 2pi-alpha is
    in [0, pi) and vice versa. Since Re(q) = w = sin(alpha/2),
    alpha in (pi, 2pi] <==> w < 0 and alpha in [0, pi) <==> w >= 0.

    This function returns the quaternions with angles in [0, pi]. Input
    quaternions q with angles already in [0, pi] are returned directly.
    Otherwise, -q is returned to reduce the rotation angle from
    (pi, 2pi] to [0, pi).

    Parameters
    ----------
    quats : (..., 4) ndarray
        Array of quaternions.

    Returns
    -------
    quats_reduced : (..., 4) ndarray
        Array of quaternions with rotation angles reduced to <= pi.
    """
    quats_reduced = np.copy(quats)
    quats_reduced[quats_reduced[..., 0] < 0] *= -1
    return quats_reduced


def multiply_quats(q1, q2):
    """
    Compute Hamilton product (of arrays) of quaternions q1 and q2.

    Parameters
    ----------
    q1, q2 : (..., 4) ndarray
        Input arrays to be multiplied, must be broadcastable to a common
        shape. The last dimension must contain quaternions in order
        (w, x, y, z).

    Returns
    -------
    q_prod : (..., 4) ndarray
        The pairwise Hamilton product of quaternions q1 and q2.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    (26th October 2023).
    """
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + w2*x1 + y1*z2 - z1*y2
    y = w1*y2 + w2*y1 + z1*x2 - z2*x1
    z = w1*z2 + w2*z1 + x1*y2 - x2*y1

    q_prod = np.moveaxis([w, x, y, z], 0, -1)
    return q_prod
