import functools
import multiprocessing as mp

import numpy as np
from tqdm.asyncio import tqdm

from . import quaternions as qops


def _correlate_i(quaternions, quaternions_inv, ndx, do_variance=False):
    """Compute the covariance matrix `Q` for one discrete correlation
    time."""
    q1 = quaternions[..., :-ndx, :]
    q2 = quaternions_inv[..., ndx:, :]
    q_corr = qops.multiply_quats(q1, q2)
    Q_not_averaged = np.matmul(q_corr[..., 1:, np.newaxis],
                               q_corr[..., np.newaxis, 1:])
    if do_variance:
        var = Q_not_averaged.var(axis=-3)
        return Q_not_averaged.mean(axis=-3), var
    return Q_not_averaged.mean(axis=-3)


def correlate(orientations, stop=None, step=1, do_variance=False,
              verbose=False):
    # TODO: Update documentation.
    """Compute six rotational correlation functions, returned as
    elements of the symmetric quaternion covariance matrix `Q`.



    The `orientations` may be passed either as an array of rotational
    matrices or of quaternions.
    Rotational matrices will be converted
    to quaternions under the hood. If passing quaternions directly, the
    scalar part must be leading. The passed array may have any
    dimensionality

    Parameters
    ----------
    orientations : ndarray
        The orientations, represented either as rotational matrices
        (shape `(..., 3, 3)`), or as quaternions (shape `(..., 4)`).
    stop : int, optional
        Maximum lag index.
    step : int, default: 1

    do_variance : bool, default: False

    verbose : bool, default: False

    Returns
    -------
    Q : ndarray, shape (..., N, 3, 3)
        The quaternion covariance matrix computed at `N` discrete
        correlation times.
    Q_var : ndarray, shape (..., N, 3, 3), optional
        The variance of `Q`.

    Notes
    -----

    """
    if orientations.shape[-2:] == (3, 3):
        orientations = qops.rotmat2quat(orientations)

    stop = int(orientations.shape[-2] / 10) + 1 if stop is None else stop
    indices = np.arange(step, stop, step)
    orientations_inv = qops.invert_quat(orientations)
    Q = np.zeros((indices.size,) + orientations.shape[:-2] + (3, 3))
    var = np.zeros(Q.shape) if do_variance else None

    # TODO (correlate): Parallelize the correlation function.
    for i, ndx in enumerate(tqdm(indices, disable=not verbose)):
        if do_variance:
            Q[i], var[i] = _correlate_i(orientations, orientations_inv, ndx,
                                        do_variance=do_variance)
        else:
            Q[i] = _correlate_i(orientations, orientations_inv, ndx,
                                do_variance=do_variance)

    if do_variance:
        return np.moveaxis(Q, 0, -3), np.moveaxis(var, 0, -3)
    return np.moveaxis(Q, 0, -3)
