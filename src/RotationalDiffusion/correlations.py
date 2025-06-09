"""
This module provides a function for computing rotational correlation
functions.
"""
import functools
import multiprocessing as mp

import numpy as np
from tqdm.asyncio import tqdm
from . import quaternions as qops


def _get_lag_indices(n_frames, stop, step):
    if stop == 'auto':
        # By default, compute lag up to 10% of the trajectory length.
        stop = 0.1
    if isinstance(stop, float):
        stop = int(n_frames * stop)
    if not isinstance(stop, int):
        raise ValueError('`stop` must be either `auto` or int or float.')
    elif stop >= n_frames:
        raise ValueError('`stop` must be less than the number of '
                         'trajectory frames.')
    return np.arange(step, stop, step)


def _correlate_i(quaternions, quaternions_inv, ndx, do_variance=False):
    """Compute the covariance matrix `Q` for one discrete correlation
    time."""
    q1 = quaternions[..., :-ndx, :]
    q2 = quaternions_inv[..., ndx:, :]
    q_corr = qops.multiply(q1, q2)
    Q_not_averaged = np.matmul(q_corr[..., 1:, np.newaxis],
                               q_corr[..., np.newaxis, 1:])
    if do_variance:
        var = Q_not_averaged.var(axis=-3)
        return Q_not_averaged.mean(axis=-3), var
    return Q_not_averaged.mean(axis=-3)


def correlate(orientations, stop=.1, step=1, do_variance=False,
              verbose=False):
    # TODO: Update documentation.
    """Compute rotational correlation functions from trajectories of
    orientations.

    The orientations must be provided as a time series (trajectory) of
    orientational (unit) quaternions in scalar first convention, i.e.,
    each quaternion is represented by a numpy array
    :math:`(w, x, y, z)`, where :math:`w` is the scalar part. If an
    :any:`array_like` of orientational trajectories is provided, all
    trajectories are processed at once using fast numpy array
    operations, which is much quicker than looping over the
    trajectories.

    Parameters
    ----------
    orientations : (..., n_frames, 4) array_like
        Quaternions representing the orientations, in scalar first
        convention. The second-to-last dimension must contain the time
        series of orientations.
    stop : float or int, default: 0.1
        Maximum lag time. If :any:`float`, ``stop`` specifies a fraction
        of the trajectory length (default). If :any:`int`, ``stop``
        specifies the maximum lag index.
    step : int, default: 1
        Increment of the lag index.
    do_variance : bool, default: False
        Whether to compute the variances of products.
    verbose : bool, default: False
        Show progress bar if set to :any:`True`.

    Returns
    -------
    Q : ndarray, shape (..., N, 3, 3)
        The quaternion covariance matrix computed at `N` discrete
        correlation times.
    Q_var : ndarray, shape (..., N, 3, 3), optional
        The variance of `Q`.

    Notes
    -----
    First, reorientations :math:`q(t, \\tau)` are computed from the
    provided trajectory of orientations :math:`q(t)` as

    .. math::

        q(t, \\tau) = q(t) \cdot q^{-1}(t + \\tau).

    Second, the covariance matrix of these reorientational quaternions
    is computed as
    :math:`{\\bf \\tilde{Q}}_{ij}(\\tau) = \\langle q_i q_j \\rangle_t`.
    Evidently, :math:`{\\bf \\tilde{Q}}_{ij}(\\tau)` is a matrix of
    correlation functions. If ``do_variance`` is :any:`True`, the


    See :footcite:t:`holtbruegge2025` for more
    details.

    References
    ----------
    .. footbibliography::
    """
    orientations = np.array(orientations)
    n_frames = orientations.shape[-2]
    if isinstance(stop, float):
        stop = int(n_frames * stop)
    lag_indices = np.arange(step, stop, step)

    # The inverse of a unit quaternion is its complex conjugate.
    orientations_inv = qops.conjugate(orientations)

    Q = np.zeros((lag_indices.size,) + n_frames + (3, 3))
    if do_variance:
        var = np.zeros(Q.shape)

    for i, ndx in enumerate(tqdm(lag_indices, disable=not verbose)):
        if do_variance:
            Q[i], var[i] = _correlate_i(orientations, orientations_inv, ndx,
                                        do_variance=do_variance)
        else:
            Q[i] = _correlate_i(orientations, orientations_inv, ndx,
                                do_variance=do_variance)

    if do_variance:
        return np.moveaxis(Q, 0, -3), np.moveaxis(var, 0, -3)
    return np.moveaxis(Q, 0, -3)
