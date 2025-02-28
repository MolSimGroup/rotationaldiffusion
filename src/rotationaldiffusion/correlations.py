import functools
import multiprocessing as mp

import numpy as np
from tqdm.asyncio import tqdm

from src.rotationaldiffusion import quaternions as qops


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


def extract_Q_data(quats, do_variance=False, stop=None, step=1, njobs=mp.cpu_count(),
                   chunksize=None, silent=False):
    """
    Compute quaternion covariance matrix Q in reference frame from quaternion trajectory.

    First, compute the correlation function q_corr(t, tau) = q(t) * q^{-1}(t+tau), where
    q is a quaternion describing the least-squares rotation from a trajectory frame to
    a reference frame. Return the covariance matrix of the axial part of q_corr, so
    Q_ij(tau) = <q_corr_i * q_corr_j>, i,j = 1,2,3, where <...> denotes the ensemble
    average over all starting times t.

    Parameters
    ----------
    quats: (..., 4) ndarray
        Quaternions in order (w, x, y, z).

    Returns
    -------
    Q: (..., N, 3, 3) ndarray
        Quaternion covariance matrix.
    """
    # TODO: add unit test.
    stop = int(quats.shape[-2] / 10) if not stop else stop
    indices = np.arange(step, stop + 1, step, dtype=int)
    inverted_quats = qops.invert_quat(quats)
    Q = np.zeros((indices.size,) + quats.shape[:-2] + (3, 3))
    var = np.zeros(
        (indices.size,) + quats.shape[:-2] + (3, 3)) if do_variance else None

    if njobs > 1 and 'fork' in mp.get_all_start_methods():
        with mp.get_context('fork').Pool(njobs) as pool:
            if not chunksize:
                chunksize, extra = divmod(len(indices), len(pool._pool) * 4)
                chunksize = min(chunksize + 1 if extra else chunksize, 100)
            func = functools.partial(_correlate_i, quats, inverted_quats,
                                     do_variance=do_variance)
            for i, res in enumerate(pool.imap(func,
                                              tqdm(indices, disable=silent),
                                              chunksize=chunksize)):
                if do_variance:
                    Q[i], var[i] = res
                else:
                    Q[i] = res
    else:
        for i, ndx in enumerate(tqdm(indices, disable=silent)):
            if do_variance:
                Q[i], var[i] = _correlate_i(quats, inverted_quats, ndx,
                                            do_variance=do_variance)
            else:
                Q[i] = _correlate_i(quats, inverted_quats, ndx,
                                    do_variance=do_variance)

    if do_variance:
        return np.moveaxis(Q, 0, -3), np.moveaxis(var, 0, -3)
    return np.moveaxis(Q, 0, -3)
