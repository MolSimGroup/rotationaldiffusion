"""
This module provides a function for computing rotational correlation
functions from orientation trajectories.
"""
import numpy as np
from tqdm.auto import tqdm
from . import quaternions as qops


def _correlate_i(quaternions, quaternions_inv, lag_ndx, do_variance=False):
    """Compute the rotational correlation functions for one discrete lag
    index."""
    q1 = quaternions[..., :-lag_ndx, :]
    q2 = quaternions_inv[..., lag_ndx:, :]
    q_corr = qops.multiply(q1, q2)
    Q_not_averaged = np.matmul(q_corr[..., 1:, np.newaxis],
                               q_corr[..., np.newaxis, 1:])
    if do_variance:
        var = Q_not_averaged.var(axis=-3)
        return Q_not_averaged.mean(axis=-3), var
    return Q_not_averaged.mean(axis=-3)


def correlate(orientations, stop=.1, step=1, do_variance=False,
              verbose=False):
    """Compute rotational correlation functions from trajectories of
    orientations.

    The orientations must be provided as a time series (trajectory) of
    orientational (unit) quaternions in scalar-first convention, i.e.,
    each quaternion is represented by a numpy array
    :math:`(w, x, y, z)`, where :math:`w` is the scalar part. If an
    :any:`array_like` of orientational trajectories is provided, all
    trajectories are processed at once using fast numpy array
    operations, which is much quicker than looping over the
    trajectories.

    The maximum lag time can be specified with ``stop``. If ``stop`` is
    a :any:`float`, then it specifies a fraction of the trajectory
    length, which is converted into a maximal lag index. For example, if
    the trajectory contains 1000 frames and ``stop`` is set to 0.1
    (default), then the correlation function is computed up to 10% of
    the trajectory length and the maximum lag index is set to 100. If
    ``stop`` is an :any:`int`, it directly specifies the maximum lag
    index.

    The computed quaternion covariance matrix contains (ensemble)
    averages of products of quaternion
    components, which form six rotational correlation
    functions.\ :footcite:`holtbruegge2025` If ``do_variance`` is
    :any:`True`, then also the corresponding variances of products of
    quaternion components are computed. In that case, a tuple of
    the quaternion covariance matrix and the variance matrix is
    returned. However, computing the variances increases the
    computational time and is seldomly necessary, hence, it is
    deactivated by default.

    Parameters
    ----------
    orientations : (..., n_frames, 4) array_like
        Quaternions representing the orientations, in scalar first
        convention. The second-to-last dimension must contain the time
        series of quaternions.
    stop : float or int, default: 0.1
        Maximum lag time. If :any:`float`, ``stop`` specifies a fraction
        of the trajectory length. If :any:`int`, ``stop`` specifies the
        maximum lag index.
    step : int, default: 1
        Increment of the lag index.
    do_variance : bool, default: False
        Whether to compute the variances of correlation matrix elements.
    verbose : bool, default: False
        Show progress bar if set to :any:`True`.

    Returns
    -------
    Q : (..., N, 3, 3) ndarray
        The quaternion covariance matrix computed at :math:`N` discrete
        correlation times.
    var : (..., N, 3, 3) ndarray, optional
        The variance matrix, returned only if ``do_variance`` is
        :any:`True`.

    Raises
    ------
    ValueError
        If ``stop`` is too large.

    See Also
    --------
    RotationalDiffusion.orientations.Orientations
        MDAnalysis-based class for extracting the orientations of a
        molecular structure from MD trajectories.
    RotationalDiffusion.quaternions
        Quaternion operations used under the hood.

    Notes
    -----
    First, reorientations :math:`q(t, \\tau)` are computed from the
    provided trajectory of orientations :math:`q(t)` as

    .. math::

        q(t, \\tau) = q(t) \cdot q^{-1}(t + \\tau).

    Second, the covariance matrix of the vector parts of these
    reorientational quaternions is computed as
    :math:`{\\bf \\tilde{Q}}_{ij}(\\tau) = \\langle q_i q_j \\rangle_t`.
    The resulting matrix :math:`{\\bf \\tilde{Q}}_{ij}(\\tau)` contains
    six rotational correlation functions. See
    :footcite:t:`holtbruegge2025` for more details.

    References
    ----------
    .. footbibliography::
    """
    orientations = np.array(orientations)
    n_frames = orientations.shape[-2]
    if isinstance(stop, float):
        stop = int(n_frames * stop)
    if stop > n_frames:
        raise ValueError('The maximum lag time must not exceed the '
                         'trajectory length.')
    lag_indices = np.arange(step, stop, step)

    # The inverse of a unit quaternion is its complex conjugate.
    orientations_inv = qops.conjugate(orientations)

    # Create output arrays.
    output_shape = lag_indices.shape + orientations.shape[:-2] +  (3, 3)
    Q = np.zeros(output_shape)
    if do_variance:
        var = np.zeros(output_shape)

    # TODO: Parallelize this loop.
    for i, lag_ndx in enumerate(tqdm(lag_indices, disable=not verbose,
                                     desc='Computing correlations')):
        if do_variance:
            Q[i], var[i] = _correlate_i(orientations, orientations_inv,
                                        lag_ndx, do_variance=do_variance)
        else:
            Q[i] = _correlate_i(orientations, orientations_inv, lag_ndx,
                                do_variance=do_variance)

    if do_variance:
        return np.moveaxis(Q, 0, -3), np.moveaxis(var, 0, -3)
    return np.moveaxis(Q, 0, -3)
