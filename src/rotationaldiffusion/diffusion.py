#! /usr/bin/env python
import functools
import multiprocessing as mp

import numpy as np
import scipy

from . import quaternions as qops
from tqdm.auto import tqdm

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


def arange_lag_times(Q, timestep):
    return np.arange(1, Q.shape[-3] + 1, 1) * timestep


def construct_Q_model(lag_times, diffusion_coeffs, PAF=np.eye(3)):
    """
    Quaternion covariance matrix of an ideal Brownian rotor with diffusion tensor D.

    Utilizes Equations 2-4 from Favro (1960) as described by Linke et al. (2017).

    Parameters
    ----------
    lag_times: (N,) ndarray
        Discrete lag times at which to compute the Q matrix.
    diffusion_coeffs: (3,) ndarray
        Trace of a diffusion tensor in its diagonal form / principal axis frame.

    Returns
    -------
    Q: (N, 3, 3) ndarray
        Quaternion covariance matrix of D.
    """
    assert np.shape(diffusion_coeffs) == (3,)
    assert np.shape(PAF) == (3, 3)

    Q = np.zeros((3, 3, len(lag_times)))
    for i, j, k in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        Q[i, i] = 1 / 4 * (1
           + np.exp(-(diffusion_coeffs[j] + diffusion_coeffs[k]) * lag_times)
           - np.exp(-(diffusion_coeffs[i] + diffusion_coeffs[j]) * lag_times)
           - np.exp(-(diffusion_coeffs[i] + diffusion_coeffs[k]) * lag_times))
    if not np.allclose(PAF - np.eye(3), np.zeros((3, 3))):
        Q = np.tensordot(np.tensordot(PAF, PAF, 0), Q, axes=((0, 2), (0, 1)))
    return np.moveaxis(Q, -1, 0)


def construct_Q_model_var(lag_time, diffusion_coeffs, PAF=np.eye(3),
                          precomputed_Q_model=None):
    """
    Compute the variances of the quaternion covariance matrix Q of an ideal Brownian rotor with diffusion tensor D.

    Utilizes Equations 2-4, 6.3, and 6.4 from Favro (1960) as described by Linke et al. (2017).

    Parameters
    ----------
    diffusion_coeffs: (3,) ndarray
        Trace of a diffusion tensor in its diagonal form / principal axis frame.
    lag_time: (N,) ndarray
        Discrete lag times at which to compute the variances of the covariance matrix.

    Returns
    -------
    qi2_qj2: (3, 3, N) ndarray
        Analytical variances of the quaternion covariance matrix corresponding to D.
    """
    # TODO: Swap order of axes in output to (N, 3, 3) for consistency.
    # Use Var(q_i * q_j) = <q_i^2 * q_j^2> - <q_i * q_j>^2 =: qi2_qj2 - qij2.
    # Helpers:
    D_av = np.mean(diffusion_coeffs)
    Dx, Dy, Dz = diffusion_coeffs
    delta = np.sqrt(Dx**2 + Dy**2 + Dz**2 - Dx*Dy - Dx*Dz - Dy*Dz)
    delta = 0 if np.isnan(delta) else delta

    # Start with <q_i^2 * q_j^2> =: qi2_qj2
    qi2_qj2 = np.zeros((3, 3, len(lag_time)))
    for ndx in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        i, j, k = ndx
        Dx, Dy, Dz = diffusion_coeffs[ndx]

        # Diagonal elements qi2_qi2 using Eq. 6.4 from Favro (1960).
        qi2_qj2[i, i] += 1 / 8 * (
            1 + 3 / 2 * np.exp(-3 * D_av * lag_time) * (
                np.exp(Dx * lag_time)
                - np.exp(Dy * lag_time)
                - np.exp(Dz * lag_time))
            + 1 / 2 * np.exp(-3 * D_av * lag_time) * (
                np.exp(-3 * Dx * lag_time)
                - np.exp(-3 * Dy * lag_time)
                - np.exp(-3 * Dz * lag_time))
            + np.exp(-6 * D_av * lag_time) * np.cosh(2 * lag_time * delta))

        # Off-diagonal elements qi2_qj2 using Eq. 6.3 from Favro (1960).
        # Catch zero-division in isotropic case (delta=0) by l'Hospital.
        if np.isclose(delta, 1e-12):
            lHospital = 2 * lag_time * (Dz - D_av)
        else:
            lHospital = (1 / delta * (Dz - D_av)
                         * np.sinh(2 * lag_time * delta))

        qi2_qj2[i, j] = qi2_qj2[j, i] = 1 / 8 * (
            1 / 3 - 1 / 2 * np.exp(-3 * D_av * lag_time) * (
                np.exp(Dz * lag_time) - np.exp(-3 * Dz * lag_time)
                )
            + np.exp(-6 * D_av * lag_time) * (
                lHospital - 1 / 3 * np.cosh(2 * lag_time * delta)
                )
            )

    # Rotate qi2_qj2 into reference frame.
    if not np.allclose(PAF - np.eye(3), np.zeros((3, 3))):
        PAF_2nd_power_shifted = PAF * PAF[(1, 2, 0),]
        PAF_4th_power_shifted = np.multiply(
            PAF_2nd_power_shifted[:, np.newaxis, :],
            PAF_2nd_power_shifted[:, :, np.newaxis]).T

        qi2_qj2_rotated = np.tensordot(
            np.tensordot(PAF ** 2, PAF ** 2, axes=0), qi2_qj2,
            axes=((0, 2), (0, 1)))
        qi2_qj2_rotated += 4 * np.matmul(PAF_4th_power_shifted,
                                         qi2_qj2[(0, 1, 0), (1, 2, 2)])
    else:
        qi2_qj2_rotated = qi2_qj2

    if precomputed_Q_model is not None:
        var = np.moveaxis(qi2_qj2_rotated, -1, 0) - precomputed_Q_model ** 2
    else:
        var = np.moveaxis(qi2_qj2_rotated, -1, 0) - construct_Q_model(lag_time, diffusion_coeffs, PAF) ** 2
    return var


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


def convert2D_and_PAF(params):
    PAF = qops.quat2rotmat(params[-4:])
    D = list(np.float_power(10, params[:-4]))
    while len(D) < 3:
        D.insert(0, D[0])
    return D, PAF


def guess_initial_params_4fitting(lag_times, Q_data, model='anisotropic'):
    # Initial guess of diffusion coefficients.
    larger_125 = np.any(np.abs(np.abs(Q_data)) > 0.2, axis=(0, 1))
    ndx = np.argmax(larger_125) if larger_125.any() else -1
    diff_coeffs_init, _ = instantaneous_tensors(lag_times[ndx], Q_data[ndx])

    # Define initial parameter set.
    match model:
        case 'anisotropic':
            diff_params_init = diff_coeffs_init
        case 'semi-isotropic':
            diff_params_init = diff_coeffs_init[(0, 2),]
        case 'isotropic':
            diff_params_init = [np.mean(diff_coeffs_init)]
        case _:
            raise KeyError(f"Model must be one of anisotropic, semi-isotropic,"
                           f" or isotropic. Is: {model}.")

    return list(np.log10(diff_params_init)) + [1, 0, 0, 0]


def chi2(params, lag_times, Q_data):
    diffusion_coeffs, PAF = convert2D_and_PAF(params)
    model = construct_Q_model(lag_times, diffusion_coeffs)
    data = np.einsum('im,tmn,jn->tij', PAF, Q_data, PAF)
    residuals = (model - data) ** 2
    return np.mean(residuals[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])


def optimize(params_init, constraints, lag_times, Q_data, tol=1e-10,
             maxiter=1000):
    res = scipy.optimize.minimize(chi2, params_init, tol=tol,
                                  args=(lag_times, Q_data),
                                  constraints=constraints,
                                  method='trust-constr',
                                  options={'disp': False,
                                           'maxiter': maxiter})
    return res


def least_squares_fit(lag_times, Q_data, model='anisotropic',
                      tol=1e-10, maxiter=1000, tmp=None):
    params_init = guess_initial_params_4fitting(lag_times, Q_data, model)

    # Constrain PAF-quaternion to norm 1 (to make it a rotational quaternion).
    def unit_quaternion_constraint(params):
        return np.sum(np.square(params[-4:])) - 1
    constraints = [{'type': 'eq', 'fun': unit_quaternion_constraint}]

    # Main optimization step.
    if model != 'semi-isotropic':
        res = optimize(params_init, constraints, lag_times, Q_data, tol=tol,
                       maxiter=maxiter)
        res.shape = 'triaxial' if model == 'anisotropic' else 'spherical'
    else:
        # Fit prolate model.
        constraints_prolate = constraints + [{'type': 'ineq',
                                              'fun': lambda x: x[1]-x[0]}]
        res_prolate = optimize(params_init, constraints_prolate, lag_times,
                               Q_data, tol=tol, maxiter=maxiter)

        # Fit oblate model.
        constraints_oblate = constraints + [{'type': 'ineq',
                                              'fun': lambda x: x[0]-x[1]}]
        params_init[0], params_init[1] = params_init[1], params_init[0]
        res_oblate = optimize(params_init, constraints_oblate, lag_times,
                              Q_data, tol=tol, maxiter=maxiter)

        # Select best fit.
        if res_prolate.fun < res_oblate.fun:
            res = res_prolate
            res.shape = 'prolate'
        else:
            res = res_oblate
            res.shape = 'oblate'

        if tmp == 'prolate':
            res = res_prolate
            res.shape = 'prolate'
        elif tmp == 'oblate':
            res = res_oblate
            res.shape = 'oblate'

    # Check that result is converged.
    # assert res.success, f"The optimization failed after {res.nit} iterations."
    res.model = model

    # Convert parameters back to D and PAF.
    # res.D = np.float_power(10, res.x[diff_params_indices,])
    # res._PAF = qops.quat2rotmat(res.x[-4:])
    D, PAF = convert2D_and_PAF(res.x)
    res.D = np.array(D)
    res._PAF = PAF

    # Sort D (and PAF accordingly, only anisotropic model).
    if model == 'anisotropic':
        res._PAF = res._PAF[np.argsort(res.D)]
        res.D = np.sort(res.D)

    # Apply PAF convention.
    res._PAF = apply_PAF_convention(res._PAF)

    # Store rotational axes.
    match model:
        case 'anisotropic':
            res.rotation_axes = res._PAF
        case 'semi-isotropic':
            res.rotation_axes = res._PAF[2]
        case 'isotropic':
            res.rotation_axes = None

    # TODO: Compute anisotropy.
    # TODO: manually test optimizer on huge variety of Ds and PAFs.
    return res


def get_error(value):
    if f"{value:.1e}"[0] == '1' and f"{value:.1e}"[2] < '5':
        return float(f"{value:.1e}")
    return float(f"{value:.0e}")


def compute_uncertainty(D, nrepeats, sim_time_max, lag_time_step, lag_time_max,
                        sim_time_step=1e-10, model='anisotropic',
                        lag_time_min=0):
    if not qsim:
        print('Warning: Computing uncertainties of diffusion tensors requires'
              'the "pydiffusion" package, which is missing. Skipping ...')
        return

    niter = int(sim_time_max/sim_time_step)
    stop = int(lag_time_max/sim_time_step/lag_time_step)

    i, new_diff_coeffs, new_PAF_angles, new_chi2 = 1, [], [], []
    new_PAFs = [] # HERE
    all_stds_converged = False
    # while not all_stds_converged:
    for i in tqdm(range(1000)):
        while True:
            quat_trajs = np.array([qsim.run(D, niter, sim_time_step)
                                   for j in range(nrepeats)])
            Q_data = correlate(quat_trajs, step=lag_time_step, stop=stop)
                                   # silent=True, njobs=1)
            Q_data_mean = np.mean(Q_data, axis=0)
            lag_times = arange_lag_times(Q_data_mean,
                                         sim_time_step*lag_time_step)
            fit = least_squares_fit(lag_times[lag_time_min:-1],
                                    Q_data_mean[lag_time_min:-1], model=model)

            if fit.success:
                new_diff_coeffs.append(fit.D)
                new_chi2.append(fit.fun)
                angles = []
                for axis in fit._PAF:
                    angles_tmp = []
                    for ref in np.eye(3):
                        cos = np.abs(np.dot(axis, ref))
                        angle = np.rad2deg(np.arccos(cos))
                        angles_tmp.append(angle)
                    angles.append(angles_tmp)
                new_PAF_angles.append(angles)
                new_PAFs.append(fit._PAF) # HERE
                # i += 1
                break
            print('FAILED')

        # if not i%10:
        #     stds_converged, errors, width_to_mean_ratios = [], [], []
        #     for diff_coeff in np.array(new_diff_coeffs).T:
        #         std = scipy.stats.bootstrap((diff_coeff,), np.std,
        #                                     confidence_level=0.9,
        #                                     n_resamples=100)
        #         std_mean = np.average(std.confidence_interval)
        #         std_diff = np.ptp(std.confidence_interval)
        #         error_low = get_error(std.confidence_interval.low)
        #         error_high = get_error(std.confidence_interval.high)
        #
        #         stds_converged.append(std_diff/std_mean < 0.1
        #                               or error_low == error_high)
        #         errors.append(get_error(std_mean))
        #         width_to_mean_ratios.append(std_diff/std_mean*100)
        #
        #     means = np.mean(new_diff_coeffs, axis=0)
        #     stds = np.std(new_diff_coeffs, axis=0)
        #     _diff_coeffs = np.array(new_diff_coeffs)
        #     aniso = 2 * _diff_coeffs[:, 2] / (_diff_coeffs[:, 1] + _diff_coeffs[:, 0])
        #
        #     print(f"Iteration {i}: {means[0]:.1e} {means[1]:.1e} {means[2]:.1e}"
        #           f", {errors[0]:.1e} {errors[1]:.1e} {errors[2]:.1e}. "
        #           f"Errors: {stds[0]:.2e} {stds[1]:.2e} {stds[2]:.2e}. "
        #           f"Aniso: {np.mean(aniso):.2f}, {np.std(aniso):.2f}. "
        #           f"(Largest interval width:) "
        #           f"{np.max(width_to_mean_ratios):.0f}%. "
        #           f"Chi2: {np.mean(new_chi2):.2e} {np.std(new_chi2):.1e}. ")
        #           # f"PAF angles: {np.mean(new_PAF_angles, axis=0)[0]:.2f}, "
        #           # f"{np.mean(new_PAF_angles, axis=0)[1]:.2f}, "
        #           # f"{np.mean(new_PAF_angles, axis=0)[2]:.2f}.")
        #     if np.all(stds_converged):
        #         break
    # return errors, new_PAF_angles
    return new_diff_coeffs, new_PAFs # HERE
