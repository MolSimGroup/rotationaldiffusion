import argparse
import functools
import multiprocessing as mp
import pickle as pkl
from collections import defaultdict
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
# from scipy.optimize import fmin_powell
import scipy

from .parser import parse_args
from . import quaternions as qops
from tqdm.auto import tqdm

def chi2_isotropic(params, lag_times, Q_data):
    diffusion_coeffs = np.float_power(10, params[(0, 0, 0),])
    PAF = qops.quat2rotmat(params[1:])
    model = construct_Q_model(lag_times, diffusion_coeffs)
    data = np.einsum('im,mnt,jn->ijt', PAF, Q_data, PAF)
    residuals = (model - data)**2
    return np.sum(residuals[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])


def chi2_semi_isotropic(params, lag_times, Q_data):
    diffusion_coeffs = np.float_power(10, params[(0, 0, 1),])
    PAF = qops.quat2rotmat(params[2:])
    model = construct_Q_model(lag_times, diffusion_coeffs)
    data = np.einsum('im,mnt,jn->ijt', PAF, Q_data, PAF)
    residuals = (model - data)**2
    return np.sum(residuals[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])


def chi2_anisotropic(params, lag_times, Q_data):
    diffusion_coeffs = np.float_power(10, params[:3])
    PAF = qops.quat2rotmat(params[3:])
    model = construct_Q_model(lag_times, diffusion_coeffs)
    data = np.einsum('im,mnt,jn->ijt', PAF, Q_data, PAF)
    residuals = (model - data)**2
    return np.sum(residuals[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])


def _chi2_anisotropic_in_REF(params, lag_times, Q_data):
    diffusion_coeffs = np.float_power(10, params[(0, 0, 0),])
    PAF = qops.quat2rotmat(params[1:])
    model = construct_Q_model(lag_times, diffusion_coeffs, PAF)
    residuals = (model - Q_data)**2
    return np.sum(residuals[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])




def compute_model_Q_chen(D, taus):
    """
    Compute the quaternion covariance matrix Q according to Chen's model.

    Uses Equation 16 from Chen et al. (2017).

    Parameters
    ----------
    D: (3,) ndarray
        Trace of a diffusion tensor in its diagonal form / principal axis frame.
    taus: (N,) ndarray
        Discrete lag times at which to compute the covariance matrix.

    Returns
    -------
    Q: (3, 3, N) ndarray
        Quaternion covariance matrix of D.
    """
    Q = np.zeros((3, 3, len(taus)))
    for i, Di in enumerate(D):
        Q[i, i] = 1 / 4 * (1 - np.exp(-2 * Di * taus))
    return Q


def square_deviation(D, taus, Q_data):
    Q_model = Q_model(D, taus)
    chi_square = np.sum((Q_data - Q_model) ** 2)
    return chi_square


def weighted_square_deviation(D, taus, Q_data):
    Q_model = Q_model(D, taus)
    Q_model_var = Q_model_var(D, taus)
    chi_square = np.sum((Q_data - Q_model) ** 2 / Q_model_var)
    return chi_square


def square_deviation_chen(D, taus, Q_data):
    Q_model = Q_model(D, taus)
    chi_square = np.sum((Q_data - Q_model) ** 2)
    return chi_square


def euler2rotmat(alpha, beta, gamma):
    rotmat = np.array([
        [np.cos(alpha) * np.cos(gamma) - np.sin(alpha) * np.cos(beta) * np.sin(
            gamma), -np.cos(alpha) * np.sin(gamma) - np.sin(alpha) * np.cos(
            beta) * np.cos(gamma), np.sin(alpha) * np.sin(beta)],
        [np.sin(alpha) * np.cos(gamma) + np.cos(alpha) * np.cos(beta) * np.sin(
            gamma), -np.sin(alpha) * np.sin(gamma) + np.cos(alpha) * np.cos(
            beta) * np.cos(gamma), -np.cos(alpha) * np.sin(beta)],
        [np.sin(beta) * np.sin(gamma), np.sin(beta) * np.cos(gamma),
         np.cos(beta)]
    ])
    return rotmat


def square_deviation_wmatrix(params, taus, Q_data):
    Dx, Dy, Dz, alpha, beta, gamma = params
    Q_model = Q_model([Dx, Dy, Dz], taus)
    rotmat = euler2rotmat(alpha, beta, gamma)
    Q_model = np.matmul(rotmat.transpose(), np.matmul(Q_model, rotmat))
    chi_square = np.sum((Q_data - Q_model) ** 2)
    return chi_square


def eq_15(T_ij):
    """Equation 15 from Chen et al. (2017)."""
    return 1 - 2 * np.trace(T_ij, axis1=-2, axis2=-1)


def eq_17(T_ij, diagonalize=False, return_eigvecs=False):
    """Equation 17 from Chen et al. (2017)."""
    if not diagonalize:
        return 1 - 2 * np.diagonal(T_ij, axis1=-2, axis2=-1).transpose()
    eigvals, eigvecs = np.linalg.eigh(T_ij)
    if not return_eigvecs:
        return 1 - 2 * eigvals.transpose()
    else:
        return 1 - 2 * eigvals.transpose(), eigvecs


def compute_data_2fit(model, T_ij, n_chunks=0,
                      analyze_individual_trajectories=False, **kwargs):
    if model == 'isotropic':
        eq = eq_15
    elif model == 'anisotropic':
        eq = eq_17
    else:
        raise ValueError(
            f"model must be 'isotropic' or 'anisotropic' (is {model})")
    data = {}
    data['all'] = eq(T_ij.mean(axis=0), **kwargs)
    if n_chunks:
        for i, T_ij_by_chunk in enumerate(np.split(T_ij, n_chunks)):
            data[f"chunk {i}"] = eq(T_ij_by_chunk.mean(axis=0), **kwargs)
    if analyze_individual_trajectories:
        for i, T_ij_by_traj in enumerate(T_ij):
            data[f"traj {i}"] = eq(T_ij_by_traj, **kwargs)
    return data


def fit_data(model, taus, data, fitC=False):
    if model == 'isotropic':
        C = 1.5
    elif model == 'anisotropic':
        C = 0.5
    else:
        raise ValueError(
            f"model must be 'isotropic' or 'anisotropic' (is {model})")
    # Prevent negative value in log in 'init_guess'.
    if data[-1] > 0.6:
        init_guess = -taus[-1] / np.log((data[-1] - 1 + C) / C)
    else:
        init_guess = -taus[data < 0.6][0] / np.log(
            (data[data < 0.6][0] - 1 + C) / C)

    if not fitC:
        fit = scipy.optimize.curve_fit(construct_exp_fct(C=C), taus, data,
                                       p0=(init_guess))
    else:
        fit = scipy.optimize.curve_fit(construct_exp_fct(), taus, data,
                                       p0=(init_guess, C))
    return fit[0]


def construct_exp_fct(C=None, tau_corr=None):
    if C and tau_corr:
        def exp_fct(tau):
            return C * np.exp(-tau / tau_corr) + 1 - C
    elif C:
        def exp_fct(tau, tau_corr):
            return C * np.exp(-tau / tau_corr) + 1 - C
    else:
        def exp_fct(tau, tau_corr, C):
            return C * np.exp(-tau / tau_corr) + 1 - C
    return exp_fct


def fct_2fit(A, B, fitC=False):
    if not fitC:
        def real_fct_to_fit(taus, tau_corr):
            return A * np.exp(-taus / tau_corr) + B
    else:
        def real_fct_to_fit(taus, tau_corr, C):
            return C * np.exp(-taus / tau_corr) + 1 - C
    return real_fct_to_fit


def main(*args):
    parser, args = parse_args(*args)
    n_trajectories = len(args.files)
    n_timesteps = int((args.ndx_stop - args.ndx_start - 1) // args.skip) + 1

    # TODO: add progress bar
    # TODO: add proper logging
    # Load rotational matrices (rotmats) from disk.
    rotmats = np.zeros((n_trajectories, n_timesteps, 3, 3))
    for i, file in enumerate(tqdm(args.files)):
        try:
            rotmats_i = np.loadtxt(file, comments=['#', '@'])[
                        args.ndx_start:args.ndx_stop + 1:args.skip, 1:]
            rotmats[i] = rotmats_i.reshape(-1, 3, 3)
        except ValueError:
            parser.error(
                f"argument -f: cannot load data from file {file} into numpy array")
    print('Files loaded.')

    # Convert rotmats to quaternions (quats).
    quats_frame2ref = qops.rotmat2quat(rotmats)

    # Invert direction of rotation by inverting quats.
    quats = qops.invert_quat(quats_frame2ref)

    # Determine discrete lag times tau.
    taus = np.arange(args.tau_step, args.tau_max + args.tau_step,
                     args.tau_step)
    n_taus = taus.size
    ndx_taus = np.arange(1, n_taus + 1) * int(args.tau_step // args.time_step)

    # Compute data2fit in moving PAF frame.
    iso_data2fit, aniso_data2fit = {}, {}
    if args.use_moving_PAF:
        # Compute quaternion correlation covariance matrix (eq. 8 and 14 in Chen et al. (2017)) in body frame.
        T_ij, T_ij_var, fullQ = extract_Q_data(quats, ndx_taus)

        # Compute moving PAF(tau)
        _, eigvecs = compute_data_2fit('anisotropic', T_ij, diagonalize=True,
                                       return_eigvecs=True)['all']
        moving_PAF_axes = eigvecs.transpose(0, 2, 1)

        # Compute data2fit to isotropic model (eq. 15 in Chen et al. (2017)).
        if args.do_iso:
            iso_data2fit = compute_data_2fit('isotropic', T_ij,
                                             n_chunks=args.n_chunks,
                                             analyze_individual_trajectories=args.analyze_individual_trajectories)

        # Compute data2fit to anisotropic model (eq. 17 in Chen et al. (2017)) in moving PAF frame.
        if args.do_aniso:
            data2fit_aniso_moving_PAF = compute_data_2fit('anisotropic', T_ij,
                                                          n_chunks=args.n_chunks,
                                                          diagonalize=True,
                                                          analyze_individual_trajectories=args.analyze_individual_trajectories)
            for key, data in data2fit_aniso_moving_PAF.items():
                for i, data_by_dim in enumerate(data):
                    aniso_data2fit[
                        ('moving PAF', key, f"dim{i}")] = data_by_dim

    # Determine indices of 'PAF_times' in time.
    ndx_fixed_PAFs = np.array([t // args.time_step for t in args.PAF_times],
                              dtype=int)
    n_fixed_PAFs = ndx_fixed_PAFs.size

    # Compute data2fit in fixed PAF frames.
    if n_fixed_PAFs:
        # Compute PAFs.
        T_ij, _, _ = extract_Q_data(quats, ndx_fixed_PAFs)
        _, eigvecs = compute_data_2fit('anisotropic', T_ij, diagonalize=True,
                                       return_eigvecs=True)['all']
        fixed_PAF_axes = eigvecs.transpose(0, 2, 1)

        # Force PAFs to be right-handed.
        fixed_PAF_axes[np.linalg.det(fixed_PAF_axes) < 0, 0] *= -1
        quats_ref2fixedPAF = qops.rotmat2quat(fixed_PAF_axes)

        if args.do_full_tensor:
            T_ij_fixed_PAF = np.zeros((n_fixed_PAFs, n_taus, 3, 3))

        # Compute data2fit in each PAF frame.
        for i, (quat_ref2paf, PAF_time) in enumerate(
                tqdm(zip(quats_ref2fixedPAF, args.PAF_times))):
            # Rotate quaternions into fixed PAF.
            quats_paf = copy(quats)
            quats_paf[..., 1:] = qops.multiply_quats(quat_ref2paf,
                                                qops.multiply_quats(quats,
                                                               qops.invert_quat(
                                                                   quat_ref2paf)))[
                                 ..., 1:]

            # Compute quaternion correlation covariance matrix (eq. 8 and 14 in Chen et al. (2017)) in fixed PAF frame.
            T_ij, T_ij_var, fullQ = extract_Q_data(quats_paf, ndx_taus)

            if args.do_full_tensor:
                T_ij_fixed_PAF[i] = T_ij.mean(axis=0)

            # Compute data2fit to anisotropic model (eq. 17 in Chen et al. (2017)) in fixed PAF frame.
            if args.do_aniso:
                data_aniso_fixed_PAF = compute_data_2fit('anisotropic', T_ij,
                                                         n_chunks=args.n_chunks,
                                                         analyze_individual_trajectories=args.analyze_individual_trajectories)
                for key, data in data_aniso_fixed_PAF.items():
                    for i, data_by_dim in enumerate(data):
                        aniso_data2fit[(
                        f"fixed PAF {PAF_time} {args.unit}", key,
                        f"dim{i}")] = data_by_dim

    # Compute data2fit to isotropic model (eq. 15 in Chen et al. (2017)).
    if args.do_iso and not args.use_moving_PAF:
        iso_data2fit = compute_data_2fit('isotropic', T_ij,
                                         n_chunks=args.n_chunks,
                                         analyze_individual_trajectories=args.analyze_individual_trajectories)
    print('data2fit computed, start fitting')

    # Fit isotropic data.
    if iso_data2fit:
        iso_tau_corr = defaultdict(dict)
        if args.fitC:
            iso_C = defaultdict(dict)
        for tau_max in tqdm(args.tau_max_4fitting):
            for key, data in iso_data2fit.items():
                ndx = int(tau_max // args.tau_step)
                fit = fit_data('isotropic', taus[:ndx], data[:ndx],
                               fitC=args.fitC)
                iso_tau_corr[key][f"{tau_max} {args.unit}"] = fit[0]
                if args.fitC:
                    iso_C[key][f"{tau_max} {args.unit}"] = fit[1]
        iso_D_iso = defaultdict(dict)
        iso_func = defaultdict(dict)
        for key, data in iso_tau_corr.items():
            for tau_max, data_by_tau_max in data.items():
                iso_D_iso[key][tau_max] = 0.5e12 / data_by_tau_max
                if not args.fitC:
                    iso_func[key][tau_max] = construct_exp_fct(1.5,
                                                               data_by_tau_max)
                else:
                    iso_func[key][tau_max] = construct_exp_fct(
                        iso_C[key][tau_max], data_by_tau_max)
        if args.n_chunks:
            data = np.array(
                pd.DataFrame.from_dict(iso_tau_corr).filter(like='chunk'))
            for key, av, std in zip(iso_tau_corr['all'].keys(),
                                    np.mean(data, axis=1),
                                    np.std(data, axis=1)):
                iso_tau_corr['chunks av.'][key] = av
                iso_tau_corr['chunks std.'][key] = std
            data = np.array(
                pd.DataFrame.from_dict(iso_D_iso).filter(like='chunk'))
            for key, av, std in zip(iso_D_iso['all'].keys(),
                                    np.mean(data, axis=1),
                                    np.std(data, axis=1)):
                iso_D_iso['chunks av.'][key] = av
                iso_D_iso['chunks std.'][key] = std
        if args.analyze_individual_trajectories:
            data = np.array(
                pd.DataFrame.from_dict(iso_tau_corr).filter(like='traj'))
            for key, av, std in zip(iso_tau_corr['all'].keys(),
                                    np.mean(data, axis=1),
                                    np.std(data, axis=1)):
                iso_tau_corr['trajs av.'][key] = av
                iso_tau_corr['trajs std.'][key] = std
            data = np.array(
                pd.DataFrame.from_dict(iso_D_iso).filter(like='traj'))
            for key, av, std in zip(iso_D_iso['all'].keys(),
                                    np.mean(data, axis=1),
                                    np.std(data, axis=1)):
                iso_D_iso['trajs av.'][key] = av
                iso_D_iso['trajs std.'][key] = std

    # Fit anisotropic data.
    if aniso_data2fit:
        aniso_tau_corr = defaultdict(dict)
        if args.fitC:
            aniso_C = defaultdict(dict)
        for tau_max in tqdm(args.tau_max_4fitting):
            for key, data in aniso_data2fit.items():
                ndx = int(tau_max // args.tau_step)
                fit = fit_data('anisotropic', taus[:ndx], data[:ndx],
                               fitC=args.fitC)
                aniso_tau_corr[key][f"{tau_max} {args.unit}"] = fit[0]
                if args.fitC:
                    aniso_C[key][f"{tau_max} {args.unit}"] = fit[1]
        aniso_D_iso = defaultdict(dict)
        aniso_func = defaultdict(dict)
        for key, data in aniso_tau_corr.items():
            for tau_max, data_by_tau_max in data.items():
                aniso_D_iso[key][tau_max] = 0.5e12 / data_by_tau_max
                if not args.fitC:
                    aniso_func[key][tau_max] = construct_exp_fct(0.5,
                                                                 data_by_tau_max)
                else:
                    aniso_func[key][tau_max] = construct_exp_fct(
                        aniso_C[key][tau_max], data_by_tau_max)

    # Compute other results.
    if args.use_moving_PAF:
        # Force moving PAF to be right-handed.
        tmp_axes = copy(moving_PAF_axes)
        tmp_axes[np.linalg.det(tmp_axes) < 0, 0] *= -1
        quats_ref2movingPAF = qops.rotmat2quat(tmp_axes)

    # Combine all results.
    results = {
        'time unit': args.unit,
        'taus': taus
    }
    if args.do_iso:
        results['iso_data2fit'] = iso_data2fit
        results['iso_tau_corr'] = iso_tau_corr
        results['iso_D_iso'] = iso_D_iso
        # results['iso_func'] = iso_func
    if args.do_iso and args.fitC:
        results['iso_C'] = iso_C
    if args.do_aniso:
        results['aniso_data2fit'] = aniso_data2fit
        results['aniso_tau_corr'] = aniso_tau_corr
        results[('aniso_D')] = aniso_D_iso
        # results['aniso_func'] = aniso_func
    if args.do_aniso and args.fitC:
        results['aniso_C'] = aniso_C
    if args.use_moving_PAF:
        results['moving_PAF_axes'] = moving_PAF_axes
        results['quats_ref2movingPAF'] = quats_ref2movingPAF
    if args.do_full_tensor and args.do_iso:
        results['full_tensor'] = T_ij_fixed_PAF
    results['Q_av'] = T_ij
    results['Q_var'] = T_ij_var
    results['Q_full'] = fullQ

    # Write to file.
    if args.out_file:
        with open(args.out_file, 'wb') as out:
            pkl.dump(results, out)

    # Print output.
    if args.print_taus:
        if args.do_iso:
            print(f'Tau corr, iso:')
            if args.n_chunks and args.analyze_individual_trajectories:
                print(pd.DataFrame.from_dict(iso_tau_corr)[
                          ['all', 'chunks av.', 'chunks std.', 'trajs av.',
                           'trajs std.']])
            elif args.n_chunks:
                print(pd.DataFrame.from_dict(iso_tau_corr)[
                          ['all', 'chunks av.', 'chunks std.']])
            elif args.analyze_individual_trajectories:
                print(pd.DataFrame.from_dict(iso_tau_corr)[
                          ['all', 'trajs av.', 'trajs std.']])
            else:
                print(pd.DataFrame.from_dict(iso_tau_corr)['all'])
            print()
        if args.do_aniso:
            print(f"Tau corr, aniso, at PAF {args.PAF_times[0]} ps:")
            print(pd.DataFrame.from_dict(aniso_tau_corr)[
                      f"fixed PAF {args.PAF_times[0]} ps"]['all'])
            print()
    if args.print_D:
        if args.do_iso:
            print(f'D_iso, iso:')
            if args.n_chunks and args.analyze_individual_trajectories:
                print(pd.DataFrame.from_dict(iso_D_iso)[
                          ['all', 'chunks av.', 'chunks std.', 'trajs av.',
                           'trajs std.']])
            elif args.n_chunks:
                print(pd.DataFrame.from_dict(iso_D_iso)[
                          ['all', 'chunks av.', 'chunks std.']])
            elif args.analyze_individual_trajectories:
                print(pd.DataFrame.from_dict(iso_D_iso)[
                          ['all', 'trajs av.', 'trajs std.']])
            else:
                print(pd.DataFrame.from_dict(iso_D_iso)['all'])
            print()
        if args.do_aniso:
            print(f"D, aniso, at PAF {args.PAF_times[0]} ps:")
            print(pd.DataFrame.from_dict(aniso_D_iso)[
                      f"fixed PAF {args.PAF_times[0]} ps"]['all'])
            print()

    return results


if __name__ == '__main__':
    res = main()