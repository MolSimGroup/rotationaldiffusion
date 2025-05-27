from copy import copy
import numpy as np
import scipy
from scipy._lib._util import check_random_state
from scipy.optimize import OptimizeResult

from . import quaternions as qops, instantaneous_tensors, \
    construct_Q, apply_PAF_convention, construct_V


def _to_D_and_PCS(params):
    """Convert the optimization parameters to diffusion coefficients and
    PCS.

    Parameters
    ----------
    params : array_like
        The optimization parameters with the last 4 elements
        representing a quaternion that defines the PAF orientation, and
        preceding elements representing log10 of diffusion coefficients.

    Returns
    -------
    D : list
        Diffusion coefficients.
    PAF : ndarray
        Principal axes frame as a 3x3 rotation matrix.
    """
    PCS = qops.quat2rotmat(params[-4:])
    D = list(np.float_power(10, params[:-4]))
    while len(D) < 3:
        D.insert(0, D[0])
    return D, PCS


def _to_params(D, PCS):
    params = list(np.log10(D)) + list(qops.rotmat2quat(PCS))
    return params


def guess_init_params(lag_times, Q_data, model='anisotropic'):
    indices = np.nonzero(np.all(np.abs(Q_data) < 0.1, axis=(1, 2)))[0]
    ndx = 0 if indices.size == 0 else indices[-1]
    diff_coeffs_init, PCS = instantaneous_tensors(lag_times[ndx], Q_data[ndx])

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
    return _to_params(diff_params_init, PCS)


def chi2_PCS(params, lag_times, Q_data, weights=1):
    diffusion_coeffs, PCS = _to_D_and_PCS(params)
    model = construct_Q(lag_times, diffusion_coeffs)
    data = np.einsum('im,tmn,jn->tij', PCS, Q_data, PCS)
    residuals = (model - data) ** 2 * weights
    return np.sum(residuals[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])


def chi2_BODY_weigh_by_variance(params, lag_times, Q_data, weights=1):
    diffusion_coeffs, PCS = _to_D_and_PCS(params)
    model = construct_Q(lag_times, diffusion_coeffs, PCS)
    var = construct_V(lag_times, np.array(diffusion_coeffs), PCS)
    residuals = (model - Q_data) ** 2 / var
    return np.sum(residuals[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])


def local_minimization(lag_times, Q_data, weights=1, model='anisotropic',
                       chi2_func=chi2_PCS, D_init=None,
                       PCS_init=None, tol=1e-10, max_iter=1000):
    """Local optimization of diffusion coefficients and principal axes
    using a least-squares fitting procedure."""
    # Get initial parameters.
    if D_init:
        _PCS = np.eye((3, 3)) if PCS_init is None else PCS_init
        params_init = _to_params(D_init, _PCS)
    else:
        params_init = guess_init_params(lag_times, Q_data, model=model)

    def unit_quaternion_constraint(params):
        return np.sum(np.square(params[-4:])) - 1
    constraints = [{'type': 'eq', 'fun': unit_quaternion_constraint}]

    if model == 'prolate':
        def prolate_constraint(params):
            D, _ = _to_D_and_PCS(params)
            return D[1] - D[0]
        constraints.append({'type': 'eq', 'fun': prolate_constraint})

    elif model == 'oblate':
        def oblate_constraint(params):
            D, _ = _to_D_and_PCS(params)
            return D[0] - D[1]
        constraints.append({'type': 'eq', 'fun': oblate_constraint})

    res = scipy.optimize.minimize(
        chi2_func,
        params_init,
        args = (lag_times, Q_data, weights),
        method = 'trust-constr',
        constraints = constraints,
        tol = tol,
        options = {'disp': False, 'maxiter': max_iter},
    )

    D, PCS = _to_D_and_PCS(res.x)
    _PCS = PCS[np.argsort(D)]
    res._PCS = apply_PAF_convention(_PCS)
    res.D = np.sort(D)

    match model:
        case 'anisotropic':
            res.principal_axes = res._PCS
        case 'oblate':
            res.principal_axes = res._PCS[0]
        case 'prolate':
            res.principal_axes = res._PCS[2]
        case 'isotropic':
            res.principal_axes = None
    return res


def _construct_generator(start, stop, step):
    cond = min if step > 0 else max
    start -= step
    while True:
        start += step
        yield cond(start, stop)


def _metropolis(dE, beta):
    return min(1.0, np.exp(-beta * dE))


def global_optimization(lag_times, Q_data, weights=1,
                        chi2_func=chi2_PCS, D_init=None, PCS_init=None,
                        seed=None, beta_params=(0.1, 20.0, 0.05),
                        D_scale_params=(0.5, 0.05, -0.005),
                        angle_params=(90.0, 1.0, -0.5), max_iter=1000,
                        switch_freq=20, use_relative_energy=True):
    """Global optimization of diffusion coefficients and principal axes
    using a simulated annealing algorithm."""

    # Get initial parameters.
    if D_init:
        D_current = D_init
        PCS_current = np.eye(3) if PCS_init is None else PCS_init
        params_init = _to_params(D_current, PCS_current)
    else:
        params_init = guess_init_params(lag_times, Q_data, model='anisotropic')
        D_current, PCS_current = _to_D_and_PCS(params_init)

    # Initialize random state and counters.
    rng = check_random_state(seed)
    chi2_current = chi2_func(params_init, lag_times, Q_data, weights=weights)
    no_improvement_count = 0
    update_count = 0

    # Initialize generators.
    beta_gen = _construct_generator(*beta_params)
    D_scale_gen = _construct_generator(*D_scale_params)
    angle_gen = _construct_generator(*angle_params)

    # Store results.
    global_chi2_min = chi2_current
    global_D_opt = D_current
    global_PCS_opt = PCS_current
    res = OptimizeResult

    # Main annealing loop. Start with optimizing D.
    mode = 'D'
    for i in range(max_iter):
        if mode == 'D':
            scale = next(D_scale_gen) * np.array(D_current)
            _D_new = D_current + scale * rng.uniform(-0.5, 0.5, size=3)
            _PCS_new = copy(PCS_current)
            _D_new = np.sort(_D_new)
        elif mode == 'PCS':
            max_angle = np.deg2rad(next(angle_gen))
            angle = rng.uniform(0, max_angle)
            axis = rng.normal(size=3)
            quat = np.array([np.cos(angle/2), *np.sin(angle/2) * axis])
            rotation = qops.quat2rotmat(quat)
            _PCS_new = np.dot(PCS_current, rotation)
            _D_new = copy(D_current)

        _params_new = _to_params(_D_new, _PCS_new)
        _chi2_new = chi2_func(_params_new, lag_times, Q_data, weights=weights)

        # Apply Metropolis criterion.
        beta = next(beta_gen)
        dE = _chi2_new - chi2_current
        if use_relative_energy:
            dE /= chi2_current

        if rng.uniform(0, 1) < _metropolis(dE, beta):
            # Accept the new params.
            D_current, PCS_current = _to_D_and_PCS(_params_new)
            chi2_current = _chi2_new

        # Change between modes.
        if (i + 1) % switch_freq == 0:
            mode = 'PCS' if mode == 'D' else 'D'

        # Update global minimum.
        if chi2_current < global_chi2_min:
            update_count += 1
            global_chi2_min = chi2_current
            global_D_opt = D_current
            global_PCS_opt = PCS_current

    return global_D_opt, global_PCS_opt, global_chi2_min

