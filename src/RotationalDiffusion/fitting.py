from copy import copy
import numpy as np
import scipy
from scipy._lib._util import check_random_state

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
    larger_125 = np.any(np.abs(np.abs(Q_data)) > 0.1, axis=(0, 1))
    ndx = np.argmax(larger_125) if larger_125.any() else -1
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


def _chi2_PCS(params, lag_times, Q_data, weights=1):
    diffusion_coeffs, PCS = _to_D_and_PCS(params)
    model = construct_Q(lag_times, diffusion_coeffs)
    data = np.einsum('im,tmn,jn->tij', PCS, Q_data, PCS)
    residuals = (model - data) ** 2 * weights
    return np.mean(residuals[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])


def _chi2_BODY(params, lag_times, Q_data, weights=1):
    diffusion_coeffs, PCS = _to_D_and_PCS(params)
    model = construct_Q(lag_times, diffusion_coeffs, PCS)
    residuals = (model - Q_data) ** 2 * weights
    return np.mean(residuals[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])


def _chi2_BODY_variance_weights(params, lag_times, Q_data, weights=1):
    diffusion_coeffs, PCS = _to_D_and_PCS(params)
    model = construct_Q(lag_times, diffusion_coeffs, PCS)
    var = construct_V(lag_times, np.array(diffusion_coeffs), PCS)
    residuals = (model - Q_data) ** 2 / var
    return np.sum(residuals[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])


def optimize(params_init, constraints, lag_times, Q_data, tol=1e-10,
             maxiter=1000):
    # Deprecated
    res = scipy.optimize.minimize(_chi2_PCS, params_init, tol=tol,
                                  args=(lag_times, Q_data),
                                  constraints=constraints,
                                  method='trust-constr',
                                  options={'disp': False,
                                           'maxiter': maxiter})
    return res


def local_minimization(lag_times, Q_data, weights=1, model='anisotropic',
                       chi2_func=_chi2_PCS, D_init=None,
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


def least_squares_fit(lag_times, Q_data, model='anisotropic',
                      tol=1e-10, maxiter=1000, tmp=None):
    # Deprecated
    params_init = guess_init_params(lag_times, Q_data, model)

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
    D, PAF = _to_D_and_PCS(res.x)
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


def _construct_generator(start, stop, step):
    cond = min if step > 0 else max
    start -= step
    while True:
        start += step
        yield cond(start, stop)


def _metropolis(dE, beta):
    return min(1.0, np.exp(-beta * dE))


def global_optimization(lag_times, Q_data, weights=1,
                        chi2_func=_chi2_PCS, D_init=None, PCS_init=None,
                        seed=None, beta_params=(0.1, 20.0, 0.05),
                        D_scale_params=(0.5, 0.05, -0.005), eps=1,
                        angle_params=(90.0, 1.0, -0.5), max_iter=1000,
                        success_iter=5, switch_freq=20):
    """Global optimization of diffusion coefficients and principal axes
    using a simulated annealing algorithm."""
    # Initialize random state.
    rng = check_random_state(seed)

    # Get initial parameters.
    if D_init:
        D_current = D_init
        PCS_current = np.eye((3, 3)) if PCS_init is None else PCS_init
        params_init = _to_params(D_current, PCS_current)
    else:
        params_init = guess_init_params(lag_times, Q_data, model='anisotropic')
        D_current, PCS_current = _to_D_and_PCS(params_init)
    chi2_current = chi2_func(params_init, lag_times, Q_data, weights=weights)
    global_chi2_min = chi2_current
    global_D_opt = D_current
    global_PCS_opt = PCS_current

    beta_gen = _construct_generator(*beta_params)
    D_scale_gen = _construct_generator(*D_scale_params)
    angle_gen = _construct_generator(*angle_params)
    no_improvement_count = 0

    # Main annealing loop. Start with optimizing D.
    mode = 'D'
    for i in range(max_iter):
        if mode == 'D':
            scale = next(D_scale_gen) * np.array(D_current)
            _D_new = D_current + scale * rng.uniform(-0.5, 0.5, size=3)
            _PCS_new = copy(PCS_current) #[np.argsort(_D_new)]
            _D_new = np.sort(_D_new)[::-1]
        elif mode == 'PCS':
            max_angle = np.deg2rad(next(angle_gen))
            angle = rng.uniform(0, max_angle)
            axis = rng.uniform(-1, 1, 3)
            axis /= np.linalg.norm(axis)
            quat = np.array([np.cos(angle/2), *np.sin(angle/2) * axis])
            rotation = qops.quat2rotmat(quat)
            _PCS_new = np.dot(PCS_current, rotation)
            _D_new = copy(D_current)

        _params_new = _to_params(_D_new, _PCS_new)
        _chi2_new = chi2_func(_params_new, lag_times, Q_data, weights=weights)

        # Apply Metropolis criterion.
        dE = _chi2_new - chi2_current
        beta = next(beta_gen)
        if rng.uniform(0, 1) < _metropolis(dE, beta):
            # Accept the new params.
            D_current, PCS_current = _to_D_and_PCS(_params_new)
            chi2_current = _chi2_new
        elif np.abs(dE) < eps * chi2_current:
            # Reject and increment rejection counter.
            no_improvement_count += 1
        else:
            # Reject and reset rejection counter, bc. change was too large.
            no_improvement_count = 0

        # Check for convergence.
        if no_improvement_count > success_iter:
            break

        # Change between modes.
        if (i + 1) % switch_freq == 0:
            mode = 'PCS' if mode == 'D' else 'D'

        if chi2_current < global_chi2_min:
            global_chi2_min = chi2_current
            global_D_opt = D_current
            global_PCS_opt = PCS_current

    return global_D_opt, global_PCS_opt, global_chi2_min

