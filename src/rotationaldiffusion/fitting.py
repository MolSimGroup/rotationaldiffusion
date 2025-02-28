import numpy as np
import scipy

from . import quaternions as qops, instantaneous_tensors, \
    construct_Q_model, apply_PAF_convention


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
