import numpy as np
from tqdm.asyncio import tqdm

from .utils import arange_lag_times
from .fitting import least_squares_fit
# from pydiffusion import quaternionsimulation as qsim
from .correlations import correlate


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
