import numpy as np

from src.rotationaldiffusion import apply_PAF_convention


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
