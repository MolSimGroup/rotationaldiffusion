import numpy as np
import scipy
import pydiffusion as pdiff

PAF = scipy.spatial.transform.Rotation.random().as_matrix()
qi2_qj2 = np.random.random_sample((3, 3, 1000))
qi2_qj2[0, 1] = qi2_qj2[1, 0]
qi2_qj2[1, 2] = qi2_qj2[2, 1]
qi2_qj2[0, 2] = qi2_qj2[2, 0]

def rotate_3_times_known_path(qi2_qj2, PAF):
    qi2_qj2_rotated = np.einsum('hi,ijx,jk->hkx', PAF.T**2, qi2_qj2, PAF**2,
                                optimize=['einsum_path', (0, 1), (0, 1)])
    qi2_qj2_rotated += 2 * np.einsum('hi,ki,ijx,jh,jk->hkx', PAF.T, PAF.T, qi2_qj2, PAF, PAF,
                                     optimize=['einsum_path', (0, 1), (1, 2), (1, 2), (0, 1)])
    qi2_qj2_rotated -= 2 * np.einsum('hi,ki,iix,ih,ik->hkx', PAF.T, PAF.T, qi2_qj2, PAF, PAF,
                                     optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    return qi2_qj2_rotated

def rotate_individually_no_loop(qi2_qj2, PAF):
    qi2_qj2_rotated = np.einsum('ih,ijx,jk->hkx', PAF**2, qi2_qj2, PAF**2,
                                optimize=['einsum_path', (0, 1), (0, 1)])
    tmp = PAF[[0, 0, 1]] * PAF[[1, 2, 2]]
    qi2_qj2_rotated += 4 * np.einsum('ai,at,aj->ijt', tmp, qi2_qj2[(0, 0, 1), (1, 2, 2)], tmp,
                                     optimize=['einsum_path', (0, 2), (0, 1)])
    return qi2_qj2_rotated


def rotate_using_tensordot(qi2_qj2, PAF):
    qi2_qj2_rotated = np.tensordot(PAF**2, np.tensordot(PAF.T**2, qi2_qj2, axes=1), axes=((0,), (1,)))
    tmp = PAF[[0, 0, 1]] * PAF[[1, 2, 2]]
    qi2_qj2_rotated += 4 * np.tensordot(np.multiply(tmp[:, :, np.newaxis], tmp[:, np.newaxis, :]), qi2_qj2[(0, 0, 1), (1, 2, 2)], axes=((0,),(0,)))
    return qi2_qj2_rotated

def rotate_using_matmul(qi2_qj2, PAF):
    axes = [[0], [1]]
    qi2_qj2_rotated = np.tensordot(PAF**2, np.tensordot(PAF**2, qi2_qj2, axes=axes), axes=axes)
    tmp = PAF[[0, 0, 1]] * PAF[[1, 2, 2]]
    qi2_qj2_rotated += 4 * np.matmul(np.multiply(tmp.T[:, np.newaxis, :], tmp.T[np.newaxis, :, :]), qi2_qj2[(0, 0, 1), (1, 2, 2)])
    return qi2_qj2_rotated

def rotate_using_tensordot2(qi2_qj2, PAF):
    qi2_qj2 = np.moveaxis(qi2_qj2, -1, 0)
    qi2_qj2_rotated = np.matmul(np.matmul(PAF.T**2, qi2_qj2), PAF**2)
    # qi2_qj2_rotated = qi2_qj2_rotated.transpose(-2, -1, 0)
    # qi2_qj2 = qi2_qj2.transpose(-2, -1, 0)
    tmp = PAF[[0, 0, 1]] * PAF[[1, 2, 2]]
    # qi2_qj2_rotated += 4 * np.matmul(np.multiply(tmp.T[:, np.newaxis, :], tmp.T[np.newaxis, :, :]), qi2_qj2[:, (0, 0, 1), (1, 2, 2)].T)
    qi2_qj2_rotated += 4 * np.matmul(qi2_qj2[:, (0, 0, 1), (1, 2, 2)], np.multiply(tmp.T[:, :, np.newaxis], tmp[np.newaxis, :, :])).transpose(1, 0, 2)
    return qi2_qj2_rotated.transpose(-2, -1, 0)

def rotate_using_matmul_reverse(qi2_qj2, PAF):
    axes = [[0], [1]]
    qi2_qj2_rotated = np.matmul(np.matmul(PAF.T**2, qi2_qj2), PAF**2)
    tmp = PAF[[0, 0, 1]] * PAF[[1, 2, 2]]
    qi2_qj2_rotated += 4 * np.tensordot(qi2_qj2[:, (0, 0, 1), (1, 2, 2)], np.multiply(tmp[:, :, np.newaxis], tmp[:, np.newaxis, :]), axes=[[1], [0]])
    return qi2_qj2_rotated

def rotate_testing(qi2_qj2, PAF):
    qi2_qj2_rotated = np.tensordot(np.tensordot(PAF**2, PAF**2, axes=0), qi2_qj2, axes=((0, 2), (0, 1)))
    tmp = PAF * PAF[(1, 2, 0),]
    qi2_qj2_rotated += 4 * np.tensordot(np.multiply(tmp[:, :, np.newaxis], tmp[:, np.newaxis, :]), qi2_qj2[(0, 1, 0), (1, 2, 2)], axes=(0, 0))
    return qi2_qj2_rotated

def _rotate_m4(qi2_qj2, PAF):
    PAF_2nd_power_shift = PAF * PAF[(1, 2, 0),]
    PAF_4th_power_shift = np.multiply(PAF_2nd_power_shift.T[:, np.newaxis, :],
                                      PAF_2nd_power_shift.T[np.newaxis, :, :])

    qi2_qj2_rotated = np.tensordot(np.tensordot(PAF**2, PAF**2, axes=0), qi2_qj2, axes=((0, 2), (0, 1)))
    qi2_qj2_rotated += 4 * np.matmul(PAF_4th_power_shift, qi2_qj2[(0, 1, 0), (1, 2, 2)])
    return qi2_qj2_rotated



from copy import copy

for i in range(10000):
    _rotate_m4(qi2_qj2, PAF)