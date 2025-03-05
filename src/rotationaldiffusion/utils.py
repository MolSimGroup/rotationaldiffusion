from typing import List, Dict, Any, Union, Optional, Type
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase


def arange_lag_times(Q, timestep):
    return np.arange(1, Q.shape[-3] + 1, 1) * timestep

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


