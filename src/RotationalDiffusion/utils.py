from typing import List, Dict, Any, Union, Optional, Type
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase


def arange_lag_times(Q, timestep):
    return np.arange(1, Q.shape[-3] + 1, 1) * timestep

def apply_PCS_convention(PCSs):
    assert PCSs.shape[-2:] == (3, 3)
    PCSs = np.copy(PCSs)

    # Enforce positive elements 11 and 22.
    PCSs[np.diagonal(PCSs, axis1=-2, axis2=-1) < 0] *= -1

    # Make PAFs right-handed (by inverting x-axis).
    if PCSs.ndim > 2:
        PCSs[np.linalg.det(PCSs) < 0, 0] *= -1
    elif np.linalg.det(PCSs) < 0:
        PCSs[0] *= -1
    return PCSs


