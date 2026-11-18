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


def REF_to_PCS(structure, PCS):
    """Rotate the reference structure into the PCS.

    The rotation happens in-place returning the altered input object.

    Parameters
    ----------
    structure : Universe or AtomGroup
        The reference structure to be rotated into the PCS.
    PCS: (3, 3) ndarray
        The PCS.

    Returns
    -------
    Universe or AtomGroup
        The structure rotated into the PCS.
    """
    new_pos = np.einsum('im,nm', PCS, structure.atoms.positions).T
    structure.atoms.positions = new_pos
    return structure