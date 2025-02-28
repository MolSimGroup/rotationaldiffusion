"""Orientations
====================
"""
import numpy as np
from MDAnalysis import AtomGroup
from MDAnalysis.analysis import rms, align
from MDAnalysis.lib import util
from MDAnalysis.analysis.base import AnalysisBase


class Orientations(AnalysisBase):
    # TODO: Parallelize the `Orientations` analysis class.
    r"""Determines the orientation of an
    :class:`AtomGroup <MDAnalysis.core.groups.AtomGroup>` along the
    trajectory.

    The orientation is the rotation that minimizes the
    root-mean-square deviations of atomic positions. Translation is
    removed beforehand. After initializing this class, call its
    :meth:`run` method to perform the analysis. Afterwards, the
    orientation matrices are available in the `results.orientations`
    attribute.

    Parameters
    ----------
    mobile : AtomGroup or Universe
        Group of atoms for which the orientations along the trajectory
        are calculated.
    reference : AtomGroup or Universe, optional
        Group of atoms defining the reference orientation. (The default
        is :any:`None`, which implies using the current frame of
        `mobile`).
    select : str or tuple or dict, default: `'all'`
        Selection defining the  :class:`AtomGroup
        <MDAnalysis.core.groups.AtomGroup>` to be analyzed. Must follow
        the `MDAnalysis Selection Syntax
        <https://userguide.mdanalysis.org/stable/selections.html>`_.
        To pass separate selection strings to `mobile` and `reference`,
        provide a 2-tuple of strings or a dictionary with keywords
        `'mobile'` and `'reference'`. The selections must yield a
        one-to-one mapping of atoms in `reference` to atoms in `mobile`.
    weights : None or 'mass' or :any:`array_like`, default: None
        Weights which will be used for removing translation and for
        determining the orientation. Weigh atoms equally with
        :any:`None`, use masses of `reference` as weights with
        `'mass'`, or provide an :any:`array_like` of weights, which must
        contain exactly one weight per atom in the selection.
    unwrap : bool, default: :any:`True`
        Make broken molecules whole using an on-the-fly transformation.
        May be set to :any:`False` if the molecules in `mobile` and
        `reference` are already whole.
    verify_match : bool, default: :any:`True`
        Whether to verify the one-to-one atom mapping of `mobile` and
        `reference` based on the residue names and atom masses using
        :func:`MDAnalysis.analysis.align.get_matching_atoms()
        <MDAnalysis.analysis.align.get_matching_atoms>`.
    tol_mass : float, default: 0.1
        Tolerance in mass, only used if `verify_match` is set to
        :any:`True`.
    strict : bool, default: True
        Only used if `verify_match` is set to :any:`True`. If
        :any:`True`, raise an error if a residue is missing an atom. If
        :any:`False`, ignore residues with missing atoms in the
        analysis.
    verbose : bool, default: False
        Set logger to show more information and show detailed progress
        of the calculation if set to :any:`True`.

    Attributes
    ----------
    results.orientations : ndarray, shape (n_frames, 3, 3)
        Orientations represented as an array of matrices.

    Notes
    -----
    This class uses the machinery of
    `MDAnalysis <https://www.mdanalysis.org/>`_. In each frame,
    Theobald's QCP method is used to determine the orientation, i.e.,
    the rotation matrix that minimizes the RMSD between mobile and
    reference after removing translational
    motion. :footcite:p:`Theobald2005, Liu2010`

    Follows loosely the AlignTraj class and RMSD class implementations
    in MDAnalysis 2.7.

    References
    ----------
    .. footbibliography::
    """
    def __init__(self, mobile, reference=None, select='all', weights=None,
                 unwrap=True, verify_match=True, tol_mass=0.1, strict=True,
                 verbose=False):
        super(Orientations, self).__init__(mobile.universe.trajectory,
                                           verbose=verbose)

        self.mobile = mobile
        self.reference = reference if reference else mobile

        select = rms.process_selection(select)
        self._mobile_atoms = self.mobile.select_atoms(*select['mobile'])
        self._ref_atoms = self.reference.select_atoms(*select['reference'])
        self._mobile_atoms, self._ref_atoms = align.get_matching_atoms(
            self._mobile_atoms, self._ref_atoms, tol_mass=tol_mass,
            strict=strict, match_atoms=verify_match)

        weights = np.array(weights) if weights is not None else None
        self.weights = util.get_weights(self._ref_atoms, weights)
        self._unwrap = unwrap

    def _prepare(self):
        # Make molecules whole.
        if self._unwrap:
            self._ref_atoms.unwrap()
        # Center the reference.
        self._ref_center = self._ref_atoms.center(self.weights)
        self._ref_coordinates = self._ref_atoms.positions - self._ref_center
        # Allocate an array for storing the orientation matrices.
        self.results.orientations = np.zeros((self.n_frames, 3, 3))
        self.results._rmsd = np.zeros((self.n_frames,))

    def _single_frame(self):
        index = self._frame_index
        # Make molecules whole.
        if self._unwrap:
            self._mobile_atoms.unwrap()
        # Remove translation.
        mobile_center = self._mobile_atoms.center(self.weights)
        mobile_coordinates = self._mobile_atoms.positions - mobile_center
        # Compute best-fit rotation matrix.
        orientation, rmsd = align.rotation_matrix(mobile_coordinates,
                                                  self._ref_coordinates,
                                                  self.weights)
        self.results.orientations[index] = orientation
        self.results._rmsd[index] = rmsd


def load_orientations(*files, start=None, stop=None, step=None):
    """
    Load orientations obtained using `gmx rotmat` from disk.

    Several files can be loaded at once by passing the path to each file
    as a separate argument. All files must contain the same number of
    orientations.

    The `start`, `stop`, and `step` arguments can be used to load only
    parts of each file using `numpy slicing <https://numpy.org/doc/
    stable/user/basics.indexing.html#slicing-and-striding>`_.

    Parameters
    ----------
    files: str
        Paths to the 'gmx rotmat' output files. Each path should be a
        separate argument.
    start, stop : int, optional
        Index of first and last orientation to load.
    step : int, optional
        Number of orientations to skip between each loaded orientation.

    Returns
    -------
    orientations : ndarray, shape (N, T, 3, 3)
        Orientation matrices loaded from `N` `files`. `T` is the number
        of orientations loaded from each file.
    time : ndarray, shape (N, T)
        Time information corresponding to the matrices in
        `orientations`.
    """
    orientations, time = [], []
    for file in files:
        data = np.loadtxt(file, comments=['#', '@'])[start:stop:step]
        orientations.append(data[:, 1:].reshape(-1, 3, 3))
        time.append(data[:, 0])
    orientations, time = np.array(orientations), np.array(time)
    return orientations, time
