"""
Orientations
============

This module provides an MDAnalysis-based class for extracting the
orientations of molecular structures from MD trajectories. The
orientations are the optimal rotation matrices that align a trajectory
frame to a reference structure.
"""
from copy import copy
import warnings
import numpy as np
from MDAnalysis import AtomGroup, NoDataError
from MDAnalysis.analysis import rms, align
from MDAnalysis.lib import util
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup


class Orientations(AnalysisBase):
    r"""Determines the orientation of a molecular structure along a
    trajectory.

    This class calculates the optimal rotation matrices that align each
    frame of a trajectory to a reference structure after removing
    translation. If no reference structure is provided, the analysis
    will use the current trajectory frame of `mobile`. A (sub-)structure
    can be selected for the analysis, e.g., one molecule out of many, or
    the backbone of a protein.

    Examples
    --------
    Basic usage with the current frame as reference:

    >>> import MDAnalysis as mda
    >>> import RotationalDiffusion as rd
    >>> u = mda.Universe('protein.pdb', 'trajectory.xtc')
    >>> orient = rd.orientations.Orientations(u, select='backbone')
    >>> orient.run()
    >>> # Orientation matrices are now in orient.results.orientations
    >>> print(orient.results.orientations.shape)
    (100, 3, 3)  # For a 100-frame trajectory
    """
    _analysis_algorithm_is_parallelizable = True

    def __init__(self, mobile, reference=None, select='all', weights='mass',
                 center=True, centering_weights='weights', unwrap=True,
                 verify_match=True, tol_mass=0.1, strict=True, verbose=False):
        """ Parameters
        ----------
        mobile : AtomGroup or Universe
            Trajectory along which the orientations are to be computed.
        reference : AtomGroup or Universe, optional
            The reference structure.
        select : str or tuple or dict, default: `'all'`
            Selection string(s) defining the  :class:`AtomGroup
            <MDAnalysis.core.groups.AtomGroup>` to be used for the
            analysis. The selection must result in identical numbers of
            atoms in both mobile and reference. Options include:

            - A single string: The same selection is applied to both
            mobile and reference.
            - A 2-tuple of strings: (mobile_selection,
            reference_selection)
            - A dictionary: {'mobile': 'mobile_selection', 'reference':
            'reference_selection'}

            Must follow the `MDAnalysis Selection Syntax
            <https://userguide.mdanalysis.org/stable/selections.html>`_.
            To pass separate selection strings to `mobile` and
            `reference`, provide a 2-tuple of strings or a dictionary
            with keywords `'mobile'` and `'reference'`. The selections
            must yield a one-to-one mapping of atoms in `reference` to
            atoms in `mobile`.
        weights : None or 'mass' or :any:`array_like`, default: `'mass'`
            Weights to be used for the analysis. Options include:

            - None: use equal weights
            - `'mass'`: use masses defined in `reference`
            - :any:`array_like`: use custom weights (must match the
            number of atoms in the selection)

            Separate weights may be chosen for the centering step using
            the 'centering_weights' option.
        center : bool, default: :any:`True`
            Center molecules. Only deactivate if molecules are already
            centered.
        centering_weights : 'weights' or None or 'mass' or :any:`array_like`, default: 'weights'
            Weights to be used for centering. This option determines
            which kind of center is used. By default, the 'weights'
            option is used, which defaults to 'mass' in turn. Then,
            the obtained orientations describe rotations around the
            center-of-mass. Specifying the None option here corresponds
            to rotations around the center-of-geometry.
        unwrap : bool, default: :any:`True`
            Unwrap molecules to repair broken structures due to periodic
            boundary conditions.
        verify_match : bool, default: :any:`True`
            Whether to verify the one-to-one atom mapping of `mobile`
            and `reference` based on the residue names and atom masses
            using :func:`MDAnalysis.analysis.align.get_matching_atoms()
            <MDAnalysis.analysis.align.get_matching_atoms>`.
        tol_mass : float, default: 0.1
            Tolerance in mass, only used if `verify_match` is set to
            :any:`True`.
        strict : bool, default: True
            Only used if `verify_match` is set to :any:`True`. If
            :any:`True`, raise an error if a residue is missing an atom.
            If  :any:`False`, ignore residues with missing atoms in the
            analysis.
        verbose : bool, default: False
            Set logger to show more information and show detailed
            progress of the calculation if set to :any:`True`.

        Attributes
        ----------
        results.orientations : ndarray, shape (n_frames, 3, 3)
            Orientations represented as an array of matrices.

        Notes
        -----
        This class uses the machinery of
        `MDAnalysis <https://www.mdanalysis.org/>`_. In each frame,
        Theobald's QCP method is used to determine the orientation,
        i.e., the rotation matrix that minimizes the RMSD between mobile
        and reference after removing translational motion.
        :footcite:p:`Theobald2005, Liu2010`

        Follows loosely the AlignTraj class and RMSD class
        implementations in MDAnalysis 2.7.

        References
        ----------
        .. footbibliography::

        """
        super(Orientations, self).__init__(mobile.universe.trajectory,
                                           verbose=verbose)

        self.mobile = mobile
        self.reference = reference if reference else mobile

        select = rms.process_selection(select)
        self._mobile_atoms = self.mobile.select_atoms(*select['mobile'])
        self._ref_atoms = self.reference.select_atoms(*select['reference'])
        self._mobile_atoms, self._ref_atoms = align.get_matching_atoms(
            self._mobile_atoms, self._ref_atoms, tol_mass=tol_mass,
            strict=strict, match_atoms=verify_match
        )

        self.weights = util.get_weights(self._ref_atoms, weights)

        if centering_weights == 'weights':
            self.centering_weights = self.weights
        else:
            self.centering_weights = util.get_weights(self._ref_atoms,
                                                      centering_weights)

        self._center = center
        self._unwrap = unwrap

    def _prepare(self):
        # Make molecules whole.
        if self._unwrap:
            try:
                self._ref_atoms.unwrap()
            except (NoDataError, ValueError):
                warnings.warn('Failed to unwrap the reference system. '
                              'Continuing without unwrapping the '
                              'reference system.')

        self._ref_coordinates = copy(self._ref_atoms.positions)

        # Center the reference.
        if self._center:
            self._ref_center = self._ref_atoms.center(self.centering_weights)
            self._ref_coordinates -= self._ref_center

        # Allocate an array for storing the orientation matrices.
        self.results.orientations = np.zeros((self.n_frames, 3, 3))
        self.results._rmsd = np.zeros((self.n_frames,))

    def _single_frame(self):
        index = self._frame_index
        # Make molecules whole.
        if self._unwrap:
            self._mobile_atoms.unwrap()

        # Remove translation.
        if self._center:
            mobile_center = self._mobile_atoms.center(self.centering_weights)
            self._mobile_atoms.positions -= mobile_center

        # Compute best-fit rotation matrix.
        orientation, rmsd = align.rotation_matrix(
            self._mobile_atoms.positions,
            self._ref_coordinates,
            self.weights
        )
        self.results.orientations[index] = orientation
        self.results._rmsd[index] = rmsd

    @classmethod
    def get_supported_backends(cls):
        return ('serial', 'multiprocessing', 'dask')

    def _get_aggregator(self):
        return ResultsGroup(lookup={
            'orientations': ResultsGroup.ndarray_vstack,
            '_rmsd': ResultsGroup.ndarray_hstack
        })
