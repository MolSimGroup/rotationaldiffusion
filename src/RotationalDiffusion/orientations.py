"""
This module provides an MDAnalysis-based class for extracting the
orientations of molecular structures from MD trajectories. The
orientation is the optimal rotation of the molecule with respect to a
reference structure. "Optimal" means that the rotation minimizes the
RMSD between both structures.
"""
from copy import copy
import warnings
import numpy as np
from MDAnalysis import AtomGroup, NoDataError
from MDAnalysis.lib import util
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from MDAnalysis.analysis import rms, align


class Orientations(AnalysisBase):
    r"""Determines the orientation of a molecular structure along the
    trajectory.

    This class calculates the optimal rotations that minimize the RMSD
    between each trajectory frame and a reference structure after
    removing translation. If the reference structure is not explicitly
    specified, the analysis will use the current trajectory frame of
    ``mobile`` as the reference. (Sub-)structures of both the ``mobile``
    and ``reference`` trajectories can be selected for the analysis
    using standard `MDAnalysis selection language
    <https://userguide.mdanalysis.org/stable/selections.html>`_.
    The selected :class:`AtomGroup <MDAnalysis.core.groups.AtomGroup>`
    is centered before the optimal rotations are calculated, which are
    collected in the ``results.orientations`` attribute of this class.
    Centering may be turned off to speed up the calculation, but
    therefore the trajectories must already be centered.

    Prior to the main analysis step, periodic boundary conditions are
    treated by unwrapping the trajectory. The analysis is significangly
    faster when unwrapping is turned off, but therefore the trajectories
    must already be unwrapped.

    Parameters
    ----------
    mobile : AtomGroup or Universe
        Trajectory along which the orientations are to be computed.
    reference : AtomGroup or Universe, optional
        The reference structure.
    select : str or tuple or dict, default: `'all'`
        Selection string(s) defining the  :class:`AtomGroup
        <MDAnalysis.core.groups.AtomGroup>`\ (s) to be used for the
        analysis. The string(s) must follow the `MDAnalysis selection
        syntax
        <https://userguide.mdanalysis.org/stable/selections.html>`_ . If
        a single string is provided, it is applied to both ``mobile``
        and ``reference``. To apply different selections, specify a
        2-:any:`tuple` of selection strings; the first string is applied
        to ``mobile``, the second to ``reference``. Alternatively,
        specify a dictionary with the keys `'mobile'` and `'reference'`
        mapped to selection strings for ``mobile`` and ``reference``.
        After the selection has been applied, there must be a
        one-to-one correspondence of atoms in ``mobile`` to atoms in
        ``reference``.
    weights : None or 'mass' or :any:`array_like`, default: `'mass'`
        Weights to be used for determining the orientations. If
        :any:`None`, use equal weights. If `'mass'`, use the atom masses
        defined in
        ``reference`` (default). Use custom weights by providing an
        :any:`array_like`, which must contain one weight for each
        selected atom.
    center : bool, default: :any:`True`
        Center molecules. Only deactivate if molecules are already
        centered.
    centering_weights : 'weights' or None or 'mass' or :any:`array_like`, default: 'weights'
        Weights to be used for centering. If `'weights'`, the same
        weights are used as specified in the ``weights`` option
        (default). Otherwise, custom weights can be specified (see
        ``weights`` option). If masses are used for centering, the
        obtained orientations describe rotations around the
        center-of-mass. Similarly, if equal weights are used here, the
        orientations describe rotations around the center-of-geometry.
    unwrap : bool, default: :any:`True`
        Unwrap molecules to repair broken structures due to periodic
        boundary conditions. Only deactivate if molecules are already
        whole.
    verify_match : bool, default: :any:`True`
        Whether to verify the one-to-one atom mapping of ``mobile``
        and ``reference`` based on the residue names and atom masses
        using :func:`MDAnalysis.analysis.align.get_matching_atoms()
        <MDAnalysis.analysis.align.get_matching_atoms>`.
    tol_mass : float, default: 0.1
        Tolerance in mass for identifying identical atoms. Only used if
        ``verify_match`` is set to :any:`True`.
    strict : bool, default: True
        Only used if `verify_match` is set to :any:`True`. If
        :any:`True`, raise an error if a residue is missing an atom.
        If  :any:`False`, ignore residues with missing atoms in the
        analysis.
    verbose : bool, default: False
        Show detailed progress if set to :any:`True`.

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
    _analysis_algorithm_is_parallelizable = True

    def __init__(self, mobile, reference=None, select='all', weights='mass',
                 center=True, centering_weights='weights', unwrap=True,
                 verify_match=True, tol_mass=0.1, strict=True, verbose=False):
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

        # Create an array for storing the orientation matrices.
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

    def run(self, **kwargs):
        # For documentation purposes only.
        """Perform the calculation.

        Parameters
        ----------
        start : int, optional
            Start frame of analysis.
        stop : int, optional
            Stop frame of analysis.
        step : int, optional
            Number of frames to skip between each analysed frame.
        frames : array_like, optional
            Array of integers or booleans to slice trajectory;
            ``frames`` can only be used *instead* of ``start``,
            ``stop``, and ``step``. Setting *both* ``frames`` and at
            least one of ``start``, ``stop``, ``step`` to a non-default
            value will raise a :exc:`ValueError`.
        verbose : bool, optional
            Turn on verbosity.
        progressbar_kwargs : dict, optional
            ProgressBar keywords with custom parameters regarding
            progress bar position, etc; see
            :class:`MDAnalysis.lib.log.ProgressBar` for full list.
        backend : str or subclass of BackendBase , default 'serial'
            Backend to be used for analysis. Supported backends are
            listed in the :any:`parallelizable <MDAnalysis.analysis.base.AnalysisBase.parallelizable>`
            property of the :class:`Orientations` class. Otherwise,
            a custom backend that is a subclass of
            :class:`BackendBase <MDAnalysis.analysis.backends.BackendBase>`
            may be provided, but then the ``unsupported_backend`` flag
            must be set.
        n_workers : int
            Positive integer with number of workers (processes, in case
            of built-in backends) to split the work between. Only
            available for backends other than 'serial'.
        n_parts : int, optional
            Number of parts to split computations across. Can be more
            than number of workers. Only available for backends other
            than 'serial'.
        unsupported_backend : bool, default :any:`False`
            Must be set to :any:`True` to run with a custom backend.
        """
        super().run(**kwargs)
        return self