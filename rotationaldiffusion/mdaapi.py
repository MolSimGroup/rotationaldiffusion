"""Rotational diffusion
====================
"""
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations as trans, AtomGroup
from MDAnalysis.analysis import rms, align
from MDAnalysis.lib import util
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.coordinates.memory import MemoryReader


def load_universes(topology, *coordinates, **kwargs):
    """Load (one or) several trajectories of one system into separate
    :class:`Universe <MDAnalysis.core.universe.Universe>` objects.

    Parameters
    ----------
    topology : str
        Path to the topology file.
    *coordinates : str
        Paths to the trajectory files. Each path should be a separate
        argument.
    **kwargs :
        Additional keyword arguments to be passed to the
        :class:`Universe <MDAnalysis.core.universe.Universe>`
        constructor.

    Returns
    -------
    universes : list
        A list of universes, one for each argument in `*coordinates`.

    Warnings
    --------
    If you pass a container containing the paths to several trajectory
    files as one argument to `*coordinates`, then all trajectories will
    be loaded into the same
    :class:`Universe <MDAnalysis.core.universe.Universe>`, which is
    likely not intended when using this function, as discussed below.
    Use star notation to pass all paths in the container separately,
    i.e., `*[traj1, traj2, ...]`, and always double check if the number
    of obtained universes matches your expectations!

    Notes
    -----
    Intended for analyzing several independent trajectories of one
    system, in cases when the trajectories must NOT be treated as
    continuous, hence NOT loaded into a single
    :class:`Universe <MDAnalysis.core.universe.Universe>`. This function
    returns a list of universes, one for each trajectory. Identical
    analyses can be applied easily to all trajectories by looping over
    that list.

    """
    universes = []
    for coords in coordinates:
        u = mda.Universe(topology, coords, **kwargs)
        universes.append(u)
    return universes


class Orientations(AnalysisBase):
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


def get_orientations(universes, reference=None, select='all', in_memory=False,
                     **kwargs):
    """Run the :class:`Orientations` analysis class in one go for
    several trajectories and/or selections.

    Analyze several trajectories at once by providing a list of
    :class:`Universe <MDAnalysis.core.universe.Universe>` or
    :class:`AtomGroup <MDAnalysis.core.groups.AtomGroup>` objects.
    The trajectories must match in length and should match in timestep.

    Analyze the orientations of several atom groups by providing a list
    selections, one for each :class:`AtomGroup
    <MDAnalysis.core.groups.AtomGroup>`. All selections are applied
    separately to each of the trajectories.

    The returned :class:`ndarray <numpy.ndarray>` has five dimensions.
    The first dimension corresponds to the trajectories, the second to
    the selections, the third to the timsteps, and the fourth and fifth
    dimension contain the orientation matrices. Example: `N`
    trajectories are analyzed using `S` selection strings each, and each
    trajectory contains `T` timesteps, then the resulting array has the
    shape `(N, S, T, 3, 3)`.

    Parameters
    ----------
    universes : Universe or AtomGroup or list thereof
        Each object must contain exactly one trajectory to be analyzed.
    reference : Universe or AtomGroup
        The reference configuration used for all trajectories.
    select : str or tuple or dict or list thereof, default: `'all'`
        Selections defining the  :class:`AtomGroup
        <MDAnalysis.core.groups.AtomGroup>` to be analyzed. If a
        :any:`list` is passed, all selections within that list will be
        applied to each entry in `universes`. Must follow the
        `MDAnalysis Selection Syntax
        <https://userguide.mdanalysis.org/stable/selections.html>`_.
        To pass separate selection strings to `mobile` and `reference`,
        provide a 2-tuple of strings or a dictionary with keywords
        `'mobile'` and `'reference'` for each selection. All selections
        must yield a one-to-one mapping of atoms in `reference` to
        atoms in `mobile`.
    in_memory : bool, default: `False`
        Load each trajectory to memory before analyzing it.
    **kwargs :
        Additional keyword arguments to be passed to the
        :class:`Orientations` analysis class. Important arguments may
        include `weights` and `unwrap`.

    Returns
    -------
    orientations : ndarray, shape (N, S, T, 3, 3)
        Orientations along all `N` trajectories considering all `S`
        selections. `T` is the number of timesteps in each trajectory.

    Warnings
    --------
    It is highly recommended to pass an explicit reference
    configuration. Otherwise, the returned orientations of different
    trajectories or selections are likely incomparable, since they
    consider different reference orientations.
    """
    if isinstance(select, str):
        select = [select]
    if isinstance(universes, mda.Universe) or isinstance(universes, AtomGroup):
        universes = [universes]

    orientations = np.zeros((len(universes), len(select),
                             universes[0].universe.trajectory.n_frames, 3, 3))

    for i, u in enumerate(universes):
        if in_memory:
            u.universe.transfer_to_memory()
        for j, sel in enumerate(select):
            orientations[i, j] = Orientations(
                u, reference=reference, select=sel, **kwargs
            ).run().results.orientations
    return orientations


def run_all():
    # Load universe.
    # (Compute average structure).
    # RMSD-fit trajectory to selection and frame.
    # Compute quaternion covariance matrix.
    # (Compute instantaneous tensors).
    # (Plot intermediate results).
    # Least-squares fit.
    # (Plot and/or report results).
    return
