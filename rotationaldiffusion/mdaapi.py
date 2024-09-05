"""Rotational diffusion
====================
"""
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.analysis import rms, align
from MDAnalysis.lib import util
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.coordinates.memory import MemoryReader


def load_universes(topology, *coordinates, **kwargs):
    """Convenience function to load (one or) several trajectories of one
    single system into separate
    :class:`Universe <MDAnalysis.core.universe.Universe>` objects.

    Exemplary use case: you want to analyze several independent
    trajectories of one system, but the trajectories must NOT be treated
    as continuous, hence NOT loaded into a single
    :class:`Universe <MDAnalysis.core.universe.Universe>`. This function
    returns a list of universes, one for each trajectory. All
    trajectories can be analyzed easily by looping over that list.

    Parameters
    ----------
    topology : str
        Path to the topology file.
    *coordinates : str
        Paths to the trajectory files. Each path should be a separate
        argument. If you provide several paths in a container such as a
        list, the container will be passed as a whole to the
        :class:`Universe <MDAnalysis.core.universe.Universe>`
        constructor, which corresponds to concatenating the
        trajectories. Use star notation to avoid this, i.e., `*[traj1,
        traj2, ...]`.
    **kwargs :
        Additional keyword arguments to be passed to the
        :class:`Universe <MDAnalysis.core.universe.Universe>`
        constructor.

    Returns
    -------
    universes : list
        A list of universes, one for each argument in `*coordinates`.
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
    removed beforehand.

    This class uses the machinery of
    `MDAnalysis <https://www.mdanalysis.org/>`_. To use it, first create
    a class instance, then call its :meth:`run` method. In each frame,
    Theobald's QCP method is used to determine the orientation, i.e.,
    the rotation matrix that minimizes the RMSD between mobile and
    reference after removing translational
    motion. :footcite:p:`Theobald2005, Liu2010` The orientation matrices
    are stored in `results.orientations`.

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
        Selection string that is passed to
        :class:`AtomGroup.select_atoms()
        <MDAnalysis.core.groups.AtomGroup>`
        to construct the :meth:`AtomGroup
        <MDAnalysis.core.groups.AtomGroup.select_atoms>` to operate on.
        Must follow the `MDAnalysis Selection Syntax
        <https://userguide.mdanalysis.org/stable/selections.html>`_.
        It is possible to pass separate selection strings to `mobile`
        and `reference` by providing a 2-tuple of strings or a
        dictionary with keywords `'mobile'` and `'reference'`. However,
        the selection strings must define a one-to-one mapping of atoms
        between mobile and reference, which will be checked using
        :func:`MDAnalysis.analysis.align.get_matching_atoms()
        <MDAnalysis.analysis.align.get_matching_atoms>` if
        `verify_match` is set to :any:`True`.
    weights : None or 'mass' or :any:`array_like`, default: None
        Weights which will be used for removing translation and for
        determining the orientation. Weigh atoms equally with
        :any:`None`, use masses of `reference` as weights with
        `'mass'`, or provide an :any:`array_like` of weights, which must
        contain exactly one weight per atom in the selection.
    unwrap : :any:`bool`, default: :any:`True`
        Make broken molecules whole using an on-the-fly transformation.
        May be set to :any:`False` if the molecules in `mobile` and
        `reference` are already whole.
    verify_match : :any:`bool`, default: :any:`True`
        Whether to verify the one-to-one atom mapping of `mobile` and
        `reference` based on the residue names and atom masses using
        :func:`MDAnalysis.analysis.align.get_matching_atoms()
        <MDAnalysis.analysis.align.get_matching_atoms>`.
    tol_mass : float, default: 0.1
        Tolerance in mass, only used if `verify_match` is set to
        :any:`True`.
    strict : :any:`bool`, default: True
        Only used if `verify_match` is set to :any:`True`. If
        :any:`True`, raise an error if a residue is missing an atom. If
        :any:`False`, ignore residues with missing atoms in the
        analysis.
    verbose : :any:`bool`, default: True
        Set logger to show more information and show detailed progress
        of the calculation if set to :any:`True`.

    Attributes
    ----------
    results.orientations : :class:`ndarray <numpy.ndarray>`
        Orientations represented as array of matrices with shape
        *(n_frames, 3, 3)*.

    Notes
    -----
    Follows loosely the AlignTraj class and RMSD class implementations
    in MDAnalysis 2.7.
    """
    def __init__(self, mobile, reference=None, select='all', weights=None,
                 unwrap=True, verify_match=True, tol_mass=0.1, strict=True,
                 verbose=True):
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

        # Fix PBC jumps by unwrapping the trajectory.
        if unwrap:
            self.mobile.universe.trajectory.add_transformations(
                trans.unwrap(self._mobile_atoms))
        if unwrap and reference:
            self.reference.universe.trajectory.add_transformations(
                trans.unwrap(self._ref_atoms))

    def _prepare(self):
        # Center the reference.
        self._ref_center = self._ref_atoms.center(self.weights)
        self._ref_coordinates = self._ref_atoms.positions - self._ref_center
        # Allocate an array for storing the orientation matrices.
        self.results.orientations = np.zeros((self.n_frames, 3, 3))
        self.results._rmsd = np.zeros((self.n_frames,))

    def _single_frame(self):
        index = self._frame_index
        # Remove translational motion, compute best-fit rotation matrix.
        mobile_center = self._mobile_atoms.center(self.weights)
        mobile_coordinates = self._mobile_atoms.positions - mobile_center
        orientation, rmsd = align.rotation_matrix(mobile_coordinates,
                                                  self._ref_coordinates,
                                                  self.weights)
        self.results.orientations[index] = orientation
        self.results._rmsd[index] = rmsd


def get_orientations(*universes, reference=None, select='all', unwrap=True,
                     **kwargs):
    """ Convenience wrapper around :class:`Orientations`, to analyze
    multiple trajectories and/or multiple selections at once.


    The trajectory of each
    :class:`Universe <MDAnalysis.core.universe.Universe>` will be
    treated as continuous. To analyze multiple independent (i.e.,
    discontinuous) trajectories, load each trajectory into its own
    :class:`Universe <MDAnalysis.core.universe.Universe>`. The
    trajectories should match in length and timestep.

    A list of selection strings may be provided instead of a single
    selection string. In that case, each string is applied to each
    trajectory, separately. The returned
    :class:`ndarray <numpy.ndarray>` has four dimensions. Multiple
    selections are considered along the first dimension, multiple
    trajectories in the second dimension.

    Parameters
    ----------
    universes : :class:`Universe <MDAnalysis.core.universe.Universe>` or AtomGroup
        Each object contains one trajectory to be analyzed.

    select : :class:`str` or :class:`list` of :class:`str`

    Returns
    -------
    orientations : ndarray, shape (n_selections, n_universes, n_frames, 3, 3)
        Orientations of each selection along each trajectory.
    """
    # If only a single universe or selection string was provided, put
    # it in a list.
    if isinstance(select, str):
        select = [select]

    # Allocate result array.
    n_trajectories, n_selections, n_frames = len(universes), len(select), universes[0].universe.trajectory.n_frames
    orientations = np.zeros((n_trajectories, n_selections, n_frames, 3, 3))

    # n_results = len(universes) * len(select)
    # n_calls_to_orientations = {
    #     'zip': max(len(universes), len(select)),
    #     'product': len(universes) * len(select)}[mapping]
    # orientations = np.zeros((n_calls_to_orientations,
    #                          universes[0].universe.trajectory.n_frames, 3, 3))

    # Main computation.
    # mapping = {'zip': zip, 'product': itertools.product}[mapping]
    for i, u in enumerate(universes):
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
