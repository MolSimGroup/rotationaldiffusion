"""Rotational diffusion
====================

"""
import itertools
import numpy as np
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import rms, align
from MDAnalysis.lib import util
from MDAnalysis.analysis.base import AnalysisBase
from .quaternions import rotmat2quat


def load_universes(topology, *coordinates, strict=True, in_memory=True,
                   unwrap=True, **kwargs):
    r"""Convenience function to load (one or) several trajectories
    into separate :class:`Universe <MDAnalysis.core.universe.Universe>`
    objects.

    Parameters
    ----------
    topology : str
        Path to topology file.

    *coordinates : str
        Paths to trajectory files.

    strict : bool
        Raise an error if trajectories differ in their number of frames
        or in the time between frames. Default: :any:`True`.

    in_memory : bool
        Load trajectories to in-memory representation. Default:
        :any:`True`.

    unwrap : bool
        Unwrap molecules to make them whole.

    **kwargs :
        Additional keyword arguments to be passed to
        :class:`Universe <MDAnalysis.core.universe.Universe>`.

    Returns
    -------
    universes : list
        The universes, one for each entry in ``*coordinates``.
    """
    assert coordinates, "At least one trajectory must be provided."
    universes = []
    for coords in coordinates:
        u = mda.Universe(topology, coords, in_memory=in_memory, **kwargs)
        if unwrap:
            u.trajectory.add_transformations(trans.unwrap(u.atoms))
        universes.append(u)
    if strict:
        assert len(set([u.trajectory.n_frames for u in universes])) == 1
        assert len(set([u.trajectory.dt for u in universes])) == 1
    return universes


class Orientations(AnalysisBase):
    r"""Determines the orientation of an
    :class:`AtomGroup <MDAnalysis.core.groups.AtomGroup>` along the trajectory.

    This class uses the machinery of `MDAnalysis <https://www.mdanalysis.org/>`_.
    To use it, first create a class instance, then call its :meth:`run` method.
    In each frame, Theobald's QCP method is used to determine the orientation,
    i.e., the rotation matrix that minimizes the RMSD between mobile and
    reference after removing translational motion. :footcite:p:`Theobald2005,
    Liu2010` The orientation matrices are stored in
    :attr:`results.orientations_as_mat` and can be optionally converted
    to quaternions by calling the :meth:`to_quaternion` method.

    Warning
    ----
    If you use trajectory data from simulations performed under periodic
    boundary conditions then you must **make your molecules whole** before
    computing orientations so that the structures are properly superimposed.

    Note
    ----
    Follows loosely the AlignTraj class and RMSD class in MDAnalysis 2.7.
    """

    def __init__(self, mobile, reference=None, select='all', weights=None,
                 verify_match=True, tol_mass=0.1, strict=True,
                 verbose=True):
        r"""Parameters
        ----------
        mobile : AtomGroup or Universe
            Group of atoms for which the orientations along the trajectory
            are calculated.

        reference : AtomGroup or Universe, optional
            Group of atoms defining the reference orientation. If :any:`None`
            then the current frame of `mobile` is used. Default: :any:`None`.

        select : str or tuple or dict, optional
            Selection string that is passed to :class:`AtomGroup.select_atoms() <MDAnalysis.core.groups.AtomGroup>`
            to construct the :meth:`AtomGroup <MDAnalysis.core.groups.AtomGroup.select_atoms>`
            to operate on. Must follow the
            `MDAnalysis Selection Syntax <https://userguide.mdanalysis.org/stable/selections.html>`_.
            It is possible to pass separate selection strings to `mobile` and
            `reference` by providing a tuple of strings or a dictionary with
            keywords `'mobile'` and `'reference'`. However, the selection strings
            must define a one-to-one mapping of atoms between mobile and
            reference, which can be checked with
            :func:`get_matching_atoms() <MDAnalysis.analysis.align.get_matching_atoms>`.
            Default: ``'all'``.

        weights : None or 'mass' or array_like, optional
            Choose weights which will be used for removing translation and for
            determining the orientation. Weigh atoms equally with :any:`None`,
            use masses of `reference` as weights with ``'mass'``, or provide
            an ``array_like`` of weights, which must provide exactly one weight
            per atom in the selection. Default: :any:`None`.

        verify_match : bool, optional
            Verify the one-to-one atom mapping of `mobile` and `reference`
            based on the residue names and atom masses using
            :func:`get_matching_atoms() <MDAnalysis.analysis.align.get_matching_atoms>`.
            Default: :any:`True`.

        tol_mass : float, optional
            Tolerance in mass when running
            :func:`get_matching_atoms() <MDAnalysis.analysis.align.get_matching_atoms>`.

        strict : bool, optional
            If :any:`True`, raise an error in
            :func:`get_matching_atoms() <MDAnalysis.analysis.align.get_matching_atoms>`
            if a residue is missing an atom. If :any:`False`, ignore residues
            with missing atoms in the analysis. Default ``True``.

        Attributes
        ----------
        results.orientations : ndarray
            Orientations represented as array of matrices with shape
            *(n_frames, 3, 3)*.
        """
        # Maybe add in_memory reader
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

    def _prepare(self):
        # Center the reference.
        self._ref_center = self._ref_atoms.center(self.weights)
        self._ref_coordinates = self._ref_atoms.positions - self._ref_center
        # Allocate an array for storing the orientation matrices.
        self.results.orientations = np.zeros((self.n_frames, 3, 3))
        self.results._rmsd = np.zeros((self.n_frames,))

    def _single_frame(self):
        index = self._frame_index
        # Remove translational motion, then compute best-fit rotation matrix.
        mobile_center = self._mobile_atoms.center(self.weights)
        mobile_coordinates = self._mobile_atoms.positions - mobile_center
        orientation, rmsd = align.rotation_matrix(mobile_coordinates,
                                                  self._ref_coordinates,
                                                  self.weights)
        self.results.orientations[index] = orientation
        self.results._rmsd[index] = rmsd


def get_orientations(*universes, reference=None, select='all',
                     mapping='zip', **kwargs):
    # If only a single universe or selection string was provided, put
    # it in a list.
    if (isinstance(universes, mda.Universe)
            or isinstance(universes, mda.AtomGroup)):
        universes = [universes]
    if isinstance(select, str):
        select = [select]

    # Some checks.
    if (mapping == 'zip' and len(universes) != len(select)
            and len(universes) != 1 and len(select) != 1):
        raise ValueError(f"With 'zip', provide exactly one selection "
                         f"string per universe, or only one selection "
                         f"string, or only one universe.")
    elif mapping not in ('zip', 'product'):
        raise ValueError(f"Mapping must be one of ('zip', 'product'). Is "
                         f"{mapping}.")

    # Allocate result array.
    n_calls_to_orientations = {
        'zip': max(len(universes), len(select)),
        'product': len(universes) * len(select)}[mapping]
    orientations = np.zeros((n_calls_to_orientations,
                             universes[0].universe.trajectory.n_frames, 3, 3))

    # Main computation.
    mapping = {'zip': zip, 'product': itertools.product}[mapping]
    for i, (u, sel) in enumerate(mapping(universes, select)):
        orientations[i] = Orientations(
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
