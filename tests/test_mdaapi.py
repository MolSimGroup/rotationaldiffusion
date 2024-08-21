"""
Unit and regression test for the mdaapi module.
"""
import os
import sys

import pytest
import numpy as np

import MDAnalysis as mda
from MDAnalysis.core.topology import Topology
from MDAnalysis.core.topologyattrs import (Atomnames, Masses, Resids, Resnames,
                                           Segids)
from MDAnalysis.exceptions import SelectionError
from MDAnalysis.lib.transformations import rotation_matrix

from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal)
sys.path.insert(0, os.path.abspath(".."))
import rotationaldiffusion as rd


def rotmat(angle):
    return rotation_matrix(angle, [0, 0, 1])[:3, :3]


def coordinates(*angles):
    """Return the dummy coordinates of a fictitious molecule.
    (Numpy array of shape (4*n_angles, 3)).

    The unit vectors in x-, y-, z-, and negative z-direction are taken
    as base coordinates. Each angle is used to rotate the base
    coordinates. All rotated configurations are concatenated to yield
    the final coordinates.
    """
    base_coords = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])
    all_coords = []
    for angle in angles:
        all_coords.append(np.matmul(rotmat(angle), base_coords.T).T)
    all_coords = np.concatenate(all_coords)
    return all_coords


def topology(n_atoms):
    """Return the dummy topology of a fictitious molecule."""
    n_atoms_per_residue = np.max([i for i in range(1, n_atoms)
                                  if not n_atoms % i])
    n_residues = n_atoms // n_atoms_per_residue
    ndx_and_mass = np.repeat(range(n_residues), n_atoms_per_residue)
    top = Topology(n_atoms, n_residues, 1,
                   atom_resindex=ndx_and_mass,
                   attrs=[Atomnames(['A' for i in range(n_atoms)]),
                          Masses(ndx_and_mass),
                          Resids(range(n_residues)),
                          Resnames([str(i) for i in range(n_residues)]),
                          Segids([0])])
    return top


@pytest.fixture
def mobile():
    n_frames = 5
    all_coords = coordinates(0, np.pi/2)
    top = topology(len(all_coords))
    u = mda.Universe(top, n_frames*[all_coords])
    return u


@pytest.fixture
def reference(mobile):
    u = mobile.copy()
    u.atoms.positions = coordinates(0, 0)
    return u


@pytest.mark.parametrize('in_memory', [True, False])
@pytest.mark.parametrize('unwrap', [True, False])
class TestLoadUniverses:
    @pytest.fixture
    def top(self):
        return 'data/ubq.tpr'

    @pytest.fixture
    def traj(self):
        return 'data/ubq.xtc'

    def test_no_trajectory_failure(self, top, in_memory, unwrap):
        # Fail if no trajectory is provided.
        with pytest.raises(AssertionError):
            rd.mdaapi.load_universes(top, in_memory=in_memory, unwrap=unwrap)

    @pytest.mark.parametrize('n_trajectories', [1, 2])
    def test_with_trajectories(self, top, traj, n_trajectories, in_memory,
                               unwrap):
        # Load one or two (i.e. several) trajectories.
        trajectories = n_trajectories * [traj]
        universes = rd.mdaapi.load_universes(top, *trajectories,
                                             in_memory=in_memory,
                                             unwrap=unwrap)
        assert_equal(len(universes), n_trajectories)
        assert_array_equal(np.array([u.atoms.n_atoms for u in universes]), 1231)
        if in_memory:
            assert_array_equal(
                np.array([
                    isinstance(u.trajectory, mda.coordinates.memory.MemoryReader)
                    for u in universes]), True)

    def test_strict_failure(self, top, traj, in_memory, unwrap):
        # Fail if trajectories are not equally long.
        with pytest.raises(AssertionError):
            rd.mdaapi.load_universes(top, 2*[traj], 3*[traj],
                                     in_memory=in_memory, unwrap=unwrap)


class TestOrientations:
    def test_mobile(self, mobile):
        ana = rd.mdaapi.Orientations(mobile, verbose=False)
        ana.run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(*np.broadcast_arrays(
            ana.results.orientations, rotmat(0)))

    def test_reference(self, mobile, reference):
        ana = rd.mdaapi.Orientations(mobile, reference, verbose=False)
        ana.run()
        assert_array_almost_equal(*np.broadcast_arrays(
            ana.results.orientations, rotmat(-np.pi/4)))

    def test_atomgroup(self, mobile, reference):
        ana = rd.mdaapi.Orientations(mobile.atoms, reference.atoms,
                                     verbose=False)
        ana.run()
        assert_array_almost_equal(*np.broadcast_arrays(
            ana.results.orientations, rotmat(-np.pi/4)))

    @pytest.mark.parametrize('selection, angle', [
        ('all', -np.pi/4), ('resid 1', -np.pi/2),
        (('resname 1', 'resid 1'), -np.pi/2),
        ({'mobile': 'resname 1', 'reference': 'resid 1'}, -np.pi/2)])
    def test_selection(self, mobile, reference, selection, angle):
        ana = rd.mdaapi.Orientations(mobile, reference, verbose=False,
                                     select=selection)
        ana.run()
        assert_array_almost_equal(*np.broadcast_arrays(
            ana.results.orientations, rotmat(angle)))

    @pytest.mark.parametrize('selection', [
        ('all', 'resid 0'), # missing residue
        ('resid 0', 'residue 0 and index 1:3'), # missing atom in a residue
        ('resid 0', 'resid 1')]) # mismatching masses
    def test_selection_failure(self, mobile, reference, selection):
        with pytest.raises(SelectionError):
            ana = rd.mdaapi.Orientations(mobile, reference, verbose=False,
                                         select=(selection))

    @pytest.mark.parametrize('weights, angle', [
        (None, -np.pi/4),  ('mass', -np.pi/2),
        ([0, 0, 0, 0, 1, 1, 1, 1], -np.pi/2),
        ([1, 1, 1, 1, 0, 0, 0, 0], 0)])
    def test_weighting(self, mobile, reference, weights, angle):
        ana = rd.mdaapi.Orientations(mobile, reference, verbose=False,
                                     weights=weights)
        ana.run()
        assert_array_almost_equal(*np.broadcast_arrays(
            ana.results.orientations, rotmat(angle)))


class TestGetOrientations:
    @pytest.mark.parametrize('mapping', ['zip', 'product'])
    def test_universe(self, mobile, mapping):
        res = rd.mdaapi.get_orientations(mobile, mapping=mapping,
                                         verbose=False)
        assert_equal(len(res), 1)
        assert_array_almost_equal(*np.broadcast_arrays(
            res, rotmat(0)))

    @pytest.mark.parametrize('mapping', ['zip', 'product'])
    def test_reference(self, mobile, reference, mapping):
        res = rd.mdaapi.get_orientations(mobile, reference=reference,
                                         mapping=mapping, verbose=False)
        assert_equal(len(res), 1)
        assert_array_almost_equal(*np.broadcast_arrays(
            res, rotmat(-np.pi/4)))


    @pytest.mark.parametrize('mapping', ['zip', 'product'])
    def test_atomgroup(self, mobile, reference, mapping):
        res = rd.mdaapi.get_orientations(mobile.atoms,
                                         reference=reference.atoms,
                                         mapping=mapping, verbose=False)
        assert_equal(len(res), 1)
        assert_array_almost_equal(*np.broadcast_arrays(
            res, rotmat(-np.pi/4)))

    @pytest.mark.parametrize('mapping', ['zip', 'product'])
    def test_universes(self, mobile, mapping):
        res = rd.mdaapi.get_orientations(mobile, mobile, mapping=mapping,
                                         verbose=False)
        assert_equal(len(res), 2)
        assert_array_almost_equal(*np.broadcast_arrays(
            res, rotmat(0)))



    # def test(self, universes, reference, select, mapping, n_res, angle):
    #
    #     res = rd.mdaapi.get_orientations(universes,
    #                                      reference=reference, select=select,
    #                                      mapping=mapping, verbose=False)
    #     assert_equal(len(res), n_res)
    #     assert_array_almost_equal(*np.broadcast_arrays(
    #         res, rotmat(angle)))

    def test_mapping_failure(self, mobile):
        with pytest.raises(ValueError):
            rd.mdaapi.get_orientations(mobile, mapping='shouldcauseerror')
        with pytest.raises(ValueError):
            rd.mdaapi.get_orientations(
                mobile, mobile, select=['backbone', 'backbone', 'backbone'],
                mapping='zip')
