"""
Unit and regression test for the mdaapi module.
"""
import os
import sys

import pytest
import numpy as np
import rotationaldiffusion as rd
import MDAnalysis as mda
from MDAnalysis.exceptions import SelectionError
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal)
sys.path.insert(0, os.path.abspath(".."))


@pytest.fixture
def top():
    return 'data/ubq.tpr'


@pytest.fixture
def traj():
    return 'data/ubq.xtc'


@pytest.fixture
def mobile(top, traj):
    return mda.Universe(top, traj)


@pytest.fixture
def reference(top):
    # Last frame of mobile.
    return mda.Universe(top, 'data/ubq.gro')


class TestLoadUniverses:
    def test_one_trajectory(self, top, traj):
        # Load one trajectory.
        universes = rd.orientations.load_universes(top, traj)
        assert_equal(len(universes), 1)
        assert_array_equal([u.atoms.n_atoms for u in universes], 1231)

    def test_several_trajectories(self, top, traj):
        # Load two (i.e. several) trajectories.
        universes = rd.orientations.load_universes(top, traj, traj)
        assert_equal(len(universes), 2)
        assert_array_equal([u.atoms.n_atoms for u in universes], 1231)

    def test_kwargs(self, top, traj):
        # Test if kwargs are passed correctly to the universe
        # constructor using 'in_memory' as an example.
        universes = rd.orientations.load_universes(top, traj, in_memory=True)
        assert isinstance(universes[0].trajectory,
                          mda.coordinates.memory.MemoryReader)
        assert_array_equal([u.atoms.n_atoms for u in universes], 1231)


class TestOrientations:
    # All expected orientations have been double-checked with GROMACS.
    def test_mobile1(self, mobile):
        ana = rd.orientations.Orientations(mobile, verbose=False).run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(ana.results.orientations[0], np.eye(3))
        assert_array_almost_equal(
            ana.results.orientations[-1], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]), decimal=4)

    def test_mobile2(self, mobile):
        mobile.trajectory[-1]  # Set trajectory to last frame.
        ana = rd.orientations.Orientations(mobile, verbose=False).run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(ana.results.orientations[-1], np.eye(3))
        assert_array_almost_equal(
            ana.results.orientations[0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.8970, -0.2317, 0.3764],
                 [0.3673, 0.0830, 0.9264]]).T, decimal=4)

    def test_broken_pbc(self, mobile):
        ana = rd.orientations.Orientations(mobile, unwrap=False, verbose=False).run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(ana.results.orientations[0], np.eye(3))
        assert_array_almost_equal(
            ana.results.orientations[-1], np.array(
                [[-0.0604, -0.8743, -0.4815],
                 [0.6933, -0.3838, 0.6098],
                 [-0.7180, -0.2970, 0.6294]]), decimal=4)

    def test_reference(self, mobile, reference):
        ana = rd.orientations.Orientations(
            mobile, reference=reference, verbose=False).run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(ana.results.orientations[-1], np.eye(3))
        assert_array_almost_equal(
            ana.results.orientations[0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)

    def test_atomgroup(self, mobile, reference):
        ana = rd.orientations.Orientations(
            mobile.atoms, reference=reference.atoms, verbose=False).run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(ana.results.orientations[-1], np.eye(3))
        assert_array_almost_equal(
            ana.results.orientations[0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)

    def test_in_memory(self, mobile, reference):
        mobile.transfer_to_memory()
        reference.transfer_to_memory()
        ana = rd.orientations.Orientations(
            mobile, reference=reference, verbose=False).run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(ana.results.orientations[-1], np.eye(3))
        assert_array_almost_equal(
            ana.results.orientations[0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)

    @pytest.mark.parametrize('selection', [
        'name CA or name C or name N',
        ('name CA or name C or name N', 'name CA or name C or name N'),
        {'mobile': 'name CA or name C or name N',
         'reference': 'name CA or name C or name N'}])
    def test_select(self, mobile, reference, selection):
        ana = rd.orientations.Orientations(
            mobile, reference, select=selection, verbose=False).run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(ana.results.orientations[-1], np.eye(3))
        assert_array_almost_equal(
            ana.results.orientations[0], np.array(
                [[-0.2217, -0.9064, 0.3595],
                 [0.975, -0.2016, 0.093],
                 [-0.0118, 0.3711, 0.9285]]), decimal=4)

    def test_select_missing_residue_error(self, mobile, reference):
        with pytest.raises(SelectionError):
            rd.orientations.Orientations(
                mobile, reference, select=('all', 'not resid 1'),
                verbose=False).run()

    def test_select_missing_atom_error(self, mobile, reference):
        with pytest.raises(SelectionError):
            rd.orientations.Orientations(
                mobile, reference, verbose=False,
                select=('all', 'not atom seg_0_Protein_chain_A 1 CA')).run()

    def test_mass_weighting(self, mobile, reference):
        ana = rd.orientations.Orientations(
            mobile, reference=reference, weights='mass', verbose=False).run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(ana.results.orientations[-1], np.eye(3))
        assert_array_almost_equal(
            ana.results.orientations[0], np.array(
                [[-0.2387, -0.8998, 0.3653],
                 [0.9711, -0.2233, 0.0844],
                 [0.0056, 0.3749, 0.927]]), decimal=4)

    def test_custom_weighting(self, mobile, reference):
        # Put weight 1 on backbone atoms, 0 on rest.
        # Analogous to select backbone.
        sel = 'name CA or name C or name N'
        weights = np.zeros((mobile.atoms.n_atoms,))
        weights[mobile.atoms.select_atoms(sel).ids] = 1
        ana = rd.orientations.Orientations(
            mobile, reference=reference, weights=weights, verbose=False).run()
        assert_equal(len(ana.results.orientations), mobile.trajectory.n_frames)
        assert_array_almost_equal(ana.results.orientations[-1], np.eye(3))
        assert_array_almost_equal(
            ana.results.orientations[0], np.array(
                [[-0.2217, -0.9064, 0.3595],
                 [0.975, -0.2016, 0.093],
                 [-0.0118, 0.3711, 0.9285]]), decimal=4)


class TestGetOrientations:
    def test_universe(self, mobile):
        ana = rd.orientations.get_orientations(mobile)
        assert_equal(ana.shape, (1, 1, mobile.trajectory.n_frames, 3, 3))
        assert_array_almost_equal(
            ana[0, 0, -1], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]), decimal=4)

    def test_reference(self, mobile, reference):
        ana = rd.orientations.get_orientations(mobile, reference)
        assert_equal(ana.shape, (1, 1, mobile.trajectory.n_frames, 3, 3))
        assert_array_almost_equal(
            ana[0, 0, 0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)

    def test_atomgroup(self, mobile, reference):
        ana = rd.orientations.get_orientations(mobile.atoms, reference.atoms)
        assert_equal(ana.shape, (1, 1, mobile.trajectory.n_frames, 3, 3))
        assert_array_almost_equal(
            ana[0, 0, 0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)

    def test_select(self, mobile, reference):
        ana = rd.orientations.get_orientations(mobile, reference,
                                               select='name CA or name C or name N')
        assert_equal(ana.shape, (1, 1, mobile.trajectory.n_frames, 3, 3))
        assert_array_almost_equal(
            ana[0, 0, 0], np.array(
                [[-0.2217, -0.9064, 0.3595],
                 [0.975, -0.2016, 0.093],
                 [-0.0118, 0.3711, 0.9285]]), decimal=4)

    def test_in_memory(self, mobile, reference):
        assert not isinstance(mobile.trajectory,
                              mda.coordinates.memory.MemoryReader)
        ana = rd.orientations.get_orientations(mobile, reference, in_memory=True)
        assert isinstance(mobile.trajectory,
                          mda.coordinates.memory.MemoryReader)
        assert_equal(ana.shape, (1, 1, mobile.trajectory.n_frames, 3, 3))
        assert_array_almost_equal(
            ana[0, 0, 0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)

    def test_multiple_trajectories(self, mobile, reference):
        ana = rd.orientations.get_orientations([mobile, mobile], reference)
        assert_equal(ana.shape, (2, 1, mobile.trajectory.n_frames, 3, 3))
        assert_array_almost_equal(
            ana[0, 0, 0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)
        assert_array_almost_equal(
            ana[1, 0, 0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)

    def test_multiple_selections(self, mobile, reference):
        ana = rd.orientations.get_orientations(
            mobile, reference, select=['all', 'name CA or name C or name N'])
        assert_equal(ana.shape, (1, 2, mobile.trajectory.n_frames, 3, 3))
        assert_array_almost_equal(
            ana[0, 0, 0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)
        assert_array_almost_equal(
            ana[0, 1, 0], np.array(
                [[-0.2217, -0.9064, 0.3595],
                 [0.975, -0.2016, 0.093],
                 [-0.0118, 0.3711, 0.9285]]), decimal=4)

    def test_multiple_trajectories_and_selections(self, mobile, reference):
        ana = rd.orientations.get_orientations(
            3*[mobile], reference,
            select=['all', 'name CA or name C or name N'])
        assert_equal(ana.shape, (3, 2, mobile.trajectory.n_frames, 3, 3))
        assert_array_almost_equal(ana[0, 0, -1], np.eye(3))
        assert_array_almost_equal(
            ana[1, 0, 0], np.array(
                [[-0.2459, 0.9692, 0.0106],
                 [-0.897, -0.2317, 0.3764],
                 [0.3673, 0.083, 0.9264]]).T, decimal=4)
        assert_array_almost_equal(
            ana[2, 1, 0], np.array(
                [[-0.2217, -0.9064, 0.3595],
                 [0.975, -0.2016, 0.093],
                 [-0.0118, 0.3711, 0.9285]]), decimal=4)


@pytest.fixture()
def gmx_rotmat_file():
    return 'data/ubq.xvg'

class TestLoadOrientations:
    def test_one_file(self, gmx_rotmat_file):
        orientations, time = rd.orientations.load_orientations(gmx_rotmat_file)
        assert_array_almost_equal(orientations[0, -1], np.eye(3))
        assert_array_almost_equal(
            orientations[0, 0], np.array(
                [[-0.2387, -0.8998, 0.3653],
                 [0.9711, -0.2233, 0.0844],
                 [0.0056, 0.3749, 0.9270]]), decimal=4)
        assert_array_almost_equal(time, np.array([[0.99e6, 1e6]]))

    def test_start_stop_step(self, gmx_rotmat_file):
        raise NotImplementedError('A test for the start, stop, and step'
                                  ' arguments is not implemented, yet.')

