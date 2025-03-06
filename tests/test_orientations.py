"""
Tests for the ``orientations`` module.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
import MDAnalysis as mda
from MDAnalysis import NoDataError, SelectionError
from MDAnalysisTests.datafiles import PSF, DCD, TPR, GRO
import RotationalDiffusion as rd


@pytest.fixture()
def universe():
    return mda.Universe(PSF, DCD)


@pytest.fixture()
def reference():
    u = mda.Universe(PSF, DCD)
    u.trajectory[-1]
    return u


@pytest.fixture()
def broken_universe_in_memory():
    return mda.Universe(TPR, GRO, in_memory=True)


ORIENTATIONS = {
    'default': np.array([
        [ 0.9995, -0.0059,  0.0304],
        [ 0.0064,  0.9998, -0.0167],
        [-0.0303,  0.0169,  0.9994]
    ]),
    'selection': np.array([
        [ 0.9995, -0.0061, -0.032 ],
        [ 0.0061,  1.    ,  0.0013],
        [ 0.032 , -0.0015,  0.9995]
    ]),
    'weighted': np.array([
        [ 0.9995,  0.005 , -0.0316],
        [-0.0045,  0.9999,  0.0159],
        [ 0.0317, -0.0157,  0.9994]
    ])
}


class TestOrientations:
    def test_defaults(self, universe):
        ana = rd.orientations.Orientations(universe, unwrap=False).run()
        assert len(ana.results.orientations) == universe.trajectory.n_frames
        assert_allclose(ana.results.orientations[0], np.eye(3), atol=1e-8)
        assert_allclose(
            ana.results.orientations[-1], ORIENTATIONS['default'], atol=1e-4
        )

    def test_uses_current_frame(self, universe):
        universe.trajectory[-1]  # Set trajectory to last frame.
        ana = rd.orientations.Orientations(universe, unwrap=False).run()
        assert_allclose(ana.results.orientations[-1], np.eye(3), atol=1e-8)
        assert_allclose(
            ana.results.orientations[0], ORIENTATIONS['default'].T, atol=1e-4
        )

    def test_uses_reference(self, universe, reference):
        ana = rd.orientations.Orientations(
            universe, reference=reference, unwrap=False
        ).run()
        assert_allclose(ana.results.orientations[-1], np.eye(3), atol=1e-8)
        assert_allclose(
            ana.results.orientations[0], ORIENTATIONS['default'].T, atol=1e-4
        )

    def test_accepts_atomgroups(self, universe, reference):
        ana = rd.orientations.Orientations(
            universe.atoms, reference=reference.atoms, unwrap=False
        ).run()
        assert_allclose(ana.results.orientations[-1], np.eye(3), atol=1e-8)
        assert_allclose(
            ana.results.orientations[0], ORIENTATIONS['default'].T, atol=1e-4
        )

    @pytest.mark.parametrize('selection', [
        'name CA',
        ('name CA', 'name CA'),
        {'mobile': 'name CA', 'reference': 'name CA'}
    ])
    def test_uses_selections(self, universe, reference, selection):
        ana = rd.orientations.Orientations(
            universe, reference=reference, select=selection, unwrap=False
        ).run()
        assert_allclose(ana.results.orientations[-1], np.eye(3), atol=1e-8)
        assert_allclose(
            ana.results.orientations[0], ORIENTATIONS['selection'], atol=1e-4
        )

    @pytest.mark.parametrize('sel2', ['not resid 1', 'not atom 4AKE 1 CA'])
    def test_raises_selection_error(self, universe, reference, sel2):
        with pytest.raises(SelectionError):
            rd.orientations.Orientations(
                universe, reference, select=('all', sel2), unwrap=False
            ).run()

    def test_mass_weighting(self, universe, reference):
        ana = rd.orientations.Orientations(
            universe, reference=reference, weights='mass', unwrap=False
        ).run()
        assert_allclose(ana.results.orientations[-1], np.eye(3), atol=1e-8)
        assert_allclose(
            ana.results.orientations[0], ORIENTATIONS['weighted'], atol=1e-4
        )

    def test_custom_weighting(self, universe, reference):
        # Choose weights such that they correspond to select 'name CA'.
        weights = np.zeros((universe.atoms.n_atoms,))
        weights[universe.atoms.select_atoms('name CA').ids] = 1
        ana = rd.orientations.Orientations(
            universe, reference=reference, weights=weights, unwrap=False
        ).run()
        assert_allclose(ana.results.orientations[-1], np.eye(3), atol=1e-8)
        assert_allclose(
            ana.results.orientations[0], ORIENTATIONS['selection'], atol=1e-4
        )

    def test_unwraps_reference(self, broken_universe_in_memory):
        ref = broken_universe_in_memory.copy()
        broken_universe_in_memory.select_atoms('protein').unwrap()

        with pytest.raises(AssertionError):
            ana = rd.orientations.Orientations(
                broken_universe_in_memory, reference=ref, select='protein',
                unwrap=False, center=False
            ).run()
            assert_allclose(ana.results.orientations[0], np.eye(3), atol=1e-8)

        ana = rd.orientations.Orientations(
            broken_universe_in_memory, reference=ref, select='protein',
            unwrap=True, center=False
        ).run()
        assert_allclose(ana.results.orientations[0], np.eye(3), atol=1e-8)

    def test_unwraps_mobile(self, broken_universe_in_memory):
        ref = broken_universe_in_memory.copy()
        ref.select_atoms('protein').unwrap()

        with pytest.raises(AssertionError):
            ana = rd.orientations.Orientations(
                broken_universe_in_memory, reference=ref, select='protein',
                unwrap=False, center=False
            ).run()
            assert_allclose(ana.results.orientations[0], np.eye(3), atol=1e-8)

        ana = rd.orientations.Orientations(
            broken_universe_in_memory, reference=ref, select='protein',
            unwrap=True, center=False
        ).run()
        assert_allclose(ana.results.orientations[0], np.eye(3), atol=1e-8)

    def test_unwrapping_fails(self):
        mobile = mda.Universe(TPR, GRO)
        ref = mobile.copy()
        ref.atoms.dimensions = None

        with pytest.warns(UserWarning):
            rd.orientations.Orientations(
                mobile, reference=ref, select='protein', unwrap=True
            ).run()

        with pytest.raises(ValueError):
            rd.orientations.Orientations(
                ref, reference=mobile, select='protein', unwrap=True
            ).run()

        ref = mda.Universe(GRO)
        with pytest.warns(UserWarning):
            rd.orientations.Orientations(
                mobile, reference=ref, select='protein', unwrap=True
            ).run()

        with pytest.raises(NoDataError):
            rd.orientations.Orientations(
                ref, reference=mobile, select='protein', unwrap=True
            ).run()

    def test_centering(self, universe):
        universe.transfer_to_memory()
        ref = universe.copy()
        universe.atoms.positions += [10, 0, 0]
        ref.atoms.positions += [0, 10, 0]

        with pytest.raises(AssertionError):
            ana = rd.orientations.Orientations(
                universe, reference=ref, unwrap=False, center=False
            ).run(stop=1)
            assert_allclose(ana.results.orientations[-1], np.eye(3), atol=1e-8)

        ana = rd.orientations.Orientations(
            universe, reference=ref, unwrap=False, center=True
        ).run(stop=1)
        assert_allclose(ana.results.orientations[-1], np.eye(3), atol=1e-8)
        assert_allclose(ana.results._rmsd[0], 0, atol=1e-6)

    @pytest.mark.parametrize('backend', ['multiprocessing', 'dask'])
    def test_parallelization(self, universe, reference, backend):
        ana = rd.orientations.Orientations(
            universe, reference=reference, unwrap=False
        ).run(backend=backend, n_workers=4)
        assert_allclose(ana.results.orientations[-1], np.eye(3), atol=1e-8)
        assert_allclose(
            ana.results.orientations[0], ORIENTATIONS['default'].T, atol=1e-4
        )
