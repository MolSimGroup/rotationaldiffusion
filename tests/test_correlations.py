"""
Unit tests for the ``correlations`` module.
"""

import pytest
from numpy.testing import assert_equal, assert_array_almost_equal
import rotationaldiffusion as rd


@pytest.fixture
def orientations_path():
    return "data/100-orientations.xvg"


class TestCorrelate:
    def test_quaternions(self, orientations_path):
        orientations, _ = rd.load_orientations(orientations_path)
        quats = rd.quaternions.rotmat2quat(orientations[0])
        corr = rd.diffusion.correlate(quats)
        assert_equal(corr.shape, (10, 3, 3))
        assert_array_almost_equal(corr[0, 0, 1], 1.86224602e-03)
        assert_array_almost_equal(corr[-1, 2, 2], 2.51958003e-01)

    def test_rotmats_and_higher_dimensionality(self, orientations_path):
        orientations, _ = rd.load_orientations(orientations_path,
                                               orientations_path)
        corr = rd.diffusion.correlate(orientations)
        assert_equal(corr.shape, (2, 10, 3, 3))
        assert_array_almost_equal(corr[0, 0, 0, 1], 1.86224602e-03)
        assert_array_almost_equal(corr[1, -1, 2, 2], 2.51958003e-01)

    def test_stop_step(self, orientations_path):
        orientations, _ = rd.load_orientations(orientations_path)
        corr = rd.diffusion.correlate(orientations[0], stop=7, step=2)
        assert_equal(corr.shape, (3, 3, 3))
        assert_array_almost_equal(corr[0, 0, 1], 1.12716661e-03)
        assert_array_almost_equal(corr[-1, 2, 2], 1.50418364e-01)

    def test_do_variance(self, orientations_path):
        return NotImplementedError
