import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from RotationalDiffusion import quaternions as qops


@pytest.fixture
def q():
    # Unit quaternion with rotation angle in [0, pi].
    return np.array([0.37476168, 0.54766763, -0.7132539, 0.225572])


@pytest.fixture
def q_conj():
    # Complex conjugate of q.
    return np.array([0.37476168, -0.54766763, 0.7132539, -0.225572])


@pytest.fixture
def R():
    # Representation of q as a rotation matrix.
    R = np.array([
        [-0.1192277 , -0.95032363, -0.28752349],
        [-0.61218066,  0.29835488, -0.7322699 ],
        [ 0.78167742,  0.08870946, -0.61734191]
    ])
    return R


class TestConjugate:
    def test_one_quaternion(self, q, q_conj):
        assert_equal(qops.conjugate(q), q_conj)

    def test_array_of_quaternions(self, q, q_conj):
        assert_equal(qops.conjugate([q, -q, q_conj]), [q_conj, -q_conj, q])

    def test_accepts_array_like(self, q, q_conj):
        assert_equal(qops.conjugate(list(q)), q_conj)


class TestRotmat2Quat:
    def test_one_quaternion(self, R, q):
        assert_allclose(qops.rotmat2quat(R), q)

    def test_array_of_quaternions(self, R, q, q_conj):
        assert_allclose(qops.rotmat2quat([R, R.T]), [q, q_conj])

    def test_accepts_array_like(self, R, q):
        assert_allclose(qops.rotmat2quat(list(R)), q)


class TestQuat2Rotmat:
    def test_one_quaternion(self, q, R):
        assert_allclose(qops.quat2rotmat(q), R)

    def test_array_of_quaternions(self, q, q_conj, R):
        assert_allclose(qops.quat2rotmat([-q, q, q_conj]), [R, R, R.T])

    def test_accepts_array_like(self, q, R):
        assert_allclose(qops.quat2rotmat(list(q)), R)


class TestLimitAngle:
    def test_angle_inside_interval(self, q):
        assert_equal(qops.limit_angle(q), q)

    def test_angle_outside_interval(self, q):
        assert_equal(qops.limit_angle(-q), q)

    def test_array_of_quaternions(self, q):
        assert_equal(qops.limit_angle([q, -q]), [q, q])

    def test_accepts_array_like(self, q):
        assert_equal(qops.limit_angle(list(q)), q)


class TestMultiply:
    @pytest.fixture
    def q2(self):
        return np.array([-0.45879293, 0.86512793, 0.1176837 , -0.16496439])

    @pytest.fixture
    def p(self):
        return np.array([-0.52459087, 0.16406611, 0.65683348, 0.51619425])

    def test_one_quaternion(self, q, q2, p):
        assert_allclose(qops.multiply(q, q2), p)

    def test_arrays_of_quaternions(self, q, q2, p):
        assert_allclose(qops.multiply([[q, q]], [[q2], [q2]]), 2*[[p, p]])

    def test_accepts_array_like(self, q, q2, p):
        assert_allclose(qops.multiply(list(q), list(q2)), p)