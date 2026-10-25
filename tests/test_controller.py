import numpy as np
import pytest
from src.controller import SafetyFilter


def test_hard_and_slew_limits():
    f = SafetyFilter("careful")
    prev = np.zeros(2)
    for _ in range(100):
        v = f.apply([10, -10], [0, 0], 0.04)
        assert np.all(np.abs(v) <= 0.525 + 1e-10)
        assert np.all(np.abs(v - prev) <= 0.04 + 1e-10)
        prev = v
    f.reset()
    np.testing.assert_array_equal(f.previous, [0, 0])


def test_modes_make_different_commands():
    careful, fast = SafetyFilter("careful"), SafetyFilter("fast")
    for _ in range(50):
        a = careful.apply([1, 1], [0, 0], 0.04)
        b = fast.apply([1, 1], [0, 0], 0.04)
    assert np.all(a < b) and np.all(b <= 1.5)


def test_joint_limit_does_not_command_outward():
    f = SafetyFilter("fast")
    for _ in range(10):
        v = f.apply([1, -1], [2.79, -2.79], 0.04)
    np.testing.assert_array_equal(v, [0, 0])


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        SafetyFilter("unlimited")
