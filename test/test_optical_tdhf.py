"""Run with 'python -m unittest discover -s test -p test_optical_tdhf.py'."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from optical_tdhf import current_modes


class OpticalResponseTest(unittest.TestCase):
    def test_independent_levels(self):
        energy = np.array([1.2, 2.7])
        j = np.array([.3, -.7])
        e, v, _ = current_modes(np.diag(energy), np.zeros((2, 2)), j, np.zeros((2, 0)))
        np.testing.assert_allclose(e, energy)
        np.testing.assert_allclose(v**2, 2*j**2/energy)

    def test_full_forward_reverse_resolvent(self):
        a = np.array([[2., .31], [.31, 2.8]])
        b = np.array([[.24, -.12], [-.12, .15]])
        h = np.array([[.40], [-.19]])
        j = np.array([.67, .31])
        e, v, u = current_modes(a, b, j, h)
        # Direct linearized density-matrix evolution, including the Hartree
        # field on forward and reverse transitions before forming the resolvent.
        aa, bb = a + h@h.T, b - h@h.T
        m = np.block([[aa, bb], [-bb, -aa]])
        source, probe = np.r_[j, -j], np.r_[j, j]
        def susceptibility(z):
            return probe @ np.linalg.solve(z*np.eye(4) - m, source)
        for z in (.32+.12j, 1.7+.21j, 2.8+.07j):
            direct = (susceptibility(0.) - susceptibility(z))/z**2
            pole = v @ np.linalg.solve(np.diag(e**2-z**2) + u@u.T, v)
            np.testing.assert_allclose(pole, direct, rtol=2e-13, atol=1e-14)

    def test_instability_and_asymmetry(self):
        for a, b in ((np.eye(2), 2*np.eye(2)), (-np.eye(2), np.zeros((2, 2))),
                     (np.array([[1., .3], [0., 2.]]), np.zeros((2, 2)))):
            with self.assertRaises(ValueError):
                current_modes(a, b, np.ones(2), np.zeros((2, 0)))
