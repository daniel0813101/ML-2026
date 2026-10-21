"""Verify optimized training against exhaustive hypothesis evaluation."""

import unittest

import numpy as np

from decision_stump import Stump, fit_stump, generate_data, out_error
from q14 import solve as solve_q14


class DecisionStumpTests(unittest.TestCase):
    """Check threshold enumeration, tie-breaking, and population error."""

    def test_matches_exhaustive_search(self):
        """Match direct predictions on random, duplicate, and boundary inputs."""
        rng = np.random.default_rng(7)
        datasets = [
            (np.array([-1., -1., 0., 1.]), np.array([1, -1, 1, -1])),
            (np.array([0., 0., 0.]), np.array([1, -1, 1])),
            (np.array([-1.]), np.array([1])),
        ]
        for n in (3, 30, 300):
            datasets.extend(generate_data(n, 0.1, rng) for _ in range(30))
        for x, y in datasets:
            xs = np.unique(x)
            thresholds = np.r_[-1., (xs[:-1] + xs[1:]) / 2]
            candidates = []
            for s in (-1, 1):
                for theta in thresholds:
                    candidate = Stump(s, float(theta), 0.)
                    errors = int(np.count_nonzero(candidate.predict(x) != y))
                    candidates.append((errors, s + theta, s, float(theta)))
            errors, _, s, theta = min(candidates)
            actual = fit_stump(x, y)
            self.assertEqual((actual.s, actual.theta, actual.ein), (s, theta, errors / len(x)))

    def test_population_error(self):
        """Compare analytic errors with integration over a dense midpoint grid."""
        x = -1 + (np.arange(100000) + 0.5) * 2 / 100000
        target = np.where(x > 0, 1, -1)
        for s in (-1, 1):
            for theta in (-1., -0.47, 0., 0.29, 1.):
                stump = Stump(s, theta, 0.)
                clean = np.mean(stump.predict(x) != target)
                for tau in (0., 0.1, 0.5, 1.):
                    expected = tau * (1 - clean) + (1 - tau) * clean
                    self.assertAlmostEqual(out_error(stump, tau), expected, places=5)

    def test_tie_break_and_sign_zero(self):
        """Choose the smallest s + theta and assign threshold equality left."""
        stump = fit_stump(np.array([0.5]), np.array([1]))
        self.assertEqual((stump.s, stump.theta), (1, -1.))
        self.assertEqual(Stump(1, 0., 0.).predict(np.array([0.]))[0], -1)
        tied = fit_stump(np.array([0., 0.]), np.array([-1, 1]))
        self.assertEqual((tied.s, tied.theta), (-1, -1.))

    def test_repeatability(self):
        """Reproduce identical summaries when using the same random seed."""
        first = solve_q14(trials=20, seed=42)
        self.assertEqual(first, solve_q14(trials=20, seed=42))


if __name__ == "__main__":
    unittest.main()

