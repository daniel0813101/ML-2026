"""Check gradients, least-squares optimality, data, and experiment wiring."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from linear_models import (classification_error, cross_entropy, fit_linear,
                           load_data, squared_error, train_logistic_sgd)
from q12 import train_linear_sgd
import q12
import q13
import q14
import q15


class LinearModelTests(unittest.TestCase):
    """Use mathematical invariants and independent numerical gradient checks."""

    def test_data_bias_and_validation(self):
        """Preserve feature/label order, add bias, and reject malformed files."""
        for split, size in (("train", 100), ("test", 400)):
            x, y = load_data(split)
            self.assertEqual(x.shape, (size, 11))
            np.testing.assert_array_equal(x[:, 0], np.ones(size))
            self.assertTrue(np.isin(y, (-1, 1)).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hw3_train.dat"
            row = list(range(10)) + [-1]
            np.savetxt(path, [row])
            x, y = load_data("train", directory)
            np.testing.assert_array_equal(x[0], [1] + list(range(10)))
            self.assertEqual(y[0], -1)
            row[-1] = 0
            np.savetxt(path, [row])
            with self.assertRaises(ValueError):
                load_data("train", directory)

    def test_least_squares_optimality_and_rank_deficiency(self):
        """Verify the normal equations and a known rank-deficient solution."""
        x, y = load_data("train")
        w = fit_linear(x, y)
        np.testing.assert_allclose(x.T @ (x @ w - y), 0, atol=1e-11)
        singular_x = np.array([[1., 1.], [2., 2.]])
        np.testing.assert_allclose(fit_linear(singular_x, np.array([2., 4.])),
                                   [1., 1.], atol=1e-12)

    def test_sgd_steps_match_numerical_gradients(self):
        """Compare a single sampled update against central finite differences."""
        x = np.array([[1., -0.5, 2.], [1., 1.5, -1.]])
        y = np.array([1., -1.])
        eta, seed = 0.001, 55
        index = np.random.Generator(np.random.PCG64(seed)).integers(len(y))
        row, label = x[index], y[index]
        for logistic in (False, True):
            initial = np.array([0.2, -0.1, 0.3]) if logistic else np.zeros(3)

            def loss(w):
                if logistic:
                    return np.log(1 + np.exp(-label * (row @ w)))
                return (row @ w - label) ** 2

            gradient = np.empty(3)
            for dimension in range(3):
                delta = np.zeros(3)
                delta[dimension] = 1e-6
                gradient[dimension] = (loss(initial + delta)
                                       - loss(initial - delta)) / 2e-6
            rng = np.random.Generator(np.random.PCG64(seed))
            if logistic:
                actual = train_logistic_sgd(x, y, rng, initial, 1, eta)
            else:
                actual = train_linear_sgd(x, y, rng, 1, eta)
            np.testing.assert_allclose(actual, initial - eta * gradient, atol=1e-12)

    def test_loss_definitions_and_extreme_margins(self):
        """Check averaging, zero-score classification, and numerical stability."""
        x = np.eye(2)
        y = np.array([1., -1.])
        self.assertEqual(squared_error(x, y, np.zeros(2)), 1.)
        self.assertEqual(classification_error(x, y, np.zeros(2)), 0.5)
        self.assertAlmostEqual(cross_entropy(x, y, np.zeros(2)), np.log(2))
        with np.errstate(over="raise", invalid="raise"):
            self.assertAlmostEqual(cross_entropy(x, y, np.array([-1000., -1000.])),
                                   500.)
            initial = np.array([-1000., 1000.])
            saved = initial.copy()
            w = train_logistic_sgd(x, y, np.random.default_rng(1), initial, 5, .001)
            self.assertTrue(np.isfinite(w).all())
            np.testing.assert_array_equal(initial, saved)

    def test_q14_q15_use_identical_models_and_absolute_gaps(self):
        """Record trained models to verify Q15 really repeats Q14's setup."""
        captured = []

        def record(*args):
            w = train_logistic_sgd(*args)
            captured.append(w.copy())
            return w

        with patch("q14.train_logistic_sgd", side_effect=record):
            result14 = q14.solve(trials=5, iterations=30)
        models14 = captured.copy()
        captured.clear()
        with patch("q15.train_logistic_sgd", side_effect=record):
            result15 = q15.solve(trials=5, iterations=30)
        np.testing.assert_array_equal(models14, captured)
        x, y = load_data("train")
        test_x, test_y = load_data("test")
        manual_gaps = []
        for w in models14:
            # Independent counting avoids calling the implementation's error helper.
            in_error = sum((x @ w >= 0) != (y == 1)) / len(y)
            out_error = sum((test_x @ w >= 0) != (test_y == 1)) / len(test_y)
            manual_gaps.append(abs(in_error - out_error))
        np.testing.assert_allclose(result15["trial_gaps"], manual_gaps)
        self.assertAlmostEqual(result15["value"], np.mean(manual_gaps))
        self.assertAlmostEqual(result14["value"],
                               np.mean([cross_entropy(x, y, w) for w in models14]))

    def test_reproducibility_and_trial_averages(self):
        """Independent solve calls reproduce the same trial records and means."""
        for solve in (q12.solve, q13.solve, q14.solve, q15.solve):
            first = solve(trials=3, iterations=20)
            self.assertEqual(first, solve(trials=3, iterations=20))
            values = first.get("trial_errors", first.get("trial_gaps"))
            self.assertAlmostEqual(first["value"], np.mean(values))
            self.assertIn(first["closest_choice"], first["choices"])

    def test_invalid_experiment_settings(self):
        """Reject empty trials, invalid seeds/steps, and nonfinite step sizes."""
        for kwargs in ({"trials": 0}, {"seed": -1}, {"iterations": 0},
                       {"learning_rate": 0}, {"learning_rate": float("nan")}):
            with self.assertRaises(ValueError):
                q12.solve(**kwargs)


if __name__ == "__main__":
    unittest.main()
