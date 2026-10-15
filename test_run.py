"""Regression checks for held-out scoring and command-line inputs."""

import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import config
import fitness
import run
from tree import Node


class TestRunner(unittest.TestCase):
    def test_evaluation_fits_only_training_targets(self):
        x_train = np.arange(1., 6.)
        x_test = np.arange(6., 9.)
        train = ({"x": x_train}, 3 + 2 * x_train)
        test = ({"x": x_test}, 100 + x_test)
        with patch("fitness.fit_scale", wraps=fitness.fit_scale) as fit:
            result = run.evaluate(Node("x"), train, test, config.Config())
        self.assertEqual(fit.call_count, 1)
        self.assertAlmostEqual(result["a"], 3.)
        self.assertAlmostEqual(result["b"], 2.)
        expected = np.mean((test[1] - (3 + 2*x_test))**2)
        self.assertAlmostEqual(result["test_rmse"]**2, expected)
        self.assertAlmostEqual(result["test_error"], expected / train[1].var())

    def test_invalid_cli_settings_fail_before_evolution(self):
        for option, value in [("--runs", "0"), ("--pop", "0"),
                              ("--gens", "0"), ("--penalty", "-1")]:
            with self.subTest(option=option):
                result = subprocess.run(
                    [sys.executable, "-B", "run.py", "dataset1.csv", option, value],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
