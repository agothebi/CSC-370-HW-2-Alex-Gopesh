"""Checks for tree operations, scoring, and the evolutionary loop."""

import random
import dataclasses
from unittest.mock import patch
import unittest

import numpy as np

import config
import data
import fitness
import gp
import tree
from tree import Node as N


def scalar_eval(t, x):
    """Independent single-point reference for the vectorized evaluator."""
    if not t.children:
        return x[t.value] if t.value in x else float(t.value)
    a, b = (scalar_eval(c, x) for c in t.children)
    if t.value == "/":
        return 1.0 if abs(b) < 1e-12 else a / b
    return {"+": a + b, "-": a - b, "*": a * b}[t.value]


class TestTree(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(370)
        self.cfg = config.PRESETS["dataset1.csv"]

    def test_evaluate_matches_scalar_reference(self):
        xs = np.linspace(-3, 3, 41)
        for _ in range(200):
            t = tree.grow(4, self.cfg, self.rng)
            expected = [scalar_eval(t, {"x": float(v)}) for v in xs]
            np.testing.assert_allclose(t.evaluate({"x": xs}), expected, rtol=1e-9)

    def test_protected_division_returns_one(self):
        t = N("/", (N("x"), N(0)))
        np.testing.assert_array_equal(t.evaluate({"x": np.array([2.0, 5.0])}), [1.0, 1.0])

    def test_constant_tree_broadcasts(self):
        self.assertEqual(len(N(3).evaluate({"x": np.zeros(7)})), 7)

    def test_replace_leaves_original_untouched(self):
        t = N("+", (N("x"), N(1)))
        self.assertEqual(str(t.replace((1,), N(99))), "(x + 99)")
        self.assertEqual(str(t), "(x + 1)")

    def test_depth_and_size(self):
        t = N("+", (N("*", (N("x"), N("x"))), N(1)))
        self.assertEqual((t.depth, t.size), (2, 5))
        self.assertEqual(N("x").depth, 0)


class TestFitness(unittest.TestCase):
    def test_linear_scaling_recovers_constants(self):
        pred = np.linspace(-2, 2, 50)
        y = 7.0 + 3.0 * pred
        a, b = fitness.fit_scale(pred, y, "least_squares")
        self.assertAlmostEqual(a, 7.0)
        self.assertAlmostEqual(b, 3.0)
        self.assertAlmostEqual(fitness.loss(a + b * pred, y, "least_squares"), 0.0)

    def test_relative_scaling_across_orders_of_magnitude(self):
        pred = np.geomspace(1, 1e20, 100)
        y = 6.67e-11 * pred
        a, b = fitness.fit_scale(pred, y, "relative")
        self.assertEqual(a, 0.0)
        self.assertAlmostEqual(b / 6.67e-11, 1.0)
        self.assertLess(fitness.loss(a + b * pred, y, "relative"), 1e-25)

    def test_fixed_predictions_are_not_refitted(self):
        y = np.arange(1., 11.)
        self.assertAlmostEqual(fitness.loss(2 * y, y, "relative"), 1.0)
        self.assertAlmostEqual(fitness.loss(-y, y, "relative"), 4.0)

    def test_zero_targets_are_kept(self):
        self.assertAlmostEqual(fitness.loss(np.array([.005, 1.]),
                                           np.array([0., 1.]), "relative"), .5)

    def test_constant_and_invalid_predictions(self):
        y = np.arange(5.)
        a, b = fitness.fit_scale(np.ones(5), y, "least_squares")
        self.assertEqual((a, b), (2., 0.))
        self.assertEqual(fitness.fit_scale(np.zeros(5), y, "relative"), (0., 0.))
        for bad in [np.inf, np.nan]:
            t = N(bad)
            self.assertEqual(fitness.error(t, {"x": y}, y, config.Config()), np.inf)


class TestData(unittest.TestCase):
    def test_split_is_disjoint_complete_and_repeatable(self):
        y = np.arange(100)
        train, test = data.split({"x": y}, y, 20, 370)
        again = data.split({"x": y}, y, 20, 370)
        self.assertEqual(len(train[1]), 80)
        self.assertEqual(len(test[1]), 20)
        self.assertFalse(set(train[1]) & set(test[1]))
        self.assertEqual(set(train[1]) | set(test[1]), set(y))
        np.testing.assert_array_equal(test[1], again[1][1])
        for invalid in (0, 100):
            with self.assertRaises(ValueError):
                data.split({"x": y}, y, invalid, 370)


class TestEvolve(unittest.TestCase):
    def test_recovers_koza_quadratic(self):
        """x^2 + x + 1 from the tutorial, which GP should find quickly."""
        cfg = dataclasses.replace(
            config.PRESETS["dataset1.csv"], population=200, generations=30
        )
        x = np.linspace(-1, 1, 100)
        best, history = gp.evolve(({"x": x}, x * x + x + 1), cfg, seed=1)
        self.assertLess(fitness.error(best, {"x": x}, x * x + x + 1, cfg), 1e-10)
        self.assertLessEqual(best.depth, cfg.max_depth)
        self.assertEqual(len(history), cfg.generations)

    def test_reproduction_counts(self):
        cfg = config.Config()
        pop = [N(i) for i in range(cfg.population)]
        with patch("gp.crossover", return_value=N("cross")) as cross, \
             patch("gp.mutate", return_value=N("mutate")) as mutate:
            nxt = gp.next_population(pop, list(range(cfg.population)), cfg, random.Random(1))
        self.assertEqual(len(nxt), 250)
        self.assertEqual(nxt[:20], pop[:20])
        self.assertEqual(cross.call_count, 225)
        self.assertEqual(mutate.call_count, 5)

    def test_cut_point_can_choose_a_leaf(self):
        t = N("+", (N("x"), N(1)))
        rng = random.Random(1)
        with patch.object(rng, "random", return_value=.99):
            self.assertFalse(t.at(gp.pick_path(t, rng)).children)
        with patch.object(rng, "random", return_value=.01):
            self.assertTrue(t.at(gp.pick_path(t, rng)).children)

    def test_generation_limit_returns_an_evaluated_tree(self):
        cfg = dataclasses.replace(config.Config(), population=7, generations=1)
        train = ({"x": np.arange(10.)}, np.arange(10.))
        pop = [N(i) for i in range(7)]
        with patch("tree.initial_population", return_value=pop), \
             patch("gp.next_population") as reproduce:
            best, history = gp.evolve(train, cfg, 370)
        self.assertIn(best, pop)
        self.assertEqual(len(history), 1)
        reproduce.assert_not_called()

    def test_seed_repeats_tree_and_history(self):
        cfg = dataclasses.replace(config.Config(), population=20, generations=3)
        x = np.linspace(-1, 1, 30)
        train = ({"x": x}, x*x+x+1)
        self.assertEqual(gp.evolve(train, cfg, 3), gp.evolve(train, cfg, 3))

    def test_invalid_settings(self):
        for changes in [{"population": 0}, {"population": 6}, {"generations": 0},
                        {"size_penalty": -1}, {"size_penalty": np.nan}]:
            with self.assertRaises(ValueError):
                dataclasses.replace(config.Config(), **changes)


if __name__ == "__main__":
    unittest.main()
