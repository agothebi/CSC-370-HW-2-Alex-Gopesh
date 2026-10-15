"""Immutable expression trees, evaluated on whole numpy arrays at once."""

from typing import NamedTuple

import numpy as np


def protected_divide(a, b):
    """Koza's protected division: 1 where the divisor is ~zero."""
    zero = np.abs(b) < 1e-12
    return np.where(zero, 1.0, a / np.where(zero, 1.0, b))


OPS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": protected_divide}


class Node(NamedTuple):
    """An operator with two children, or a leaf (variable name or number)."""

    value: object
    children: tuple = ()

    def evaluate(self, env):
        """env maps variable name -> ndarray. Returns an ndarray."""
        n = len(next(iter(env.values())))
        with np.errstate(all="ignore"):
            # broadcast so that a constant-only tree still returns n values
            out = np.broadcast_to(self._eval(env), n)
        return out  # fitness rejects nonfinite outputs

    def _eval(self, env):
        if not self.children:
            return env[self.value] if self.value in env else float(self.value)
        a, b = (c._eval(env) for c in self.children)
        return OPS[self.value](a, b)

    @property
    def depth(self):
        return 1 + max((c.depth for c in self.children), default=-1)

    @property
    def size(self):
        return 1 + sum(c.size for c in self.children)

    def paths(self, prefix=()):
        """Path (tuple of child indices) to every node, pre-order."""
        yield prefix
        for i, child in enumerate(self.children):
            yield from child.paths(prefix + (i,))

    def at(self, path):
        node = self
        for i in path:
            node = node.children[i]
        return node

    def replace(self, path, new):
        """New tree with the subtree at path swapped out; the rest is shared."""
        if not path:
            return new
        i, *rest = path
        children = list(self.children)
        children[i] = children[i].replace(tuple(rest), new)
        return Node(self.value, tuple(children))

    def __str__(self):
        if not self.children:
            return str(self.value)
        return f"({self.children[0]} {self.value} {self.children[1]})"


def grow(depth, cfg, rng):
    """Ragged tree: branches may stop early."""
    if depth == 0 or rng.random() < 0.3:
        return leaf(cfg, rng)
    return Node(rng.choice(list(OPS)), tuple(grow(depth - 1, cfg, rng) for _ in range(2)))


def full(depth, cfg, rng):
    """Bushy tree: every branch runs to depth."""
    if depth == 0:
        return leaf(cfg, rng)
    return Node(rng.choice(list(OPS)), tuple(full(depth - 1, cfg, rng) for _ in range(2)))


def leaf(cfg, rng):
    if rng.random() < 0.5:
        return Node(rng.choice(cfg.variables))
    return Node(cfg.constant(rng))


def initial_population(cfg, rng):
    """Ramped half-and-half (Koza): depths ramped across the population,
    alternating grow and full, for a diverse generation 0."""
    depths = range(cfg.init_min_depth, cfg.init_max_depth + 1)
    builders = [grow, full]
    return [
        builders[i % 2](depths[i % len(depths)], cfg, rng)
        for i in range(cfg.population)
    ]
