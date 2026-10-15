"""Dataset loading and the train/test split."""

import numpy as np


def load(path):
    """Read a CSV whose last column is f(...). Returns (env, y).

    env maps each variable name from the header to its column.
    """
    with open(path) as f:
        names = f.readline().strip().split(",")
    table = np.genfromtxt(path, delimiter=",", skip_header=1)
    # The target's own header contains commas ("f(x1,x2,x3)"), so trust the
    # column count rather than the header's field count.
    env = {name: table[:, i] for i, name in enumerate(names[: table.shape[1] - 1])}
    return env, table[:, -1]


def split(env, y, n_test, seed):
    """Hold out n_test points. The test set is never seen during evolution."""
    if not 0 < n_test < len(y):
        raise ValueError("test size must leave nonempty training and test sets")
    order = np.random.default_rng(seed).permutation(len(y))
    test, train = order[:n_test], order[n_test:]
    def take(idx):
        return {k: v[idx] for k, v in env.items()}, y[idx]

    return take(train), take(test)
