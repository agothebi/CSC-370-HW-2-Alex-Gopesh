"""Fit scale constants on training data, then score fixed predictions."""

import numpy as np

# Half of one unit in the last decimal place of dataset2's recorded targets.
RESOLUTION = 0.005


def fit_scale(predictions, targets, metric):
    """Return (a, b) for a + b*predictions. Only call this on training data."""
    if not np.all(np.isfinite(predictions)):
        return np.nan, np.nan
    scale = float(np.max(np.abs(predictions)))
    if scale == 0:
        return (float(targets.mean()), 0.0) if metric == "least_squares" else (0.0, 0.0)
    # Normalizing avoids squaring enormous raw tree outputs during fitting.
    q = predictions / scale
    with np.errstate(all="ignore"):
        if metric == "least_squares":
            centered = q - q.mean()
            variance = np.mean(centered * centered)
            if variance == 0:
                return float(targets.mean()), 0.0
            b = np.mean(centered * (targets - targets.mean())) / variance
            a = targets.mean() - b * q.mean()
        elif metric == "relative":
            denominator = np.maximum(np.abs(targets), RESOLUTION)
            weighted = q / denominator
            weight_sum = np.mean(weighted * weighted)
            if weight_sum == 0:
                return 0.0, 0.0
            b = np.mean(weighted * (targets / denominator)) / weight_sum
            a = 0.0
        else:
            raise ValueError(f"unknown metric: {metric}")
        return float(a), float(b / scale)


def loss(predictions, targets, metric, normalizer=1.0):
    """Score predictions without fitting anything to the supplied targets."""
    if not np.all(np.isfinite(predictions)):
        return np.inf
    with np.errstate(all="ignore"):
        residual = predictions - targets
        if metric == "least_squares":
            result = np.mean(residual * residual) / normalizer
        elif metric == "relative":
            residual = residual / np.maximum(np.abs(targets), RESOLUTION)
            result = np.mean(residual * residual)
        else:
            raise ValueError(f"unknown metric: {metric}")
    return float(result) if np.isfinite(result) else np.inf


def score(t, env, y, cfg, normalizer=None):
    """Fit and score one training candidate; return (error, a, b)."""
    if t.depth > cfg.max_depth:
        return np.inf, 0.0, 0.0
    if normalizer is None:
        normalizer = max(float(y.var()), 1e-30)
    raw = t.evaluate(env)
    a, b = fit_scale(raw, y, cfg.metric)
    with np.errstate(all="ignore"):
        predictions = a + b * raw
    return loss(predictions, y, cfg.metric, normalizer), a, b


def error(t, env, y, cfg):
    return score(t, env, y, cfg)[0]
