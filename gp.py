"""Tournament selection, subtree operators, and the evolutionary loop."""

import random

import numpy as np

import fitness
import tree


def tournament(pop, keys, k, rng):
    """Pick the best of k distinct randomly sampled individuals."""
    best = min(rng.sample(range(len(pop)), k), key=lambda i: keys[i])
    return pop[best]


def pick_path(t, rng):
    """Choose an internal node 90% of the time, otherwise a leaf."""
    paths = list(t.paths())
    internal = [p for p in paths if t.at(p).children]
    leaves = [p for p in paths if not t.at(p).children]
    return rng.choice(internal if internal and rng.random() < 0.9 else leaves)


def crossover(a, b, cfg, rng):
    """Replace a subtree in the first parent with one from the second."""
    child = a.replace(pick_path(a, rng), b.at(pick_path(b, rng)))
    return child if child.depth <= cfg.max_depth else a


def mutate(t, cfg, rng):
    """Replace a random subtree with a newly grown one."""
    child = t.replace(pick_path(t, rng), tree.grow(cfg.init_max_depth, cfg, rng))
    return child if child.depth <= cfg.max_depth else t


def next_population(pop, keys, cfg, rng):
    """Copy elites, then create the configured number of crossover children."""
    order = sorted(range(len(pop)), key=lambda i: keys[i])
    n_elite = round(cfg.elite_frac * cfg.population)
    n_cross = round(cfg.crossover_frac * cfg.population)
    nxt = [pop[i] for i in order[:n_elite]]
    for _ in range(n_cross):
        nxt.append(crossover(
            tournament(pop, keys, cfg.tournament_k, rng),
            tournament(pop, keys, cfg.tournament_k, rng), cfg, rng,
        ))
    while len(nxt) < cfg.population:
        nxt.append(mutate(tournament(pop, keys, cfg.tournament_k, rng), cfg, rng))
    return nxt


def evolve(train, cfg, seed):
    """Evaluate a fixed number of populations on all training rows."""
    env, y = train
    rng = random.Random(seed)
    pop = tree.initial_population(cfg, rng)
    normalizer = max(float(y.var()), 1e-30)
    best, best_key = None, np.inf
    history = []

    for gen in range(cfg.generations):
        errors = [fitness.score(t, env, y, cfg, normalizer)[0] for t in pop]
        keys = [penalized(t, e, cfg) for t, e in zip(pop, errors)]
        winner = min(range(len(pop)), key=lambda i: keys[i])
        if best is None or keys[winner] < best_key:
            best, best_key = pop[winner], keys[winner]
        history.append(stats(gen, pop, errors, keys[winner], winner))
        if gen + 1 < cfg.generations:
            pop = next_population(pop, keys, cfg, rng)

    if not np.isfinite(best_key):
        raise ValueError("no individual had finite training fitness")
    return best, history


def penalized(t, err, cfg):
    return err * (1 + cfg.size_penalty * t.size)


def stats(gen, pop, errors, best_fitness, winner):
    sizes = [t.size for t in pop]
    finite = [e for e in errors if np.isfinite(e)]
    return {
        "gen": gen,
        "best": errors[winner],
        "best_fitness": best_fitness,
        "median": float(np.median(finite)) if finite else None,
        "mean_size": float(np.mean(sizes)),
        "max_size": max(sizes),
        "unique": len({str(t) for t in pop}),
    }
