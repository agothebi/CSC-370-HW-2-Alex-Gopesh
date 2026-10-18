"""Symbolic regression for CSC 370 HW2. Alex Shields and Gopesh Baheti."""

import argparse
import csv
from dataclasses import dataclass
import dataclasses
import hashlib
from pathlib import Path
import platform
import random
import time

import numpy as np


@dataclass(frozen=True)
class Config:
    dataset: str = "dataset1.csv"
    variables: tuple = ("x",)
    metric: str = "least_squares"

    population: int = 250
    generations: int = 50
    tournament_k: int = 7
    elite_frac: float = 0.08  # reproduction: copied unchanged
    crossover_frac: float = 0.90  # the remaining 2% is mutation

    init_min_depth: int = 2
    init_max_depth: int = 6
    max_depth: int = 8
    # Fitness = error * (1 + penalty * number of tree nodes).
    size_penalty: float = 0.01
    n_test: int = 5000

    def __post_init__(self):
        if self.population < self.tournament_k or self.tournament_k < 1:
            raise ValueError("population must be at least tournament_k, which must be positive")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if not 0 <= self.elite_frac <= 1 or not 0 <= self.crossover_frac <= 1:
            raise ValueError("reproduction fractions must be between 0 and 1")
        if self.elite_frac + self.crossover_frac > 1:
            raise ValueError("elite and crossover fractions must sum to at most 1")
        n_elite = round(self.elite_frac * self.population)
        n_cross = round(self.crossover_frac * self.population)
        if n_elite + n_cross > self.population:
            raise ValueError("rounded reproduction counts exceed population")
        if not 0 <= self.size_penalty < float("inf"):
            raise ValueError("size penalty must be finite and nonnegative")
        if not 0 <= self.init_min_depth <= self.init_max_depth <= self.max_depth:
            raise ValueError("initial depths must be within the depth limit")
        if self.metric not in ("least_squares", "relative"):
            raise ValueError("unknown error metric")


PRESETS = {
    "dataset1.csv": Config(),
    "dataset2.csv": Config(
        dataset="dataset2.csv",
        variables=("x1", "x2", "x3"),
        metric="relative",
    ),
}


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


def protected_divide(a, b):
    """Koza's protected division: 1 where the divisor is ~zero."""
    zero = np.abs(b) < 1e-12
    return np.where(zero, 1.0, a / np.where(zero, 1.0, b))


OPS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": protected_divide}


@dataclass(frozen=True)
class ExpressionTree:
    value: object
    left: "ExpressionTree | None" = None
    right: "ExpressionTree | None" = None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def children(self):
        return () if self.is_leaf else (self.left, self.right)

    def evaluate(self, env):
        n = len(next(iter(env.values())))
        with np.errstate(all="ignore"):
            return np.broadcast_to(self._eval(env), n)

    def _eval(self, env):
        if self.is_leaf:
            return env[self.value] if self.value in env else float(self.value)
        return OPS[self.value](self.left._eval(env), self.right._eval(env))

    @property
    def size(self):
        if self.is_leaf:
            return 1
        return 1 + self.left.size + self.right.size

    @property
    def depth(self):
        if self.is_leaf:
            return 0
        return 1 + max(self.left.depth, self.right.depth)

    def paths(self, prefix=()):
        yield prefix
        for i, child in enumerate(self.children):
            yield from child.paths(prefix + (i,))

    def at(self, path):
        node = self
        for i in path:
            node = node.children[i]
        return node

    def replace(self, path, replacement):
        if not path:
            return replacement
        children = list(self.children)
        i = path[0]
        children[i] = children[i].replace(path[1:], replacement)
        return ExpressionTree(self.value, *children)

    def __str__(self):
        if self.is_leaf:
            return str(self.value)
        return f"({self.left} {self.value} {self.right})"


def grow(depth, cfg, rng):
    """Ragged tree: branches may stop early."""
    if depth == 0 or rng.random() < 0.3:
        return leaf(cfg, rng)
    return ExpressionTree(rng.choice(list(OPS)), *(grow(depth - 1, cfg, rng) for _ in range(2)))


def full(depth, cfg, rng):
    """Bushy tree: every branch runs to depth."""
    if depth == 0:
        return leaf(cfg, rng)
    return ExpressionTree(rng.choice(list(OPS)), *(full(depth - 1, cfg, rng) for _ in range(2)))


def leaf(cfg, rng):
    if rng.random() < 0.5:
        return ExpressionTree(rng.choice(cfg.variables))
    return ExpressionTree(rng.randint(-5, 5))


def initial_population(cfg, rng):
    """Alternate grow and full trees at different depths."""
    depths = range(cfg.init_min_depth, cfg.init_max_depth + 1)
    builders = [grow, full]
    return [
        builders[i % 2](depths[i % len(depths)], cfg, rng)
        for i in range(cfg.population)
    ]


RESOLUTION = 0.005


def fit_scale(predictions, targets, metric):
    """Fit a + b*predictions using training data only."""
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
    child = t.replace(pick_path(t, rng), grow(cfg.init_max_depth, cfg, rng))
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
    pop = initial_population(cfg, rng)
    normalizer = max(float(y.var()), 1e-30)
    best, best_key = None, np.inf
    history = []

    for gen in range(cfg.generations):
        errors = [score(t, env, y, cfg, normalizer)[0] for t in pop]
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


def result_name(cfg, seed, runs):
    return (f"{Path(cfg.dataset).stem}_pop{cfg.population}_gens{cfg.generations}"
            f"_pen{cfg.size_penalty:g}_seed{seed}_runs{runs}")


def evaluate(t, train, test, cfg):
    """Fit on train once, then use those constants for both sets."""
    normalizer = max(float(train[1].var()), 1e-30)
    _, a, b = score(t, *train, cfg, normalizer)
    train_predictions = a + b * t.evaluate(train[0])
    test_predictions = a + b * t.evaluate(test[0])
    return {
        "tree": str(t),
        "a": a,
        "b": b,
        "expression": f"{a:.6g} + {b:.6g} * {t}",
        "train_error": loss(train_predictions, train[1], cfg.metric, normalizer),
        "test_error": loss(test_predictions, test[1], cfg.metric, normalizer),
        "test_rmse": float(np.sqrt(np.mean((test[1] - test_predictions) ** 2))),
        "size": t.size,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=list(PRESETS))
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--pop", type=int)
    p.add_argument("--gens", type=int)
    p.add_argument("--penalty", type=float)
    p.add_argument("--seed", type=int, default=370)
    p.add_argument("--plot", action="store_true", help="replot the matching saved experiment")
    args = p.parse_args()
    if args.runs < 1 or args.seed < 0:
        p.error("runs must be positive and seed must be nonnegative")
    overrides = {key: value for key, value in {
        "population": args.pop, "generations": args.gens, "size_penalty": args.penalty,
    }.items() if value is not None}
    try:
        cfg = dataclasses.replace(PRESETS[args.dataset], **overrides)
    except ValueError as exc:
        p.error(str(exc))
    path = Path("results") / result_name(cfg, args.seed, args.runs)
    if args.plot:
        plot(path)
        return

    env, y = load(args.dataset)
    train, test = split(env, y, cfg.n_test, args.seed)
    print(f"{args.dataset}: {args.runs} runs, population {cfg.population}, "
          f"{cfg.generations} evaluated generations", flush=True)
    runs = []
    for i in range(args.runs):
        seed = args.seed + i
        started = time.perf_counter()
        t, history = evolve(train, cfg, seed)
        r = evaluate(t, train, test, cfg)
        r.update(seed=seed, history=history, generations=len(history),
                 elapsed_seconds=time.perf_counter() - started)
        runs.append(r)
        print(f"  seed {seed}: test={r['test_error']:.6g}, size={t.size}, "
              f"{r['elapsed_seconds']:.1f}s  {r['expression']}", flush=True)
    best_index = min(range(len(runs)), key=lambda i:
                     runs[i]["test_error"] * (1 + cfg.size_penalty * runs[i]["size"]))
    result = {
        "config": dataclasses.asdict(cfg),
        "split_seed": args.seed,
        "n_train": len(train[1]),
        "n_test": len(test[1]),
        "normalizer": max(float(train[1].var()), 1e-30),
        "zero_distance_rows": int(np.sum(env["x3"] == 0)) if "x3" in env else 0,
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "best_run": best_index,
        "runs": runs,
    }
    save_results(path, result)
    print(f"best: {runs[best_index]['expression']}\nwrote CSVs in {path.parent}", flush=True)
    plot(path)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def save_results(path, result):
    cfg, runs = result["config"], result["runs"]
    common = {"experiment": path.name, "dataset": cfg["dataset"],
              "population": cfg["population"], "size_penalty": cfg["size_penalty"]}
    settings = {"experiment": path.name, "source_json": ""}
    settings.update({k: v for k, v in result.items() if k not in ("config", "runs")})
    settings.update({f"config_{k}": ",".join(v) if isinstance(v, (tuple, list)) else v
                     for k, v in cfg.items()})
    run_rows, generation_rows = [], []
    for i, run in enumerate(runs):
        row = {**common, "run_index": i, "selected_best_run": i == result["best_run"]}
        row.update({k: v for k, v in run.items() if k != "history"})
        run_rows.append(row)
        generation_rows.extend({**common, "run_index": i, "seed": run["seed"], **point}
                               for point in run["history"])
    best = runs[result["best_run"]]
    summary = {**common, "run_count": len(runs), "generations": cfg["generations"],
               "selected_run_index": result["best_run"], "selected_seed": best["seed"],
               "selected_expression": best["expression"],
               "selected_test_error": best["test_error"], "selected_test_rmse": best["test_rmse"],
               "selected_tree_size": best["size"],
               "median_test_error": float(np.median([r["test_error"] for r in runs])),
               "median_tree_size": float(np.median([r["size"] for r in runs])),
               "mean_initial_diversity": float(np.mean([r["history"][0]["unique"] for r in runs]) / cfg["population"]),
               "mean_final_diversity": float(np.mean([r["history"][-1]["unique"] for r in runs]) / cfg["population"])}
    tables = {"experiments.csv": [settings], "runs.csv": run_rows,
              "generations.csv": generation_rows, "summary.csv": [summary]}
    for rows in tables.values():
        for row in rows:
            if any(isinstance(v, float) and not np.isfinite(v) for v in row.values()):
                raise ValueError("cannot save nonfinite experiment results")
    path.parent.mkdir(exist_ok=True)
    for name, rows in tables.items():
        target = path.parent / name
        existing = read_csv(target) if target.exists() else []
        existing = [r for r in existing if r["experiment"] != path.name]
        columns = list(dict.fromkeys(k for row in existing + rows for k in row))
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(existing + rows)
        temporary.replace(target)


def summarize(directory="results"):
    """Make the results table from summary.csv."""
    lines = [
        "| Dataset | Population | Penalty | Selected test error | Median test error | Median nodes | Mean diversity, initial → final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(read_csv(Path(directory) / "summary.csv"), key=lambda r: r["experiment"]):
        lines.append(
            f"| {Path(row['dataset']).stem} | {row['population']} | {float(row['size_penalty']):g} "
            f"| {float(row['selected_test_error']):.6g} | {float(row['median_test_error']):.6g} "
            f"| {float(row['median_tree_size']):g} | {float(row['mean_initial_diversity']):.1%} → {float(row['mean_final_diversity']):.1%} |"
        )
    return "\n".join(lines)


def plot(path):
    """Plot training convergence, tree size, and expression diversity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    rows = [r for r in read_csv(path.parent / "generations.csv") if r["experiment"] == path.name]
    if not rows:
        raise ValueError(f"no saved generations for {path.name}")
    run_ids = sorted({int(r["run_index"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key, label, ax in zip(
        ["best", "mean_size", "unique"],
        ["training error of generation winner", "mean tree size", "distinct expressions"], axes,
    ):
        for run_id in run_ids:
            points = sorted((r for r in rows if int(r["run_index"]) == run_id), key=lambda r: int(r["gen"]))
            ax.plot([int(r["gen"]) for r in points], [float(r[key]) for r in points], lw=1)
        ax.set(xlabel="generation (0 is initial)", ylabel=label, title=label)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    fig.suptitle(f"{path.name}: {len(run_ids)} runs")
    fig.tight_layout()
    output = path.parent / (path.name + ".png")
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
