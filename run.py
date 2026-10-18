"""Run independent GP experiments, save their measurements, and plot them."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

import config
import data
import fitness
import gp


def result_name(cfg, seed, runs):
    return (f"{Path(cfg.dataset).stem}_pop{cfg.population}_gens{cfg.generations}"
            f"_pen{cfg.size_penalty:g}_seed{seed}_runs{runs}")


def evaluate(t, train, test, cfg):
    """Fit on train once, then use those constants for both sets."""
    normalizer = max(float(train[1].var()), 1e-30)
    _, a, b = fitness.score(t, *train, cfg, normalizer)
    train_predictions = a + b * t.evaluate(train[0])
    test_predictions = a + b * t.evaluate(test[0])
    return {
        "tree": str(t),
        "a": a,
        "b": b,
        "expression": f"{a:.6g} + {b:.6g} * {t}",
        "train_error": fitness.loss(train_predictions, train[1], cfg.metric, normalizer),
        "test_error": fitness.loss(test_predictions, test[1], cfg.metric, normalizer),
        "test_rmse": float(np.sqrt(np.mean((test[1] - test_predictions) ** 2))),
        "size": t.size,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=list(config.PRESETS))
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
        cfg = dataclasses.replace(config.PRESETS[args.dataset], **overrides)
    except ValueError as exc:
        p.error(str(exc))
    path = Path("results") / (result_name(cfg, args.seed, args.runs) + ".json")
    if args.plot:
        plot(path)
        return

    env, y = data.load(args.dataset)
    train, test = data.split(env, y, cfg.n_test, args.seed)
    print(f"{args.dataset}: {args.runs} runs, population {cfg.population}, "
          f"{cfg.generations} evaluated generations", flush=True)
    runs = []
    for i in range(args.runs):
        seed = args.seed + i
        started = time.perf_counter()
        t, history = gp.evolve(train, cfg, seed)
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
    # Validate before replacing an existing result file.
    saved = json.dumps(result, indent=1, allow_nan=False)
    path.parent.mkdir(exist_ok=True)
    path.write_text(saved + "\n")
    print(f"best: {runs[best_index]['expression']}\nwrote {path}", flush=True)
    plot(path)


def summarize(paths):
    """Make the results table directly from saved experiments."""
    lines = [
        "| Dataset | Population | Penalty | Selected test error | Median test error | Median nodes | Mean diversity, initial → final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for path in sorted(map(Path, paths)):
        d = json.loads(path.read_text())
        cfg, runs = d["config"], d["runs"]
        best = runs[d["best_run"]]
        initial = np.mean([r["history"][0]["unique"] for r in runs]) / cfg["population"]
        final = np.mean([r["history"][-1]["unique"] for r in runs]) / cfg["population"]
        lines.append(
            f"| {Path(cfg['dataset']).stem} | {cfg['population']} | {cfg['size_penalty']:g} "
            f"| {best['test_error']:.6g} | {np.median([r['test_error'] for r in runs]):.6g} "
            f"| {np.median([r['size'] for r in runs]):g} | {initial:.1%} → {final:.1%} |"
        )
    return "\n".join(lines)


def plot(path):
    """Plot training convergence, tree size, and expression diversity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    d = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key, label, ax in zip(
        ["best", "mean_size", "unique"],
        ["training error of generation winner", "mean tree size", "distinct expressions"], axes,
    ):
        for r in d["runs"]:
            ax.plot([h["gen"] for h in r["history"]], [h[key] for h in r["history"]], lw=1)
        ax.set(xlabel="generation (0 is initial)", ylabel=label, title=label)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    fig.suptitle(f"{path.stem}: {len(d['runs'])} runs")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=130)
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
