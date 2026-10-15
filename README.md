# CSC 370 Homework 2: Symbolic Regression

Contributors: Alex Shields and Gopesh Baheti

This program evolves arithmetic expression trees for the two required datasets.
The trees use `+`, `-`, `*`, protected division, variables, and integer constants
from -5 through 5. Fitted real-valued scale constants supply the final model's
offset and multiplier. Dataset3 is not attempted.

## Setup and running

Use Python 3.10 or newer. From the repository directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -m unittest -v
python run.py dataset1.csv
python run.py dataset2.csv
```

Defaults are ten runs, population 250, and 50 evaluated generations. These
commands reproduce all six reported settings:

```sh
python run.py dataset1.csv
python run.py dataset1.csv --pop 50
python run.py dataset1.csv --penalty 0
python run.py dataset2.csv
python run.py dataset2.csv --pop 50
python run.py dataset2.csv --penalty 0
```

`--runs`, `--pop`, `--gens`, `--penalty`, and `--seed` change the experiment.
Every filename includes those settings, so different seeds and sweep values
have separate files. Repeating exactly the same command replaces that file.
Add `--plot` to replot the matching JSON without running evolution again:

```sh
python run.py dataset1.csv --plot
python -c 'from pathlib import Path; import run; print(run.summarize(Path("results").glob("*.json")))'
```

The six JSON files are committed. PNG plots are generated locally. Results from
the earlier implementation are superseded; local copies are in the ignored
`results/legacy/` directory.

## How it works

The CSV data is split into 20,000 training rows and 5,000 held-out rows with
seed 370. Each independent run starts with seed 370 through 379. Evolution
uses every training row at every generation. Held-out targets are used only
after a run has finished, and never to fit scale constants.

Initialization alternates full and grow trees at depths 2 through 6. Trees
are immutable: crossover and mutation return new trees. Tournament selection
chooses the best of seven distinct sampled individuals. Each new population
contains 8% copied elites, 90% crossover children, and 2% mutation children
(20/225/5 at population 250; 4/45/1 at population 50). Crossover and mutation
choose internal nodes 90% of the time when available, and leaves otherwise.
A child past depth 8 is replaced by its parent. Protected division returns 1
when the absolute divisor is below `1e-12`.

For dataset1, each tree's outputs `f` are scaled as `a + b*f` by least squares.
Error is MSE divided by the variance of the training targets; that same
normalizer is used for held-out error. This lets evolution search for the
expression's shape while fitting its offset and multiplier on training data.

For dataset2, the prediction is `b*f`, fitted by weighted least squares.
With `d = max(abs(y), 0.005)` for each row, error is
`mean(((prediction - y) / d)**2)`. The weights stop the largest measurements
from dominating ordinary squared error. The 0.005 denominator floor reflects
the targets' two-decimal recording precision and keeps zero targets usable.
All rows are retained, including the 13 whose recorded `x3` is zero. Protected
division makes the program finite there, but information lost by input rounding
cannot be recovered by that arithmetic rule. This scoring choice is a modeling
assumption, not a guarantee of recovering the generating function.

Training fitness is `error * (1 + size_penalty * tree_size)`, with penalty 0.01
by default. This trades accuracy against complexity. Tree size counts the raw
tree's nodes; the fitted outer constants are not included. A run evaluates
50 populations (generation 0 is the initial population) and returns the best
training-fitness individual seen. There are no accuracy-based early stops.

Among the ten returned models, the reported winner minimizes held-out error
with the same size penalty. Because those held-out rows select the winner,
they are not an additional untouched final assessment. Population and penalty
comparisons are descriptive experiments, not proof of optimal parameter values.

## Code and results

- `tree.py`: expression trees, initialization, and protected arithmetic.
- `data.py`: CSV loading and splitting.
- `config.py`: dataset settings and parameter checks.
- `fitness.py`: training-only fitting and prediction errors.
- `gp.py`: selection, crossover, mutation, and evolution.
- `run.py`: independent runs, JSON output, tables, and plots.
- `test_gp.py` and `test_run.py`: tree, numerical, loop, and scoring checks.

See [RESULTS.md](RESULTS.md) for measured models and comparisons. The assignment's
[Koza tutorial](https://www.genetic-programming.com/gpquadraticexample.html)
is the background for tree-based symbolic regression and its genetic operators.
Our weighted error, scale fitting, elitism, and fixed generation budget are the
specific choices used here.
