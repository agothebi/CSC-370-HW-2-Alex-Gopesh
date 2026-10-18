# CSC 370 Homework 2

Alex Shields and Gopesh Baheti

Symbolic regression with expression trees for dataset1 and dataset2.
Dataset3 is optional and we did not use it.

## Running

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -m unittest -v
python run.py dataset1.csv
python run.py dataset2.csv
```

The defaults are 10 runs, population 250, and 50 generations. Each run uses
20,000 training rows and 5,000 test rows. The split seed is 370 and the run
seeds are 370 through 379.

For the population and size penalty comparisons:

```sh
python run.py dataset1.csv --pop 50
python run.py dataset1.csv --penalty 0
python run.py dataset2.csv --pop 50
python run.py dataset2.csv --penalty 0
```

Results go in `results/` as JSON and PNG files. Running the same command
again overwrites its files. Use `--plot` to plot an existing JSON:

```sh
python run.py dataset1.csv --plot
```

## Code

`tree.py` builds and evaluates trees. `gp.py` handles selection, crossover,
mutation, and the generation loop. `fitness.py` scores trees, `data.py` loads
the CSVs, and `config.py` has the settings. `run.py` runs the experiments.

Trees use +, -, *, protected division, variables, and integers from -5 to 5.
Division returns 1 when the divisor is smaller than 1e-12 in absolute value.
Initialization mixes full and grow trees at depths 2 to 6. Tournament size is 7.
Each generation keeps 8% elites, makes 90% crossover children, and fills the
remaining 2% with mutation. Children deeper than 8 are replaced by their parent.
Fitness is `error * (1 + 0.01 * tree_size)` by default.

Scale constants are fitted on the training rows. Dataset1 uses `a + b*tree`
and MSE divided by training target variance. Dataset2 uses `b*tree` and mean
squared relative error, with denominator `max(abs(y), 0.005)`. The floor is
half the last recorded decimal unit. All rows are kept, including zero targets
and zero distances. Test predictions use the same fitted constants.

`symbolic_regression.py` and `symreg.ipynb` are Gopesh's implementation and
notebook. They use a separate GP loop and scoring method. The saved results
in [RESULTS.md](RESULTS.md) come from `run.py`.

The expression trees and operators follow the assignment's
[Koza tutorial](https://www.genetic-programming.com/gpquadraticexample.html).
