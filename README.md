# CSC 370 Homework 2

Alex Shields and Gopesh Baheti

Symbolic regression for dataset1 and dataset2. Dataset3 is not used.

## Running

```sh
python -m venv env
source env/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn ipykernel
```

Open `symreg.ipynb`, select the an environment interpreter, and run the cells in order.
The notebook contains the data split, experiment settings, scoring,
CSV output, and graph code. It imports `run_gp` from `symbolic_regression.py`,
which contains the expression trees, training fitness, and genetic operators.

The experiments use these settings: population 1,000, at most 40 generations,
tournament size 4, two elites, and 80% crossover / 20% mutation for the remaining
children. Initial depths are 1 to 5 and maximum depth is 8. There is no fitted
outer scale or size penalty. Training fitness below 0.0001 stops a run early.
Initial constants are integers.

The notebook runs ten seeds, 212 through 221, for each of three conditions:
dataset1, full dataset2, and dataset2 restricted to recorded distance at least 0.1.
The split uses seed 42 and 5,000 held-out rows before the distance filter.

`results.csv` holds every run score, generation history, and held-out prediction.
Each row has a `condition` and a `kind` (run, history, or prediction). The notebook
loads this file instead of rerunning; comment the two lines in the experiment cell
to regenerate it.
