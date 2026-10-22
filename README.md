# CSC 370 Homework 2

Alex Shields and Gopesh Baheti

Symbolic regression with genetic programming on dataset1 and dataset2. Dataset3 is not used.

## Files

- `symbolic_regression.py`: expression trees, fitness, crossover, mutation and the GP loop (`run_gp`)
- `symreg.ipynb`: data split, experiments and plots
- `results.csv`: saved results, so the notebook doesn't have to rerun the GP

## Running

```sh
python3 -m venv env
source env/bin/activate
python3 -m pip install numpy pandas matplotlib scikit-learn ipykernel
```

Open `symreg.ipynb`, select the env interpreter, and run all cells. It loads `results.csv`.
To rerun the experiments (about 10 minutes), uncomment the run lines in the cell that loads `results.csv`.

## Settings

- Population 1,000, up to 40 generations, 10 seeds (212 to 221)
- Tournament size 4, 2 elites, 80% crossover and 20% mutation
- Starting depth 1 to 5, max depth 8
- Fitness: MSE / Var(y), plus 0.00001 per node. A run stops early when the error is below 0.0001
- 5,000 rows held out for testing (split seed 42)

## Experiments

- Dataset1
- Dataset2, all rows
- Dataset2, rows with x3 >= 0.1
- Dataset2, rows with x3 >= 0.1, with linear scaling (each tree is scored as `a + b * tree`)

## Note

This README and some of the tests were done with the help of an LLM.
