# Experiment data

- `experiments.csv`: six experiments with settings, split seeds, and dataset hashes.
- `runs.csv`: 60 runs with formulas, scale constants, training/test errors, and timings.
- `generations.csv`: 3,000 generation records for plotting error, tree size, and diversity.
- `summary.csv`: six rows for comparing population size and size penalty.

Filter by `dataset`, `population`, and `size_penalty` to choose a setting.
Use `experiment` and `run_index` to join runs to their generation records.
`gen`, `run_index`, and `best_run` start at zero. `selected_best_run` marks
the selected model. Diversity in `summary.csv` is a fraction from 0 to 1.
Empty cells represent missing values. Floating-point precision is preserved.
`source_json` names the original JSON export when available.

For a convergence graph, use `gen` for x and `best` for y in `generations.csv`.
`best` is the raw training error of that generation's penalized winner.
`best_fitness` includes the size penalty. `mean_size` and `unique` track tree
size and diversity. For final accuracy comparisons, use `test_error` in
`runs.csv`. Dataset1 and dataset2 use different error metrics.

The original input datasets are in the repository root.
