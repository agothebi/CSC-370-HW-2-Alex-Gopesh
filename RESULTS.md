# Results

Each setting was run 10 times with seeds 370 through 379. We used the same
20,000/5,000 train/test split (seed 370) and 50 generations for every setting.

## Functions found

| Dataset | Selected model | Test error | Test RMSE | Tree nodes |
|---|---|---:|---:|---:|
| dataset1 | `13.99997085 + 0.999976112 * ((x * (x - 5)) - x)` | 1.66457e-06 | 0.0242904 | 7 |
| dataset2 | `6.665645961e-11 * (((x1 / x3) / x3) * x2)` | 0.0015837 | 1.29674e+07 | 7 |

Dataset1 is approximately `x*x - 6*x + 14`. Eight runs found seven-node trees
and two found nine-node trees with the same quadratic shape.

Dataset2 found the inverse-square form `G*x1*x2/(x3*x3)` in nine of ten runs,
with G approximately 6.66565e-11. This looks like Newton's gravitational law,
though we do not know the original generating formula. The remaining run
found an 11-node expression with slightly lower raw test error. The size
penalty selected a seven-node tree overall.

## Changing the settings

The selected run minimizes `test_error * (1 + penalty * tree_size)`.
Dataset1's error is normalized MSE. Dataset2's is squared relative error
with denominator floor 0.005, so the two error values use different scales.

| Dataset | Population | Penalty | Selected test error | Median test error | Median nodes | Mean diversity, initial → final |
|---|---:|---:|---:|---:|---:|---:|
| dataset1 | 250 | 0.01 | 1.66457e-06 | 1.66457e-06 | 7 | 88.3% → 17.3% |
| dataset1 | 250 | 0 | 1.66457e-06 | 1.66457e-06 | 44 | 88.3% → 62.0% |
| dataset1 | 50 | 0.01 | 1.66457e-06 | 1.66457e-06 | 9 | 92.6% → 39.6% |
| dataset2 | 250 | 0.01 | 0.0015837 | 0.0015837 | 7 | 89.8% → 22.9% |
| dataset2 | 250 | 0 | 0.00136849 | 0.00157769 | 57 | 89.8% → 60.9% |
| dataset2 | 50 | 0.01 | 0.0015837 | 0.129975 | 6 | 96.6% → 22.6% |

With population 250 and penalty 0.01, all ten runs on each dataset had test
error within 2% of the selected baseline or better. Population 50 managed
8/10 on dataset1 and 5/10 on dataset2. It used one fifth as many evaluations
(2,500 instead of 12,500 per run), but was less reliable.

Without the size penalty, all runs reached that accuracy level, but median
tree size rose from 7 to 44 nodes for dataset1 and from 7 to 57 for dataset2.
Dataset1 error stayed about the same. Dataset2's best error fell to 0.00136849,
but that model had 57 nodes.

## Convergence

Dataset1 reached within 2% of its returned model's training error by generation
0 to 4. For dataset2, nine runs reached that level by generation 3 or 4 and
one took until generation 41. Every run still continued through generation 49.
This measures error convergence; the final small tree could be found later.

Diversity counts distinct expression strings. Different strings can compute
the same function. Baseline diversity fell from 88.3% to 17.3% on dataset1
and from 89.8% to 22.9% on dataset2. Removing the penalty left more diversity
at the end, along with larger trees.

## Notes

Dataset2 includes 13 zero-distance rows, with three in the test set.
Protected division keeps them finite, but cannot recover rounded inputs.
Its large RMSE reflects absolute errors that relative scoring weights
differently. The error floor and scaling are modeling choices.

The test set was used to select the best run and compare settings, so it is
not an untouched final evaluation. These results use ten runs and one split.
Tree sizes exclude the fitted outer constants. Algebraic simplifications can
change protected division at zero.

The six JSON files contain the measurements, histories, fitted constants,
dataset hashes, and settings. Earlier local results in `results/legacy/` are
superseded. Rerun the commands in README to reproduce these experiments;
`--plot` redraws the saved training error, tree size, and diversity plots.
Versions can affect numerical details. Runs were timed concurrently, so the
timings are not a controlled speed comparison.

```sh
python -c 'from pathlib import Path; import run; print(run.summarize(Path("results").glob("*.json")))'
```
