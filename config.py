"""Parameters shared by the two required datasets."""

from dataclasses import dataclass


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

    def constant(self, rng):
        return rng.randint(-5, 5)


PRESETS = {
    "dataset1.csv": Config(),
    "dataset2.csv": Config(
        dataset="dataset2.csv",
        variables=("x1", "x2", "x3"),
        metric="relative",
    ),
}
