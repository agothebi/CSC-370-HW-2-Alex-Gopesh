"""Genetic programming for CSC 370 HW2. Alex Shields and Gopesh Baheti."""

from dataclasses import dataclass
import random
from typing import Any, Optional

import numpy as np

EARLY_STOP_THRESHOLD = 1e-4


@dataclass(frozen=True)
class ExpressionTree:
    value: Any
    left: Optional['ExpressionTree'] = None
    right: Optional['ExpressionTree'] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        # if it is a variable, return the associated column from the dataset
        if isinstance(self.value, str) and self.value.startswith('x'):
            col = int(self.value[1:]) - 1
            return X[:, col]

        # for constant, return a np array of constant of same size as dataset
        if self.is_leaf:
            return np.full(len(X), float(self.value))

        # evaluate left and right children, then apply the operator to the final results
        left_val = self.left.evaluate(X)
        right_val = self.right.evaluate(X)

        if self.value == '+':
            return left_val + right_val
        if self.value == '-':
            return left_val - right_val
        if self.value == '*':
            return left_val * right_val
        if self.value == '/':
            safe_divisor = np.where(right_val == 0, 1.0, right_val)
            return np.where(right_val == 0, 1.0, left_val / safe_divisor)
        else:
            raise ValueError(f'The operator {self.value} is not valid!')

    def size(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def get_subtree(self, index: int) -> 'ExpressionTree':
        """Returns the subtree at the given index."""
        if index == 0:
            return self

        left_tree_size = self.left.size()
        # check which side of tree the index falls in
        if index <= left_tree_size:
            return self.left.get_subtree(index - 1)
        return self.right.get_subtree(index - 1 - left_tree_size)

    def replace_subtree(self, index: int, replacement: 'ExpressionTree') -> 'ExpressionTree':
        """ Finds the subtree at given index and replaces it with the given subtree."""
        if index == 0:
            return replacement

        left_size = self.left.size()
        if index <= left_size:
            new_left = self.left.replace_subtree(index - 1, replacement)
            return ExpressionTree(self.value, new_left, self.right)
        else:
            new_right = self.right.replace_subtree(index - 1 - left_size, replacement)
            return ExpressionTree(self.value, self.left, new_right)

    def __str__(self) -> str:
        if self.is_leaf:
            return str(self.value)
        return f"({self.left} {self.value} {self.right})"

    def __repr__(self) -> str:
        return self.__str__()


def make_random_leaf(num_vars: int, real_constants: bool) -> ExpressionTree:
    """Returns either a variable or a numerical constant with equal probability."""
    if random.random() < 0.5:
        return ExpressionTree(f"x{random.randint(1, num_vars)}")
    val = random.randint(-5, 5) if not real_constants else round(random.uniform(-5, 5), 2)
    return ExpressionTree(val)


def make_random_tree(max_depth: int, num_vars: int, real_constants: bool, is_root: bool = False) -> ExpressionTree:
    """Recursively builds an expression tree."""
    if max_depth == 0 or (not is_root and random.random() < 0.3):
        return make_random_leaf(num_vars, real_constants)

    operator = random.choice(['+', '-' , '*', '/'])
    left = make_random_tree(max_depth - 1, num_vars, real_constants, is_root=False)
    right = make_random_tree(max_depth - 1, num_vars, real_constants, is_root=False)

    return ExpressionTree(operator, left, right)


def initialize_population(
    population_size: int,
    num_vars: int,
    min_depth: int = 1,
    max_depth: int = 5,
    real_constants : bool = False
    ) -> list[ExpressionTree]:
    """Generates a collection of randomly generated unique trees of varying heights and contents."""
    population = []
    seen = set()

    while len(population) < population_size:
        depth_budget = random.randint(min_depth, max_depth)
        tree = make_random_tree(depth_budget, num_vars, real_constants, is_root=True)

        if tree in seen:
            continue

        seen.add(tree)
        population.append(tree)

    return population


def linear_scale(y_pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Returns the least-squares a, b for y = a + b * y_pred (Keijzer 2003)."""
    var = np.var(y_pred)
    if not np.isfinite(var) or var == 0:
        return float(np.mean(y)), 0.0
    b = np.mean((y_pred - y_pred.mean()) * (y - y.mean())) / var
    return float(y.mean() - b * y_pred.mean()), float(b)


def get_fitness(expression: ExpressionTree, x: np.ndarray, y: np.ndarray, scale: bool = False) -> float:
    """Returns normalized mean squared error between predicted and true values.
    With scale, the predictions are first linearly scaled to fit y."""
    y_pred = expression.evaluate(x)

    if not np.isfinite(y_pred).all():
        return float('inf')

    if scale:
        a, b = linear_scale(y_pred, y)
        y_pred = a + b * y_pred

    mse = np.mean((y_pred - y) ** 2)
    normalized_error = mse / (np.var(y) + 1e-9)

    return normalized_error


def mutate(expression: ExpressionTree, num_vars: int, real_constants: bool, max_depth: int = 8) -> ExpressionTree:
    # Pick a random index in the expression tree
    index = random.randrange(expression.size())
    # generate a small random replacement subtree
    subtree_depth = random.randint(1, 2)
    replacement = make_random_tree(subtree_depth, num_vars, real_constants)
    new_expression = expression.replace_subtree(index, replacement)

    return expression if new_expression.depth() > max_depth else new_expression


def crossover(parent1: ExpressionTree, parent2: ExpressionTree, max_depth: int = 8) -> ExpressionTree:
    index1 = random.randrange(parent1.size())
    index2 = random.randrange(parent2.size())

    donor = parent2.get_subtree(index2)
    child = parent1.replace_subtree(index1, donor)

    return parent1 if child.depth() > max_depth else child


def select_parent(population: list[ExpressionTree], scores: list[float], k: int = 4) -> ExpressionTree:
    """Returns the tree with lowest error among k random trees to add diversity to next generation"""
    chosen_species = random.sample(range(len(population)), k)
    best_species = min(chosen_species, key=lambda i: scores[i])
    return population[best_species]


def run_gp(x_train: np.ndarray, y_train: np.ndarray, num_vars: int, real_constants: bool, population_size=1000, generations=40, scale=False):
    population = initialize_population(population_size, num_vars)
    history = []

    for gen in range(generations):
        scores = [get_fitness(species, x_train, y_train, scale) for species in population]

        # Sort the population by fitness
        ranked = [tree for _, tree in sorted(zip(scores, population), key=lambda x: x[0])]

        best_tree = ranked[0]
        best_score = min(scores)

        history.append(float(best_score))

        # Early stop if we found an essentially perfect formula
        if best_score < EARLY_STOP_THRESHOLD:
            break

        # generate new population
        next_population = [ranked[0], ranked[1]] # take the top 2 of this population
        # fill the remaining population with crossover (0.8) and mutation (0.2)
        while len(next_population) < population_size:
            p1 = select_parent(population, scores)
            if random.random() < 0.8:
                p2 = select_parent(population, scores)
                child = crossover(p1, p2)
            else:
                child = mutate(p1, num_vars, real_constants)

            next_population.append(child)

        population = next_population

    return best_tree, history
