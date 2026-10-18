from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import random

@dataclass(frozen=True)
class EquationTree:
    value: Any
    left: Optional['EquationTree'] = None
    right: Optional['EquationTree'] = None

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

        # evaluate left and right children, then apply thhe operator to the final results
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

    def __str__(self) -> str:
        if self.is_leaf:
            return str(self.value)
        return f"({self.left} {self.value} {self.right})"
    
    def __repr__(self) -> str:
        return self.__str__()
    

def make_random_leaf(num_vars: int, real_constants: bool) -> EquationTree:
    """Returns either a variable or a numerical constnt with equal probability."""
    if random.random() < 0.5:
        return EquationTree(f"x{random.randint(1, num_vars)}")
    val = random.randint(-5, 5) if not real_constants else round(random.uniform(-5, 5), 2)
    return EquationTree(val)


def make_random_tree(max_depth: int, num_vars: int, real_constants: bool, is_root: bool = False) -> EquationTree:
    """Recursively builds an expression tree."""
    if max_depth == 0 or (not is_root and random.random() < 0.3):
        return make_random_leaf(num_vars, real_constants)

    operator = random.choice(['+', '-' , '*', '/'])
    left = make_random_tree(max_depth - 1, num_vars, real_constants, is_root=False)
    right = make_random_tree(max_depth - 1, num_vars, real_constants, is_root=False)

    return EquationTree(operator, left, right)


def initialize_population(
    population_size: int,
    num_vars: int,
    min_depth: int = 1,
    max_depth: int = 5,
    real_constants : bool = False
    ) -> list[EquationTree]:
    """Generates a collection of randomly generated unique trees of varying heights & contents."""
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

def get_fitness(equation: EquationTree, x: np.ndarray, y: np.ndarray) -> float:
    """Returns normalized mean squared error between predicted and true values."""
    with np.errstate(all='ignore'):
        y_pred = equation.evaluate(x)

    # Normalized Mean Squared Error
    y_variance = np.var(y)
    mse = np.mean((y_pred - y) ** 2)
    normalized_error = mse / y_variance if y_variance > 1e-9 else mse

    return normalized_error