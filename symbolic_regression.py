from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

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