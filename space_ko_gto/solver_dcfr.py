"""DCFR regret-matching primitives for a future complete tree traversal.

The class implements per-information-set discounted regret matching. It is not
yet a complete poker-game solver; callers must supply counterfactual action
utilities from a traversal that respects card removal and reaches.
"""

from __future__ import annotations

import numpy as np


class DCFR:
    def __init__(self, alpha: float = 1.5, beta: float = 0.0, gamma: float = 2.0):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.iteration = 0
        self.regrets: dict[str, np.ndarray] = {}
        self.strategy_sum: dict[str, np.ndarray] = {}

    @staticmethod
    def _strategy(regrets: np.ndarray) -> np.ndarray:
        positive = np.maximum(regrets, 0.0)
        total = positive.sum(axis=-1, keepdims=True)
        uniform = np.full_like(positive, 1.0 / positive.shape[-1])
        return np.divide(positive, total, out=uniform, where=total > 0)

    def strategy(self, key: str, shape: tuple[int, int]) -> np.ndarray:
        regrets = self.regrets.setdefault(key, np.zeros(shape, dtype=np.float64))
        return self._strategy(regrets)

    def update(self, key: str, action_utilities: np.ndarray, reach: np.ndarray | None = None) -> np.ndarray:
        """Update one information set; utilities have shape (hands, actions)."""
        utilities = np.asarray(action_utilities, dtype=np.float64)
        if utilities.ndim != 2 or utilities.shape[1] < 2:
            raise ValueError("action_utilities must have shape (hands, >=2 actions)")
        current = self.regrets.setdefault(key, np.zeros_like(utilities))
        if current.shape != utilities.shape:
            raise ValueError("information-set action shape changed")
        self.iteration += 1
        t = self.iteration
        positive_discount = t**self.alpha / (t**self.alpha + 1.0)
        negative_discount = t**self.beta / (t**self.beta + 1.0)
        current *= np.where(current > 0, positive_discount, negative_discount)
        strategy = self._strategy(current)
        node_value = (strategy * utilities).sum(axis=1, keepdims=True)
        current += utilities - node_value
        weight = t**self.gamma
        if reach is not None:
            weight = weight * np.asarray(reach, dtype=np.float64)[:, None]
        self.strategy_sum.setdefault(key, np.zeros_like(utilities))[:] += strategy * weight
        return strategy

    def average_strategy(self, key: str) -> np.ndarray:
        sums = self.strategy_sum[key]
        total = sums.sum(axis=1, keepdims=True)
        return np.divide(sums, total, out=np.full_like(sums, 1.0 / sums.shape[1]), where=total > 0)
