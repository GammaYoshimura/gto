"""Pluggable terminal evaluator interface and phase-1 EQR approximation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

EQR = {
    "IP_AGGRESSOR": 1.12,
    "OOP_AGGRESSOR": 0.92,
    "IP_CALLER": 0.95,
    "OOP_CALLER": 0.76,
    "MULTIWAY_3WAY": 0.68,
}


class BaseLeafEvaluator(ABC):
    @abstractmethod
    def evaluate_postflop_leaf(self, active_players, reaches, pot_bb, contributions,
                               stacks_rem, payoff_matrix) -> np.ndarray:
        """Return EVs with shape (3, 1326), one value per player and combo."""


class PreflopEQRLeafEvaluator(BaseLeafEvaluator):
    """Approximate postflop EV from preflop equity and realization factors.

    This is a heuristic phase-1 leaf model; its values are not exact postflop
    solutions and must be calibrated before treating exported strategies as GTO.
    """

    def __init__(self, equity: np.ndarray, card_removal: np.ndarray, epsilon: float = 1e-12):
        self.equity = np.asarray(equity, dtype=np.float32)
        self.card_removal = np.asarray(card_removal, dtype=np.float32)
        if self.equity.shape != (1326, 1326) or self.card_removal.shape != (1326, 1326):
            raise ValueError("equity and card_removal must have shape (1326, 1326)")
        self.epsilon = epsilon

    def evaluate_postflop_leaf(self, active_players, reaches, pot_bb, contributions,
                               stacks_rem, payoff_matrix) -> np.ndarray:
        reach = np.asarray(reaches, dtype=np.float32)
        bounty = np.asarray(payoff_matrix, dtype=np.float32)
        if reach.shape != (3, 1326) or bounty.shape != (3, 3):
            raise ValueError("reaches must be (3,1326), payoff_matrix must be (3,3)")
        active = tuple(active_players)
        if len(active) not in (2, 3) or any(p not in range(3) for p in active):
            raise ValueError("two or three valid active players are required")
        out = np.zeros((3, 1326), dtype=np.float32)
        for p in active:
            opponents = [q for q in active if q != p]
            pair_values = []
            for q in opponents:
                weights = reach[q][None, :] * self.card_removal
                denom = weights.sum(axis=1) + self.epsilon
                pair_values.append((self.equity * weights).sum(axis=1) / denom)
            equity = pair_values[0] if len(pair_values) == 1 else 1.0 / (sum(1.0 / np.clip(x, 1e-5, 1.0) for x in pair_values))
            role = "MULTIWAY_3WAY" if len(active) == 3 else "IP_CALLER"
            eqr = EQR[role]
            ev = float(pot_bb) * np.clip(equity * eqr, 0.0, 1.0)
            spr = max((float(stacks_rem[q]) for q in opponents), default=0.0) / max(float(pot_bb), self.epsilon)
            stackoff = float(np.clip(1.0 / (1.0 + spr), 0.0, 1.0))
            for q in opponents:
                if stacks_rem[p] >= stacks_rem[q]:
                    ev += bounty[p, q] * stackoff / len(opponents)
            out[p] = ev.astype(np.float32)
        return out
