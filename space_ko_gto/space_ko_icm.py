"""Three-player ICM and Space KO elimination values."""

from __future__ import annotations

import numpy as np
from itertools import permutations


def icm_equities(stacks: list[float] | tuple[float, ...], payouts: list[float] | tuple[float, ...]) -> np.ndarray:
    """Return exact three-player ICM equity, allowing zero stacks."""
    s = np.asarray(stacks, dtype=np.float64)
    p = np.asarray(payouts, dtype=np.float64)
    if (s.shape != (3,) or p.shape != (3,) or not np.isfinite(s).all() or
            not np.isfinite(p).all() or np.any(s < 0) or np.any(p < 0)):
        raise ValueError("stacks and payouts must each contain three non-negative values")
    if s.sum() <= 0:
        raise ValueError("At least one stack must be positive")
    result = np.zeros(3, dtype=np.float64)
    alive = tuple(int(player) for player in np.flatnonzero(s > 0))
    eliminated = tuple(int(player) for player in np.flatnonzero(s == 0))
    for order in permutations(alive):
        probability = 1.0
        remaining = float(s.sum())
        for player in order:
            probability *= s[player] / remaining
            remaining -= s[player]
        for place, player in enumerate(order):
            result[player] += probability * p[place]
    # Zero-stack players have already been eliminated; assign the remaining
    # fixed finishing places in stable player-index order.
    for offset, player in enumerate(eliminated):
        result[player] = p[len(alive) + offset]
    return result


def elimination_payoffs(
    stacks: list[float] | tuple[float, ...],
    token_evs: list[float] | tuple[float, ...],
    prizepool_payouts: list[float] | tuple[float, ...],
) -> np.ndarray:
    """Build token-only KO values B[i,j] for eliminating covered player j.

    Tournament chip value is handled separately: chipEV for winner-takes-all,
    and terminal-stack ICM for multi-place payouts. Including an ICM stack
    transfer here would count the eliminated chips a second time.
    """
    s = np.asarray(stacks, dtype=np.float64)
    t = np.asarray(token_evs, dtype=np.float64)
    if (s.shape != (3,) or t.shape != (3,) or not np.isfinite(s).all() or
            not np.isfinite(t).all() or np.any(s < 0) or np.any(t < 0)):
        raise ValueError("stacks and token_evs must each contain three non-negative values")
    payouts = np.asarray(prizepool_payouts, dtype=np.float64)
    if payouts.shape != (3,) or not np.isfinite(payouts).all() or np.any(payouts < 0):
        raise ValueError("prizepool_payouts must contain three finite non-negative values")
    out = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            if i == j or s[i] < s[j]:
                continue
            out[i, j] = 0.5 * t[j]
    return out
