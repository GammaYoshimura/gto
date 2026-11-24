"""Three-player ICM and Space KO elimination values."""

from __future__ import annotations

import numpy as np
from itertools import permutations


def icm_equities(stacks: list[float] | tuple[float, ...], payouts: list[float] | tuple[float, ...]) -> np.ndarray:
    """Return exact three-player ICM equity, allowing zero stacks."""
    s = np.asarray(stacks, dtype=np.float64)
    p = np.asarray(payouts, dtype=np.float64)
    if s.shape != (3,) or p.shape != (3,) or np.any(s < 0) or np.any(p < 0):
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
    """Build B[i,j], immediate KO value to i for eliminating covered j.

    B includes half of j's token EV as immediate reward and the ICM equity
    change from transferring j's stack to i. The other token half is treated
    as future value and represented by the supplied token EV; callers can
    replace this policy if their tournament token model differs.
    """
    s = np.asarray(stacks, dtype=np.float64)
    t = np.asarray(token_evs, dtype=np.float64)
    if s.shape != (3,) or t.shape != (3,) or np.any(s < 0) or np.any(t < 0):
        raise ValueError("stacks and token_evs must each contain three non-negative values")
    base = icm_equities(s, prizepool_payouts)
    out = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            if i == j or s[i] < s[j]:
                continue
            after = s.copy()
            after[i] += after[j]
            after[j] = 0.0
            delta_icm = icm_equities(after, prizepool_payouts)[i] - base[i]
            out[i, j] = max(0.0, 0.5 * t[j] + delta_icm)
    return out
