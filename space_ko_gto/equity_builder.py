"""Build a preflop hand-vs-hand equity matrix with eval7 Monte Carlo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .constants import CARD_REMOVAL_MASK, COMBO_CARDS, N_COMBOS, RANKS, SUITS


def build_equity_matrix(trials: int = 10_000, seed: int = 7) -> np.ndarray:
    """Estimate all pair equities from shared random boards using eval7.

    A board is reused for every pair of hole-card combos it can coexist with.
    Thus each valid pair receives about 66% of ``trials`` observations on
    average, while avoiding billions of repeated hand evaluations.
    """
    try:
        import eval7
    except ImportError as exc:
        raise RuntimeError("Install optional dependency with `pip install '.[equity]'`") from exc
    if trials < 1:
        raise ValueError("trials must be positive")
    deck = [eval7.Card(rank + suit) for rank in RANKS for suit in SUITS]
    rng = np.random.default_rng(seed)
    equity_sum = np.zeros((N_COMBOS, N_COMBOS), dtype=np.float32)
    observations = np.zeros((N_COMBOS, N_COMBOS), dtype=np.float32)
    for _ in range(trials):
        board_ids = rng.choice(52, size=5, replace=False)
        board_cards = [deck[int(card_id)] for card_id in board_ids]
        valid_ids = np.flatnonzero(~np.isin(COMBO_CARDS, board_ids).any(axis=1))
        ranks = np.fromiter(
            (eval7.evaluate([deck[int(COMBO_CARDS[i, 0])],
                             deck[int(COMBO_CARDS[i, 1])], *board_cards])
             for i in valid_ids),
            dtype=np.int32,
            count=len(valid_ids),
        )
        win = ranks[:, None] > ranks[None, :]
        tie = ranks[:, None] == ranks[None, :]
        outcomes = win.astype(np.float32) + 0.5 * tie
        pair_ids = np.ix_(valid_ids, valid_ids)
        compatible = CARD_REMOVAL_MASK[pair_ids]
        equity_sum[pair_ids] += outcomes * compatible
        observations[pair_ids] += compatible
    equity = np.full_like(equity_sum, 0.5)
    np.divide(equity_sum, observations, out=equity, where=observations > 0)
    equity *= CARD_REMOVAL_MASK
    return equity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("equity_1326x1326.npy"))
    parser.add_argument("--trials", type=int, default=10_000,
                        help="shared random boards; each valid hand pair sees about 66% of them")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    matrix = build_equity_matrix(args.trials, args.seed)
    np.save(args.output, matrix)
    print(f"Saved {matrix.shape} float32 equity matrix to {args.output}")


if __name__ == "__main__":
    main()
