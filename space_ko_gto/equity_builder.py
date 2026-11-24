"""Build a preflop hand-vs-hand equity matrix with eval7 Monte Carlo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .constants import CARD_REMOVAL_MASK, COMBO_CARDS, N_COMBOS, RANKS, SUITS


def build_equity_matrix(trials: int = 10_000, seed: int = 7) -> np.ndarray:
    """Estimate equity for each legal pair; requires optional dependency eval7."""
    try:
        import eval7
    except ImportError as exc:
        raise RuntimeError("Install optional dependency with `pip install '.[equity]'`") from exc
    if trials < 1:
        raise ValueError("trials must be positive")
    deck = [eval7.Card(rank + suit) for rank in RANKS for suit in SUITS]
    card_ids = {str(card): i for i, card in enumerate(deck)}
    rng = np.random.default_rng(seed)
    equity = np.zeros((N_COMBOS, N_COMBOS), dtype=np.float32)
    for i, pair_i in enumerate(COMBO_CARDS):
        hand_i = [deck[int(pair_i[0])], deck[int(pair_i[1])]]
        for j in range(i + 1, N_COMBOS):
            if not CARD_REMOVAL_MASK[i, j]:
                continue
            pair_j = COMBO_CARDS[j]
            hand_j = [deck[int(pair_j[0])], deck[int(pair_j[1])]]
            dead = set(map(int, pair_i)) | set(map(int, pair_j))
            available = [card for cid, card in enumerate(deck) if cid not in dead]
            scores = np.zeros(trials, dtype=np.float64)
            for n in range(trials):
                board = rng.choice(len(available), size=5, replace=False)
                cards = [available[int(k)] for k in board]
                a = eval7.evaluate(hand_i + cards)
                b = eval7.evaluate(hand_j + cards)
                scores[n] = 1.0 if a > b else 0.5 if a == b else 0.0
            value = float(scores.mean())
            equity[i, j], equity[j, i] = value, 1.0 - value
    return equity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("equity_1326x1326.npy"))
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    matrix = build_equity_matrix(args.trials, args.seed)
    np.save(args.output, matrix)
    print(f"Saved {matrix.shape} float32 equity matrix to {args.output}")


if __name__ == "__main__":
    main()
