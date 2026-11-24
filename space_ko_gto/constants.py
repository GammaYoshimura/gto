"""Canonical cards, combos and preflop sizing constants."""

from itertools import combinations

import numpy as np

POSITIONS = ("BTN", "SB", "BB")
BLINDS = {"BTN": 0.0, "SB": 0.5, "BB": 1.0}
RANKS = "23456789TJQKA"
SUITS = "cdhs"
N_CARDS = 52
N_COMBOS = 1326

# Strict lexicographic order: (0, 1), (0, 2), ..., (50, 51).
COMBO_CARDS = np.asarray(list(combinations(range(N_CARDS), 2)), dtype=np.int16)
COMBO_MASKS = (np.left_shift(np.uint64(1), COMBO_CARDS[:, 0].astype(np.uint64)) |
               np.left_shift(np.uint64(1), COMBO_CARDS[:, 1].astype(np.uint64)))
CARD_REMOVAL_MASK = ((COMBO_MASKS[:, None] & COMBO_MASKS[None, :]) == 0).astype(np.float32)

LIMP_SIZE = 1.0
OPEN_RAISE_SIZE = {"BTN": 2.0, "SB": 2.5}
THREE_BET_MULT = {"IP": 2.75, "OOP": 3.25}
SQUEEZE_MULT = 3.5
FOUR_BET_MULT = 2.25
ALL_IN_THRESHOLD = 0.40


def combo_index(card_a: int, card_b: int) -> int:
    """Return the canonical combo index for two distinct card IDs."""
    if card_a == card_b or not (0 <= card_a < N_CARDS and 0 <= card_b < N_CARDS):
        raise ValueError("Two distinct card IDs in [0, 51] are required")
    a, b = sorted((card_a, card_b))
    # Number of pairs in all preceding rows, plus offset in this row.
    return a * (2 * N_CARDS - a - 1) // 2 + (b - a - 1)


def combo_label(index: int) -> str:
    """Readable e.g. 'AsKd' label for a canonical combo index."""
    if not 0 <= index < N_COMBOS:
        raise IndexError(index)
    return "".join(RANKS[int(card) // 4] + SUITS[int(card) % 4] for card in COMBO_CARDS[index])
