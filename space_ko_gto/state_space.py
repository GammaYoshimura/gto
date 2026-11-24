"""Serializable grids of stack, token and payout states."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import hashlib
import json

STACK_BUCKETS_BB = (2, 3, 4, 6, 8, 10, 13, 17, 22, 30, 45)
TOKEN_PROFILES = {
    "EQUAL_LOW": (0.02, 0.02, 0.02),
    "EQUAL_HIGH": (0.20, 0.20, 0.20),
    "BTN_WHALE_TOKEN": (0.50, 0.05, 0.05),
    "SB_WHALE_TOKEN": (0.05, 0.50, 0.05),
    "BB_WHALE_TOKEN": (0.05, 0.05, 0.50),
    "HERO_COVERED_WHALE": (0.05, 0.50, 0.05),
}
PRIZE_STRUCTURES = {
    "WTA": (1.0, 0.0, 0.0),
    "ICM_TOP2": (0.7, 0.3, 0.0),
    "ICM_ALL3": (0.6, 0.3, 0.1),
}


@dataclass(frozen=True)
class State:
    stacks: tuple[int, int, int]
    token_profile: str
    prize_structure: str

    @property
    def state_id(self) -> str:
        payload = json.dumps({"stacks": self.stacks, "tokens": self.token_profile,
                              "prizes": self.prize_structure}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    @property
    def token_evs(self) -> tuple[float, float, float]:
        return TOKEN_PROFILES[self.token_profile]

    @property
    def payouts(self) -> tuple[float, float, float]:
        return PRIZE_STRUCTURES[self.prize_structure]


def iter_states(stacks: tuple[int, ...] = STACK_BUCKETS_BB):
    """Yield the full Cartesian state grid in stable order."""
    for stack_set in product(stacks, repeat=3):
        for token_name in TOKEN_PROFILES:
            for prize_name in PRIZE_STRUCTURES:
                yield State(tuple(stack_set), token_name, prize_name)
