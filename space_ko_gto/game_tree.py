"""Preflop betting-state representation and legal-action geometry.

This module deliberately models public betting state only. A complete
imperfect-information tree traversal is still required before strategy export.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import ALL_IN_THRESHOLD, BLINDS


@dataclass
class BettingState:
    stacks: tuple[float, float, float]
    contributions: tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.5, 1.0))
    active: tuple[int, ...] = (0, 1, 2)
    actor: int = 0
    history: tuple[str, ...] = ()
    raises: int = 0
    last_raise_bb: float = 1.0

    @property
    def pot_bb(self) -> float:
        return sum(self.contributions)


def initial_state(stacks: tuple[float, float, float]) -> BettingState:
    if len(stacks) != 3 or any(s <= 0 for s in stacks):
        raise ValueError("Three positive starting stacks are required")
    return BettingState(stacks=tuple(map(float, stacks)))


def legal_actions(state: BettingState) -> tuple[str, ...]:
    """Return the supported action labels for the current public state."""
    actor = state.actor
    if actor not in state.active:
        raise ValueError("Current actor must be active")
    to_call = max(state.contributions[p] for p in state.active) - state.contributions[actor]
    if to_call <= 1e-9:
        if state.raises == 0 and not state.history:
            return ("FOLD", "LIMP", "OPEN_RAISE", "JAM")
        return ("CHECK", "LIMP_BEHIND", "ISO_RAISE", "JAM")
    if state.raises >= 3:
        return ("FOLD", "CALL", "JAM")
    if state.raises == 2:
        return ("FOLD", "CALL", "4BET", "JAM")
    if len(state.active) == 3 and any("CALL" in a for a in state.history):
        return ("FOLD", "CALL", "SQUEEZE", "JAM")
    if state.raises == 1:
        return ("FOLD", "CALL", "3BET", "JAM")
    return ("FOLD", "CALL", "JAM")


def jam_if_committed(raise_to_bb: float, effective_stack_bb: float) -> bool:
    """Whether a raise would commit at least the configured stack fraction."""
    if effective_stack_bb <= 0:
        return True
    return raise_to_bb / effective_stack_bb >= ALL_IN_THRESHOLD
