"""Finite public preflop betting tree for three-handed play.

Amounts are total contributions in big blinds. ``stacks`` are the players'
starting stacks, including chips posted as blinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .constants import (ALL_IN_THRESHOLD, FOUR_BET_MULT, OPEN_RAISE_SIZE,
                        SQUEEZE_MULT, THREE_BET_MULT)

_EPS = 1e-9
_MAX_RAISES = 4


@dataclass(frozen=True)
class Action:
    name: str
    target: float | None = None  # new total contribution for calls/raises

    def __str__(self) -> str:
        return self.name if self.target is None else f"{self.name}_{self.target:g}"


@dataclass(frozen=True)
class BettingState:
    stacks: tuple[float, float, float]
    contributions: tuple[float, float, float] = (0.0, 0.5, 1.0)
    active: tuple[int, ...] = (0, 1, 2)  # not folded; includes all-in players
    all_in: tuple[int, ...] = ()
    actor: int = 0
    pending: tuple[int, ...] = (0, 1, 2)
    current_bet: float = 1.0
    raises: int = 0
    last_raise_bb: float = 1.0
    history: tuple[str, ...] = ()

    @property
    def pot_bb(self) -> float:
        return sum(self.contributions)

    @property
    def stacks_rem(self) -> tuple[float, float, float]:
        return tuple(max(0.0, self.stacks[p] - self.contributions[p]) for p in range(3))

    @property
    def to_call(self) -> float:
        return max(0.0, self.current_bet - self.contributions[self.actor])


@dataclass(frozen=True)
class TreeNode:
    state: BettingState
    actions: tuple[Action, ...] = ()
    children: tuple["TreeNode", ...] = ()
    terminal: str | None = None


def initial_state(stacks: tuple[float, float, float]) -> BettingState:
    if len(stacks) != 3 or any(float(s) < 1.0 for s in stacks):
        raise ValueError("Three stacks of at least 1 bb are required")
    stacks = tuple(map(float, stacks))
    return BettingState(stacks=stacks)


def _active_order_after(actor: int, candidates: tuple[int, ...]) -> tuple[int, ...]:
    order = ((actor + 1) % 3, (actor + 2) % 3, actor)
    return tuple(p for p in order if p in candidates)


def _raise_target(state: BettingState, name: str) -> float:
    actor = state.actor
    active_others = [p for p in state.active if p != actor]
    last_aggressor = next((int(item.split(":")[0]) for item in reversed(state.history)
                           if item.split(":", 1)[1].startswith(("OPEN_RAISE", "ISO_RAISE", "3BET", "SQUEEZE", "4BET"))), None)
    effective = min([state.stacks_rem[actor]] + [state.stacks_rem[p] for p in active_others])
    if name == "OPEN_RAISE":
        target = OPEN_RAISE_SIZE["BTN" if actor == 0 else "SB"]
    elif name == "ISO_RAISE":
        # BB isolates a limp out of position; SB isolation over a BTN limp is smaller.
        target = 3.0 if actor == 1 else 3.5
    elif name == "3BET":
        opener_size = max(state.contributions[p] for p in active_others)
        # BTN (0) is in position over SB (1) and BB (2); SB is IP over BB.
        ip = last_aggressor is not None and actor < last_aggressor
        target = opener_size * THREE_BET_MULT["IP" if ip else "OOP"]
    elif name == "SQUEEZE":
        opener_size = max((state.contributions[p] for p in active_others), default=2.0)
        callers = sum(state.contributions[p] >= state.current_bet - _EPS for p in active_others) - 1
        target = opener_size * SQUEEZE_MULT + max(0, callers) * 1.0
    elif name == "4BET":
        target = state.current_bet * FOUR_BET_MULT
    elif name == "JAM":
        return state.stacks[actor]
    else:
        raise ValueError(f"Not a raise action: {name}")
    if effective > 0 and (target - state.contributions[actor]) / effective >= ALL_IN_THRESHOLD:
        return state.stacks[actor]
    return min(float(target), state.stacks[actor])


def legal_action_options(state: BettingState) -> tuple[Action, ...]:
    """List action labels and their total-contribution targets at this node."""
    actor = state.actor
    if actor not in state.active or actor in state.all_in or actor not in state.pending:
        raise ValueError("Current actor must be active, able to act, and pending")
    owed = state.to_call
    prior_limp = any(":LIMP:" in h or ":LIMP_BEHIND:" in h for h in state.history)
    unopened = state.raises == 0 and not prior_limp
    if owed <= _EPS:
        if unopened and actor == 0 and not state.history:
            names = ["FOLD", "LIMP", "OPEN_RAISE", "JAM"]
        elif state.raises == 0:
            names = ["CHECK", "ISO_RAISE", "JAM"] if prior_limp else ["CHECK"]
        elif state.raises < _MAX_RAISES:
            names = ["CHECK", "3BET", "JAM"]
        else:
            names = ["CHECK", "JAM"]
    else:
        if state.raises >= _MAX_RAISES:
            names = ["FOLD", "CALL", "JAM"]
        elif state.raises == 0 and prior_limp:
            names = ["FOLD", "LIMP_BEHIND", "ISO_RAISE", "JAM"]
        elif state.raises == 0 and actor == 1:
            names = ["FOLD", "LIMP_BEHIND", "OPEN_RAISE", "JAM"]
        elif state.raises == 1 and len(state.active) == 3 and any("CALL" in h for h in state.history):
            names = ["FOLD", "CALL", "SQUEEZE", "JAM"]
        elif state.raises == 1:
            names = ["FOLD", "CALL", "3BET", "JAM"]
        elif state.raises == 2:
            names = ["FOLD", "CALL", "4BET", "JAM"]
        else:
            names = ["FOLD", "CALL", "JAM"]
    result = []
    for name in names:
        if name in ("LIMP", "LIMP_BEHIND", "CALL"):
            result.append(Action(name, min(state.current_bet, state.stacks[actor])))
        elif name in ("OPEN_RAISE", "ISO_RAISE", "3BET", "SQUEEZE", "4BET", "JAM"):
            target = _raise_target(state, name)
            if target > state.current_bet + _EPS or name == "JAM":
                normalized = "JAM" if name == "JAM" or target >= state.stacks[actor] - _EPS else name
                result.append(Action(normalized, target))
        else:
            result.append(Action(name))
    # Deduplicate equivalent raises caused by a short effective stack.
    deduped = []
    seen = set()
    for action in result:
        key = (action.name, round(action.target or 0.0, 8))
        if key not in seen:
            deduped.append(action)
            seen.add(key)
    return tuple(deduped)


def legal_actions(state: BettingState) -> tuple[str, ...]:
    """Compatibility helper returning only action names."""
    return tuple(action.name for action in legal_action_options(state))


def apply_action(state: BettingState, action: Action) -> BettingState:
    if action not in legal_action_options(state):
        raise ValueError(f"Illegal action {action} at {state.history}")
    actor = state.actor
    stacks = state.stacks
    contrib = list(state.contributions)
    active = list(state.active)
    all_in = list(state.all_in)
    pending = list(state.pending)
    raises, last_raise, current_bet = state.raises, state.last_raise_bb, state.current_bet
    pending.remove(actor)
    if action.name == "FOLD":
        active.remove(actor)
        if actor in pending:
            pending.remove(actor)
    elif action.name in ("CALL", "LIMP", "LIMP_BEHIND"):
        contrib[actor] = float(action.target)
        if contrib[actor] >= stacks[actor] - _EPS:
            all_in.append(actor)
            pending = [p for p in pending if p not in all_in]
    elif action.name == "CHECK":
        pass
    else:  # raise or jam
        target = min(float(action.target), stacks[actor])
        last_raise = max(_EPS, target - current_bet)
        current_bet = max(current_bet, target)
        contrib[actor] = target
        raises += 1
        if target >= stacks[actor] - _EPS:
            all_in.append(actor)
        pending = _active_order_after(actor, tuple(p for p in active if p != actor and p not in all_in))
    history = state.history + (f"{actor}:{action.name}:{action.target or 0:g}",)
    pending = [p for p in pending if p in active and p not in all_in]
    next_actor = pending[0] if pending else actor
    return BettingState(stacks, tuple(contrib), tuple(active), tuple(sorted(set(all_in))),
                        next_actor, tuple(pending), current_bet, raises, last_raise, history)


def terminal_kind(state: BettingState) -> str | None:
    if len(state.active) == 1:
        return "FOLD_WIN"
    if not state.pending:
        return "ALL_IN_SHOWDOWN" if len(state.all_in) >= len(state.active) - 1 else "PREFLOP_CALL"
    return None


def build_tree(stacks: tuple[float, float, float], max_nodes: int = 100_000) -> TreeNode:
    """Construct the complete bounded public tree for the supplied stacks."""
    count = 0

    def visit(state: BettingState) -> TreeNode:
        nonlocal count
        count += 1
        if count > max_nodes:
            raise RuntimeError(f"Tree exceeded max_nodes={max_nodes}; reduce raise depth or cap")
        terminal = terminal_kind(state)
        if terminal:
            return TreeNode(state=state, terminal=terminal)
        actions = legal_action_options(state)
        children = tuple(visit(apply_action(state, action)) for action in actions)
        return TreeNode(state=state, actions=actions, children=children)

    return visit(initial_state(stacks))


def walk_tree(root: TreeNode) -> Iterator[TreeNode]:
    yield root
    for child in root.children:
        yield from walk_tree(child)


def jam_if_committed(raise_to_bb: float, effective_stack_bb: float) -> bool:
    return effective_stack_bb <= 0 or raise_to_bb / effective_stack_bb >= ALL_IN_THRESHOLD
