"""Vector DCFR over the full-combo public tree plus legacy MCCFR helpers."""

from __future__ import annotations

import numpy as np
import json
import zlib
import hashlib
from itertools import permutations

from .constants import CARD_REMOVAL_MASK, COMBO_CARDS
from .game_tree import TreeNode, build_tree
from .leaf_evaluator import (BaseLeafEvaluator, EQR, PreflopEQRLeafEvaluator,
                             TorchPreflopEQRLeafEvaluator)
from .space_ko_icm import icm_equities


def _torch_icm_equity(stacks, payouts):
    """Vectorized three-player ICM for a batch shaped ``(N,3)``."""
    torch = __import__("torch")
    total = stacks.sum(dim=1, keepdim=True)
    first = stacks / torch.clamp(total, min=1e-12)
    second = torch.zeros_like(stacks)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            remaining_after_j = total[:, 0] - stacks[:, j]
            probability_j_first = stacks[:, j] / torch.clamp(total[:, 0], min=1e-12)
            probability_i_second = stacks[:, i] / torch.clamp(remaining_after_j, min=1e-12)
            valid = remaining_after_j > 1e-12
            second[:, i] += torch.where(valid, probability_j_first * probability_i_second,
                                        torch.zeros_like(probability_i_second))
    third = torch.clamp(1.0 - first - second, min=0.0)
    return first * payouts[0] + second * payouts[1] + third * payouts[2]


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
        weight = t**self.gamma / (t**self.gamma + 1.0)
        if reach is not None:
            weight = weight * np.asarray(reach, dtype=np.float64)[:, None]
        self.strategy_sum.setdefault(key, np.zeros_like(utilities))[:] += strategy * weight
        return strategy

    def average_strategy(self, key: str) -> np.ndarray:
        sums = self.strategy_sum[key]
        total = sums.sum(axis=1, keepdims=True)
        return np.divide(sums, total, out=np.full_like(sums, 1.0 / sums.shape[1]), where=total > 0)


class _LegacySampledMCCFRSolver:
    """Retained only to deserialize old application code; use VectorDCFRSolver.

    The public betting tree is complete for its configured sizing abstraction.
    Each iteration samples one legal three-hand deal and traverses the tree once
    per player. Pairwise preflop equity supplies terminal values; three-way
    equity and non-all-in EQR leaves remain approximations.
    """

    def __init__(self, stacks, equity: np.ndarray, payoff_matrix: np.ndarray,
                 seed: int = 7, max_nodes: int = 100_000, payoff_scale: float = 1.0,
                 leaf_evaluator: BaseLeafEvaluator | None = None,
                 prizepool_payouts=(1.0, 0.0, 0.0)):
        self.stacks = tuple(map(float, stacks))
        self.prizepool_payouts = np.asarray(prizepool_payouts, dtype=np.float64)
        if self.prizepool_payouts.shape != (3,) or not np.isfinite(self.prizepool_payouts).all() or np.any(self.prizepool_payouts < 0):
            raise ValueError("prizepool_payouts must have three finite non-negative payouts")
        self.chip_ev_mode = np.array_equal(self.prizepool_payouts, np.asarray((1.0, 0.0, 0.0)))
        self.initial_icm = icm_equities(self.stacks, self.prizepool_payouts)
        self.equity = np.asarray(equity, dtype=np.float32)
        if not np.isfinite(payoff_scale) or payoff_scale < 0:
            raise ValueError("payoff_scale must be finite and non-negative")
        self.payoffs = np.asarray(payoff_matrix, dtype=np.float32) * float(payoff_scale)
        if self.equity.shape != (1326, 1326) or self.payoffs.shape != (3, 3):
            raise ValueError("equity must be (1326,1326) and payoff_matrix (3,3)")
        if not np.isfinite(self.equity).all() or not np.isfinite(self.payoffs).all():
            raise ValueError("equity and payoff_matrix must be finite")
        if np.any(self.equity < 0.0) or np.any(self.equity > 1.0):
            raise ValueError("equity values must lie in [0,1]")
        if not np.allclose(self.equity + self.equity.T, CARD_REMOVAL_MASK, atol=1e-3):
            raise ValueError("equity matrix must be complementary and zero for card collisions")
        self.root: TreeNode = build_tree(self.stacks, max_nodes=max_nodes)
        self.leaf_evaluator = leaf_evaluator or PreflopEQRLeafEvaluator(self.equity, CARD_REMOVAL_MASK)
        if not callable(getattr(self.leaf_evaluator, "checkpoint_fingerprint", None)):
            raise TypeError("leaf_evaluator must implement checkpoint_fingerprint()")
        self._checkpoint_config = {
            "solver": "mccfr-v6",
            "stacks": list(self.stacks),
            "prizepool_payouts": self.prizepool_payouts.tolist(),
            "equity_sha256": hashlib.sha256(self.equity.tobytes()).hexdigest(),
            "payoff_sha256": hashlib.sha256(self.payoffs.tobytes()).hexdigest(),
            "tree_sha256": self._tree_fingerprint(),
            "eqr_factors": EQR,
            "leaf_evaluator": self.leaf_evaluator.checkpoint_fingerprint(),
        }
        self.rng = np.random.default_rng(seed)
        # Sparse per-information-set rows avoid allocating every combo at every node.
        self.regrets: dict[str, dict[int, np.ndarray]] = {}
        self.regret_last_t: dict[str, dict[int, int]] = {}
        self.strategy_sums: dict[str, dict[int, np.ndarray]] = {}
        self.ev_sums: dict[str, dict[int, np.ndarray]] = {}
        self.row_visits: dict[str, dict[int, float]] = {}
        self.iteration = 0
        self._positive_log_discount = [0.0]

    def _strategy(self, key: str, combo: int, n_actions: int, iteration: int) -> np.ndarray:
        row = self.regrets.setdefault(key, {}).setdefault(combo, np.zeros(n_actions, dtype=np.float64))
        last_t = self.regret_last_t.setdefault(key, {}).get(combo, 0)
        while len(self._positive_log_discount) <= iteration:
            t = len(self._positive_log_discount)
            self._positive_log_discount.append(
                self._positive_log_discount[-1] + np.log(t**1.5 / (t**1.5 + 1.0))
            )
        positive_factor = np.exp(self._positive_log_discount[iteration] -
                                 self._positive_log_discount[last_t])
        negative_factor = 0.5 ** (iteration - last_t)
        row *= np.where(row > 0.0, positive_factor, negative_factor)
        self.regret_last_t[key][combo] = iteration
        positive = np.maximum(row, 0.0)
        total = positive.sum()
        return positive / total if total > 0 else np.full(n_actions, 1.0 / n_actions)

    def _deal(self) -> tuple[int, int, int]:
        chosen = []
        used = set()
        for _ in range(3):
            available = np.flatnonzero(~np.isin(COMBO_CARDS[:, 0], tuple(used)) &
                                       ~np.isin(COMBO_CARDS[:, 1], tuple(used)))
            combo = int(self.rng.choice(available))
            chosen.append(combo)
            used.update(map(int, COMBO_CARDS[combo]))
        return tuple(chosen)

    def _tree_fingerprint(self) -> str:
        digest = hashlib.sha256()

        def visit(node: TreeNode, path: str) -> None:
            state = node.state
            record = {
                "path": path,
                "actor": state.actor,
                "active": state.active,
                "all_in": state.all_in,
                "pending": state.pending,
                "contributions": state.contributions,
                "current_bet": state.current_bet,
                "raises": state.raises,
                "last_raise": state.last_raise_bb,
                "last_aggressor": state.last_aggressor,
                "raise_locked": state.raise_locked,
                "terminal": node.terminal,
                "actions": [str(action) for action in node.actions],
            }
            digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
            for action, child in zip(node.actions, node.children):
                visit(child, f"{path}/{action}")

        visit(self.root, "ROOT")
        return digest.hexdigest()

    def _showdown_shares(self, players: tuple[int, ...], deal: tuple[int, ...], approximate: bool = False) -> dict[int, float]:
        if len(players) == 1:
            return {players[0]: 1.0}
        if len(players) == 2:
            p, q = players
            eq = float(self.equity[deal[p], deal[q]])
            if approximate:
                eq = float(np.clip(eq * 0.95, 0.0, 1.0))
            return {p: eq, q: 1.0 - eq}
        strengths = {}
        for p in players:
            score = 1.0
            for q in players:
                if p != q:
                    score *= max(1e-6, float(self.equity[deal[p], deal[q]]))
            strengths[p] = score
        total = sum(strengths.values())
        return {p: score / total for p, score in strengths.items()}

    def _showdown_rankings(self, players: tuple[int, ...], deal: tuple[int, ...]):
        """Approximate complete showdown orders from preflop equity strengths."""
        if len(players) == 1:
            yield players, 1.0
            return
        if len(players) == 2:
            p, q = players
            equity = float(self.equity[deal[p], deal[q]])
            yield (p, q), equity
            yield (q, p), 1.0 - equity
            return
        shares = self._showdown_shares(players, deal)
        total = 0.0
        for order in permutations(players):
            remaining = 1.0
            probability = 1.0
            for player in order:
                denominator = sum(shares[p] for p in players if p not in order[:order.index(player)])
                probability *= shares[player] / denominator if denominator > 0 else 0.0
            total += probability
            yield order, probability

    def _terminal_values(self, node: TreeNode, deal: tuple[int, int, int]) -> np.ndarray:
        state = node.state
        if not self.chip_ev_mode:
            return self._terminal_icm_values(node, deal)
        if node.terminal == "FOLD_WIN":
            winner = state.active[0]
            values = -np.asarray(state.contributions, dtype=np.float32)
            values[winner] += state.pot_bb
            return values
        if node.terminal == "PREFLOP_CALL":
            sampled = getattr(self.leaf_evaluator, "evaluate_sampled_leaf", None)
            if callable(sampled):
                leaf_ev = sampled(
                    active_players=state.active,
                    combo_ids=deal,
                    pot_bb=state.pot_bb,
                    contributions=state.contributions,
                    stacks_rem=state.stacks_rem,
                    payoff_matrix=self.payoffs,
                    aggressor=state.last_aggressor,
                )
            else:
                reaches = np.zeros((3, 1326), dtype=np.float32)
                for player in state.active:
                    reaches[player, deal[player]] = 1.0
                leaf_ev = self.leaf_evaluator.evaluate_postflop_leaf(
                    active_players=state.active,
                    reaches=reaches,
                    pot_bb=state.pot_bb,
                    contributions=state.contributions,
                    stacks_rem=state.stacks_rem,
                    payoff_matrix=self.payoffs,
                    aggressor=state.last_aggressor,
                )
                if hasattr(leaf_ev, "detach"):
                    leaf_ev = leaf_ev.detach().cpu().numpy()
                leaf_ev = np.asarray([leaf_ev[player, deal[player]] for player in range(3)], dtype=np.float32)
            result = np.zeros(3, dtype=np.float32)
            result[[p for p in range(3) if p not in state.active]] = -np.asarray(state.contributions, dtype=np.float32)[[p for p in range(3) if p not in state.active]]
            for player in state.active:
                result[player] = leaf_ev[player]
            return result
        values = -np.asarray(state.contributions, dtype=np.float64)
        awards = np.zeros(3, dtype=np.float64)
        levels = sorted(set(c for c in state.contributions if c > 0))
        previous = 0.0
        for level in levels:
            contributors = [p for p, contribution in enumerate(state.contributions) if contribution >= level - 1e-9]
            amount = (level - previous) * len(contributors)
            previous = level
            eligible = tuple(p for p in state.active if state.contributions[p] >= level - 1e-9)
            if not eligible or amount <= 0:
                continue
            shares = self._showdown_shares(eligible, deal, approximate=False)
            for p, share in shares.items():
                awards[p] += amount * share
        values += awards
        if node.terminal == "ALL_IN_SHOWDOWN":
            # KO value is added only for all-in opponents who can be covered.
            showdown_shares = self._showdown_shares(tuple(state.active), deal, approximate=False)
            for q in state.all_in:
                covering = [p for p in state.active if p != q and state.stacks[p] >= state.stacks[q]
                            and self.payoffs[p, q] > 0]
                covering_mass = sum(showdown_shares.get(p, 0.0) for p in covering)
                if not covering or covering_mass <= 0:
                    continue
                elimination_chance = float(np.clip(1.0 - showdown_shares.get(q, 0.0), 0.0, 1.0))
                for p in covering:
                    share = showdown_shares.get(p, 0.0) / covering_mass
                    values[p] += float(self.payoffs[p, q]) * elimination_chance * share
        return values.astype(np.float32)

    def _terminal_icm_values(self, node: TreeNode, deal: tuple[int, int, int]) -> np.ndarray:
        state = node.state
        if node.terminal == "FOLD_WIN":
            winner = state.active[0]
            final_stacks = np.asarray(state.stacks, dtype=np.float64) - np.asarray(state.contributions, dtype=np.float64)
            final_stacks[winner] += state.pot_bb
            return (icm_equities(final_stacks, self.prizepool_payouts) - self.initial_icm).astype(np.float32)

        if node.terminal == "PREFLOP_CALL":
            zero_bounties = np.zeros((3, 3), dtype=np.float32)
            sampled = getattr(self.leaf_evaluator, "evaluate_sampled_leaf", None)
            if callable(sampled):
                leaf_ev = sampled(
                    active_players=state.active,
                    combo_ids=deal,
                    pot_bb=state.pot_bb,
                    contributions=state.contributions,
                    stacks_rem=state.stacks_rem,
                    payoff_matrix=zero_bounties,
                    aggressor=state.last_aggressor,
                )
            else:
                reaches = np.zeros((3, 1326), dtype=np.float32)
                for player in state.active:
                    reaches[player, deal[player]] = 1.0
                leaf_ev = self.leaf_evaluator.evaluate_postflop_leaf(
                    active_players=state.active,
                    reaches=reaches,
                    pot_bb=state.pot_bb,
                    contributions=state.contributions,
                    stacks_rem=state.stacks_rem,
                    payoff_matrix=zero_bounties,
                    aggressor=state.last_aggressor,
                )
                if hasattr(leaf_ev, "detach"):
                    leaf_ev = leaf_ev.detach().cpu().numpy()
                leaf_ev = np.asarray([leaf_ev[p, deal[p]] for p in range(3)], dtype=np.float32)
            if hasattr(leaf_ev, "detach"):
                leaf_ev = leaf_ev.detach().cpu().numpy()
            net = np.asarray(leaf_ev, dtype=np.float64)
            final_stacks = np.maximum(0.0, np.asarray(state.stacks) + net)
            return (icm_equities(final_stacks, self.prizepool_payouts) - self.initial_icm).astype(np.float32)

        expected_utility = np.zeros(3, dtype=np.float64)
        levels = sorted(set(c for c in state.contributions if c > 0))
        for order, probability in self._showdown_rankings(tuple(state.active), deal):
            if probability <= 0:
                continue
            rank = {player: index for index, player in enumerate(order)}
            awards = np.zeros(3, dtype=np.float64)
            layer_winners = {}
            previous = 0.0
            for level in levels:
                contributors = [p for p, contribution in enumerate(state.contributions)
                                if contribution >= level - 1e-9]
                amount = (level - previous) * len(contributors)
                previous = level
                eligible = [p for p in state.active
                            if state.contributions[p] >= level - 1e-9]
                if not eligible or amount <= 0:
                    continue
                winner = min(eligible, key=lambda p: rank[p])
                awards[winner] += amount
                layer_winners[round(level, 9)] = winner
            final_stacks = np.maximum(0.0, np.asarray(state.stacks, dtype=np.float64) -
                                      np.asarray(state.contributions, dtype=np.float64) + awards)
            utility = icm_equities(final_stacks, self.prizepool_payouts) - self.initial_icm
            if node.terminal == "ALL_IN_SHOWDOWN":
                for busted in state.all_in:
                    if final_stacks[busted] > 1e-9:
                        continue
                    level = round(float(state.contributions[busted]), 9)
                    recipient = layer_winners.get(level)
                    if recipient is not None and recipient != busted:
                        utility[recipient] += float(self.payoffs[recipient, busted])
            expected_utility += probability * utility
        return expected_utility.astype(np.float32)

    def _visit(self, node: TreeNode, traverser: int, deal: tuple[int, int, int],
               iteration: int, own_reach: float, path: str) -> np.ndarray:
        if node.terminal is not None:
            return self._terminal_values(node, deal)
        actor = node.state.actor
        key = path
        combo = deal[actor]
        strategy = self._strategy(key, combo, len(node.actions), iteration)
        if actor != traverser:
            action_index = int(self.rng.choice(len(node.actions), p=strategy))
            return self._visit(node.children[action_index], traverser, deal, iteration,
                               own_reach, f"{path}/{node.actions[action_index]}")

        action_values = np.empty((len(node.actions), 3), dtype=np.float32)
        for index, action in enumerate(node.actions):
            action_values[index] = self._visit(node.children[index], traverser, deal, iteration,
                                               own_reach * float(strategy[index]), f"{path}/{action}")
        hero_values = action_values[:, traverser].astype(np.float64)
        node_value = float(np.dot(strategy, hero_values))
        regrets = self.regrets[key][combo]
        regrets += hero_values - node_value
        t = max(1, iteration)
        weight = (t**2.0 / (t**2.0 + 1.0)) * own_reach
        self.strategy_sums.setdefault(key, {}).setdefault(combo, np.zeros(len(node.actions), dtype=np.float64))[:] += strategy * weight
        self.ev_sums.setdefault(key, {}).setdefault(combo, np.zeros(len(node.actions), dtype=np.float64))[:] += hero_values * weight
        self.row_visits.setdefault(key, {})[combo] = self.row_visits.setdefault(key, {}).get(combo, 0.0) + weight
        return np.sum(action_values * strategy[:, None], axis=0)

    def solve(self, iterations: int = 100) -> "MCCFRSolver":
        """Run sampled MCCFR iterations. Strategy tables are combo-specific."""
        if iterations < 1:
            raise ValueError("iterations must be positive")
        for _ in range(iterations):
            self.iteration += 1
            deal = self._deal()
            for traverser in range(3):
                self._visit(self.root, traverser, deal, self.iteration, 1.0, "ROOT")
        return self

    def average_strategy(self, key: str, combo: int) -> np.ndarray:
        sums = self.strategy_sums.get(key, {}).get(combo)
        if sums is None or sums.sum() <= 0:
            n_actions = self._action_count(key)
            return np.full(n_actions, 1.0 / n_actions, dtype=np.float32)
        return (sums / sums.sum()).astype(np.float32)

    def _action_count(self, key: str) -> int:
        node = self._find_node(key)
        return len(node.actions)

    def _find_node(self, key: str) -> TreeNode:
        def search(node: TreeNode, path: str):
            if path == key:
                return node
            for action, child in zip(node.actions, node.children):
                found = search(child, f"{path}/{action}")
                if found is not None:
                    return found
            return None
        result = search(self.root, "ROOT")
        if result is None:
            raise KeyError(key)
        return result

    def export(self, exporter, state_id: str) -> None:
        """Write every visited decision node to a BatchExporter instance."""
        def records():
            def visit(node: TreeNode, path: str):
                yield node, path
                for action, child in zip(node.actions, node.children):
                    yield from visit(child, f"{path}/{action}")

            for node, path in visit(self.root, "ROOT"):
                if node.terminal:
                    continue
                n_actions = len(node.actions)
                strategy = np.full((1326, n_actions), 1.0 / n_actions, dtype=np.float32)
                ev = np.zeros_like(strategy)
                for combo, sums in self.strategy_sums.get(path, {}).items():
                    if sums.sum() > 0:
                        strategy[combo] = sums / sums.sum()
                    visits = self.row_visits.get(path, {}).get(combo, 0.0)
                    if visits > 0:
                        ev[combo] = self.ev_sums[path][combo] / visits
                yield (path, node.state.actor, [str(action) for action in node.actions], strategy, ev)

        if hasattr(exporter, "export_nodes"):
            exporter.export_nodes(state_id, records())
            return

        for path, hero_pos, actions, strategy, ev in records():
            exporter.export_node(state_id, path, hero_pos, actions, strategy, ev)

    def checkpoint(self) -> bytes:
        """Serialize sparse regrets and average strategy for SQLite checkpointing."""
        def rows(source):
            return [[key, combo, values.tolist()]
                    for key, combos in source.items() for combo, values in combos.items()]
        payload = {
            "config": self._checkpoint_config,
            "iteration": self.iteration,
            "rng": self.rng.bit_generator.state,
            "regrets": rows(self.regrets),
            "regret_last_t": [[key, combo, last_t]
                              for key, combos in self.regret_last_t.items()
                              for combo, last_t in combos.items()],
            "strategy_sums": rows(self.strategy_sums),
            "ev_sums": rows(self.ev_sums),
            "row_visits": [[key, combo, value]
                           for key, combos in self.row_visits.items() for combo, value in combos.items()],
        }
        return zlib.compress(json.dumps(payload, separators=(",", ":")).encode(), level=6)

    def restore_checkpoint(self, blob: bytes) -> None:
        payload = json.loads(zlib.decompress(blob))
        if payload.get("config") != self._checkpoint_config:
            raise ValueError("checkpoint configuration does not match stacks, payoffs, equity, or solver version")
        self.iteration = int(payload["iteration"])
        self.rng.bit_generator.state = payload["rng"]
        for name in ("regrets", "strategy_sums", "ev_sums"):
            target = getattr(self, name)
            target.clear()
            for key, combo, values in payload[name]:
                target.setdefault(key, {})[int(combo)] = np.asarray(values, dtype=np.float64)
        self.regret_last_t.clear()
        for key, combo, last_t in payload["regret_last_t"]:
            self.regret_last_t.setdefault(key, {})[int(combo)] = int(last_t)
        self._positive_log_discount = [0.0]
        self.row_visits.clear()
        for key, combo, value in payload["row_visits"]:
            self.row_visits.setdefault(key, {})[int(combo)] = float(value)

    def _average_policy(self, key: str, combo: int, n_actions: int) -> np.ndarray:
        sums = self.strategy_sums.get(key, {}).get(combo)
        if sums is None or sums.sum() <= 0:
            return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
        return sums / sums.sum()

    def _profile_value(self, node: TreeNode, deal: tuple[int, int, int], path: str) -> np.ndarray:
        if node.terminal is not None:
            return self._terminal_values(node, deal)
        combo = deal[node.state.actor]
        policy = self._average_policy(path, combo, len(node.actions))
        value = np.zeros(3, dtype=np.float64)
        for probability, action, child in zip(policy, node.actions, node.children):
            value += probability * self._profile_value(child, deal, f"{path}/{action}")
        return value

    def _best_response_value(self, node: TreeNode, deal: tuple[int, int, int],
                             hero: int, path: str) -> float:
        if node.terminal is not None:
            return float(self._terminal_values(node, deal)[hero])
        actor = node.state.actor
        if actor == hero:
            values = [self._best_response_value(child, deal, hero, f"{path}/{action}")
                      for action, child in zip(node.actions, node.children)]
            return max(values)
        policy = self._average_policy(path, deal[actor], len(node.actions))
        return sum(probability * self._best_response_value(child, deal, hero, f"{path}/{action}")
                   for probability, action, child in zip(policy, node.actions, node.children))

    def estimate_nashconv(self, samples: int = 100, seed: int = 73) -> dict:
        """Estimate a clairvoyant exploitability upper bound on sampled deals.

        The best response observes all three private hands, so this is not the
        information-set-constrained NashConv. It is useful only as a loose
        progress diagnostic and inherits terminal-model and Monte Carlo error.
        """
        if samples < 1:
            raise ValueError("samples must be positive")
        rng = self.rng
        self.rng = np.random.default_rng(seed)
        baseline = np.zeros(3, dtype=np.float64)
        best_response = np.zeros(3, dtype=np.float64)
        try:
            for _ in range(samples):
                deal = self._deal()
                baseline += self._profile_value(self.root, deal, "ROOT")
                for hero in range(3):
                    best_response[hero] += self._best_response_value(self.root, deal, hero, "ROOT")
        finally:
            self.rng = rng
        baseline /= samples
        best_response /= samples
        gaps = np.maximum(0.0, best_response - baseline)
        return {
            "metric": "sampled_clairvoyant_exploitability_upper_bound",
            "samples": samples,
            "baseline_ev_bb": baseline.tolist(),
            "clairvoyant_best_response_ev_bb": best_response.tolist(),
            "player_gaps_bb": gaps.tolist(),
            "clairvoyant_exploitability_upper_bound_bb_per_deal": float(gaps.sum()),
        }

    def _path_for(self, target: TreeNode) -> str:
        def search(node: TreeNode, path: str):
            if node is target:
                return path
            for action, child in zip(node.actions, node.children):
                found = search(child, f"{path}/{action}")
                if found is not None:
                    return found
            return None
        result = search(self.root, "ROOT")
        if result is None:
            raise ValueError("node is not part of this solver's tree")
        return result


class VectorDCFRSolver:
    """Full-combo, full-tree DCFR using PyTorch range vectors.

    Every decision node stores one regret/average-strategy row for all 1,326
    combos. Each iteration traverses the public tree once per player, propagates
    `(3,1326)` realization reaches, and applies exact card-removal masks to
    pairwise range equity. Three-way equity and multiway ICM are range-based
    approximations; heads-up conditional equity uses the supplied matrix.
    """

    def __init__(self, stacks, equity: np.ndarray, payoff_matrix: np.ndarray,
                 prizepool_payouts=(1.0, 0.0, 0.0), payoff_scale: float = 1.0,
                 device: str | None = None, max_nodes: int = 100_000,
                 leaf_evaluator: BaseLeafEvaluator | None = None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install PyTorch with `pip install '.[gpu]'` to use VectorDCFRSolver") from exc
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(device)
        self.stacks = tuple(map(float, stacks))
        self.payouts = np.asarray(prizepool_payouts, dtype=np.float64)
        if len(self.stacks) != 3 or any(not np.isfinite(s) or s <= 0 for s in self.stacks):
            raise ValueError("three finite positive stacks are required")
        if self.payouts.shape != (3,) or not np.isfinite(self.payouts).all() or np.any(self.payouts < 0):
            raise ValueError("prizepool_payouts must have three finite non-negative values")
        if not np.isfinite(payoff_scale) or payoff_scale < 0:
            raise ValueError("payoff_scale must be finite and non-negative")
        self.chip_ev_mode = np.array_equal(self.payouts, np.asarray((1.0, 0.0, 0.0)))
        self.initial_icm = icm_equities(self.stacks, self.payouts)
        equity = np.asarray(equity, dtype=np.float32)
        payoffs = np.asarray(payoff_matrix, dtype=np.float32) * float(payoff_scale)
        if equity.shape != (1326, 1326) or payoffs.shape != (3, 3):
            raise ValueError("equity must be (1326,1326) and payoff_matrix (3,3)")
        if (not np.isfinite(equity).all() or np.any(equity < 0) or np.any(equity > 1) or
                not np.allclose(equity + equity.T, CARD_REMOVAL_MASK, atol=1e-3)):
            raise ValueError("equity must be finite, complementary and zero on card collisions")
        if not np.isfinite(payoffs).all() or np.any(payoffs < 0):
            raise ValueError("payoff_matrix must be finite and non-negative")
        self.equity = torch.as_tensor(equity, dtype=torch.float32, device=self.device)
        self.card_mask = torch.as_tensor(CARD_REMOVAL_MASK, dtype=torch.float32, device=self.device)
        self.payoffs = torch.as_tensor(payoffs, dtype=torch.float32, device=self.device)
        self.root = build_tree(self.stacks, max_nodes=max_nodes)
        self.leaf_evaluator = leaf_evaluator or TorchPreflopEQRLeafEvaluator(
            equity, CARD_REMOVAL_MASK, device=str(self.device))
        if not callable(getattr(self.leaf_evaluator, "checkpoint_fingerprint", None)):
            raise TypeError("leaf_evaluator must implement checkpoint_fingerprint()")
        self._checkpoint_config = {
            "solver": "vector-dcfr-v1",
            "stacks": list(self.stacks),
            "payouts": self.payouts.tolist(),
            "payoff_scale": float(payoff_scale),
            "equity_sha256": hashlib.sha256(equity.tobytes()).hexdigest(),
            "payoff_sha256": hashlib.sha256(payoffs.tobytes()).hexdigest(),
            "tree_sha256": self._tree_fingerprint(),
            "leaf_evaluator": self.leaf_evaluator.checkpoint_fingerprint(),
            "dcfr": {"alpha": 1.5, "beta": 0.0, "gamma": 2.0},
        }
        self.iteration = 0
        self._public_nodes: dict[str, TreeNode] = {}
        self._index_public_tree(self.root, "ROOT")
        self.regrets: dict[str, torch.Tensor] = {}
        self.strategy_sums: dict[str, torch.Tensor] = {}
        self.ev_sums: dict[str, torch.Tensor] = {}
        self.ev_weights: dict[str, torch.Tensor] = {}
        self._node_actions: dict[str, tuple[str, ...]] = {}
        self.alpha, self.beta, self.gamma = 1.5, 0.0, 2.0
        self._log_positive = [0.0]

    def _tree_fingerprint(self) -> str:
        digest = hashlib.sha256()

        def visit(node: TreeNode, path: str):
            state = node.state
            record = (path, state.actor, state.active, state.all_in, state.pending,
                      state.contributions, state.current_bet, state.raises,
                      state.last_raise_bb, state.last_aggressor, state.raise_locked,
                      node.terminal, tuple(map(str, node.actions)))
            digest.update(json.dumps(record, separators=(",", ":")).encode())
            for action, child in zip(node.actions, node.children):
                visit(child, f"{path}/{action}")
        visit(self.root, "ROOT")
        return digest.hexdigest()

    def _index_public_tree(self, node: TreeNode, path: str) -> None:
        self._public_nodes[path] = node
        for action, child in zip(node.actions, node.children):
            self._index_public_tree(child, f"{path}/{action}")

    def _node_key(self, path: str, actions: tuple) -> str:
        self._node_actions[path] = tuple(map(str, actions))
        if path not in self.regrets:
            n_actions = len(actions)
            self.regrets[path] = self.torch.zeros((1326, n_actions), dtype=self.torch.float32, device=self.device)
            self.strategy_sums[path] = self.torch.zeros_like(self.regrets[path])
            self.ev_sums[path] = self.torch.zeros_like(self.regrets[path])
            self.ev_weights[path] = self.torch.zeros((1326,), dtype=self.torch.float32, device=self.device)
        return path

    def _strategy(self, key: str, iteration: int, discount: bool = False) -> "torch.Tensor":
        torch = self.torch
        regrets = self.regrets[key]
        if discount:
            while len(self._log_positive) <= iteration:
                t = len(self._log_positive)
                self._log_positive.append(self._log_positive[-1] + np.log(t**self.alpha / (t**self.alpha + 1.0)))
            positive = float(np.exp(self._log_positive[iteration] - self._log_positive[iteration - 1]))
            negative = iteration**self.beta / (iteration**self.beta + 1.0)
            regrets.mul_(torch.where(regrets > 0, positive, negative))
        positive_regrets = torch.clamp(regrets, min=0.0)
        totals = positive_regrets.sum(dim=1, keepdim=True)
        uniform = torch.full_like(positive_regrets, 1.0 / positive_regrets.shape[1])
        return torch.where(totals > 0, positive_regrets / torch.clamp(totals, min=1e-30), uniform)

    def _average_strategy(self, key: str) -> "torch.Tensor":
        sums = self.strategy_sums[key]
        totals = sums.sum(dim=1, keepdim=True)
        uniform = self.torch.full_like(sums, 1.0 / sums.shape[1])
        regrets = self.torch.clamp(self.regrets[key], min=0.0)
        regret_totals = regrets.sum(dim=1, keepdim=True)
        current = self.torch.where(regret_totals > 0,
                                   regrets / self.torch.clamp(regret_totals, min=1e-30), uniform)
        average = sums / self.torch.clamp(totals, min=1e-30)
        return self.torch.where(totals > 0, average, current)

    def _pair_equity(self, hero: int, villain: int, reach: "torch.Tensor") -> "torch.Tensor":
        weights = reach[villain].unsqueeze(0) * self.card_mask
        denominator = weights.sum(dim=1)
        numerator = (self.equity * weights).sum(dim=1)
        return numerator / self.torch.clamp(denominator, min=1e-12)

    def _conditional_pair_strength(self, player: int, opponent: int,
                                   reach: "torch.Tensor") -> "torch.Tensor":
        """Opponent's strength vs a third range, conditioned on blocker combos."""
        torch = self.torch
        base = self._pair_equity(player, opponent, reach)
        weighted_opponent_range = reach[player] * base
        numerator = (self.card_mask * weighted_opponent_range.unsqueeze(0) * self.equity.T).sum(dim=1)
        denominator = (self.card_mask * reach[player].unsqueeze(0)).sum(dim=1)
        return numerator / torch.clamp(denominator, min=1e-12)

    def _conditional_pair_win_by_blocker(self, winner: int, loser: int,
                                         reach: "torch.Tensor") -> "torch.Tensor":
        """P(winner beats loser) for each blocker combo, averaging legal ranges."""
        torch = self.torch
        win_weights = self.equity * self.card_mask * reach[loser].unsqueeze(0)
        legal_weights = self.card_mask * reach[loser].unsqueeze(0)
        # Rows are winner combos and columns are blocker combos after projection.
        win_by_winner_blocker = win_weights @ self.card_mask.T
        mass_by_winner_blocker = legal_weights @ self.card_mask.T
        win_by_winner_blocker = win_by_winner_blocker / torch.clamp(
            mass_by_winner_blocker, min=1e-12)
        winner_legal = self.card_mask.T
        winner_weights = reach[winner][:, None] * winner_legal
        denominator = winner_weights.sum(dim=0)
        return (winner_weights * win_by_winner_blocker).sum(dim=0) / torch.clamp(
            denominator, min=1e-12)

    def _mix_action_values(self, values: "torch.Tensor", strategy: "torch.Tensor",
                           actor: int, reach: "torch.Tensor", perspective: int) -> "torch.Tensor":
        """Mix action values for one player's private hand, respecting blockers."""
        if perspective == actor:
            return (values[perspective] * strategy).sum(dim=1)
        weights = self.card_mask * reach[actor].unsqueeze(0)
        denominator = weights.sum(dim=1)
        conditional_strategy = (weights @ strategy) / self.torch.clamp(denominator[:, None], min=1e-12)
        return (values[perspective] * conditional_strategy).sum(dim=1)

    def _multiway_shares(self, eligible: tuple[int, ...], reach: "torch.Tensor") -> dict[int, "torch.Tensor"]:
        if len(eligible) == 1:
            return {eligible[0]: self.torch.ones((1326,), dtype=self.torch.float32, device=self.device)}
        if len(eligible) == 2:
            p, q = eligible
            eq_p = self._pair_equity(p, q, reach)
            eq_q = self._pair_equity(q, p, reach)
            return {p: eq_p, q: eq_q}
        scores = {}
        normalizers = {}
        for p in eligible:
            q, r = [player for player in eligible if player != p]
            scores[p] = self._pair_equity(p, q, reach) * self._pair_equity(p, r, reach)
            cond_q = self._conditional_pair_strength(q, r, reach)
            cond_r = self._conditional_pair_strength(r, q, reach)
            normalizers[p] = scores[p] + cond_q + cond_r
        return {p: scores[p] / self.torch.clamp(normalizers[p], min=1e-12)
                for p in eligible}

    def _terminal_chip_values(self, node: TreeNode, reach: "torch.Tensor") -> "torch.Tensor":
        torch = self.torch
        state = node.state
        if node.terminal == "FOLD_WIN":
            values = torch.tensor(-np.asarray(state.contributions), dtype=torch.float32, device=self.device)
            values[state.active[0]] += state.pot_bb
            return values[:, None].expand(3, 1326).clone()
        if node.terminal == "PREFLOP_CALL":
            evaluator = self.leaf_evaluator
            if not hasattr(evaluator, "evaluate_postflop_leaf"):
                raise TypeError("vector solver requires a range-based evaluate_postflop_leaf evaluator")
            if isinstance(evaluator, TorchPreflopEQRLeafEvaluator):
                evaluator_reach = reach
                evaluator_payoffs = self.payoffs if self.chip_ev_mode else torch.zeros_like(self.payoffs)
            else:
                evaluator_reach = reach.detach().cpu().numpy()
                evaluator_payoffs = (self.payoffs if self.chip_ev_mode else torch.zeros_like(self.payoffs))
                evaluator_payoffs = evaluator_payoffs.detach().cpu().numpy()
            output = evaluator.evaluate_postflop_leaf(
                active_players=state.active,
                reaches=evaluator_reach,
                pot_bb=state.pot_bb,
                contributions=state.contributions,
                stacks_rem=state.stacks_rem,
                payoff_matrix=evaluator_payoffs,
                aggressor=state.last_aggressor,
            )
            if hasattr(output, "detach"):
                return output.to(device=self.device, dtype=torch.float32)
            return torch.as_tensor(output, dtype=torch.float32, device=self.device)

        awards = torch.zeros((3, 1326), dtype=torch.float32, device=self.device)
        levels = sorted(set(c for c in state.contributions if c > 0))
        previous = 0.0
        for level in levels:
            contributors = [p for p, amount in enumerate(state.contributions) if amount >= level - 1e-9]
            pot_layer = (level - previous) * len(contributors)
            previous = level
            eligible = tuple(p for p in state.active if state.contributions[p] >= level - 1e-9)
            if not eligible or pot_layer <= 0:
                continue
            shares = self._multiway_shares(eligible, reach)
            for player, share in shares.items():
                awards[player] += pot_layer * share
        values = awards - torch.tensor(state.contributions, dtype=torch.float32,
                                       device=self.device)[:, None]
        if node.terminal == "ALL_IN_SHOWDOWN" and self.chip_ev_mode:
            values += self._terminal_token_values(node, reach)
        return values

    def _terminal_token_values(self, node: TreeNode, reach: "torch.Tensor") -> "torch.Tensor":
        values = self.torch.zeros((3, 1326), dtype=self.torch.float32, device=self.device)
        state = node.state
        for busted in state.all_in:
            cap = round(float(state.contributions[busted]), 9)
            eligible = tuple(p for p in state.active if state.contributions[p] >= cap - 1e-9)
            if not eligible:
                continue
            shares = self._multiway_shares(eligible, reach)
            covering = [p for p in eligible if p != busted and self.stacks[p] >= self.stacks[busted]]
            for winner in covering:
                values[winner] += self.payoffs[winner, busted] * shares[winner]
        return values

    def _terminal_values(self, node: TreeNode, reach: "torch.Tensor") -> "torch.Tensor":
        if not self.chip_ev_mode and node.terminal == "ALL_IN_SHOWDOWN":
            return self._terminal_allin_icm_values(node, reach)
        chip_values = self._terminal_chip_values(node, reach)
        if self.chip_ev_mode:
            return chip_values
        if node.terminal == "PREFLOP_CALL":
            return self._preflop_call_icm_values(node, reach, chip_values)
        torch = self.torch
        if node.terminal == "FOLD_WIN":
            final_stacks = torch.as_tensor(self.stacks, dtype=torch.float32,
                                           device=self.device) + chip_values[:, 0]
            utility = _torch_icm_equity(final_stacks.unsqueeze(0),
                                       torch.as_tensor(self.payouts, dtype=torch.float32,
                                                       device=self.device))[0]
            utility -= torch.as_tensor(self.initial_icm, dtype=torch.float32, device=self.device)
            return utility[:, None].expand(3, 1326).clone()
        raise ValueError(f"unsupported terminal node in ICM mode: {node.terminal}")

    def _preflop_call_icm_values(self, node: TreeNode, reach: "torch.Tensor",
                                 chip_values: "torch.Tensor") -> "torch.Tensor":
        """Evaluate discrete win/loss ICM states for the preflop-call leaf."""
        torch = self.torch
        state = node.state
        active = tuple(state.active)
        contributions = torch.as_tensor(state.contributions, dtype=torch.float32, device=self.device)
        base_stacks = torch.as_tensor(self.stacks, dtype=torch.float32, device=self.device) - contributions
        pot = float(state.pot_bb)
        denominator_pot = max(pot, 1e-12)
        payouts = torch.as_tensor(self.payouts, dtype=torch.float32, device=self.device)
        result = torch.zeros_like(chip_values)

        for hero in active:
            p_win = torch.clamp((chip_values[hero] + contributions[hero]) / denominator_pot,
                                min=0.0, max=1.0)
            win_stacks = base_stacks[None, :].expand(1326, 3).clone()
            win_stacks[:, hero] += pot
            lose_stacks = base_stacks[None, :].expand(1326, 3).clone()
            opponents = [player for player in active if player != hero]
            if len(opponents) == 1:
                lose_stacks[:, opponents[0]] += pot
            else:
                q, r = opponents
                q_strength = self._conditional_pair_strength(q, r, reach)
                r_strength = self._conditional_pair_strength(r, q, reach)
                total = torch.clamp(q_strength + r_strength, min=1e-12)
                lose_stacks[:, q] += pot * q_strength / total
                lose_stacks[:, r] += pot * r_strength / total

            win_icm = _torch_icm_equity(win_stacks, payouts)
            lose_icm = _torch_icm_equity(lose_stacks, payouts)
            result[hero] = (p_win * win_icm[:, hero] +
                            (1.0 - p_win) * lose_icm[:, hero] - float(self.initial_icm[hero]))

        inactive = [player for player in range(3) if player not in active]
        if len(active) == 2 and len(inactive) == 1:
            q, r = active
            folded = inactive[0]
            p_q_wins = self._conditional_pair_win_by_blocker(q, r, reach)
            stacks_q_wins = base_stacks.clone()
            stacks_q_wins[q] += pot
            stacks_r_wins = base_stacks.clone()
            stacks_r_wins[r] += pot
            icm_q_wins = _torch_icm_equity(stacks_q_wins.unsqueeze(0), payouts)[0, folded]
            icm_r_wins = _torch_icm_equity(stacks_r_wins.unsqueeze(0), payouts)[0, folded]
            result[folded] = (p_q_wins * icm_q_wins + (1.0 - p_q_wins) * icm_r_wins -
                              float(self.initial_icm[folded]))
        return result

    def _allin_outcome_utility(self, node: TreeNode, order: tuple[int, ...]) -> np.ndarray:
        state = node.state
        rank = {player: index for index, player in enumerate(order)}
        awards = np.zeros(3, dtype=np.float64)
        layer_winners = {}
        levels = sorted(set(c for c in state.contributions if c > 0))
        previous = 0.0
        for level in levels:
            contributors = [p for p, amount in enumerate(state.contributions) if amount >= level - 1e-9]
            layer = (level - previous) * len(contributors)
            previous = level
            eligible = [p for p in state.active if state.contributions[p] >= level - 1e-9]
            if layer <= 0 or not eligible:
                continue
            winner = min(eligible, key=lambda p: rank[p])
            awards[winner] += layer
            layer_winners[round(level, 9)] = winner
        final_stacks = np.maximum(0.0, np.asarray(state.stacks, dtype=np.float64) -
                                  np.asarray(state.contributions, dtype=np.float64) + awards)
        utility = icm_equities(final_stacks, self.payouts) - self.initial_icm
        for busted in state.all_in:
            if final_stacks[busted] > 1e-9:
                continue
            recipient = layer_winners.get(round(float(state.contributions[busted]), 9))
            if recipient is not None and recipient != busted:
                utility[recipient] += float(self.payoffs[recipient, busted])
        return utility

    def _terminal_allin_icm_values(self, node: TreeNode, reach: "torch.Tensor") -> "torch.Tensor":
        torch = self.torch
        active = tuple(node.state.active)
        result = torch.zeros((3, 1326), dtype=torch.float32, device=self.device)
        outcomes = {}
        if len(active) == 2:
            p, q = active
            p_win = self._pair_equity(p, q, reach)
            q_win = self._pair_equity(q, p, reach)
            outcomes[(p, q)] = self._allin_outcome_utility(node, (p, q))
            outcomes[(q, p)] = self._allin_outcome_utility(node, (q, p))
            result[p] = p_win * outcomes[(p, q)][p] + (1.0 - p_win) * outcomes[(q, p)][p]
            result[q] = q_win * outcomes[(q, p)][q] + (1.0 - q_win) * outcomes[(p, q)][q]
            inactive = [player for player in range(3) if player not in active]
            if len(inactive) == 1:
                folded = inactive[0]
                p_wins = self._conditional_pair_win_by_blocker(p, q, reach)
                result[folded] = (p_wins * outcomes[(p, q)][folded] +
                                  (1.0 - p_wins) * outcomes[(q, p)][folded])
            return result

        scores = {}
        for player in active:
            q, r = [p for p in active if p != player]
            scores[player] = self._pair_equity(player, q, reach) * self._pair_equity(player, r, reach)
        for order in permutations(active):
            outcomes[order] = self._allin_outcome_utility(node, order)
        for hero in active:
            q, r = [p for p in active if p != hero]
            cond_q = self._conditional_pair_strength(q, r, reach)
            cond_r = self._conditional_pair_strength(r, q, reach)
            weight_by_player = {hero: scores[hero], q: cond_q, r: cond_r}
            total = scores[hero] + cond_q + cond_r
            expected = torch.zeros((1326,), dtype=torch.float32, device=self.device)
            for order, utility in outcomes.items():
                first, second, _ = order
                p_first = weight_by_player[first] / torch.clamp(total, min=1e-12)
                p_second = weight_by_player[second] / torch.clamp(total - weight_by_player[first], min=1e-12)
                probability = p_first * p_second
                expected += probability * float(utility[hero])
            result[hero] = expected
        # Folded player's private combo blocks both live players' showdown hands.
        inactive = [player for player in range(3) if player not in active]
        if len(inactive) == 1:
            folded = inactive[0]
            p, q = active
            p_wins = self._conditional_pair_win_by_blocker(p, q, reach)
            result[folded] = (p_wins * outcomes[(p, q)][folded] +
                              (1.0 - p_wins) * outcomes[(q, p)][folded])
        return result

    def _walk(self, node: TreeNode, traverser: int, reach: "torch.Tensor",
              iteration: int, path: str) -> "torch.Tensor":
        torch = self.torch
        if node.terminal is not None:
            return self._terminal_values(node, reach)
        actor = node.state.actor
        key = self._node_key(path, node.actions)
        strategy = self._strategy(key, iteration, discount=(actor == traverser))
        action_values = []
        for action_index, action in enumerate(node.actions):
            next_reach = reach.clone()
            next_reach[actor] = next_reach[actor] * strategy[:, action_index]
            action_values.append(self._walk(node.children[action_index], traverser,
                                            next_reach, iteration,
                                            f"{path}/{action}"))
        child_values = torch.stack(action_values, dim=2)  # (player, combo, action)
        if actor == traverser:
            hand_action_values = child_values[actor].transpose(0, 1)
            node_values = (strategy * hand_action_values).sum(dim=1, keepdim=True)
            regrets = self.regrets[key]
            counterfactual_reach = torch.ones((1326,), dtype=torch.float32, device=self.device)
            for opponent in range(3):
                if opponent != actor:
                    counterfactual_reach *= self.card_mask @ reach[opponent]
            regrets.add_((hand_action_values - node_values) * counterfactual_reach[:, None])
            t = float(iteration)
            average_weight = t**self.gamma / (t**self.gamma + 1.0)
            own_reach = reach[actor]
            weighted_strategy = own_reach[:, None] * average_weight
            self.strategy_sums[key].add_(strategy * weighted_strategy)
            weighted_counterfactual = counterfactual_reach[:, None] * average_weight
            self.ev_sums[key].add_(hand_action_values * weighted_counterfactual)
            self.ev_weights[key].add_(weighted_counterfactual[:, 0])
        result = torch.zeros((3, 1326), dtype=torch.float32, device=self.device)
        for perspective in range(3):
            result[perspective] = self._mix_action_values(
                child_values, strategy, actor, reach, perspective)
        return result

    def solve(self, iterations: int = 1) -> "VectorDCFRSolver":
        if iterations < 1:
            raise ValueError("iterations must be positive")
        with self.torch.no_grad():
            reach = self.torch.full((3, 1326), 1.0 / 1326.0,
                                    dtype=self.torch.float32, device=self.device)
            for _ in range(iterations):
                self.iteration += 1
                for traverser in range(3):
                    self._walk(self.root, traverser, reach, self.iteration, "ROOT")
        return self

    def export(self, exporter, state_id: str) -> None:
        records = []
        for path, actions in self._node_actions.items():
            node = self._find_public_node(path)
            strategy = self._average_strategy(path)
            ev = self.ev_sums[path] / self.torch.clamp(self.ev_weights[path][:, None], min=1e-30)
            records.append((path, node.state.actor, list(actions),
                            strategy.detach().cpu().numpy(), ev.detach().cpu().numpy()))
        exporter.export_nodes(state_id, records)

    def _find_public_node(self, target_path: str) -> TreeNode:
        result = self._public_nodes.get(target_path)
        if result is None:
            raise KeyError(target_path)
        return result

    def checkpoint(self) -> bytes:
        import io
        metadata = {"config": self._checkpoint_config, "iteration": self.iteration,
                    "nodes": [], "actions": self._node_actions}
        arrays = {}
        for index, key in enumerate(self.regrets):
            names = {name: f"{name}_{index}" for name in ("regrets", "strategy_sums", "ev_sums", "ev_weights")}
            metadata["nodes"].append({"key": key, "names": names})
            for name, array_name in names.items():
                arrays[array_name] = getattr(self, name)[key].detach().cpu().numpy()
        arrays["metadata"] = np.frombuffer(json.dumps(metadata, separators=(",", ":")).encode(), dtype=np.uint8)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return zlib.compress(buffer.getvalue(), level=3)

    def restore_checkpoint(self, blob: bytes) -> None:
        import io
        with np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False) as payload:
            metadata = json.loads(payload["metadata"].tobytes())
            if metadata.get("config") != self._checkpoint_config:
                raise ValueError("checkpoint configuration does not match vector solver, payouts, equity, or tree")
            iteration = int(metadata["iteration"])
            if iteration < 0:
                raise ValueError("checkpoint iteration cannot be negative")
            node_actions = {key: tuple(actions) for key, actions in metadata["actions"].items()}
            loaded = []
            for record in metadata["nodes"]:
                key = record["key"]
                required_arrays = {"regrets", "strategy_sums", "ev_sums", "ev_weights"}
                if set(record.get("names", {})) != required_arrays:
                    raise ValueError(f"checkpoint arrays are incomplete at {key}")
                node = self._public_nodes.get(key)
                if node is None or node.terminal is not None:
                    raise ValueError(f"checkpoint contains unknown decision node {key}")
                expected_actions = tuple(map(str, node.actions))
                if node_actions.get(key) != expected_actions:
                    raise ValueError(f"checkpoint action list does not match tree at {key}")
                arrays = {}
                for name, array_name in record["names"].items():
                    array = payload[array_name].copy()
                    expected_shape = ((1326,) if name == "ev_weights" else
                                      (1326, len(expected_actions)))
                    if array.shape != expected_shape or not np.isfinite(array).all():
                        raise ValueError(f"invalid {name} array in checkpoint at {key}")
                    arrays[name] = array
                loaded.append((key, arrays))
        self.iteration = iteration
        self._node_actions = node_actions
        for name in ("regrets", "strategy_sums", "ev_sums", "ev_weights"):
            getattr(self, name).clear()
        for key, arrays in loaded:
            for name, array in arrays.items():
                getattr(self, name)[key] = self.torch.as_tensor(array, dtype=self.torch.float32,
                                                                 device=self.device)

    def _average_profile(self, node: TreeNode, reach: "torch.Tensor", path: str) -> "torch.Tensor":
        torch = self.torch
        if node.terminal is not None:
            return self._terminal_values(node, reach)
        actor = node.state.actor
        key = path
        if key not in self.strategy_sums:
            self._node_key(key, node.actions)
        strategy = self._average_strategy(key)
        values = []
        for index, action in enumerate(node.actions):
            child_reach = reach.clone()
            child_reach[actor] *= strategy[:, index]
            values.append(self._average_profile(node.children[index], child_reach, f"{path}/{action}"))
        children = torch.stack(values, dim=2)
        result = torch.zeros((3, 1326), dtype=torch.float32, device=self.device)
        for perspective in range(3):
            result[perspective] = self._mix_action_values(
                children, strategy, actor, reach, perspective)
        return result

    def _best_response(self, node: TreeNode, hero: int, reach: "torch.Tensor",
                       path: str) -> "torch.Tensor":
        torch = self.torch
        if node.terminal is not None:
            return self._terminal_values(node, reach)[hero]
        actor = node.state.actor
        key = path
        if key not in self.strategy_sums:
            self._node_key(key, node.actions)
        if actor == hero:
            action_values = []
            for action, child in zip(node.actions, node.children):
                action_values.append(self._best_response(child, hero, reach, f"{path}/{action}"))
            return torch.stack(action_values, dim=1).max(dim=1).values
        strategy = self._average_strategy(key)
        action_values = []
        for action, child in zip(node.actions, node.children):
            child_reach = reach.clone()
            child_reach[actor] *= strategy[:, len(action_values)]
            action_values.append(self._best_response(child, hero, child_reach, f"{path}/{action}"))
        values = torch.stack(action_values, dim=1)
        weights = self.card_mask * reach[actor].unsqueeze(0)
        denominator = weights.sum(dim=1)
        conditional_strategy = (weights @ strategy) / torch.clamp(denominator[:, None], min=1e-12)
        return (values * conditional_strategy).sum(dim=1)

    def _root_chance_mass(self, hero: int, reach: "torch.Tensor") -> "torch.Tensor":
        torch = self.torch
        opponents = [p for p in range(3) if p != hero]
        if len(opponents) == 1:
            return self.card_mask @ reach[opponents[0]]
        q, r = opponents
        incidence = torch.zeros((52, 1326), dtype=torch.float32, device=self.device)
        cards = torch.as_tensor(COMBO_CARDS, dtype=self.torch.long, device=self.device)
        incidence[cards[:, 0], torch.arange(1326, device=self.device)] = 1.0
        incidence[cards[:, 1], torch.arange(1326, device=self.device)] = 1.0
        valid_r = reach[r][None, :] * self.card_mask
        card_mass = valid_r @ incidence.T
        total_r = valid_r.sum(dim=1, keepdim=True)
        remaining_r = (total_r - card_mass[:, cards[:, 0]] -
                       card_mass[:, cards[:, 1]] + valid_r)
        return (reach[q][None, :] * self.card_mask *
                torch.clamp(remaining_r, min=0.0)).sum(dim=1)

    def estimate_nashconv(self) -> dict:
        with self.torch.no_grad():
            reach = self.torch.full((3, 1326), 1.0 / 1326.0,
                                    dtype=self.torch.float32, device=self.device)
            profile = self._average_profile(self.root, reach, "ROOT")
            base = []
            best = []
            for hero in range(3):
                chance = self._root_chance_mass(hero, reach)
                weights = reach[hero] * chance
                denominator = self.torch.clamp(weights.sum(), min=1e-30)
                base.append(float((profile[hero] * weights).sum() / denominator))
                best_response = self._best_response(self.root, hero, reach, "ROOT")
                best.append(float((best_response * weights).sum() / denominator))
            gaps = [max(0.0, best[p] - base[p]) for p in range(3)]
        return {"metric": "range_based_nashconv_approx",
                "utility_units": "bb" if self.chip_ev_mode else "prize_units",
                "baseline_ev": base, "best_response_ev": best,
                "player_gaps": gaps, "nashconv": float(sum(gaps))}


class MCCFRSolver(VectorDCFRSolver):
    """Compatibility name for the vector DCFR solver; no combo sampling is used."""

    def __init__(self, stacks, equity, payoff_matrix, seed: int = 7,
                 max_nodes: int = 100_000, payoff_scale: float = 1.0,
                 leaf_evaluator: BaseLeafEvaluator | None = None,
                 prizepool_payouts=(1.0, 0.0, 0.0), device: str | None = None):
        # ``seed`` remains accepted for callers of the prior sampled solver;
        # vector range traversals are deterministic and do not consume it.
        super().__init__(stacks, equity, payoff_matrix,
                         prizepool_payouts=prizepool_payouts,
                         payoff_scale=payoff_scale, device=device,
                         max_nodes=max_nodes, leaf_evaluator=leaf_evaluator)
