"""DCFR regret matching and external-sampling MCCFR over the public tree."""

from __future__ import annotations

import numpy as np
import json
import zlib
import hashlib

from .constants import CARD_REMOVAL_MASK, COMBO_CARDS
from .game_tree import TreeNode, build_tree
from .leaf_evaluator import EQR, PreflopEQRLeafEvaluator


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


class MCCFRSolver:
    """External-sampling MCCFR for sampled, non-overlapping private combos.

    The public betting tree is complete for its configured sizing abstraction.
    Each iteration samples one legal three-hand deal and traverses the tree once
    per player. Pairwise preflop equity supplies terminal values; three-way
    equity and non-all-in EQR leaves remain approximations.
    """

    def __init__(self, stacks, equity: np.ndarray, payoff_matrix: np.ndarray,
                 seed: int = 7, max_nodes: int = 100_000, payoff_scale: float = 1.0):
        self.stacks = tuple(map(float, stacks))
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
        self.leaf_evaluator = PreflopEQRLeafEvaluator(self.equity, CARD_REMOVAL_MASK)
        self._checkpoint_config = {
            "solver": "mccfr-v2",
            "stacks": list(self.stacks),
            "equity_sha256": hashlib.sha256(self.equity.tobytes()).hexdigest(),
            "payoff_sha256": hashlib.sha256(self.payoffs.tobytes()).hexdigest(),
            "tree_sha256": self._tree_fingerprint(),
            "eqr_factors": EQR,
        }
        self.rng = np.random.default_rng(seed)
        # Sparse per-information-set rows avoid allocating every combo at every node.
        self.regrets: dict[str, dict[int, np.ndarray]] = {}
        self.strategy_sums: dict[str, dict[int, np.ndarray]] = {}
        self.ev_sums: dict[str, dict[int, np.ndarray]] = {}
        self.row_visits: dict[str, dict[int, float]] = {}
        self.iteration = 0

    def _strategy(self, key: str, combo: int, n_actions: int) -> np.ndarray:
        row = self.regrets.setdefault(key, {}).setdefault(combo, np.zeros(n_actions, dtype=np.float64))
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
                "terminal": node.terminal,
                "actions": [str(action) for action in node.actions],
            }
            digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
            for action, child in zip(node.actions, node.children):
                visit(child, f"{path}/{action}")

        visit(self.root, "ROOT")
        return digest.hexdigest()

    def _showdown_shares(self, players: tuple[int, ...], deal: tuple[int, ...], approximate: bool) -> dict[int, float]:
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

    def _terminal_values(self, node: TreeNode, deal: tuple[int, int, int]) -> np.ndarray:
        state = node.state
        if node.terminal == "FOLD_WIN":
            winner = state.active[0]
            values = -np.asarray(state.contributions, dtype=np.float32)
            values[winner] += state.pot_bb
            return values
        if node.terminal == "PREFLOP_CALL":
            leaf_ev = self.leaf_evaluator.evaluate_sampled_leaf(
                active_players=state.active,
                combo_ids=deal,
                pot_bb=state.pot_bb,
                contributions=state.contributions,
                stacks_rem=state.stacks_rem,
                payoff_matrix=self.payoffs,
                aggressor=state.last_aggressor,
            )
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

    def _visit(self, node: TreeNode, traverser: int, deal: tuple[int, int, int],
               iteration: int, own_reach: float, path: str) -> np.ndarray:
        if node.terminal is not None:
            return self._terminal_values(node, deal)
        actor = node.state.actor
        key = path
        combo = deal[actor]
        strategy = self._strategy(key, combo, len(node.actions))
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
        t = max(1, iteration)
        positive_discount = t**1.5 / (t**1.5 + 1.0)
        negative_discount = t**0.0 / (t**0.0 + 1.0)
        regrets *= np.where(regrets > 0, positive_discount, negative_discount)
        regrets += hero_values - node_value
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
        """Estimate NashConv by sampled deals and exact public-tree best responses.

        The estimator is useful for progress tracking, but inherits terminal
        equity/EQR approximations and has Monte Carlo sampling error.
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
            "samples": samples,
            "baseline_ev_bb": baseline.tolist(),
            "best_response_ev_bb": best_response.tolist(),
            "player_gaps_bb": gaps.tolist(),
            "nashconv_bb_per_deal": float(gaps.sum()),
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
