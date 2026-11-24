"""Pluggable terminal evaluator interface and phase-1 EQR approximation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .constants import is_ip

EQR = {
    "IP_AGGRESSOR": 1.12,
    "OOP_AGGRESSOR": 0.92,
    "IP_CALLER": 0.95,
    "OOP_CALLER": 0.76,
    "MULTIWAY_3WAY": 0.68,
}


class BaseLeafEvaluator(ABC):
    @abstractmethod
    def evaluate_postflop_leaf(self, active_players, reaches, pot_bb, contributions,
                               stacks_rem, payoff_matrix, aggressor=None) -> Any:
        """Return net EVs shaped (3,1326), as NumPy or a device tensor."""

    def checkpoint_fingerprint(self) -> str:
        """Stable evaluator identity included in solver checkpoint compatibility."""
        return f"{type(self).__module__}.{type(self).__qualname__}:v1"


class PreflopEQRLeafEvaluator(BaseLeafEvaluator):
    """Approximate postflop EV from preflop equity and realization factors.

    This is a heuristic phase-1 leaf model; its values are not exact postflop
    solutions and must be calibrated before treating exported strategies as GTO.
    """

    def __init__(self, equity: np.ndarray, card_removal: np.ndarray, epsilon: float = 1e-12):
        self.equity = np.asarray(equity, dtype=np.float32)
        self.card_removal = np.asarray(card_removal, dtype=np.float32)
        if self.equity.shape != (1326, 1326) or self.card_removal.shape != (1326, 1326):
            raise ValueError("equity and card_removal must have shape (1326, 1326)")
        if (not np.isfinite(self.equity).all() or np.any(self.equity < 0) or
                np.any(self.equity > 1) or not np.isfinite(self.card_removal).all() or
                not np.isin(self.card_removal, (0.0, 1.0)).all() or
                not np.allclose(self.equity + self.equity.T, self.card_removal, atol=1e-3)):
            raise ValueError("equity must be complementary in [0,1] and match the binary card-removal mask")
        self.epsilon = epsilon

    def checkpoint_fingerprint(self) -> str:
        return f"preflop-eqr-v2:{self.epsilon}:{tuple(sorted(EQR.items()))}"

    def evaluate_postflop_leaf(self, active_players, reaches, pot_bb, contributions,
                               stacks_rem, payoff_matrix, aggressor=None) -> np.ndarray:
        reach = np.asarray(reaches, dtype=np.float32)
        bounty = np.asarray(payoff_matrix, dtype=np.float32)
        invested = np.asarray(contributions, dtype=np.float32)
        remaining = np.asarray(stacks_rem, dtype=np.float32)
        if (reach.shape != (3, 1326) or bounty.shape != (3, 3) or
                invested.shape != (3,) or remaining.shape != (3,)):
            raise ValueError("reaches must be (3,1326), contributions (3,), payoff_matrix (3,3)")
        active = tuple(active_players)
        if (len(active) not in (2, 3) or len(set(active)) != len(active) or
                any(not isinstance(p, (int, np.integer)) or p not in range(3) for p in active)):
            raise ValueError("two or three valid active players are required")
        if (not np.isfinite(reach).all() or np.any(reach < 0) or
                not np.isfinite(bounty).all() or np.any(bounty < 0) or
                not np.isfinite(invested).all() or np.any(invested < 0) or
                not np.isfinite(remaining).all() or np.any(remaining < 0) or
                not np.isfinite(pot_bb) or pot_bb < 0):
            raise ValueError("reaches, bounties, pot, contributions and remaining stacks must be finite and non-negative")
        out = np.broadcast_to(-invested[:, None], (3, 1326)).copy()
        pair_equities = {}
        for p in active:
            opponents = [q for q in active if q != p]
            pair_values = []
            for q in opponents:
                weights = reach[q][None, :] * self.card_removal
                denom = weights.sum(axis=1) + self.epsilon
                pair_values.append((self.equity * weights).sum(axis=1) / denom)
            pair_equities[p] = pair_values
        if len(active) == 3:
            scores = {p: np.prod(pair_equities[p], axis=0) for p in active}
            normalizer = sum(scores.values()) + self.epsilon
        else:
            scores, normalizer = {}, None
        for p in active:
            opponents = [q for q in active if q != p]
            equity = (pair_equities[p][0] if len(active) == 2
                      else scores[p] / normalizer)
            role = _eqr_role(p, opponents, active, aggressor)
            eqr = EQR[role]
            ev = float(pot_bb) * np.clip(equity * eqr, 0.0, 1.0)
            spr = max((float(stacks_rem[q]) for q in opponents), default=0.0) / max(float(pot_bb), self.epsilon)
            stackoff = float(np.clip(1.0 / (1.0 + spr), 0.0, 1.0))
            for q in opponents:
                if stacks_rem[p] >= stacks_rem[q]:
                    ev += bounty[p, q] * stackoff / len(opponents)
            out[p] = (ev - invested[p]).astype(np.float32)
        return out

    def evaluate_sampled_leaf(self, active_players, combo_ids, pot_bb, contributions, stacks_rem,
                              payoff_matrix, aggressor=None) -> np.ndarray:
        """Fast EQR evaluation for one already-sampled, non-overlapping deal."""
        active = tuple(active_players)
        bounty = np.asarray(payoff_matrix, dtype=np.float32)
        invested = np.asarray(contributions, dtype=np.float32)
        remaining = np.asarray(stacks_rem, dtype=np.float32)
        combo_ids = tuple(map(int, combo_ids))
        if (len(active) not in (2, 3) or len(set(active)) != len(active) or
                bounty.shape != (3, 3) or invested.shape != (3,) or remaining.shape != (3,)):
            raise ValueError("two or three active players, contributions (3,), and payoff matrix (3,3) are required")
        if (len(active) not in (2, 3) or len(set(active)) != len(active) or
                any(p not in range(3) for p in active) or
                len(combo_ids) != 3 or any(combo < 0 or combo >= 1326 for combo in combo_ids) or
                any(not self.card_removal[combo_ids[p], combo_ids[q]]
                    for p in range(3) for q in range(p + 1, 3))):
            raise ValueError("combo_ids must identify a legal three-player deal")
        if (not np.isfinite(bounty).all() or np.any(bounty < 0) or
                not np.isfinite(invested).all() or np.any(invested < 0) or
                not np.isfinite(remaining).all() or np.any(remaining < 0) or
                not np.isfinite(pot_bb) or pot_bb < 0):
            raise ValueError("bounties, pot, contributions and remaining stacks must be finite and non-negative")
        out = -invested.copy()
        pair_equities = {}
        for p in active:
            opponents = [q for q in active if q != p]
            pair_equities[p] = [float(self.equity[combo_ids[p], combo_ids[q]])
                                if self.card_removal[combo_ids[p], combo_ids[q]] else 0.0
                                for q in opponents]
        if len(active) == 3:
            scores = {p: float(np.prod(pair_equities[p])) for p in active}
            normalizer = sum(scores.values()) or self.epsilon
        for p in active:
            opponents = [q for q in active if q != p]
            equity = (pair_equities[p][0] if len(active) == 2
                      else scores[p] / normalizer)
            role = _eqr_role(p, opponents, active, aggressor)
            ev = float(pot_bb) * float(np.clip(equity * EQR[role], 0.0, 1.0))
            spr = max((float(stacks_rem[q]) for q in opponents), default=0.0) / max(float(pot_bb), self.epsilon)
            stackoff = float(np.clip(1.0 / (1.0 + spr), 0.0, 1.0))
            for q in opponents:
                if stacks_rem[p] >= stacks_rem[q]:
                    ev += float(bounty[p, q]) * stackoff / len(opponents)
            out[p] = ev - invested[p]
        return out


class TorchPreflopEQRLeafEvaluator(BaseLeafEvaluator):
    """PyTorch implementation of the EQR matrix kernel.

    Importing PyTorch is deferred until construction, so CPU-only installations
    can use the NumPy evaluator. CUDA is selected automatically when available.
    """

    def __init__(self, equity, card_removal, device: str | None = None, epsilon: float = 1e-12):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install PyTorch with `pip install '.[gpu]'`") from exc
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.torch = torch
        self.device = torch.device(device)
        self.equity = torch.as_tensor(equity, dtype=torch.float32, device=self.device)
        self.card_removal = torch.as_tensor(card_removal, dtype=torch.float32, device=self.device)
        if tuple(self.equity.shape) != (1326, 1326) or tuple(self.card_removal.shape) != (1326, 1326):
            raise ValueError("equity and card_removal must have shape (1326, 1326)")
        if (not torch.isfinite(self.equity).all() or torch.any(self.equity < 0) or
                torch.any(self.equity > 1) or not torch.isfinite(self.card_removal).all() or
                not torch.all((self.card_removal == 0) | (self.card_removal == 1)) or
                not torch.allclose(self.equity + self.equity.T, self.card_removal, atol=1e-3)):
            raise ValueError("equity must be complementary in [0,1] and match the binary card-removal mask")
        self.epsilon = float(epsilon)

    def checkpoint_fingerprint(self) -> str:
        return f"torch-preflop-eqr-v2:{self.epsilon}:{self.device}:{tuple(sorted(EQR.items()))}"

    def evaluate_postflop_leaf(self, active_players, reaches, pot_bb, contributions,
                               stacks_rem, payoff_matrix, aggressor=None):
        torch = self.torch
        reach = torch.as_tensor(reaches, dtype=torch.float32, device=self.device)
        bounty = torch.as_tensor(payoff_matrix, dtype=torch.float32, device=self.device)
        invested = torch.as_tensor(contributions, dtype=torch.float32, device=self.device)
        remaining = torch.as_tensor(stacks_rem, dtype=torch.float32, device=self.device)
        if (tuple(reach.shape) != (3, 1326) or tuple(bounty.shape) != (3, 3) or
                tuple(invested.shape) != (3,) or tuple(remaining.shape) != (3,)):
            raise ValueError("reaches must be (3,1326), contributions and stacks_rem (3,), payoff_matrix (3,3)")
        active = tuple(active_players)
        if len(active) not in (2, 3) or len(set(active)) != len(active) or any(p not in range(3) for p in active):
            raise ValueError("two or three valid active players are required")
        if (not torch.isfinite(reach).all() or torch.any(reach < 0) or
                not torch.isfinite(bounty).all() or torch.any(bounty < 0) or
                not torch.isfinite(invested).all() or torch.any(invested < 0) or
                not torch.isfinite(remaining).all() or torch.any(remaining < 0) or
                not np.isfinite(pot_bb) or pot_bb < 0):
            raise ValueError("reaches, bounties, pot, contributions and remaining stacks must be finite and non-negative")
        out = -invested.unsqueeze(1).expand(3, 1326).clone()
        pair_equities = {}
        for p in active:
            pair_values = []
            for q in active:
                if p == q:
                    continue
                weights = reach[q].unsqueeze(0) * self.card_removal
                numerator = torch.sum(self.equity * weights, dim=1)
                denominator = torch.sum(weights, dim=1) + self.epsilon
                pair_values.append(numerator / denominator)
            pair_equities[p] = pair_values
        if len(active) == 3:
            scores = {p: torch.prod(torch.stack(pair_equities[p]), dim=0) for p in active}
            normalizer = sum(scores.values()) + self.epsilon
        for p in active:
            equity = (pair_equities[p][0] if len(active) == 2
                      else scores[p] / normalizer)
            role = _eqr_role(p, [q for q in active if q != p], active, aggressor)
            ev = float(pot_bb) * torch.clamp(equity * EQR[role], min=0.0, max=1.0)
            spr = max((float(stacks_rem[q]) for q in active if q != p), default=0.0) / max(float(pot_bb), self.epsilon)
            stackoff = min(1.0, max(0.0, 1.0 / (1.0 + spr)))
            for q in active:
                if q != p and stacks_rem[p] >= stacks_rem[q]:
                    ev = ev + bounty[p, q] * stackoff / (len(active) - 1)
            out[p] = ev - invested[p]
        return out


def _eqr_role(player: int, opponents: list[int], active: tuple[int, ...], aggressor: int | None) -> str:
    if len(active) == 3:
        return "MULTIWAY_3WAY"
    in_position = is_ip(player, opponents[0])
    if aggressor == player:
        return "IP_AGGRESSOR" if in_position else "OOP_AGGRESSOR"
    return "IP_CALLER" if in_position else "OOP_CALLER"
