"""Run full-range vector DCFR and checkpoint strategy exports to SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .batch_exporter import BatchExporter
from .solver_dcfr import VectorDCFRSolver
from .space_ko_icm import elimination_payoffs


def state_id_for(stacks, tokens, payouts, payoff_scale) -> str:
    payload = json.dumps({"stacks": list(map(float, stacks)),
                          "tokens": list(map(float, tokens)),
                          "payouts": list(map(float, payouts)),
                          "token_utility_scale": float(payoff_scale)},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def run(equity_path: Path, db_path: Path, stacks, tokens, payouts,
        iterations: int, checkpoint_every: int, seed: int, resume: bool,
        nashconv_every: int = 0, payoff_scale: float = 1.0,
        device: str | None = None) -> tuple[str, dict | None]:
    if iterations < 1 or checkpoint_every < 1:
        raise ValueError("iterations and checkpoint_every must be positive")
    if nashconv_every < 0:
        raise ValueError("nashconv_every must be non-negative")
    if not np.isfinite(payoff_scale) or payoff_scale < 0:
        raise ValueError("payoff_scale must be finite and non-negative")
    equity = np.load(equity_path, mmap_mode="r")
    if equity.shape != (1326, 1326):
        raise ValueError("equity file must have shape (1326,1326)")
    payoff = elimination_payoffs(stacks, tokens, payouts)
    state_id = state_id_for(stacks, tokens, payouts, payoff_scale)
    exporter = BatchExporter(db_path)
    exporter.register_state(state_id, {
        "stacks_bb": list(map(float, stacks)),
        "token_evs_prize_units": list(map(float, tokens)),
        "prizepool_payouts": list(map(float, payouts)),
        "token_utility_scale": float(payoff_scale),
        "equity_matrix_sha256": hashlib.sha256(equity.tobytes()).hexdigest(),
        "solver": "vector-dcfr-v1",
    })
    solver = VectorDCFRSolver(stacks, equity, payoff, payoff_scale=payoff_scale,
                              prizepool_payouts=payouts, device=device)
    checkpoint = exporter.load_checkpoint(state_id) if resume else None
    if checkpoint:
        previous_iteration, payload = checkpoint
        solver.restore_checkpoint(payload)
        if solver.iteration != previous_iteration:
            raise ValueError("checkpoint iteration metadata does not match its payload")
    exporter.set_status(state_id, "RUNNING")
    metrics = None
    last_checkpoint_iteration = solver.iteration if checkpoint else -1
    while solver.iteration < iterations:
        count = min(checkpoint_every, iterations - solver.iteration)
        if nashconv_every:
            next_metric = ((solver.iteration // nashconv_every) + 1) * nashconv_every
            count = min(count, next_metric - solver.iteration)
        solver.solve(count)
        if solver.iteration % checkpoint_every == 0 or solver.iteration == iterations:
            exporter.save_checkpoint(state_id, solver.iteration, solver.checkpoint())
            last_checkpoint_iteration = solver.iteration
        if nashconv_every and solver.iteration % nashconv_every == 0:
            metrics = solver.estimate_nashconv()
            exporter.save_metrics(state_id, solver.iteration, metrics)
    exporter.export(solver, state_id)
    if last_checkpoint_iteration != solver.iteration:
        exporter.save_checkpoint(state_id, solver.iteration, solver.checkpoint())
    exporter.set_status(state_id, "DONE")
    if nashconv_every and (metrics is None or solver.iteration % nashconv_every != 0):
        metrics = solver.estimate_nashconv()
    if metrics is not None:
        exporter.save_metrics(state_id, solver.iteration, metrics)
    return state_id, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", required=True, type=Path, help="float32 equity .npy file")
    parser.add_argument("--db", type=Path, default=Path("space_ko_preflop.sqlite"))
    parser.add_argument("--stacks", nargs=3, type=float, required=True, metavar=("BTN", "SB", "BB"))
    parser.add_argument("--tokens", nargs=3, type=float, default=(0.02, 0.02, 0.02))
    parser.add_argument("--payouts", nargs=3, type=float, default=(0.6, 0.3, 0.1))
    parser.add_argument("--token-scale", "--payoff-to-bb", dest="token_scale", type=float, default=1.0,
                        help="convert KO token EVs to solver units (bb for WTA, payout units for multi-place ICM)")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7,
                        help="accepted for compatibility; vector DCFR is deterministic")
    parser.add_argument("--device", default=None,
                        help="PyTorch device (default: CUDA when available, otherwise CPU)")
    parser.add_argument("--resume", action="store_true", help="continue from the SQLite checkpoint")
    parser.add_argument("--nashconv-every", type=int, default=0,
                        help="calculate range-based imperfect-information NashConv at this iteration interval and at finish (0 disables)")
    args = parser.parse_args()
    state_id, metrics = run(args.equity, args.db, args.stacks, args.tokens, args.payouts,
                            args.iterations, args.checkpoint_every, args.seed, args.resume,
                            args.nashconv_every, args.token_scale, args.device)
    print(f"Saved vector DCFR state {state_id} after {args.iterations} iterations to {args.db}")
    if metrics is not None:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
