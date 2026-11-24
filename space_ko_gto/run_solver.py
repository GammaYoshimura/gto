"""Run sampled DCFR and checkpoint strategy exports to SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .batch_exporter import BatchExporter
from .solver_dcfr import MCCFRSolver
from .space_ko_icm import elimination_payoffs


def state_id_for(stacks, tokens, payouts, payoff_scale) -> str:
    payload = json.dumps({"stacks": list(map(float, stacks)),
                          "tokens": list(map(float, tokens)),
                          "payouts": list(map(float, payouts)),
                          "payoff_to_bb": float(payoff_scale)},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def run(equity_path: Path, db_path: Path, stacks, tokens, payouts,
        iterations: int, checkpoint_every: int, seed: int, resume: bool,
        nashconv_samples: int = 0, payoff_scale: float = 1.0) -> tuple[str, dict | None]:
    if iterations < 1 or checkpoint_every < 1:
        raise ValueError("iterations and checkpoint_every must be positive")
    if not np.isfinite(payoff_scale) or payoff_scale < 0:
        raise ValueError("payoff_scale must be finite and non-negative")
    equity = np.load(equity_path, mmap_mode="r")
    if equity.shape != (1326, 1326):
        raise ValueError("equity file must have shape (1326,1326)")
    payoff = elimination_payoffs(stacks, tokens, payouts)
    state_id = state_id_for(stacks, tokens, payouts, payoff_scale)
    exporter = BatchExporter(db_path)
    solver = MCCFRSolver(stacks, equity, payoff, seed=seed, payoff_scale=payoff_scale)
    checkpoint = exporter.load_checkpoint(state_id) if resume else None
    if checkpoint:
        previous_iteration, payload = checkpoint
        solver.restore_checkpoint(payload)
        if solver.iteration != previous_iteration:
            raise ValueError("checkpoint iteration metadata does not match its payload")
    exporter.set_status(state_id, "RUNNING")
    while solver.iteration < iterations:
        count = min(checkpoint_every, iterations - solver.iteration)
        solver.solve(count)
        exporter.export(solver, state_id)
        exporter.save_checkpoint(state_id, solver.iteration, solver.checkpoint())
        if nashconv_samples and solver.iteration % 200 == 0:
            metrics = solver.estimate_nashconv(nashconv_samples)
            exporter.save_metrics(state_id, solver.iteration, metrics)
    exporter.export(solver, state_id)
    exporter.save_checkpoint(state_id, solver.iteration, solver.checkpoint())
    exporter.set_status(state_id, "DONE")
    metrics = solver.estimate_nashconv(nashconv_samples) if nashconv_samples else None
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
    parser.add_argument("--payoff-to-bb", type=float, default=1.0,
                        help="convert prize-unit KO values to bb-equivalent utility units")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true", help="continue from the SQLite checkpoint")
    parser.add_argument("--nashconv-samples", type=int, default=0,
                        help="estimate NashConv on this many deals every 200 iterations and at finish (slow; 0 disables)")
    args = parser.parse_args()
    state_id, metrics = run(args.equity, args.db, args.stacks, args.tokens, args.payouts,
                            args.iterations, args.checkpoint_every, args.seed, args.resume,
                            args.nashconv_samples, args.payoff_to_bb)
    print(f"Saved sampled DCFR state {state_id} after {args.iterations} iterations to {args.db}")
    if metrics is not None:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
