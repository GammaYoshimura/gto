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


def _state_id(stacks, tokens, payouts) -> str:
    payload = json.dumps({"stacks": stacks, "tokens": tokens, "payouts": payouts},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def run(equity_path: Path, db_path: Path, stacks, tokens, payouts,
        iterations: int, checkpoint_every: int, seed: int, resume: bool) -> str:
    if iterations < 1 or checkpoint_every < 1:
        raise ValueError("iterations and checkpoint_every must be positive")
    equity = np.load(equity_path, mmap_mode="r")
    if equity.shape != (1326, 1326):
        raise ValueError("equity file must have shape (1326,1326)")
    payoff = elimination_payoffs(stacks, tokens, payouts)
    state_id = _state_id(stacks, tokens, payouts)
    exporter = BatchExporter(db_path)
    solver = MCCFRSolver(stacks, equity, payoff, seed=seed)
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
    exporter.export(solver, state_id)
    exporter.save_checkpoint(state_id, solver.iteration, solver.checkpoint())
    exporter.set_status(state_id, "DONE")
    return state_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", required=True, type=Path, help="float32 equity .npy file")
    parser.add_argument("--db", type=Path, default=Path("space_ko_preflop.sqlite"))
    parser.add_argument("--stacks", nargs=3, type=float, required=True, metavar=("BTN", "SB", "BB"))
    parser.add_argument("--tokens", nargs=3, type=float, default=(0.02, 0.02, 0.02))
    parser.add_argument("--payouts", nargs=3, type=float, default=(0.6, 0.3, 0.1))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true", help="continue from the SQLite checkpoint")
    args = parser.parse_args()
    state_id = run(args.equity, args.db, args.stacks, args.tokens, args.payouts,
                   args.iterations, args.checkpoint_every, args.seed, args.resume)
    print(f"Saved sampled DCFR state {state_id} after {args.iterations} iterations to {args.db}")


if __name__ == "__main__":
    main()
