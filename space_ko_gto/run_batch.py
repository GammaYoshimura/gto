"""Solve a selected grid of stack, token and payout states sequentially."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from .batch_exporter import BatchExporter
from .run_solver import run, state_id_for
from .state_space import PRIZE_STRUCTURES, STACK_BUCKETS_BB, TOKEN_PROFILES


def run_batch(equity_path: Path, db_path: Path, stack_buckets, token_profiles,
              prize_structures, iterations: int, checkpoint_every: int, seed: int,
              resume: bool, payoff_scale: float, limit: int | None = None) -> int:
    stack_buckets = tuple(stack_buckets)
    token_profiles = tuple(token_profiles)
    prize_structures = tuple(prize_structures)
    if iterations < 1 or checkpoint_every < 1:
        raise ValueError("iterations and checkpoint_every must be positive")
    if not np.isfinite(payoff_scale) or payoff_scale < 0:
        raise ValueError("payoff_scale must be finite and non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if not stack_buckets or any(stack < 1 for stack in stack_buckets):
        raise ValueError("stack_buckets must contain positive stacks")
    if not token_profiles or not prize_structures:
        raise ValueError("at least one token profile and prize structure are required")
    configurations = product(product(stack_buckets, repeat=3), token_profiles, prize_structures)
    completed = 0
    skipped = 0
    failed = 0
    selected = 0
    progress = BatchExporter(db_path).state_progress()
    for stacks, token_name, prize_name in configurations:
        if limit is not None and selected >= limit:
            break
        selected += 1
        state_id = state_id_for(stacks, TOKEN_PROFILES[token_name],
                                PRIZE_STRUCTURES[prize_name], payoff_scale)
        prior_status, prior_iteration = progress.get(state_id, ("PENDING", 0))
        if resume and prior_status == "DONE" and prior_iteration >= iterations:
            skipped += 1
            continue
        try:
            state_id, _ = run(
                equity_path=equity_path,
                db_path=db_path,
                stacks=stacks,
                tokens=TOKEN_PROFILES[token_name],
                payouts=PRIZE_STRUCTURES[prize_name],
                iterations=iterations,
                checkpoint_every=checkpoint_every,
                seed=seed,
                resume=resume,
                payoff_scale=payoff_scale,
            )
        except Exception as exc:
            failed += 1
            exporter = BatchExporter(db_path)
            exporter.set_status(state_id, "FAILED")
            exporter.save_error(state_id, f"{type(exc).__name__}: {exc}")
            print(f"FAILED {state_id} stacks={stacks}: {type(exc).__name__}: {exc}")
            continue
        completed += 1
        print(f"[{completed}] {state_id} stacks={stacks} tokens={token_name} prizes={prize_name}")
    if skipped:
        print(f"Skipped {skipped} states already marked DONE")
    if failed:
        print(f"Failed {failed} states; inspect state_errors in {db_path}")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", required=True, type=Path)
    parser.add_argument("--db", type=Path, default=Path("space_ko_preflop.sqlite"))
    parser.add_argument("--stack-buckets", nargs="+", type=int, default=STACK_BUCKETS_BB)
    parser.add_argument("--token-profiles", nargs="+", choices=tuple(TOKEN_PROFILES),
                        default=tuple(TOKEN_PROFILES))
    parser.add_argument("--prize-structures", nargs="+", choices=tuple(PRIZE_STRUCTURES),
                        default=tuple(PRIZE_STRUCTURES))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--token-scale", "--payoff-to-bb", dest="token_scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="maximum number of states in this invocation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    count = run_batch(args.equity, args.db, args.stack_buckets, args.token_profiles,
                      args.prize_structures, args.iterations, args.checkpoint_every,
                      args.seed, args.resume, args.token_scale, args.limit)
    print(f"Processed {count} states into {args.db}")


if __name__ == "__main__":
    main()
