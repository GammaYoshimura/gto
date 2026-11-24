"""Solve a selected grid of stack, token and payout states sequentially."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from .run_solver import run
from .state_space import PRIZE_STRUCTURES, STACK_BUCKETS_BB, TOKEN_PROFILES


def run_batch(equity_path: Path, db_path: Path, stack_buckets, token_profiles,
              prize_structures, iterations: int, checkpoint_every: int, seed: int,
              resume: bool, payoff_scale: float, limit: int | None = None) -> int:
    configurations = product(product(stack_buckets, repeat=3), token_profiles, prize_structures)
    completed = 0
    for stacks, token_name, prize_name in configurations:
        if limit is not None and completed >= limit:
            break
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
        completed += 1
        print(f"[{completed}] {state_id} stacks={stacks} tokens={token_name} prizes={prize_name}")
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
    parser.add_argument("--payoff-to-bb", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="maximum number of states in this invocation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    count = run_batch(args.equity, args.db, args.stack_buckets, args.token_profiles,
                      args.prize_structures, args.iterations, args.checkpoint_every,
                      args.seed, args.resume, args.payoff_to_bb, args.limit)
    print(f"Processed {count} states into {args.db}")


if __name__ == "__main__":
    main()
