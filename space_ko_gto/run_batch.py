"""Solve a selected grid of stack, token and payout states sequentially."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from .batch_exporter import BatchExporter
from .run_solver import run, state_id_for
from .state_space import PRIZE_STRUCTURES, STACK_BUCKETS_BB, TOKEN_PROFILES


def run_batch(equity_path: Path, db_path: Path, stack_buckets, token_profiles,
              prize_structures, iterations: int, checkpoint_every: int, seed: int,
              resume: bool, payoff_scale: float, limit: int | None = None) -> int:
    configurations = product(product(stack_buckets, repeat=3), token_profiles, prize_structures)
    completed = 0
    skipped = 0
    failed = 0
    selected = 0
    done_states = {state_id for state_id, status in BatchExporter(db_path).list_states("DONE")}
    for stacks, token_name, prize_name in configurations:
        if limit is not None and selected >= limit:
            break
        selected += 1
        state_id = state_id_for(stacks, TOKEN_PROFILES[token_name],
                                PRIZE_STRUCTURES[prize_name], payoff_scale)
        if resume and state_id in done_states:
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
