"""Sanity checks for exported SQLite strategy nodes."""

from __future__ import annotations

import argparse
import json
import sqlite3

import numpy as np

from .batch_exporter import decode_blob


def verify_database(db_path: str, tolerance: float = 0.01) -> list[str]:
    issues = []
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT state_id,node_path,actions_json,strategy_blob,ev_blob FROM nodes").fetchall()
    for state_id, node_path, actions_json, strategy_blob, ev_blob in rows:
        label = f"{state_id}/{node_path}"
        try:
            actions = json.loads(actions_json)
        except (TypeError, json.JSONDecodeError):
            issues.append(f"{label}: invalid actions_json")
            continue
        if not isinstance(actions, list) or not actions or any(not isinstance(action, str) for action in actions):
            issues.append(f"{label}: actions_json must be a non-empty string list")
            continue
        if len(set(actions)) != len(actions):
            issues.append(f"{label}: duplicate action labels")
            continue
        shape = (1326, len(actions))
        expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
        if len(strategy_blob) != expected_bytes or len(ev_blob) != expected_bytes:
            issues.append(f"{label}: strategy or EV blob has incorrect byte length")
            continue
        strategy, ev = decode_blob(strategy_blob, shape), decode_blob(ev_blob, shape)
        if not np.isfinite(strategy).all() or not np.isfinite(ev).all():
            issues.append(f"{label}: NaN or Inf")
            continue
        sums = strategy.sum(axis=1)
        if np.any(strategy < -tolerance) or np.any(strategy > 1 + tolerance):
            issues.append(f"{label}: strategy frequency outside [0,1]")
        if np.any(np.abs(sums - 1.0) > tolerance):
            issues.append(f"{label}: strategy rows do not sum to 1")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    args = parser.parse_args()
    issues = verify_database(args.database)
    if issues:
        print("\n".join(issues))
        raise SystemExit(1)
    print("Database strategy blobs passed structural checks")


if __name__ == "__main__":
    main()
