"""Sanity checks for exported SQLite strategy nodes."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import zlib

import numpy as np

from .batch_exporter import decode_blob


def verify_database(db_path: str, tolerance: float = 0.01) -> list[str]:
    issues = []
    try:
        with sqlite3.connect(db_path) as db:
            rows = db.execute("SELECT state_id,node_path,actions_json,strategy_blob,ev_blob FROM nodes").fetchall()
            queue = db.execute("SELECT state_id,status FROM states_queue").fetchall()
            metadata_rows = db.execute("SELECT state_id,metadata_json FROM state_metadata").fetchall()
            checkpoints = db.execute("SELECT state_id,iteration,payload FROM solver_checkpoints").fetchall()
            metrics = db.execute("SELECT state_id,iteration,metric_json FROM solver_metrics").fetchall()
            errors = db.execute("SELECT state_id,error_text FROM state_errors").fetchall()
    except sqlite3.Error as exc:
        return [f"SQLite schema or read error: {exc}"]

    node_counts = {}
    node_action_counts = {}
    for state_id, node_path, actions_json, strategy_blob, ev_blob in rows:
        label = f"{state_id}/{node_path}"
        node_counts[state_id] = node_counts.get(state_id, 0) + 1
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
        node_action_counts[(state_id, node_path)] = len(actions)
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

    for state_id, status in queue:
        if status not in ("PENDING", "RUNNING", "DONE", "FAILED"):
            issues.append(f"{state_id}: unknown queue status {status!r}")
        if status == "DONE" and node_counts.get(state_id, 0) == 0:
            issues.append(f"{state_id}: marked DONE with no exported nodes")
    statuses = dict(queue)
    for state_id, metadata_json in metadata_rows:
        try:
            metadata = json.loads(metadata_json)
            for key, length in (("stacks_bb", 3), ("token_evs_prize_units", 3),
                                ("prizepool_payouts", 3)):
                values = metadata[key]
                if len(values) != length or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                    issues.append(f"{state_id}: invalid metadata field {key}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"{state_id}: invalid state metadata ({exc})")
    for state_id, error_text in errors:
        if statuses.get(state_id) != "FAILED":
            issues.append(f"{state_id}: stored error without FAILED status")
        if not error_text:
            issues.append(f"{state_id}: empty stored error")

    for state_id, iteration, payload in checkpoints:
        label = f"{state_id}@{iteration}"
        try:
            checkpoint = json.loads(zlib.decompress(payload))
            if int(checkpoint.get("iteration", -1)) != iteration:
                issues.append(f"{label}: iteration metadata mismatch")
            for required in ("config", "rng", "regrets", "regret_last_t",
                             "strategy_sums", "ev_sums", "row_visits"):
                if required not in checkpoint:
                    issues.append(f"{label}: checkpoint missing {required}")
            row_keys = set()
            for table_name in ("regrets", "strategy_sums", "ev_sums"):
                for row in checkpoint.get(table_name, []):
                    if not isinstance(row, list) or len(row) != 3:
                        issues.append(f"{label}: malformed row in {table_name}")
                        continue
                    key, combo, values = row
                    node_actions = node_action_counts.get((state_id, key))
                    if not isinstance(combo, int) or not 0 <= combo < 1326:
                        issues.append(f"{label}: invalid combo index in {table_name}")
                    if node_actions is not None and (not isinstance(values, list) or len(values) != node_actions):
                        issues.append(f"{label}: action dimension mismatch in {table_name} at {key}")
                    if not isinstance(values, list) or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                        issues.append(f"{label}: non-finite or invalid values in {table_name} at {key}")
                    row_keys.add((table_name, key, combo))
            for key, combo, last_t in checkpoint.get("regret_last_t", []):
                if not isinstance(combo, int) or not 0 <= combo < 1326 or not isinstance(last_t, int) or not 0 <= last_t <= iteration:
                    issues.append(f"{label}: invalid regret discount timestamp at {key}")
                if ("regrets", key, combo) not in row_keys:
                    issues.append(f"{label}: discount timestamp has no regret row at {key}")
            for key, combo, visits in checkpoint.get("row_visits", []):
                if (not isinstance(combo, int) or not 0 <= combo < 1326 or
                        not math.isfinite(float(visits)) or float(visits) < 0):
                    issues.append(f"{label}: invalid row visit count at {key}")
        except (ValueError, zlib.error, json.JSONDecodeError, AttributeError, TypeError) as exc:
            issues.append(f"{label}: invalid compressed checkpoint ({exc})")

    for state_id, iteration, metric_json in metrics:
        label = f"{state_id}@{iteration}"
        try:
            metric = json.loads(metric_json)
            value = float(metric["clairvoyant_exploitability_upper_bound_bb_per_deal"])
            if not math.isfinite(value) or value < 0:
                issues.append(f"{label}: exploitability upper bound must be finite and non-negative")
            if metric.get("metric") != "sampled_clairvoyant_exploitability_upper_bound":
                issues.append(f"{label}: unsupported exploitability metric label")
            if int(metric["samples"]) < 1:
                issues.append(f"{label}: sample count must be positive")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            issues.append(f"{label}: invalid metrics ({exc})")
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
