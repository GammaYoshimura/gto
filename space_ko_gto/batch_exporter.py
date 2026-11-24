"""Portable SQLite queue and node strategy export helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS states_queue (
    state_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'PENDING'
);
CREATE TABLE IF NOT EXISTS nodes (
    state_id TEXT NOT NULL,
    node_path TEXT NOT NULL,
    hero_pos INTEGER NOT NULL,
    actions_json TEXT NOT NULL,
    strategy_blob BLOB NOT NULL,
    ev_blob BLOB NOT NULL,
    PRIMARY KEY (state_id, node_path),
    FOREIGN KEY (state_id) REFERENCES states_queue(state_id)
);
CREATE TABLE IF NOT EXISTS solver_checkpoints (
    state_id TEXT PRIMARY KEY,
    iteration INTEGER NOT NULL,
    payload BLOB NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (state_id) REFERENCES states_queue(state_id)
);
CREATE TABLE IF NOT EXISTS solver_metrics (
    state_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    metric_json TEXT NOT NULL,
    PRIMARY KEY (state_id, iteration),
    FOREIGN KEY (state_id) REFERENCES states_queue(state_id)
);
"""


class BatchExporter:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self):
        db = sqlite3.connect(self.db_path)
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def enqueue(self, state_ids) -> None:
        with self._connect() as db:
            db.executemany("INSERT OR IGNORE INTO states_queue(state_id) VALUES (?)",
                           ((str(state_id),) for state_id in state_ids))

    def export_node(self, state_id: str, node_path: str, hero_pos: int,
                    actions: list[str], strategy: np.ndarray, ev: np.ndarray) -> None:
        strategy = np.asarray(strategy, dtype=np.float16, order="C")
        ev = np.asarray(ev, dtype=np.float16, order="C")
        if strategy.ndim != 2 or ev.shape != strategy.shape or strategy.shape[1] != len(actions):
            raise ValueError("strategy and EV must share shape (1326, number of actions)")
        if strategy.shape[0] != 1326 or hero_pos not in (0, 1, 2):
            raise ValueError("expected 1326 combos and hero_pos in {0,1,2}")
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO states_queue(state_id) VALUES (?)", (state_id,))
            db.execute("INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
                       (state_id, node_path, hero_pos, json.dumps(actions),
                        strategy.tobytes(), ev.tobytes()))

    def set_status(self, state_id: str, status: str) -> None:
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO states_queue(state_id) VALUES (?)", (state_id,))
            db.execute("UPDATE states_queue SET status=? WHERE state_id=?", (status, state_id))

    def save_checkpoint(self, state_id: str, iteration: int, payload: bytes) -> None:
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO states_queue(state_id) VALUES (?)", (state_id,))
            db.execute("INSERT OR REPLACE INTO solver_checkpoints(state_id,iteration,payload) VALUES (?,?,?)",
                       (state_id, int(iteration), sqlite3.Binary(payload)))

    def load_checkpoint(self, state_id: str) -> tuple[int, bytes] | None:
        with self._connect() as db:
            row = db.execute("SELECT iteration,payload FROM solver_checkpoints WHERE state_id=?", (state_id,)).fetchone()
        return None if row is None else (int(row[0]), bytes(row[1]))

    def save_metrics(self, state_id: str, iteration: int, metrics: dict) -> None:
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO states_queue(state_id) VALUES (?)", (state_id,))
            db.execute("INSERT OR REPLACE INTO solver_metrics(state_id,iteration,metric_json) VALUES (?,?,?)",
                       (state_id, int(iteration), json.dumps(metrics, sort_keys=True)))


def decode_blob(blob: bytes, shape: tuple[int, int]) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).reshape(shape).astype(np.float32)
