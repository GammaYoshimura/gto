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

    def list_states(self, status: str | None = None) -> list[tuple[str, str]]:
        with self._connect() as db:
            if status is None:
                rows = db.execute("SELECT state_id,status FROM states_queue ORDER BY state_id").fetchall()
            else:
                rows = db.execute("SELECT state_id,status FROM states_queue WHERE status=? ORDER BY state_id",
                                  (status,)).fetchall()
        return [(str(state_id), str(state_status)) for state_id, state_status in rows]

    def list_node_paths(self, state_id: str) -> list[str]:
        with self._connect() as db:
            rows = db.execute("SELECT node_path FROM nodes WHERE state_id=? ORDER BY node_path",
                              (state_id,)).fetchall()
        return [str(row[0]) for row in rows]

    def load_node(self, state_id: str, node_path: str):
        """Return hero position, ordered actions, strategy and EV arrays, or None."""
        with self._connect() as db:
            row = db.execute("SELECT hero_pos,actions_json,strategy_blob,ev_blob FROM nodes "
                             "WHERE state_id=? AND node_path=?", (state_id, node_path)).fetchone()
        if row is None:
            return None
        hero_pos, actions_json, strategy_blob, ev_blob = row
        actions = json.loads(actions_json)
        shape = (1326, len(actions))
        return (int(hero_pos), actions, decode_blob(strategy_blob, shape), decode_blob(ev_blob, shape))

    def combo_strategy(self, state_id: str, node_path: str, combo_index: int):
        """Return action probabilities for one combo as an ordered mapping."""
        if not 0 <= combo_index < 1326:
            raise IndexError(combo_index)
        node = self.load_node(state_id, node_path)
        if node is None:
            return None
        _, actions, strategy, _ = node
        return dict(zip(actions, map(float, strategy[combo_index])))


def decode_blob(blob: bytes, shape: tuple[int, int]) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).reshape(shape).astype(np.float32)
