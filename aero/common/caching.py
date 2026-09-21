"""Persistent polar cache: RL must never re-pay for a CFD eval it already made.

Keyed by (airfoil-coordinate hash, Re, alpha, backend) -> (cl, cd, cm).
SQLite at aero/results/polar_cache.db.
"""
import hashlib
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS polar (
    key     TEXT PRIMARY KEY,
    cl      REAL NOT NULL,
    cd      REAL NOT NULL,
    cm      REAL NOT NULL,
    backend TEXT NOT NULL
)
"""


def polar_key(x, y, re, alpha, backend):
    h = hashlib.sha1()
    h.update(np_bytes(x))
    h.update(np_bytes(y))
    h.update(f"{re:.0f}|{alpha:.3f}|{backend}".encode())
    return h.hexdigest()


def np_bytes(a):
    import numpy as np
    return np.ascontiguousarray(a, dtype=np.float32).tobytes()


class PolarCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute(_SCHEMA)
        self._db.commit()

    def get(self, key):
        row = self._db.execute(
            "SELECT cl, cd, cm, backend FROM polar WHERE key = ?", (key,)
        ).fetchone()
        return row if row is not None else None

    def put(self, key, cl, cd, cm, backend):
        self._db.execute(
            "INSERT OR REPLACE INTO polar (key, cl, cd, cm, backend) VALUES (?,?,?,?,?)",
            (key, float(cl), float(cd), float(cm), backend),
        )
        self._db.commit()

    def __len__(self):
        return self._db.execute("SELECT COUNT(*) FROM polar").fetchone()[0]
