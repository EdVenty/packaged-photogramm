"""SQLite persistence layer (single process, thread-safe via a lock)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    preset TEXT,
    params TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stages (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    ord INTEGER NOT NULL,
    deps TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    worker_id TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    started_at REAL,
    finished_at REAL,
    heartbeat_at REAL
);
CREATE INDEX IF NOT EXISTS idx_stages_job ON stages(job_id);
CREATE INDEX IF NOT EXISTS idx_stages_status ON stages(status, kind);
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    kinds TEXT NOT NULL,
    hostname TEXT NOT NULL DEFAULT '',
    info TEXT NOT NULL DEFAULT '{}',
    last_seen REAL NOT NULL,
    current_stage TEXT
);
"""

ACTIVE = ("pending", "running")
RETRYABLE = ("failed", "cancelled", "skipped")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        rows = self.query(sql, args)
        return rows[0] if rows else None

    # ---------------- datasets ----------------
    def create_dataset(self, name: str) -> str:
        did = new_id()
        with self.tx() as c:
            c.execute("INSERT INTO datasets(id,name,created_at) VALUES(?,?,?)",
                      (did, name, time.time()))
        return did

    def delete_dataset(self, did: str) -> None:
        with self.tx() as c:
            job_ids = [r["id"] for r in c.execute("SELECT id FROM jobs WHERE dataset_id=?", (did,))]
            for jid in job_ids:
                c.execute("DELETE FROM stages WHERE job_id=?", (jid,))
            c.execute("DELETE FROM jobs WHERE dataset_id=?", (did,))
            c.execute("DELETE FROM datasets WHERE id=?", (did,))

    # ---------------- jobs ----------------
    def create_job(self, name: str, dataset_id: str, pipeline: str, preset: str | None,
                   params: dict[str, Any], stages: list[dict[str, Any]]) -> str:
        jid = new_id()
        now = time.time()
        with self.tx() as c:
            c.execute("INSERT INTO jobs(id,name,dataset_id,pipeline,preset,params,created_at) "
                      "VALUES(?,?,?,?,?,?,?)",
                      (jid, name, dataset_id, pipeline, preset, json.dumps(params), now))
            for i, s in enumerate(stages):
                c.execute("INSERT INTO stages(id,job_id,name,kind,ord,deps,status) "
                          "VALUES(?,?,?,?,?,?, 'pending')",
                          (new_id(), jid, s["name"], s["kind"], i, json.dumps(s["deps"])))
        return jid

    def job_stages(self, jid: str, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM stages WHERE job_id=? ORDER BY ord"
        rows = (conn.execute(sql, (jid,)).fetchall() if conn else self.query(sql, (jid,)))
        out = []
        for r in rows:
            d = dict(r)
            d["deps"] = json.loads(d["deps"])
            d["cancel_requested"] = bool(d["cancel_requested"])
            out.append(d)
        return out

    @staticmethod
    def _descendants(stages: list[dict[str, Any]], name: str) -> set[str]:
        out: set[str] = set()
        frontier = {name}
        while frontier:
            nxt = {s["name"] for s in stages if set(s["deps"]) & frontier} - out
            out |= nxt
            frontier = nxt
        return out

    def _cascade(self, c: sqlite3.Connection, jid: str, stage_name: str, new_status: str,
                 reason: str) -> None:
        stages = self.job_stages(jid, c)
        for d in self._descendants(stages, stage_name):
            c.execute("UPDATE stages SET status=?, message=?, finished_at=? "
                      "WHERE job_id=? AND name=? AND status='pending'",
                      (new_status, reason, time.time(), jid, d))

    def cancel_job(self, jid: str) -> None:
        now = time.time()
        with self.tx() as c:
            c.execute("UPDATE stages SET status='cancelled', message='cancelled by user', "
                      "finished_at=? WHERE job_id=? AND status='pending'", (now, jid))
            c.execute("UPDATE stages SET cancel_requested=1 WHERE job_id=? AND status='running'",
                      (jid,))

    def retry_job(self, jid: str) -> int:
        with self.tx() as c:
            cur = c.execute(
                "UPDATE stages SET status='pending', progress=0, message='', error='', "
                "worker_id=NULL, cancel_requested=0, started_at=NULL, finished_at=NULL, "
                "heartbeat_at=NULL WHERE job_id=? AND status IN (?,?,?)", (jid, *RETRYABLE))
            return cur.rowcount

    def delete_job(self, jid: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM stages WHERE job_id=?", (jid,))
            c.execute("DELETE FROM jobs WHERE id=?", (jid,))

    # ---------------- workers / scheduling ----------------
    def worker_heartbeat(self, wid: str, kinds: list[str], hostname: str,
                         info: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO workers(id,kinds,hostname,info,last_seen) VALUES(?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET kinds=excluded.kinds, hostname=excluded.hostname, "
                "info=excluded.info, last_seen=excluded.last_seen",
                (wid, json.dumps(kinds), hostname, json.dumps(info), time.time()))

    def claim(self, wid: str, kinds: list[str]) -> dict[str, Any] | None:
        if not kinds:
            return None
        now = time.time()
        marks = ",".join("?" * len(kinds))
        with self.tx() as c:
            # A worker owns at most one stage: anything still "running" under its id is stale
            # (worker restarted mid-stage).
            stale = c.execute("SELECT id, job_id, name FROM stages "
                              "WHERE worker_id=? AND status='running'", (wid,)).fetchall()
            for s in stale:
                c.execute("UPDATE stages SET status='failed', error='worker restarted', "
                          "finished_at=? WHERE id=?", (now, s["id"]))
                self._cascade(c, s["job_id"], s["name"], "skipped", "dependency failed")
            rows = c.execute(
                f"SELECT s.* FROM stages s JOIN jobs j ON j.id=s.job_id "
                f"WHERE s.status='pending' AND s.kind IN ({marks}) "
                f"ORDER BY j.created_at, s.ord", tuple(kinds)).fetchall()
            for r in rows:
                deps = json.loads(r["deps"])
                if deps:
                    st = {x["name"]: x["status"] for x in c.execute(
                        "SELECT name,status FROM stages WHERE job_id=?", (r["job_id"],))}
                    if any(st.get(d) != "done" for d in deps):
                        continue
                c.execute("UPDATE stages SET status='running', worker_id=?, started_at=?, "
                          "heartbeat_at=?, progress=0, message='started', error='' WHERE id=?",
                          (wid, now, now, r["id"]))
                c.execute("UPDATE workers SET current_stage=?, last_seen=? WHERE id=?",
                          (r["id"], now, wid))
                return dict(r)
        return None

    def stage_heartbeat(self, sid: str, wid: str, progress: float | None,
                        message: str | None) -> dict[str, Any] | None:
        now = time.time()
        with self.tx() as c:
            r = c.execute("SELECT * FROM stages WHERE id=?", (sid,)).fetchone()
            if r is None:
                return None
            if r["status"] == "running" and r["worker_id"] == wid:
                c.execute("UPDATE stages SET heartbeat_at=?, progress=COALESCE(?,progress), "
                          "message=COALESCE(?,message) WHERE id=?",
                          (now, progress, message, sid))
                c.execute("UPDATE workers SET last_seen=? WHERE id=?", (now, wid))
            return dict(r)

    def finish_stage(self, sid: str, wid: str, status: str, error: str,
                     message: str | None) -> bool:
        now = time.time()
        with self.tx() as c:
            r = c.execute("SELECT * FROM stages WHERE id=?", (sid,)).fetchone()
            if r is None or r["status"] != "running" or r["worker_id"] != wid:
                return False
            c.execute("UPDATE stages SET status=?, error=?, message=COALESCE(?,message), "
                      "finished_at=?, heartbeat_at=?, progress=CASE WHEN ?='done' THEN 1 "
                      "ELSE progress END WHERE id=?",
                      (status, error, message, now, now, status, sid))
            c.execute("UPDATE workers SET current_stage=NULL WHERE id=?", (wid,))
            if status == "failed":
                self._cascade(c, r["job_id"], r["name"], "skipped", "dependency failed")
            elif status == "cancelled":
                self._cascade(c, r["job_id"], r["name"], "cancelled", "cancelled by user")
            return True

    def expire_stale(self, lease_sec: float) -> int:
        cutoff = time.time() - lease_sec
        n = 0
        with self.tx() as c:
            rows = c.execute("SELECT * FROM stages WHERE status='running' AND heartbeat_at<?",
                             (cutoff,)).fetchall()
            for r in rows:
                c.execute("UPDATE stages SET status='failed', finished_at=?, "
                          "error='worker lost (no heartbeat)' WHERE id=?", (time.time(), r["id"]))
                self._cascade(c, r["job_id"], r["name"], "skipped", "dependency failed")
                n += 1
        return n


def job_status(stages: list[dict[str, Any]]) -> str:
    sts = [s["status"] for s in stages]
    if not sts:
        return "done"
    if any(s == "running" for s in sts):
        return "running"
    if all(s == "done" for s in sts):
        return "done"
    if any(s == "pending" for s in sts):
        return "running" if any(s != "pending" for s in sts) else "queued"
    if any(s == "failed" for s in sts):
        return "failed"
    if any(s == "cancelled" for s in sts):
        return "cancelled"
    return "failed"


def job_progress(stages: list[dict[str, Any]]) -> float:
    weights = {"preprocess": 0.5, "sfm": 2.0, "mvs": 3.0, "splat": 3.0}
    total = sum(weights.get(s["name"], 1.0) for s in stages) or 1.0
    acc = 0.0
    for s in stages:
        w = weights.get(s["name"], 1.0)
        p = 1.0 if s["status"] == "done" else float(s["progress"] or 0)
        acc += w * max(0.0, min(1.0, p))
    return round(acc / total, 4)
