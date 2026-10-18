"""Durable local records. Every write uses a fresh SQLite transaction."""
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from . import config


@contextmanager
def db():
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.WORK_DIR / "chiptrick.sqlite3", timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    except BaseException:
        c.rollback()
        raise
    finally:
        c.close()


def init():
    with db() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript('''
        CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','admin')), created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id TEXT REFERENCES users(id), expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, owner TEXT NOT NULL REFERENCES users(id),
            filename TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, created REAL NOT NULL,
            started REAL, finished REAL, status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL DEFAULT 0,
            version TEXT, result TEXT, error TEXT, training_allowed INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS feedback(job_id TEXT PRIMARY KEY REFERENCES jobs(id),
            user_id TEXT NOT NULL REFERENCES users(id), created REAL NOT NULL, verdict TEXT NOT NULL,
            events TEXT NOT NULL, note TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS feedback_immutable_update BEFORE UPDATE ON feedback
            BEGIN SELECT RAISE(ABORT, 'feedback is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS feedback_immutable_delete BEFORE DELETE ON feedback
            BEGIN SELECT RAISE(ABORT, 'feedback is immutable'); END;
        CREATE TABLE IF NOT EXISTS reviews(job_id TEXT PRIMARY KEY REFERENCES feedback(job_id),
            reviewer TEXT NOT NULL REFERENCES users(id), status TEXT NOT NULL, note TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS versions(id TEXT PRIMARY KEY, created REAL NOT NULL,
            parent TEXT, status TEXT NOT NULL, manifest TEXT NOT NULL, report TEXT, artifact TEXT);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS releases(id INTEGER PRIMARY KEY, created REAL NOT NULL,
            actor TEXT NOT NULL, previous TEXT, version TEXT NOT NULL, reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS training_runs(id TEXT PRIMARY KEY, created REAL NOT NULL,
            status TEXT NOT NULL, epoch INTEGER DEFAULT 0, epochs INTEGER DEFAULT 0,
            loss REAL, version TEXT, error TEXT);
        ''')
        if 'training_allowed' not in {r['name'] for r in c.execute('PRAGMA table_info(jobs)')}:
            c.execute('ALTER TABLE jobs ADD COLUMN training_allowed INTEGER NOT NULL DEFAULT 0')
        if 'pid' not in {r['name'] for r in c.execute('PRAGMA table_info(training_runs)')}:
            c.execute('ALTER TABLE training_runs ADD COLUMN pid INTEGER')


def one(sql, args=()):
    with db() as c:
        r = c.execute(sql, args).fetchone()
        return dict(r) if r else None


def many(sql, args=()):
    with db() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def execute(sql, args=()):
    with db() as c:
        c.execute(sql, args)


def unpack(row):
    if row:
        for key in ("result", "events", "manifest", "report"):
            if key in row and row[key] is not None:
                row[key] = json.loads(row[key])
    return row


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
