import hashlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import httpx
from dotenv import dotenv_values, find_dotenv

from .data import rows, unpack
from .providers import headers, read_answer, request


def wire(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def records(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def run(profile, source, revision, out, workers, limit):
    import fcntl

    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("Workers and sample limit must be positive")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock, closing(sqlite3.connect(out / "responses.sqlite")) as db:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {"profile": asdict(profile), "source": source, "revision": revision}
        db.execute("CREATE TABLE IF NOT EXISTS config (value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS answers (id TEXT PRIMARY KEY, hash TEXT, record TEXT)")
        previous = db.execute("SELECT value FROM config").fetchone()
        if previous and json.loads(previous[0]) != config:
            raise ValueError("Output directory belongs to another model or dataset configuration")
        if not previous:
            db.execute("INSERT INTO config VALUES (?)", (json.dumps(config),))
            db.commit()
        (out / "run.json").write_text(json.dumps(config, indent=2) + "\n")
        key = os.environ.get(profile.key_env) if profile.key_env else None
        if profile.key_env and not key:
            env_file = find_dotenv(usecwd=True)
            if env_file:
                key = dotenv_values(env_file, interpolate=False).get(profile.key_env)

        def infer(sid, path, payload):
            if profile.key_env and not key:
                raise ValueError(f"Add {profile.key_env} to .env to run this profile")
            response = client.post(profile.base_url.rstrip("/") + path, content=payload)
            response.raise_for_status()
            value = response.json()
            if value.get("error"):
                raise RuntimeError(f"Provider returned an error for {sid}")
            return {"id": sid, **read_answer(profile.api, value)}

        selected = []
        pending = []
        def save_next():
            sid, digest, future = pending.pop(0)
            record = future.result()
            db.execute("INSERT INTO answers VALUES (?, ?, ?)", (sid, digest, json.dumps(record)))
            db.commit()

        with httpx.Client(headers={"Content-Type": "application/json", **headers(profile.api, key or "")},
                          timeout=1800) as client, ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                for row in islice(rows(source, revision), limit):
                    sid = row["id"]
                    selected.append(sid)
                    path, body = request(profile, *unpack(row))
                    payload = wire(body)
                    digest = hashlib.sha256(payload).hexdigest()
                    cached = db.execute("SELECT hash FROM answers WHERE id=?", (sid,)).fetchone()
                    if cached:
                        if cached[0] != digest:
                            raise ValueError(f"Request changed for saved action {sid}")
                        continue
                    pending.append((sid, digest, pool.submit(infer, sid, path, payload)))
                    if len(pending) >= workers:
                        save_next()
            finally:
                # Preserve successes already in flight when an endpoint fails.
                first_error = None
                while pending:
                    try:
                        save_next()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                if first_error:
                    raise first_error
        if not selected:
            raise ValueError("Empty input split")
        if db.execute("SELECT COUNT(*) FROM answers").fetchone()[0] > len(selected):
            raise ValueError("This run contains more actions. Use the full split or a new output directory.")
        tmp = out / "predictions.jsonl.tmp"
        with tmp.open("w") as f:
            for sid in selected:
                record, = db.execute("SELECT record FROM answers WHERE id=?", (sid,)).fetchone()
                f.write(record + "\n")
        tmp.replace(out / "predictions.jsonl")
        return len(selected)
