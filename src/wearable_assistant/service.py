"""SQLite session journal and bounded local HTTP/API entry point."""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import uuid
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from wearable_tasks.contract import canonical
from .backends import HTTPBackend, LocalBackend
from .runtime import CONTROLLER_REVISION, Session, token_usage


class SessionStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=20)
        self.lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS requests(scope TEXT, request_id TEXT, body_hash TEXT, session_id TEXT, PRIMARY KEY(scope,request_id))")
        self.db.commit()

    def save(self, session, endpoint_hash):
        payload = canonical({"endpoint_hash": endpoint_hash, "updated_at": time.time(), "snapshot": session.snapshot()})
        with self.lock, self.db:
            self.db.execute("INSERT INTO sessions VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (session.id, payload))

    def load(self, session_id, endpoint, endpoint_hash, runtime_defaults=None):
        with self.lock:
            row = self.db.execute("SELECT payload FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None: raise KeyError("Unknown session")
        data = json.loads(row[0])
        if data["endpoint_hash"] != endpoint_hash: raise ValueError("Session belongs to a different model configuration")
        snapshot = data["snapshot"]
        for key in ("context_window", "context_tokenizer", "local_fast_path"):
            if runtime_defaults and key in runtime_defaults:
                snapshot["config"].setdefault(key, runtime_defaults[key])
        return Session.restore(snapshot, endpoint)

    def request(self, scope, request_id, body, session_id=None):
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128: raise ValueError("request_id is required (1-128 characters)")
        digest = hashlib.sha256(canonical(body).encode()).hexdigest()
        with self.lock, self.db:
            row = self.db.execute("SELECT body_hash,session_id FROM requests WHERE scope=? AND request_id=?", (scope, request_id)).fetchone()
            if row:
                if row[0] != digest: raise ValueError("request_id was reused with a different body")
                return row[1]
            if session_id is not None:
                self.db.execute("INSERT INTO requests VALUES(?,?,?,?)", (scope, request_id, digest, session_id))
        return None

    def close(self):
        self.db.close()

    def public_snapshot(self, session_id, endpoint_hash):
        with self.lock:
            row = self.db.execute("SELECT payload FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None: raise KeyError("Unknown session")
        data = json.loads(row[0])
        if data["endpoint_hash"] != endpoint_hash: raise ValueError("Session belongs to a different model configuration")
        snapshot = data["snapshot"]
        # Reading progress must not acquire the executor lock or reconstruct/replay a backend.
        result = {key: copy.deepcopy(snapshot[key]) for key in ("session_id", "profile", "state", "messages", "timeline", "counters", "final")}
        result["updated_at"] = data.get("updated_at")
        result["final_origin"] = snapshot.get("final_origin")
        turn_start = None
        if snapshot["backend"]["kind"] == "local":
            turn_start = snapshot["state"].get("turn_usage_index", 0)
        result["token_usage"] = token_usage(snapshot["usages"], snapshot["config"], turn_start)
        result["verification_scope"] = "公开参数/对象/结构化工具事实；不保证自由文本完整语义忠实度"
        return result

    def recent(self, endpoint_hash):
        with self.lock:
            rows = self.db.execute("SELECT id,payload FROM sessions ORDER BY rowid DESC LIMIT 20").fetchall()
        values = []
        for sid, payload in rows:
            data = json.loads(payload)
            if data["endpoint_hash"] == endpoint_hash:
                values.append({"session_id": sid, "phase": data["snapshot"]["state"]["phase"], "updated_at": data.get("updated_at")})
        return values


class SessionManager:
    def __init__(self, config, endpoint, database):
        self.config, self.endpoint, self.store = copy.deepcopy(config), endpoint, SessionStore(database)
        self.endpoint_hash = hashlib.sha256(canonical(config["endpoint"]).encode()).hexdigest()
        self.locks, self.cancellations, self.global_lock = {}, {}, threading.RLock()

    def _cancel_event(self, session_id):
        with self.global_lock: return self.cancellations.setdefault(session_id, threading.Event())

    def _lock(self, session_id):
        with self.global_lock: return self.locks.setdefault(session_id, threading.RLock())

    def _save(self, session):
        self.store.save(session, self.endpoint_hash)

    def get(self, session_id):
        return self.store.public_snapshot(session_id, self.endpoint_hash)

    def create(self, request, request_id, profile="full", defer=False):
        if not isinstance(request, str) or not request.strip() or len(request) > 12000: raise ValueError("Invalid request")
        body = {"request": request, "profile": profile}
        with self.global_lock:
            existing = self.store.request("create", request_id, body)
            if existing: return self.get(existing)
            if self.config.get("tool_backend", {}).get("kind") == "http":
                backend = HTTPBackend(self.config["tool_backend"], request)
            else:
                world = json.loads(Path(self.config["world_file"]).read_text(encoding="utf-8"))
                backend = LocalBackend(world, request)
            session = Session(backend, self.endpoint, self.config["runtime"], profile)
            self._save(session)
            self.store.request("create", request_id, body, session.id)
        if defer: return session.public()
        with self._lock(session.id): return session.run(self._save, self._cancel_event(session.id).is_set)

    def cancel(self, session_id, request_id):
        with self.store.lock:
            if not self.store.db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone(): raise KeyError("Unknown session")
        self.store.request(session_id + ":cancel", request_id, {}, session_id)
        self._cancel_event(session_id).set()
        return {"session_id": session_id, "cancellation_requested": True, "note": "正在执行的调用无法回滚；其回执返回后停止后续步骤"}

    def run(self, session_id, request_id):
        with self._lock(session_id):
            session = self.store.load(session_id, self.endpoint, self.endpoint_hash, self.config["runtime"])
            if self.store.request(session_id + ":run", request_id, {}): return session.public()
            self.store.request(session_id + ":run", request_id, {}, session_id)
            return session.run(self._save, self._cancel_event(session_id).is_set)

    def message(self, session_id, text, request_id):
        with self._lock(session_id):
            session = self.store.load(session_id, self.endpoint, self.endpoint_hash, self.config["runtime"])
            if self.store.request(session_id, request_id, {"text": text}): return session.public()
            if session.state["pending_call"]:
                session.run(self._save)
                return session.public()
            self.store.request(session_id, request_id, {"text": text}, session_id)
            session.user_message(text)
            self._cancel_event(session_id).clear()
            self._save(session)
            return session.run(self._save, self._cancel_event(session_id).is_set)


def make_handler(manager, token=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Prompts, tokens and authorization headers are not written to access logs.

        def send_json(self, status, value):
            body = canonical(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_json(401, {"error": "Unauthorized"})
                return False
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send_json(403, {"error": "Cross-origin request refused"})
                return False
            return True

        def do_GET(self):
            route = urlsplit(self.path).path
            if route in {"/", "/web.js"}:
                body = Path(__file__).with_name("web.html" if route == "/" else "web.js").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8" if route == "/" else "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
                self.end_headers(); self.wfile.write(body); return
            if not self.authorized(): return
            try:
                if self.path == "/health":
                    self.send_json(200, {"status": "ok", "model": manager.config["endpoint"]["model"],
                        "backend": manager.config.get("tool_backend", {}).get("kind", "local"), "training": False,
                        "controller_revision": CONTROLLER_REVISION,
                        "runtime_limits": {key: manager.config["runtime"].get(key, 3)
                                           for key in ("max_repeated_read_attempts", "max_consecutive_tool_failures")}})
                elif self.path == "/sessions": self.send_json(200, {"sessions": manager.store.recent(manager.endpoint_hash)})
                elif self.path.startswith("/sessions/") and self.path.count("/") == 2:
                    self.send_json(200, manager.get(self.path.split("/")[2]))
                else: self.send_json(404, {"error": "Not found"})
            except KeyError: self.send_json(404, {"error": "Unknown session"})
            except ValueError as exc: self.send_json(409, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized(): return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16000: raise ValueError("Request size must be 1-16000 bytes")
                from .protocol import strict_json
                data = strict_json(self.rfile.read(size).decode("utf-8"))
                if not isinstance(data, dict): raise ValueError("JSON object required")
                if self.path == "/sessions":
                    result = manager.create(data["request"], data["request_id"], data.get("profile", "full"), defer=data.get("defer") is True)
                elif self.path.startswith("/sessions/") and self.path.endswith("/run") and self.path.count("/") == 3:
                    result = manager.run(self.path.split("/")[2], data["request_id"])
                elif self.path.startswith("/sessions/") and self.path.endswith("/cancel") and self.path.count("/") == 3:
                    result = manager.cancel(self.path.split("/")[2], data["request_id"])
                elif self.path.startswith("/sessions/") and self.path.endswith("/messages") and self.path.count("/") == 3:
                    result = manager.message(self.path.split("/")[2], data["text"], data["request_id"])
                else: self.send_json(404, {"error": "Not found"}); return
                self.send_json(200, result)
            except KeyError: self.send_json(400, {"error": "Missing field or unknown session"})
            except (ValueError, UnicodeError) as exc: self.send_json(400, {"error": str(exc)})
            except Exception: self.send_json(500, {"error": "Internal error; inspect the session journal before retrying a write"})
    return Handler


def serve(config, endpoint, host, port, database):
    token = os.environ.get("ASSISTANT_API_TOKEN")
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise ValueError("Non-loopback binding requires ASSISTANT_API_TOKEN and an appropriate TLS reverse proxy")
    manager = SessionManager(config, endpoint, database)
    server = ThreadingHTTPServer((host, port), make_handler(manager, token))
    server.daemon_threads = True
    print(f"Assistant UI/API http://{host}:{port}; backend={config.get('tool_backend', {}).get('kind', 'local')}", flush=True)
    try: server.serve_forever()
    finally:
        server.server_close()
        manager.store.close()
