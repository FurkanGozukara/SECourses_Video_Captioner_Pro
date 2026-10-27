"""Real loopback HTTP transfers with deterministic network and integrity faults."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
import threading

import pytest
import requests


@pytest.fixture(scope="module")
def downloader_module():
    path = Path(__file__).resolve().parents[1] / "Models_Downloader.py"
    spec = importlib.util.spec_from_file_location("vcap_test_transfer_downloader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATA = bytes(range(256)) * 8192


@contextmanager
def _server(mode: str):
    state = {"requests": [], "faults": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            start, end = (int(match[1]), int(match[2])) if match else (0, len(DATA) - 1)
            with lock:
                state["requests"].append((start, end, bool(match)))
                first = len(state["requests"]) == 1
            if first and mode in {"503", "429", "expired"}:
                self.send_response({"503": 503, "429": 429, "expired": 403}[mode])
                self.send_header("Content-Length", "0")
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            ranged = match is not None and mode != "ignore_range"
            if not ranged:
                start, end = 0, len(DATA) - 1
            body = DATA[start : end + 1]
            if mode == "corrupt_always" or (first and mode == "corrupt_once"):
                body = bytes([body[0] ^ 1]) + body[1:]
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Length", str(len(body)))
            if ranged:
                header_start = start + 1 if first and mode == "bad_range" else start
                self.send_header("Content-Range", f"bytes {header_start}-{end}/{len(DATA)}")
            self.end_headers()
            try:
                if first and mode == "truncate":
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/weights", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _downloader(module, tmp_path, monkeypatch, url, **extra):
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path / "cache")
    downloader = module.RobustDownloader({
        "xet_enabled": 0, "num_connections": 3, "http_adaptive_connections": 0,
        "piece_size": 262144, "piece_size_max": 262144,
        "read_block_size": 16384, "write_block_size": 16384,
        "parallel_threshold": 1, "retry_delay": 0.01, "max_retry_delay": 0.02,
        "max_retries": 3, "file_attempts": 2, "progress_interval": 0.02,
        "state_flush_interval": 0.02, **extra,
    })
    monkeypatch.setattr(downloader, "_download_target", lambda _remote: (url, {}))
    remote = module.RemoteFile("fixture/repo", "weights.bin", len(DATA), "sha256", hashlib.sha256(DATA).hexdigest(), "fixture-commit")
    return downloader, remote


@pytest.mark.parametrize("mode", ["normal", "503", "429", "expired", "truncate", "ignore_range", "bad_range", "corrupt_once"])
def test_http_recovery_installs_only_exact_verified_bytes(downloader_module, tmp_path, monkeypatch, mode):
    with _server(mode) as (url, state):
        downloader, remote = _downloader(downloader_module, tmp_path, monkeypatch, url)
        target = tmp_path / "weights ü.bin"
        assert downloader.ensure_file(remote, target, show_progress=False)
        assert target.read_bytes() == DATA
        assert downloader.is_file_verified(remote, target)
        assert not Path(str(target) + ".part").exists()
        assert not Path(str(target) + ".part.json").exists()
        count = len(state["requests"])
        assert downloader.ensure_file(remote, target, show_progress=False)
        assert len(state["requests"]) == count  # Verified files are reused.


def test_bad_hash_is_never_published(downloader_module, tmp_path, monkeypatch):
    with _server("corrupt_always") as (url, _state):
        downloader, remote = _downloader(downloader_module, tmp_path, monkeypatch, url)
        target = tmp_path / "bad.bin"
        assert not downloader.ensure_file(remote, target, show_progress=False)
        assert not target.exists()


def test_interrupted_transfer_resumes_only_missing_bytes(downloader_module, tmp_path, monkeypatch):
    with _server("normal") as (url, state):
        downloader, remote = _downloader(downloader_module, tmp_path, monkeypatch, url,
                                        num_connections=1, max_bytes_for_test=131072)
        target = tmp_path / "resume.bin"
        with pytest.raises(KeyboardInterrupt):
            downloader.ensure_file(remote, target, show_progress=False)
        assert not target.exists()
        assert Path(str(target) + ".part").is_file()
        assert Path(str(target) + ".part.json").is_file()
        saved = downloader._read_state(target, remote).total
        assert 0 < saved < len(DATA)
        state["requests"].clear()
        resumed, _ = _downloader(downloader_module, tmp_path, monkeypatch, url, num_connections=1)
        assert resumed.ensure_file(remote, target, show_progress=False)
        assert target.read_bytes() == DATA
        assert all(start >= saved for start, _end, _range in state["requests"])


@pytest.mark.parametrize("route", ["api", "metadata"])
def test_rate_limit_uses_retry_after_with_falsey_http_response(downloader_module, tmp_path, monkeypatch, route):
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "7"
    failure = requests.HTTPError("rate limited", response=response)
    calls, sleeps = [], []

    def operation(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise failure
        return "metadata"

    monkeypatch.setattr(downloader_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(downloader_module, "_hf_token", lambda: None)
    if route == "api":
        result = downloader_module._hf_call(operation, "test metadata", quiet=True)
    else:
        monkeypatch.setattr(downloader_module, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(downloader_module, "get_hf_file_metadata", operation)
        result = downloader_module.RobustDownloader({})._get_file_metadata("https://example.invalid/file")
    assert result == "metadata"
    assert sleeps == [7.0]
