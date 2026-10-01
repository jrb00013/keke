"""Tests for the RTOS bridge client/daemon protocol.

These tests deliberately invoke the bridge as a *subprocess* per command, which
is how Express uses it. That is the only way to catch the original bug: run
``enqueue`` and ``results`` in separate interpreters and confirm the queued job
is still visible. Testing the kernel in-process (as test_freertos_integration.py
does) cannot catch it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE = REPO_ROOT / "api" / "rtos_bridge.py"


@pytest.fixture(scope="module")
def rtos_socket(tmp_path_factory):
    """A private socket + daemon for this test module, torn down afterwards."""
    sock = str(tmp_path_factory.mktemp("rtos") / "rtos.sock")
    env = {**os.environ, "KEKE_RTOS_SOCKET": sock}
    yield sock, env

    subprocess.run(
        [sys.executable, str(BRIDGE), "shutdown"],
        env=env, capture_output=True, timeout=30,
    )


def run_bridge(env, *args, timeout=40):
    """Run the bridge as a fresh process and return parsed stdout."""
    proc = subprocess.run(
        [sys.executable, str(BRIDGE), *args],
        env=env, capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, f"bridge failed: {proc.stderr}"
    return json.loads(proc.stdout)


def test_status_autostarts_daemon(rtos_socket):
    _sock, env = rtos_socket
    status = run_bridge(env, "status")
    assert status["running"] is True
    assert "excel_processing" in status["semaphores"]
    assert "house_lock" in status["mutexes"]
    assert "system_watchdog" in status["watchdogs"]


def test_enqueue_result_survives_across_processes(rtos_socket, tmp_path):
    """The regression guard: separate processes must share kernel state."""
    _sock, env = rtos_socket

    workbook = tmp_path / "book.xlsx"
    workbook.write_bytes(b"not-a-real-workbook")  # path is metadata here

    queued = run_bridge(
        env, "enqueue", str(workbook), json.dumps([{"type": "remove_duplicates"}])
    )
    assert queued["queued"] is True

    results = []
    for _ in range(40):
        results = run_bridge(env, "results")
        if results:
            break
        time.sleep(0.25)

    assert results, "queued job never appeared in results across processes"
    assert results[0]["file_path"] == str(workbook)
    assert results[0]["status"] == "completed"


def test_status_reflects_queue_traffic(rtos_socket, tmp_path):
    _sock, env = rtos_socket
    workbook = tmp_path / "traffic.xlsx"
    workbook.write_bytes(b"x")

    run_bridge(
        env, "enqueue", str(workbook), json.dumps([{"type": "remove_duplicates"}])
    )
    for _ in range(40):
        run_bridge(env, "results")
        status = run_bridge(env, "status")
        if status["queues"].get("excel_jobs", {}).get("sent", 0) >= 1:
            break
        time.sleep(0.25)

    jobs = status["queues"]["excel_jobs"]
    # The original per-process design always reported sent=0 for a fresh process.
    assert jobs["sent"] >= 1
    assert jobs["received"] >= 1


def test_watchdog_feed_persists(rtos_socket):
    _sock, env = rtos_socket
    # Trip the clock forward by not feeding, then feed and confirm the daemon
    # observes it (a fresh process would always report ~0s since feed).
    run_bridge(env, "status")
    time.sleep(1.2)
    before = run_bridge(env, "status")["watchdogs"]["system_watchdog"]
    fed = run_bridge(env, "feed_watchdog", "system_watchdog")
    assert fed["fed"] is True
    after = run_bridge(env, "status")["watchdogs"]["system_watchdog"]
    assert after["seconds_since_feed"] < before["seconds_since_feed"]


def test_boot_is_idempotent(rtos_socket):
    _sock, env = rtos_socket
    result = run_bridge(env, "boot", "true")
    assert result["status"] == "booted"
    assert run_bridge(env, "status")["running"] is True


def test_daemon_recovers_after_sigkill(rtos_socket):
    _sock, env = rtos_socket
    assert run_bridge(env, "status")["running"] is True

    # Find and hard-kill the daemon, then confirm the next command restarts it.
    import glob

    daemons = []
    for entry in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(entry, "rb") as handle:
                cmdline = handle.read().split(b"\x00")
        except OSError:
            continue
        if any(b"rtos_bridge.py" in part for part in cmdline) and any(
            part == b"_daemon" for part in cmdline
        ):
            daemons.append(int(entry.split("/")[2]))

    assert daemons, "could not locate the running daemon"
    for pid in daemons:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    time.sleep(0.5)

    status = run_bridge(env, "status")
    assert status["running"] is True
    assert status["uptime"] < 30  # a freshly restarted daemon
