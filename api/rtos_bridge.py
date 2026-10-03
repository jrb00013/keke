"""Bridge between Express API and the FreeRTOS-style kernel.

The Express layer talks to this module over a Unix domain socket by spawning
``python3 api/rtos_bridge.py <command> ...`` per request. The kernel itself lives
in a long-running daemon process that this module auto-starts on first use.

Why a daemon: the kernel's state (task table, message queues, watchdog feed
times, semaphore counts) is in-process. Spawning a fresh interpreter per HTTP
request meant every request booted a brand-new kernel, so ``enqueue`` then
``results`` always returned an empty list and ``boot`` had no observable effect.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LEGACY_DIR = os.path.join(os.path.dirname(__file__), "..", "legacy")
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

import freertos_integration as freertos  # noqa: E402
from freertos_integration import (  # noqa: E402
    boot_freertos,
    get_excel_processor,
    get_kernel,
)

_LOG = logging.getLogger("keke.rtos")

# AF_UNIX paths are capped at ~108 bytes on Linux. Fall back to /tmp when the
# repository path is too deep to hold a socket under data/.
_MAX_SOCKET_PATH = 100


def _default_socket_path() -> str:
    explicit = os.environ.get("KEKE_RTOS_SOCKET")
    if explicit:
        return explicit

    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "data" / "rtos.sock"
    if len(str(candidate)) <= _MAX_SOCKET_PATH:
        return str(candidate)

    digest = hashlib.sha1(str(repo_root).encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"keke-rtos-{digest}.sock")


SOCKET_PATH = _default_socket_path()
LOG_PATH = SOCKET_PATH + ".log"
_CONNECT_TIMEOUT = 30.0
_DAEMON_BOOT_TIMEOUT = 20.0


# --------------------------------------------------------------------------- #
# Server side
# --------------------------------------------------------------------------- #


def ensure_rtos() -> None:
    get_kernel()


def status() -> Dict[str, Any]:
    kernel = get_kernel()
    return kernel.get_system_stats()


def feed_watchdog(name: str = "system_watchdog") -> bool:
    return get_kernel().feed_watchdog(name)


def enqueue_excel_job(
    file_path: str, operations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    excel = get_excel_processor()
    ok = excel.enqueue(file_path, operations)
    return {"queued": ok, "file_path": file_path}


def collect_results() -> List[Dict[str, Any]]:
    return get_excel_processor().get_results()


def boot(start_scheduler: bool = True) -> Dict[str, Any]:
    """(Re)boot the kernel, stopping any previous scheduler first.

    Repeated boots must not leak scheduler/worker threads, so the old kernel is
    stopped once the replacement is in place.
    """
    previous = freertos._default_kernel
    boot_freertos(start_scheduler=start_scheduler)
    if previous is not None and previous is not freertos._default_kernel:
        try:
            previous.stop_scheduler()
        except Exception:  # never fail a boot because old teardown misbehaved
            pass
    return {"status": "booted", "scheduler": start_scheduler}


def dispatch(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one bridge request against the live kernel."""
    command = request.get("command")
    args = request.get("args") or {}

    if command == "boot":
        start = str(args.get("start_scheduler", True)).lower() != "false"
        return boot(start_scheduler=start)
    if command == "status":
        ensure_rtos()
        return status()
    if command == "feed_watchdog":
        ensure_rtos()
        return {"fed": feed_watchdog(args.get("name") or "system_watchdog")}
    if command == "enqueue":
        ensure_rtos()
        return enqueue_excel_job(args["file_path"], args.get("operations") or [])
    if command == "results":
        ensure_rtos()
        return {"results": collect_results()}
    if command == "shutdown":
        return {"status": "shutting_down"}
    raise ValueError(f"Unknown command: {command}")


def _handle_connection(conn: socket.socket) -> None:
    try:
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            buffer += chunk
        if not buffer.strip():
            return

        request = json.loads(buffer.decode("utf-8"))
        response = dispatch(request)
        if request.get("command") == "shutdown":
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
            os._exit(0)
        conn.sendall((json.dumps(response, default=str) + "\n").encode("utf-8"))
    except Exception as exc:  # never let a bad request kill the daemon
        try:
            conn.sendall((json.dumps({"error": str(exc)}) + "\n").encode("utf-8"))
        except OSError:
            pass
    finally:
        conn.close()


def serve_forever(socket_path: Optional[str] = None) -> None:
    """Run the RTOS daemon until killed. Boots the kernel once, then serves."""
    import threading

    path = socket_path or SOCKET_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # A previous daemon may have died without cleaning up its socket file.
    if os.path.exists(path):
        os.unlink(path)

    boot_freertos(start_scheduler=True)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o600)
    server.listen(64)

    logging.basicConfig(
        level=logging.INFO,
        filename=LOG_PATH,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("keke.rtos").info("RTOS daemon listening on %s", path)

    while True:
        try:
            conn, _ = server.accept()
        except OSError:
            continue
        # One thread per connection so a slow request cannot block `status`.
        threading.Thread(target=_handle_connection, args=(conn,), daemon=True).start()


# --------------------------------------------------------------------------- #
# Client side
# --------------------------------------------------------------------------- #


def _try_request(
    request: Dict[str, Any], timeout: float = _CONNECT_TIMEOUT
) -> Dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(SOCKET_PATH)
        client.sendall((json.dumps(request) + "\n").encode("utf-8"))
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            buffer += chunk
        if not buffer.strip():
            raise ConnectionError("RTOS daemon closed the connection without a reply")
        return json.loads(buffer.decode("utf-8"))
    finally:
        client.close()


def _spawn_daemon() -> None:
    """Start the daemon fully detached from this process's stdio.

    The stdio detachment matters: Express waits for the client process's stdout
    to close before resolving the request. If the daemon inherited those pipes
    the parent would never see EOF and the HTTP request would hang.
    """
    os.makedirs(os.path.dirname(SOCKET_PATH) or ".", exist_ok=True)
    log = open(LOG_PATH, "ab")
    try:
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "_daemon"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()


def _ensure_daemon() -> None:
    if os.path.exists(SOCKET_PATH):
        try:
            _try_request({"command": "status"}, timeout=2.0)
            return
        except (OSError, ConnectionError, json.JSONDecodeError):
            # Stale socket left by a dead daemon; remove and restart.
            try:
                os.unlink(SOCKET_PATH)
            except OSError:
                pass

    _spawn_daemon()

    deadline = time.time() + _DAEMON_BOOT_TIMEOUT
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        try:
            _try_request({"command": "status"}, timeout=2.0)
            return
        except (OSError, ConnectionError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.15)

    raise TimeoutError(
        f"RTOS daemon did not start within {_DAEMON_BOOT_TIMEOUT}s: {last_error}"
    )


def request(command: str, **args: Any) -> Dict[str, Any]:
    """Send a command to the daemon, starting it first if necessary."""
    _ensure_daemon()
    return _try_request({"command": command, "args": args})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: List[str]) -> Dict[str, Any]:
    command = argv[0]
    if command == "enqueue":
        if len(argv) < 3:
            raise ValueError("enqueue requires <file_path> <operations_json>")
        return {
            "command": "enqueue",
            "args": {
                "file_path": argv[1],
                "operations": json.loads(argv[2]),
            },
        }
    if command == "feed_watchdog":
        return {
            "command": "feed_watchdog",
            "args": {
                "name": argv[1] if len(argv) > 1 else "system_watchdog",
            },
        }
    if command == "boot":
        return {
            "command": "boot",
            "args": {
                "start_scheduler": (
                    (argv[1].lower() != "false") if len(argv) > 1 else True
                ),
            },
        }
    return {"command": command, "args": {}}


def main(argv: List[str]) -> int:
    if not argv:
        print("Usage: python rtos_bridge.py <command> [args]", file=sys.stderr)
        return 1

    if argv[0] == "_daemon":
        try:
            serve_forever()
        except KeyboardInterrupt:
            return 0
        return 0

    try:
        if argv[0] == "shutdown":
            print(json.dumps(request("shutdown")))
            return 0

        parsed = _parse_args(argv)
        result = request(parsed["command"], **parsed["args"])

        # `results` returns the list directly on the bridge's old contract.
        if parsed["command"] == "results":
            print(json.dumps(result.get("results", result), default=str))
        else:
            print(json.dumps(result, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
