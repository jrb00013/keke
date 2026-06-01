"""Bridge between Express API and the FreeRTOS-style kernel."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

LEGACY_DIR = os.path.join(os.path.dirname(__file__), "..", "legacy")
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

from freertos_integration import (  # noqa: E402
    boot_freertos,
    get_excel_processor,
    get_kernel,
)


def ensure_rtos() -> None:
    get_kernel()


def status() -> Dict[str, Any]:
    kernel = get_kernel()
    return kernel.get_system_stats()


def feed_watchdog(name: str = "system_watchdog") -> bool:
    return get_kernel().feed_watchdog(name)


def enqueue_excel_job(file_path: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    excel = get_excel_processor()
    ok = excel.enqueue(file_path, operations)
    return {"queued": ok, "file_path": file_path}


def collect_results() -> List[Dict[str, Any]]:
    return get_excel_processor().get_results()


def boot(start_scheduler: bool = True) -> Dict[str, str]:
    boot_freertos(start_scheduler=start_scheduler)
    return {"status": "booted", "scheduler": start_scheduler}


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rtos_bridge.py <command> [args]", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "boot":
            start = sys.argv[2].lower() != "false" if len(sys.argv) > 2 else True
            print(json.dumps(boot(start_scheduler=start)))
        elif command == "status":
            ensure_rtos()
            print(json.dumps(status(), default=str))
        elif command == "feed_watchdog":
            name = sys.argv[2] if len(sys.argv) > 2 else "system_watchdog"
            ensure_rtos()
            print(json.dumps({"fed": feed_watchdog(name)}))
        elif command == "enqueue":
            file_path = sys.argv[2]
            operations = json.loads(sys.argv[3])
            ensure_rtos()
            print(json.dumps(enqueue_excel_job(file_path, operations), default=str))
        elif command == "results":
            ensure_rtos()
            print(json.dumps(collect_results(), default=str))
        else:
            print(f"Unknown command: {command}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)
