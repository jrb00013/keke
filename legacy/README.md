# Legacy / RTOS

## FreeRTOS kernel (`freertos_integration.py`)

Python simulation of FreeRTOS primitives used by Keke for concurrent Excel job handling.

| Primitive | API |
|-----------|-----|
| **Tasks** | `create_task`, `start_scheduler`, priority queue (CRITICAL first) |
| **Semaphores** | `create_semaphore`, `take_semaphore`, `give_semaphore` |
| **Mutex** | `create_mutex`, `lock_mutex`, `unlock_mutex` — includes `house_lock` |
| **Queues** | `create_queue`, `send_to_queue`, `receive_from_queue` — `excel_jobs`, `excel_results` |
| **Watchdog** | `create_watchdog`, `feed_watchdog` — `system_watchdog` (30s) |
| **Timers** | `create_timer`, `start_timer` — `health_check` (5s) |
| **Memory pool** | `excel_buffer` blocks |

Boot via `boot_freertos()` or HTTP:

- `GET /api/rtos/status`
- `POST /api/rtos/watchdog/feed`
- `POST /api/rtos/excel/enqueue`
- `GET /api/rtos/excel/results`

## C stubs

`rtos/` and `boot/` are reference stubs, not linked to the Node/Python runtime.
