"""
FreeRTOS-style kernel for Keke (Python simulation).

Provides: tasks, priority scheduler, semaphores, named mutexes, message queues,
software timers, watchdog, and memory pools.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskState(Enum):
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class TaskPriority(Enum):
    IDLE = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskControlBlock:
    task_id: str
    name: str
    function: Callable
    priority: TaskPriority
    state: TaskState
    stack_size: int
    created_at: float
    last_run: Optional[float]
    run_count: int
    cpu_time: float
    data: Dict[str, Any] = field(default_factory=dict)


class NamedMutex:
    """FreeRTOS-style recursive mutex with owner tracking."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.owner: Optional[int] = None
        self.hold_count = 0

    def lock(self, timeout: Optional[float] = None) -> bool:
        if timeout is None:
            acquired = self._lock.acquire()
        else:
            acquired = self._lock.acquire(timeout=timeout)
        if acquired:
            if self.owner is None:
                self.owner = threading.get_ident()
            self.hold_count += 1
        return acquired

    def unlock(self) -> bool:
        if self.owner != threading.get_ident() or self.hold_count == 0:
            return False
        self.hold_count -= 1
        if self.hold_count == 0:
            self.owner = None
        self._lock.release()
        return True

    def try_lock(self) -> bool:
        return self.lock(timeout=0)


class MessageQueue:
    """Thread-safe message queue with stats."""

    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = maxsize
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.sent = 0
        self.received = 0

    def send(self, item: Any, timeout: Optional[float] = None) -> bool:
        try:
            self._queue.put(item, timeout=timeout)
            self.sent += 1
            return True
        except queue.Full:
            return False

    def receive(self, timeout: Optional[float] = None) -> Optional[Any]:
        try:
            item = self._queue.get(timeout=timeout)
            self.received += 1
            return item
        except queue.Empty:
            return None

    def size(self) -> int:
        return self._queue.qsize()

    def full(self) -> bool:
        if self.maxsize <= 0:
            return False
        return self._queue.qsize() >= self.maxsize


class WatchdogTimer:
    """Watchdog that must be fed before timeout or the callback runs."""

    def __init__(
        self,
        name: str,
        timeout_sec: float,
        callback: Callable[[], None],
        auto_reset: bool = True,
    ) -> None:
        self.name = name
        self.timeout_sec = timeout_sec
        self.callback = callback
        self.auto_reset = auto_reset
        self.enabled = True
        self.last_feed = time.time()
        self.trip_count = 0

    def feed(self) -> None:
        self.last_feed = time.time()

    def check(self, now: Optional[float] = None) -> bool:
        """Return True if watchdog tripped this tick."""
        if not self.enabled:
            return False
        now = now or time.time()
        if (now - self.last_feed) < self.timeout_sec:
            return False
        self.trip_count += 1
        logger.warning("Watchdog %s tripped (%.2fs since feed)", self.name, now - self.last_feed)
        try:
            self.callback()
        except Exception as exc:
            logger.error("Watchdog %s callback error: %s", self.name, exc)
        if self.auto_reset:
            self.feed()
        return True


class FreeRTOSKernel:
    """Python implementation of a FreeRTOS-like kernel."""

    def __init__(self, max_tasks: int = 50) -> None:
        self.max_tasks = max_tasks
        self.tasks: Dict[str, TaskControlBlock] = {}
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.running = False
        self.current_task: Optional[str] = None
        self.scheduler_thread: Optional[threading.Thread] = None

        self._kernel_lock = threading.RLock()
        self.semaphores: Dict[str, threading.Semaphore] = {}
        self.mutexes: Dict[str, NamedMutex] = {}
        self.message_queues: Dict[str, MessageQueue] = {}
        self.event_groups: Dict[str, threading.Event] = {}
        self.timers: Dict[str, Dict[str, Any]] = {}
        self.watchdogs: Dict[str, WatchdogTimer] = {}
        self.memory_pools: Dict[str, queue.Queue] = {}

        self.total_switches = 0
        self.idle_time = 0.0
        self.start_time = time.time()

    def _priority_key(self, priority: TaskPriority) -> int:
        """Lower sort key = runs sooner (CRITICAL first)."""
        return -priority.value

    def create_task(
        self,
        name: str,
        function: Callable,
        priority: TaskPriority = TaskPriority.NORMAL,
        stack_size: int = 4096,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        if len(self.tasks) >= self.max_tasks:
            raise RuntimeError("Maximum number of tasks reached")

        task_id = str(uuid.uuid4())
        tcb = TaskControlBlock(
            task_id=task_id,
            name=name,
            function=function,
            priority=priority,
            state=TaskState.READY,
            stack_size=stack_size,
            created_at=time.time(),
            last_run=None,
            run_count=0,
            cpu_time=0.0,
            data=data or {},
        )

        with self._kernel_lock:
            self.tasks[task_id] = tcb
            self.task_queue.put((self._priority_key(priority), time.time(), task_id))

        logger.info("Created task %s (%s)", name, task_id)
        return task_id

    def delete_task(self, task_id: str) -> bool:
        with self._kernel_lock:
            if task_id in self.tasks:
                self.tasks[task_id].state = TaskState.DELETED
                return True
        return False

    def suspend_task(self, task_id: str) -> bool:
        with self._kernel_lock:
            if task_id in self.tasks:
                self.tasks[task_id].state = TaskState.SUSPENDED
                return True
        return False

    def resume_task(self, task_id: str) -> bool:
        with self._kernel_lock:
            if task_id not in self.tasks:
                return False
            task = self.tasks[task_id]
            if task.state != TaskState.SUSPENDED:
                return False
            task.state = TaskState.READY
            self.task_queue.put((self._priority_key(task.priority), time.time(), task_id))
            return True

    def yield_task(self) -> None:
        if not self.current_task:
            return
        with self._kernel_lock:
            task = self.tasks.get(self.current_task)
            if task and task.state == TaskState.RUNNING:
                task.state = TaskState.READY
                self.task_queue.put(
                    (self._priority_key(task.priority), time.time(), self.current_task)
                )

    # --- Semaphores ---

    def create_semaphore(self, name: str, initial_count: int = 1) -> bool:
        self.semaphores[name] = threading.Semaphore(initial_count)
        logger.info("Created semaphore %s (count=%s)", name, initial_count)
        return True

    def take_semaphore(self, name: str, timeout: Optional[float] = None) -> bool:
        sem = self.semaphores.get(name)
        if sem is None:
            return False
        return sem.acquire(timeout=timeout)

    def give_semaphore(self, name: str) -> bool:
        sem = self.semaphores.get(name)
        if sem is None:
            return False
        sem.release()
        return True

    # --- Mutexes ---

    def create_mutex(self, name: str) -> bool:
        self.mutexes[name] = NamedMutex()
        logger.info("Created mutex %s", name)
        return True

    def lock_mutex(self, name: str, timeout: Optional[float] = None) -> bool:
        mutex = self.mutexes.get(name)
        if mutex is None:
            return False
        return mutex.lock(timeout=timeout)

    def unlock_mutex(self, name: str) -> bool:
        mutex = self.mutexes.get(name)
        if mutex is None:
            return False
        return mutex.unlock()

    def try_lock_mutex(self, name: str) -> bool:
        mutex = self.mutexes.get(name)
        if mutex is None:
            return False
        return mutex.try_lock()

    # --- Message queues ---

    def create_queue(self, name: str, maxsize: int = 0) -> bool:
        self.message_queues[name] = MessageQueue(maxsize=maxsize)
        logger.info("Created queue %s (maxsize=%s)", name, maxsize)
        return True

    def send_to_queue(self, name: str, item: Any, timeout: Optional[float] = None) -> bool:
        mq = self.message_queues.get(name)
        if mq is None:
            return False
        return mq.send(item, timeout=timeout)

    def receive_from_queue(self, name: str, timeout: Optional[float] = None) -> Optional[Any]:
        mq = self.message_queues.get(name)
        if mq is None:
            return None
        return mq.receive(timeout=timeout)

    def queue_stats(self, name: str) -> Optional[Dict[str, Any]]:
        mq = self.message_queues.get(name)
        if mq is None:
            return None
        return {
            "name": name,
            "size": mq.size(),
            "maxsize": mq.maxsize,
            "full": mq.full(),
            "sent": mq.sent,
            "received": mq.received,
        }

    # --- Events ---

    def create_event_group(self, name: str) -> bool:
        self.event_groups[name] = threading.Event()
        return True

    def set_event(self, name: str) -> bool:
        ev = self.event_groups.get(name)
        if ev is None:
            return False
        ev.set()
        return True

    def wait_event(self, name: str, timeout: Optional[float] = None) -> bool:
        ev = self.event_groups.get(name)
        if ev is None:
            return False
        return ev.wait(timeout)

    def clear_event(self, name: str) -> bool:
        ev = self.event_groups.get(name)
        if ev is None:
            return False
        ev.clear()
        return True

    # --- Software timers ---

    def create_timer(
        self,
        name: str,
        period_ms: int,
        auto_reload: bool = True,
        callback: Optional[Callable[[], None]] = None,
    ) -> bool:
        self.timers[name] = {
            "period": period_ms / 1000.0,
            "auto_reload": auto_reload,
            "callback": callback,
            "last_tick": time.time(),
            "active": False,
            "fire_count": 0,
        }
        return True

    def start_timer(self, name: str) -> bool:
        timer = self.timers.get(name)
        if timer is None:
            return False
        timer["active"] = True
        timer["last_tick"] = time.time()
        return True

    def stop_timer(self, name: str) -> bool:
        timer = self.timers.get(name)
        if timer is None:
            return False
        timer["active"] = False
        return True

    # --- Watchdog ---

    def create_watchdog(
        self,
        name: str,
        timeout_ms: int,
        callback: Callable[[], None],
        auto_reset: bool = True,
    ) -> bool:
        self.watchdogs[name] = WatchdogTimer(
            name=name,
            timeout_sec=timeout_ms / 1000.0,
            callback=callback,
            auto_reset=auto_reset,
        )
        logger.info("Created watchdog %s (%sms)", name, timeout_ms)
        return True

    def feed_watchdog(self, name: str) -> bool:
        wd = self.watchdogs.get(name)
        if wd is None:
            return False
        wd.feed()
        return True

    # --- Memory pools ---

    def create_memory_pool(self, name: str, block_size: int, num_blocks: int) -> bool:
        pool: queue.Queue = queue.Queue(maxsize=num_blocks)
        for _ in range(num_blocks):
            pool.put(bytearray(block_size))
        self.memory_pools[name] = pool
        return True

    def allocate_memory(self, pool_name: str, timeout: Optional[float] = None) -> Optional[bytearray]:
        pool = self.memory_pools.get(pool_name)
        if pool is None:
            return None
        try:
            return pool.get(timeout=timeout)
        except queue.Empty:
            return None

    def free_memory(self, pool_name: str, memory: bytearray) -> bool:
        pool = self.memory_pools.get(pool_name)
        if pool is None:
            return False
        try:
            pool.put_nowait(memory)
            return True
        except queue.Full:
            return False

    # --- Scheduler ---

    def start_scheduler(self) -> None:
        if self.running:
            return
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("FreeRTOS scheduler started")

    def stop_scheduler(self) -> None:
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=2.0)
        logger.info("FreeRTOS scheduler stopped")

    def _scheduler_loop(self) -> None:
        while self.running:
            try:
                now = time.time()
                self._process_timers(now)
                self._process_watchdogs(now)

                try:
                    _prio, _ts, task_id = self.task_queue.get(timeout=0.01)
                except queue.Empty:
                    self.idle_time += 0.01
                    continue

                with self._kernel_lock:
                    task = self.tasks.get(task_id)
                    if task is None or task.state == TaskState.DELETED:
                        continue
                    if task.state == TaskState.SUSPENDED:
                        self.task_queue.put(
                            (self._priority_key(task.priority), time.time(), task_id)
                        )
                        continue
                    if task.state != TaskState.READY:
                        self.task_queue.put(
                            (self._priority_key(task.priority), time.time(), task_id)
                        )
                        continue
                    self._run_task(task_id)

            except Exception as exc:
                logger.error("Scheduler error: %s", exc)
                time.sleep(0.01)

    def _run_task(self, task_id: str) -> None:
        task = self.tasks[task_id]
        self.current_task = task_id
        task.state = TaskState.RUNNING
        task.last_run = time.time()
        task.run_count += 1
        start = time.time()

        try:
            if task.data:
                task.function(**task.data)
            else:
                task.function()
        except Exception as exc:
            logger.error("Task %s error: %s", task.name, exc)
        finally:
            task.cpu_time += time.time() - start
            if task.state == TaskState.RUNNING and task.state != TaskState.DELETED:
                task.state = TaskState.READY
                self.task_queue.put(
                    (self._priority_key(task.priority), time.time(), task_id)
                )
            self.current_task = None
            self.total_switches += 1

    def _process_timers(self, now: float) -> None:
        for name, timer in self.timers.items():
            if not timer["active"]:
                continue
            if (now - timer["last_tick"]) < timer["period"]:
                continue
            timer["last_tick"] = now
            timer["fire_count"] += 1
            cb = timer.get("callback")
            if cb:
                try:
                    cb()
                except Exception as exc:
                    logger.error("Timer %s callback error: %s", name, exc)
            if not timer["auto_reload"]:
                timer["active"] = False

    def _process_watchdogs(self, now: float) -> None:
        for wd in self.watchdogs.values():
            wd.check(now)

    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._kernel_lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            return {
                "id": task.task_id,
                "name": task.name,
                "priority": task.priority.name,
                "state": task.state.value,
                "stack_size": task.stack_size,
                "created_at": task.created_at,
                "last_run": task.last_run,
                "run_count": task.run_count,
                "cpu_time": task.cpu_time,
            }

    def get_system_stats(self) -> Dict[str, Any]:
        with self._kernel_lock:
            uptime = time.time() - self.start_time
            active_tasks = sum(
                1 for t in self.tasks.values() if t.state != TaskState.DELETED
            )
            queue_info = {
                name: self.queue_stats(name) for name in self.message_queues
            }
            watchdog_info = {
                name: {
                    "timeout_sec": wd.timeout_sec,
                    "enabled": wd.enabled,
                    "trip_count": wd.trip_count,
                    "seconds_since_feed": time.time() - wd.last_feed,
                }
                for name, wd in self.watchdogs.items()
            }
            return {
                "uptime": uptime,
                "total_tasks": len(self.tasks),
                "active_tasks": active_tasks,
                "total_switches": self.total_switches,
                "idle_time": self.idle_time,
                "cpu_usage": ((uptime - self.idle_time) / uptime * 100) if uptime > 0 else 0,
                "current_task": self.current_task,
                "semaphores": list(self.semaphores.keys()),
                "mutexes": list(self.mutexes.keys()),
                "queues": queue_info,
                "event_groups": list(self.event_groups.keys()),
                "timers": {
                    n: {"active": t["active"], "fire_count": t["fire_count"]}
                    for n, t in self.timers.items()
                },
                "watchdogs": watchdog_info,
                "memory_pools": list(self.memory_pools.keys()),
                "running": self.running,
            }


class ExcelProcessingTask:
    """Excel jobs via RTOS queues, semaphores, and house mutex."""

    def __init__(self, kernel: FreeRTOSKernel) -> None:
        self.kernel = kernel
        self._worker_started = False
        self._worker_threads: List[threading.Thread] = []

    def _ensure_workers(self) -> None:
        if self._worker_started:
            return
        if "excel_processing" not in self.kernel.semaphores:
            self.kernel.create_semaphore("excel_processing", 3)
        if "house_lock" not in self.kernel.mutexes:
            self.kernel.create_mutex("house_lock")
        if "system_watchdog" not in self.kernel.watchdogs:
            self.kernel.create_watchdog(
                "system_watchdog",
                30000,
                lambda: logger.error("SYSTEM WATCHDOG TRIPPED"),
                auto_reset=True,
            )
        self.kernel.create_queue("excel_jobs", maxsize=100)
        self.kernel.create_queue("excel_results", maxsize=100)
        for index in range(3):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"excel_worker_{index}",
                daemon=True,
            )
            thread.start()
            self._worker_threads.append(thread)
        self._worker_started = True

    def _worker_loop(self) -> None:
        while self.kernel.running:
            job = self.kernel.receive_from_queue("excel_jobs", timeout=0.5)
            if job is None:
                continue

            if not self.kernel.take_semaphore("excel_processing", timeout=30):
                self._emit_result(
                    {
                        **job,
                        "status": "error",
                        "error": "Failed to acquire excel_processing semaphore",
                    }
                )
                continue

            if not self.kernel.lock_mutex("house_lock", timeout=10):
                self.kernel.give_semaphore("excel_processing")
                self._emit_result(
                    {
                        **job,
                        "status": "error",
                        "error": "Failed to acquire house_lock mutex",
                    }
                )
                continue

            try:
                self.kernel.feed_watchdog("system_watchdog")
                time.sleep(0.05)
                self._emit_result(
                    {
                        **job,
                        "status": "completed",
                        "processed_at": time.time(),
                    }
                )
            except Exception as exc:
                self._emit_result(
                    {
                        **job,
                        "status": "error",
                        "error": str(exc),
                        "processed_at": time.time(),
                    }
                )
            finally:
                self.kernel.unlock_mutex("house_lock")
                self.kernel.give_semaphore("excel_processing")

    def _emit_result(self, payload: Dict[str, Any]) -> None:
        self.kernel.send_to_queue("excel_results", payload, timeout=5)

    def enqueue(self, file_path: str, operations: List[Dict[str, Any]]) -> bool:
        self._ensure_workers()
        job = {
            "job_id": str(uuid.uuid4()),
            "file_path": file_path,
            "operations": operations,
            "enqueued_at": time.time(),
        }
        return self.kernel.send_to_queue("excel_jobs", job, timeout=5)

    def create_processing_task(self, file_path: str, operations: List[Dict[str, Any]]) -> str:
        """Legacy API: enqueue and return synthetic task id."""
        self.enqueue(file_path, operations)
        return f"queued:{file_path}"

    def get_results(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        while True:
            item = self.kernel.receive_from_queue("excel_results", timeout=0)
            if item is None:
                break
            results.append(item)
        return results


_default_kernel: Optional[FreeRTOSKernel] = None
_default_excel: Optional[ExcelProcessingTask] = None


def boot_freertos(start_scheduler: bool = True) -> FreeRTOSKernel:
    """Initialize kernel primitives and optionally start the scheduler."""
    global _default_kernel, _default_excel

    kernel = FreeRTOSKernel()
    kernel.create_semaphore("excel_processing", 3)
    kernel.create_mutex("house_lock")
    kernel.create_event_group("file_upload_complete")
    kernel.create_memory_pool("excel_buffer", 1024, 100)
    kernel.create_timer(
        "health_check",
        5000,
        True,
        lambda: logger.info("RTOS health_check timer tick"),
    )
    kernel.start_timer("health_check")
    kernel.create_watchdog(
        "system_watchdog",
        30000,
        lambda: logger.error("SYSTEM WATCHDOG TRIPPED — scheduler may be stalled"),
        auto_reset=True,
    )
    kernel.feed_watchdog("system_watchdog")

    if start_scheduler:
        kernel.start_scheduler()

    _default_kernel = kernel
    _default_excel = ExcelProcessingTask(kernel)
    return kernel


def get_kernel() -> FreeRTOSKernel:
    global _default_kernel
    if _default_kernel is None:
        boot_freertos()
    return _default_kernel


def get_excel_processor() -> ExcelProcessingTask:
    global _default_excel
    if _default_excel is None:
        boot_freertos()
    return _default_excel


class _LazyKernel:
    def __getattr__(self, item: str) -> Any:
        return getattr(get_kernel(), item)

    def __repr__(self) -> str:
        return repr(get_kernel())


class _LazyExcel:
    def __getattr__(self, item: str) -> Any:
        return getattr(get_excel_processor(), item)

    def __repr__(self) -> str:
        return repr(get_excel_processor())


freertos_kernel = _LazyKernel()
excel_processor = _LazyExcel()
