import asyncio
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json
import uuid

# Configure logging
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
    data: Dict[str, Any]

class FreeRTOSKernel:
    """
    Python implementation of FreeRTOS kernel for Excel data processing
    """
    
    def __init__(self, max_tasks: int = 50):
        self.max_tasks = max_tasks
        self.tasks: Dict[str, TaskControlBlock] = {}
        self.task_queue = queue.PriorityQueue()
        self.running = False
        self.current_task: Optional[str] = None
        self.scheduler_thread: Optional[threading.Thread] = None
        self.mutex = threading.Lock()
        self.semaphores: Dict[str, threading.Semaphore] = {}
        self.event_groups: Dict[str, threading.Event] = {}
        self.timers: Dict[str, Dict] = {}
        self.memory_pools: Dict[str, queue.Queue] = {}
        
        # Performance metrics
        self.total_switches = 0
        self.idle_time = 0
        self.start_time = time.time()
        
    def create_task(self, name: str, function: Callable, priority: TaskPriority = TaskPriority.NORMAL, 
                   stack_size: int = 4096, data: Dict[str, Any] = None) -> str:
        """Create a new task"""
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
            cpu_time=0,
            data=data or {}
        )
        
        with self.mutex:
            self.tasks[task_id] = tcb
            self.task_queue.put((priority.value, time.time(), task_id))
        
        logger.info(f"Created task {name} with ID {task_id}")
        return task_id
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        with self.mutex:
            if task_id in self.tasks:
                self.tasks[task_id].state = TaskState.DELETED
                logger.info(f"Deleted task {task_id}")
                return True
        return False
    
    def suspend_task(self, task_id: str) -> bool:
        """Suspend a task"""
        with self.mutex:
            if task_id in self.tasks:
                self.tasks[task_id].state = TaskState.SUSPENDED
                logger.info(f"Suspended task {task_id}")
                return True
        return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a suspended task"""
        with self.mutex:
            if task_id in self.tasks:
                if self.tasks[task_id].state == TaskState.SUSPENDED:
                    self.tasks[task_id].state = TaskState.READY
                    self.task_queue.put((self.tasks[task_id].priority.value, time.time(), task_id))
                    logger.info(f"Resumed task {task_id}")
                    return True
        return False
    
    def yield_task(self):
        """Yield control to scheduler"""
        if self.current_task:
            with self.mutex:
                if self.current_task in self.tasks:
                    self.tasks[self.current_task].state = TaskState.READY
                    self.task_queue.put((self.tasks[self.current_task].priority.value, time.time(), self.current_task))
    
    def create_semaphore(self, name: str, initial_count: int = 1) -> bool:
        """Create a semaphore"""
        self.semaphores[name] = threading.Semaphore(initial_count)
        logger.info(f"Created semaphore {name} with count {initial_count}")
        return True
    
    def take_semaphore(self, name: str, timeout: float = None) -> bool:
        """Take a semaphore"""
        if name in self.semaphores:
            return self.semaphores[name].acquire(timeout=timeout)
        return False
    
    def give_semaphore(self, name: str) -> bool:
        """Give a semaphore"""
        if name in self.semaphores:
            self.semaphores[name].release()
            return True
        return False
    
    def create_event_group(self, name: str) -> bool:
        """Create an event group"""
        self.event_groups[name] = threading.Event()
        logger.info(f"Created event group {name}")
        return True
    
    def set_event(self, name: str) -> bool:
        """Set an event"""
        if name in self.event_groups:
            self.event_groups[name].set()
            return True
        return False
    
    def wait_event(self, name: str, timeout: float = None) -> bool:
        """Wait for an event"""
        if name in self.event_groups:
            return self.event_groups[name].wait(timeout)
        return False
    
    def create_timer(self, name: str, period_ms: int, auto_reload: bool = True, 
                    callback: Callable = None) -> bool:
        """Create a timer"""
        self.timers[name] = {
            'period': period_ms / 1000.0,
            'auto_reload': auto_reload,
            'callback': callback,
            'last_tick': time.time(),
            'active': True
        }
        logger.info(f"Created timer {name} with period {period_ms}ms")
        return True
    
    def start_timer(self, name: str) -> bool:
        """Start a timer"""
        if name in self.timers:
            self.timers[name]['active'] = True
            self.timers[name]['last_tick'] = time.time()
            return True
        return False
    
    def stop_timer(self, name: str) -> bool:
        """Stop a timer"""
        if name in self.timers:
            self.timers[name]['active'] = False
            return True
        return False
    
    def create_memory_pool(self, name: str, block_size: int, num_blocks: int) -> bool:
        """Create a memory pool"""
        pool = queue.Queue(maxsize=num_blocks)
        for _ in range(num_blocks):
            pool.put(bytearray(block_size))
        self.memory_pools[name] = pool
        logger.info(f"Created memory pool {name} with {num_blocks} blocks of {block_size} bytes")
        return True
    
    def allocate_memory(self, pool_name: str, timeout: float = None) -> Optional[bytearray]:
        """Allocate memory from pool"""
        if pool_name in self.memory_pools:
            try:
                return self.memory_pools[pool_name].get(timeout=timeout)
            except queue.Empty:
                return None
        return None
    
    def free_memory(self, pool_name: str, memory: bytearray) -> bool:
        """Free memory back to pool"""
        if pool_name in self.memory_pools:
            self.memory_pools[pool_name].put(memory)
            return True
        return False
    
    def start_scheduler(self):
        """Start the FreeRTOS scheduler"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("FreeRTOS scheduler started")
    
    def stop_scheduler(self):
        """Stop the FreeRTOS scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("FreeRTOS scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Process timers
                self._process_timers()
                
                # Get next task from queue
                if not self.task_queue.empty():
                    priority, timestamp, task_id = self.task_queue.get_nowait()
                    
                    with self.mutex:
                        if task_id in self.tasks and self.tasks[task_id].state == TaskState.READY:
                            self._run_task(task_id)
                        elif task_id in self.tasks:
                            # Task is not ready, put it back
                            self.task_queue.put((priority, timestamp, task_id))
                
                # Small delay to prevent busy waiting
                time.sleep(0.001)
                
            except queue.Empty:
                # No tasks ready, idle
                self.idle_time += 0.001
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(0.01)
    
    def _run_task(self, task_id: str):
        """Run a specific task"""
        task = self.tasks[task_id]
        self.current_task = task_id
        task.state = TaskState.RUNNING
        task.last_run = time.time()
        task.run_count += 1
        
        start_time = time.time()
        
        try:
            # Run the task function
            if task.data:
                result = task.function(**task.data)
            else:
                result = task.function()
            
            # Update CPU time
            task.cpu_time += time.time() - start_time
            
            # Task completed successfully
            if task.state == TaskState.RUNNING:
                task.state = TaskState.READY
                self.task_queue.put((task.priority.value, time.time(), task_id))
            
        except Exception as e:
            logger.error(f"Task {task.name} error: {e}")
            task.state = TaskState.READY
            self.task_queue.put((task.priority.value, time.time(), task_id))
        
        finally:
            self.current_task = None
            self.total_switches += 1
    
    def _process_timers(self):
        """Process all active timers"""
        current_time = time.time()
        
        for name, timer in self.timers.items():
            if timer['active'] and (current_time - timer['last_tick']) >= timer['period']:
                timer['last_tick'] = current_time
                
                if timer['callback']:
                    try:
                        timer['callback']()
                    except Exception as e:
                        logger.error(f"Timer {name} callback error: {e}")
                
                if not timer['auto_reload']:
                    timer['active'] = False
    
    def get_task_info(self, task_id: str) -> Optional[Dict]:
        """Get information about a task"""
        with self.mutex:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                return {
                    'id': task.task_id,
                    'name': task.name,
                    'priority': task.priority.name,
                    'state': task.state.value,
                    'stack_size': task.stack_size,
                    'created_at': task.created_at,
                    'last_run': task.last_run,
                    'run_count': task.run_count,
                    'cpu_time': task.cpu_time
                }
        return None
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        with self.mutex:
            uptime = time.time() - self.start_time
            active_tasks = sum(1 for task in self.tasks.values() if task.state != TaskState.DELETED)
            
            return {
                'uptime': uptime,
                'total_tasks': len(self.tasks),
                'active_tasks': active_tasks,
                'total_switches': self.total_switches,
                'idle_time': self.idle_time,
                'cpu_usage': (uptime - self.idle_time) / uptime * 100 if uptime > 0 else 0,
                'current_task': self.current_task,
                'semaphores': len(self.semaphores),
                'event_groups': len(self.event_groups),
                'timers': len(self.timers),
                'memory_pools': len(self.memory_pools)
            }

class ExcelProcessingTask:
    """
    Excel processing task that integrates with FreeRTOS
    """
    
    def __init__(self, kernel: FreeRTOSKernel):
        self.kernel = kernel
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
    def create_processing_task(self, file_path: str, operations: List[Dict]) -> str:
        """Create a task for processing Excel file"""
        
        def process_file():
            try:
                # Simulate Excel processing
                logger.info(f"Processing file: {file_path}")
                
                # Take semaphore to limit concurrent processing
                if not self.kernel.take_semaphore('excel_processing', timeout=10):
                    logger.error("Failed to acquire processing semaphore")
                    return
                
                try:
                    # Process the file (simplified)
                    time.sleep(0.1)  # Simulate processing time
                    
                    result = {
                        'file_path': file_path,
                        'operations': operations,
                        'status': 'completed',
                        'processed_at': time.time()
                    }
                    
                    self.results_queue.put(result)
                    logger.info(f"Completed processing: {file_path}")
                    
                finally:
                    self.kernel.give_semaphore('excel_processing')
                    
            except Exception as e:
                logger.error(f"Processing error: {e}")
                error_result = {
                    'file_path': file_path,
                    'operations': operations,
                    'status': 'error',
                    'error': str(e),
                    'processed_at': time.time()
                }
                self.results_queue.put(error_result)
        
        task_id = self.kernel.create_task(
            name=f"excel_process_{file_path}",
            function=process_file,
            priority=TaskPriority.HIGH,
            data={'file_path': file_path, 'operations': operations}
        )
        
        return task_id
    
    def get_results(self) -> List[Dict]:
        """Get processing results"""
        results = []
        while not self.results_queue.empty():
            try:
                results.append(self.results_queue.get_nowait())
            except queue.Empty:
                break
        return results

# Global FreeRTOS kernel instance
freertos_kernel = FreeRTOSKernel()

# Initialize Excel processing
excel_processor = ExcelProcessingTask(freertos_kernel)

# Create system resources
freertos_kernel.create_semaphore('excel_processing', 3)  # Allow 3 concurrent Excel processes
freertos_kernel.create_event_group('file_upload_complete')
freertos_kernel.create_timer('health_check', 5000, True, lambda: logger.info("System health check"))
freertos_kernel.create_memory_pool('excel_buffer', 1024, 100)  # 100KB buffer pool

# Start the kernel
freertos_kernel.start_scheduler()
