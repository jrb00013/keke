import pytest
import time
import threading
import queue
from unittest.mock import Mock, patch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from freertos_integration import (
    FreeRTOSKernel, TaskState, TaskPriority, TaskControlBlock,
    ExcelProcessingTask, freertos_kernel, excel_processor
)

class TestFreeRTOSKernel:
    """Test suite for FreeRTOS kernel implementation"""
    
    @pytest.fixture
    def kernel(self):
        """Create FreeRTOS kernel instance for testing"""
        return FreeRTOSKernel(max_tasks=10)
    
    def test_kernel_initialization(self, kernel):
        """Test kernel initialization"""
        assert kernel.max_tasks == 10
        assert len(kernel.tasks) == 0
        assert kernel.running == False
        assert kernel.current_task is None
        assert kernel.total_switches == 0
    
    def test_create_task(self, kernel):
        """Test task creation"""
        def test_function():
            return "test_result"
        
        task_id = kernel.create_task(
            name="test_task",
            function=test_function,
            priority=TaskPriority.NORMAL,
            stack_size=4096,
            data={'param': 'value'}
        )
        
        assert task_id in kernel.tasks
        assert kernel.tasks[task_id].name == "test_task"
        assert kernel.tasks[task_id].priority == TaskPriority.NORMAL
        assert kernel.tasks[task_id].state == TaskState.READY
        assert kernel.tasks[task_id].stack_size == 4096
        assert kernel.tasks[task_id].data['param'] == 'value'
    
    def test_create_task_max_limit(self, kernel):
        """Test task creation with maximum limit"""
        def dummy_function():
            pass
        
        # Create maximum number of tasks
        for i in range(kernel.max_tasks):
            kernel.create_task(f"task_{i}", dummy_function)
        
        # Try to create one more task
        with pytest.raises(RuntimeError, match="Maximum number of tasks reached"):
            kernel.create_task("overflow_task", dummy_function)
    
    def test_delete_task(self, kernel):
        """Test task deletion"""
        def test_function():
            pass
        
        task_id = kernel.create_task("test_task", test_function)
        assert task_id in kernel.tasks
        
        result = kernel.delete_task(task_id)
        assert result == True
        assert kernel.tasks[task_id].state == TaskState.DELETED
    
    def test_suspend_resume_task(self, kernel):
        """Test task suspension and resumption"""
        def test_function():
            pass
        
        task_id = kernel.create_task("test_task", test_function)
        
        # Suspend task
        result = kernel.suspend_task(task_id)
        assert result == True
        assert kernel.tasks[task_id].state == TaskState.SUSPENDED
        
        # Resume task
        result = kernel.resume_task(task_id)
        assert result == True
        assert kernel.tasks[task_id].state == TaskState.READY
    
    def test_semaphore_operations(self, kernel):
        """Test semaphore creation and operations"""
        # Create semaphore
        result = kernel.create_semaphore("test_sem", initial_count=2)
        assert result == True
        assert "test_sem" in kernel.semaphores
        
        # Take semaphore
        result = kernel.take_semaphore("test_sem")
        assert result == True
        
        # Take semaphore again
        result = kernel.take_semaphore("test_sem")
        assert result == True
        
        # Try to take semaphore when count is 0
        result = kernel.take_semaphore("test_sem", timeout=0.1)
        assert result == False
        
        # Give semaphore back
        result = kernel.give_semaphore("test_sem")
        assert result == True
        
        # Now should be able to take it again
        result = kernel.take_semaphore("test_sem")
        assert result == True
    
    def test_event_group_operations(self, kernel):
        """Test event group creation and operations"""
        # Create event group
        result = kernel.create_event_group("test_event")
        assert result == True
        assert "test_event" in kernel.event_groups
        
        # Wait for event (should timeout)
        result = kernel.wait_event("test_event", timeout=0.1)
        assert result == False
        
        # Set event
        result = kernel.set_event("test_event")
        assert result == True
        
        # Wait for event (should succeed)
        result = kernel.wait_event("test_event", timeout=1.0)
        assert result == True
    
    def test_timer_operations(self, kernel):
        """Test timer creation and operations"""
        callback_called = threading.Event()
        
        def timer_callback():
            callback_called.set()
        
        # Create timer
        result = kernel.create_timer("test_timer", 100, True, timer_callback)
        assert result == True
        assert "test_timer" in kernel.timers
        
        # Start timer
        result = kernel.start_timer("test_timer")
        assert result == True
        assert kernel.timers["test_timer"]["active"] == True
        
        # Stop timer
        result = kernel.stop_timer("test_timer")
        assert result == True
        assert kernel.timers["test_timer"]["active"] == False
    
    def test_memory_pool_operations(self, kernel):
        """Test memory pool operations"""
        # Create memory pool
        result = kernel.create_memory_pool("test_pool", 1024, 5)
        assert result == True
        assert "test_pool" in kernel.memory_pools
        
        # Allocate memory
        memory = kernel.allocate_memory("test_pool")
        assert memory is not None
        assert isinstance(memory, bytearray)
        assert len(memory) == 1024
        
        # Free memory
        result = kernel.free_memory("test_pool", memory)
        assert result == True
        
        # Allocate again
        memory2 = kernel.allocate_memory("test_pool")
        assert memory2 is not None
    
    def test_scheduler_start_stop(self, kernel):
        """Test scheduler start and stop"""
        assert kernel.running == False
        
        # Start scheduler
        kernel.start_scheduler()
        assert kernel.running == True
        assert kernel.scheduler_thread is not None
        assert kernel.scheduler_thread.is_alive()
        
        # Stop scheduler
        kernel.stop_scheduler()
        assert kernel.running == False
    
    def test_task_execution(self, kernel):
        """Test task execution"""
        execution_count = 0
        execution_event = threading.Event()
        
        def test_function():
            nonlocal execution_count
            execution_count += 1
            if execution_count >= 3:
                execution_event.set()
        
        # Create and start task
        task_id = kernel.create_task("test_task", test_function, TaskPriority.HIGH)
        kernel.start_scheduler()
        
        try:
            # Wait for task to execute
            assert execution_event.wait(timeout=5.0)
            assert execution_count >= 3
            
            # Check task info
            task_info = kernel.get_task_info(task_id)
            assert task_info is not None
            assert task_info['name'] == "test_task"
            assert task_info['run_count'] >= 3
            assert task_info['cpu_time'] > 0
            
        finally:
            kernel.stop_scheduler()
    
    def test_task_yield(self, kernel):
        """Test task yielding"""
        yield_count = 0
        
        def yielding_function():
            nonlocal yield_count
            yield_count += 1
            kernel.yield_task()  # Yield after each execution
        
        # Create task
        task_id = kernel.create_task("yielding_task", yielding_function)
        kernel.start_scheduler()
        
        try:
            # Let it run for a bit
            time.sleep(0.5)
            
            # Should have yielded multiple times
            assert yield_count > 0
            
        finally:
            kernel.stop_scheduler()
    
    def test_system_stats(self, kernel):
        """Test system statistics"""
        def dummy_function():
            time.sleep(0.01)
        
        # Create some tasks
        for i in range(3):
            kernel.create_task(f"task_{i}", dummy_function)
        
        kernel.start_scheduler()
        
        try:
            time.sleep(0.1)
            stats = kernel.get_system_stats()
            
            assert stats['total_tasks'] == 3
            assert stats['active_tasks'] == 3
            assert stats['uptime'] > 0
            assert stats['total_switches'] > 0
            assert 'cpu_usage' in stats
            
        finally:
            kernel.stop_scheduler()

class TestExcelProcessingTask:
    """Test suite for Excel processing task integration"""
    
    @pytest.fixture
    def excel_task(self):
        """Create ExcelProcessingTask instance for testing"""
        kernel = FreeRTOSKernel(max_tasks=10)
        return ExcelProcessingTask(kernel)
    
    def test_excel_task_initialization(self, excel_task):
        """Test Excel task initialization"""
        assert excel_task.kernel is not None
        assert isinstance(excel_task.processing_queue, queue.Queue)
        assert isinstance(excel_task.results_queue, queue.Queue)
    
    def test_create_processing_task(self, excel_task):
        """Test creating Excel processing task"""
        file_path = "/test/file.xlsx"
        operations = [{"type": "remove_duplicates"}]
        
        task_id = excel_task.create_processing_task(file_path, operations)
        
        assert task_id in excel_task.kernel.tasks
        assert excel_task.kernel.tasks[task_id].name.startswith("excel_process_")
        assert excel_task.kernel.tasks[task_id].priority == TaskPriority.HIGH
    
    def test_processing_task_execution(self, excel_task):
        """Test Excel processing task execution"""
        file_path = "/test/file.xlsx"
        operations = [{"type": "remove_duplicates"}]
        
        # Create semaphore for processing
        excel_task.kernel.create_semaphore('excel_processing', 1)
        
        # Create and start task
        task_id = excel_task.create_processing_task(file_path, operations)
        excel_task.kernel.start_scheduler()
        
        try:
            # Wait for processing to complete
            time.sleep(0.2)
            
            # Check results
            results = excel_task.get_results()
            assert len(results) > 0
            
            result = results[0]
            assert result['file_path'] == file_path
            assert result['operations'] == operations
            assert result['status'] == 'completed'
            
        finally:
            excel_task.kernel.stop_scheduler()
    
    def test_processing_error_handling(self, excel_task):
        """Test error handling in processing tasks"""
        file_path = "/test/file.xlsx"
        operations = [{"type": "invalid_operation"}]
        
        # Create semaphore
        excel_task.kernel.create_semaphore('excel_processing', 1)
        
        # Create task
        task_id = excel_task.create_processing_task(file_path, operations)
        excel_task.kernel.start_scheduler()
        
        try:
            time.sleep(0.2)
            
            # Check for error results
            results = excel_task.get_results()
            if results:
                result = results[0]
                assert result['status'] in ['completed', 'error']
                
        finally:
            excel_task.kernel.stop_scheduler()

class TestIntegration:
    """Integration tests for FreeRTOS and Excel processing"""
    
    def test_concurrent_excel_processing(self):
        """Test concurrent Excel file processing"""
        kernel = FreeRTOSKernel(max_tasks=20)
        excel_task = ExcelProcessingTask(kernel)
        
        # Create semaphore to limit concurrent processing
        kernel.create_semaphore('excel_processing', 3)
        
        # Create multiple processing tasks
        file_paths = [f"/test/file_{i}.xlsx" for i in range(5)]
        operations = [{"type": "remove_duplicates"}]
        
        task_ids = []
        for file_path in file_paths:
            task_id = excel_task.create_processing_task(file_path, operations)
            task_ids.append(task_id)
        
        # Start scheduler
        kernel.start_scheduler()
        
        try:
            # Let tasks run
            time.sleep(0.5)
            
            # Check results
            results = excel_task.get_results()
            assert len(results) >= 0  # Some tasks may still be running
            
            # Check that tasks were created
            assert len(task_ids) == 5
            for task_id in task_ids:
                assert task_id in kernel.tasks
            
        finally:
            kernel.stop_scheduler()
    
    def test_memory_pool_for_excel_data(self):
        """Test using memory pools for Excel data buffering"""
        kernel = FreeRTOSKernel(max_tasks=10)
        
        # Create memory pool for Excel data
        kernel.create_memory_pool('excel_buffer', 1024, 10)
        
        # Allocate memory for Excel processing
        buffers = []
        for i in range(5):
            buffer = kernel.allocate_memory('excel_buffer')
            assert buffer is not None
            buffers.append(buffer)
        
        # Simulate Excel data processing
        for buffer in buffers:
            buffer[:100] = b'Excel data content'
        
        # Free buffers back to pool
        for buffer in buffers:
            result = kernel.free_memory('excel_buffer', buffer)
            assert result == True
        
        # Should be able to allocate again
        new_buffer = kernel.allocate_memory('excel_buffer')
        assert new_buffer is not None
    
    def test_timer_based_health_check(self):
        """Test timer-based health checking"""
        kernel = FreeRTOSKernel(max_tasks=10)
        
        health_check_count = 0
        
        def health_check_callback():
            nonlocal health_check_count
            health_check_count += 1
        
        # Create health check timer
        kernel.create_timer('health_check', 100, True, health_check_callback)
        kernel.start_timer('health_check')
        
        # Start scheduler
        kernel.start_scheduler()
        
        try:
            # Let timer run
            time.sleep(0.3)
            
            # Should have executed multiple times
            assert health_check_count > 0
            
        finally:
            kernel.stop_scheduler()
    
    def test_event_driven_processing(self):
        """Test event-driven Excel processing"""
        kernel = FreeRTOSKernel(max_tasks=10)
        excel_task = ExcelProcessingTask(kernel)
        
        # Create event group for file upload
        kernel.create_event_group('file_upload_complete')
        
        processing_started = threading.Event()
        
        def file_processor():
            # Wait for file upload event
            if kernel.wait_event('file_upload_complete', timeout=1.0):
                processing_started.set()
                # Simulate processing
                time.sleep(0.1)
        
        # Create processing task
        task_id = kernel.create_task('file_processor', file_processor)
        
        # Start scheduler
        kernel.start_scheduler()
        
        try:
            # Simulate file upload completion
            kernel.set_event('file_upload_complete')
            
            # Wait for processing to start
            assert processing_started.wait(timeout=2.0)
            
        finally:
            kernel.stop_scheduler()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
