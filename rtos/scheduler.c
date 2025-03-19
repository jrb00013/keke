#include "scheduler.h"
#include "task_manager.h"
#include <stddef.h>

#define MAX_TASKS 10  // Define the max number of tasks in the system

static task_t task_list[MAX_TASKS];  // Task list (array of tasks)
static int current_task = -1;        // Index of the current running task
static int total_tasks = 0;          // Total registered tasks

// Initialize the scheduler
void init_scheduler(void) {
    total_tasks = 0;
    current_task = -1;
}

// Add a new task to the scheduler
int add_task(void (*task_function)(void), int priority) {
    if (total_tasks >= MAX_TASKS) {
        return -1;  // Scheduler is full
    }

    task_list[total_tasks].id = total_tasks;
    task_list[total_tasks].function = task_function;
    task_list[total_tasks].priority = priority;
    task_list[total_tasks].state = TASK_READY;
    
    total_tasks++;
    return total_tasks - 1;  // Return task ID
}

// Simple round-robin scheduler
void schedule_next_task(void) {
    if (total_tasks == 0) {
        return;  // No tasks to schedule
    }

    // Find next ready task
    int next_task = (current_task + 1) % total_tasks;
    while (task_list[next_task].state != TASK_READY) {
        next_task = (next_task + 1) % total_tasks;
        if (next_task == current_task) {
            return;  // No ready tasks
        }
    }

    current_task = next_task;
    run_task(&task_list[current_task]);
}

// Run the selected task
void run_task(task_t *task) {
    if (task->state == TASK_READY) {
        task->state = TASK_RUNNING;
        task->function();
        task->state = TASK_FINISHED;  // Mark as finished after execution
    }
}

// Yield control to the scheduler
void yield(void) {
    schedule_next_task();
}
