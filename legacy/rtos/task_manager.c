#include "task_manager.h"
#include <stddef.h>  // For NULL

#define MAX_TASKS 10  // Maximum number of tasks

static task_t task_list[MAX_TASKS];  // Array of task structures
static int total_tasks = 0;          // Counter for total registered tasks

// Initialize the task manager
void init_task_manager(void) {
    total_tasks = 0;
    for (int i = 0; i < MAX_TASKS; i++) {
        task_list[i].id = -1;
        task_list[i].function = NULL;
        task_list[i].state = TASK_UNUSED;
        task_list[i].priority = 0;
    }
}

// Create a new task
int create_task(void (*task_function)(void), int priority) {
    if (total_tasks >= MAX_TASKS) {
        return -1;  // Task list is full
    }

    for (int i = 0; i < MAX_TASKS; i++) {
        if (task_list[i].state == TASK_UNUSED) {  // Find an empty slot
            task_list[i].id = i;
            task_list[i].function = task_function;
            task_list[i].priority = priority;
            task_list[i].state = TASK_READY;
            total_tasks++;
            return i;  // Return task ID
        }
    }

    return -1;  // No available slots
}

// Delete a task
void delete_task(int task_id) {
    if (task_id < 0 || task_id >= MAX_TASKS || task_list[task_id].state == TASK_UNUSED) {
        return;  // Invalid task ID or already unused
    }

    task_list[task_id].id = -1;
    task_list[task_id].function = NULL;
    task_list[task_id].state = TASK_UNUSED;
    task_list[task_id].priority = 0;
    total_tasks--;
}

// Get a task by ID
task_t *get_task(int task_id) {
    if (task_id < 0 || task_id >= MAX_TASKS || task_list[task_id].state == TASK_UNUSED) {
        return NULL;  // Invalid ID or task not found
    }
    return &task_list[task_id];
}
