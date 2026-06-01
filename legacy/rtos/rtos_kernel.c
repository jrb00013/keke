#include "rtos_kernel.h"

#define MAX_TASKS 10

static task_t task_list[MAX_TASKS];
static int task_count = 0;

void rtos_init() {
    // Initialize system components and scheduler
    task_count = 0;
}

int create_task(void (*task_function)(), int priority) {
    if (task_count < MAX_TASKS) {
        task_list[task_count].task_function = task_function;
        task_list[task_count].priority = priority;
        task_count++;
        return 0;  // Task created successfully
    }
    return -1;  // Task creation failed
}

void rtos_start() {
    while (1) {
        // Run tasks based on scheduler
        schedule_tasks();
    }
}

void schedule_tasks() {
    // Simple round-robin scheduling
    for (int i = 0; i < task_count; i++) {
        task_list[i].task_function();
    }
}
