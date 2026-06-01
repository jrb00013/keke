#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "task_manager.h"

// Initialize scheduler
void init_scheduler(void);

// Add a task to the scheduler
int add_task(void (*task_function)(void), int priority);

// Schedule the next task
void schedule_next_task(void);

// Run a given task
void run_task(task_t *task);

// Yield control to the scheduler
void yield(void);

#endif
