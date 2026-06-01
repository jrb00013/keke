#ifndef TASK_MANAGER_H
#define TASK_MANAGER_H

// Task states
typedef enum {
    TASK_UNUSED,   // Task slot is free
    TASK_READY,    // Task is ready to run
    TASK_RUNNING,  // Task is currently running
    TASK_FINISHED  // Task has finished execution
} task_state_t;

// Task structure
typedef struct {
    int id;                     // Task ID
    void (*function)(void);      // Function pointer to task
    int priority;                // Task priority (future use)
    task_state_t state;          // Current task state
} task_t;

// Initialize task manager
void init_task_manager(void);

// Create a new task
int create_task(void (*task_function)(void), int priority);

// Delete a task
void delete_task(int task_id);

// Get task by ID
task_t *get_task(int task_id);

#endif
