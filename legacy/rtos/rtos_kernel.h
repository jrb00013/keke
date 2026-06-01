#ifndef RTOS_KERNEL_H
#define RTOS_KERNEL_H

typedef struct {
    void (*task_function)();
    int priority;
} task_t;

void rtos_init();
int create_task(void (*task_function)(), int priority);
void rtos_start();
void schedule_tasks();

#endif
