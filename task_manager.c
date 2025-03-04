#include "rtos_kernel.h"

int create_task(void (*task_function)(), int priority) {
    if (task_count < MAX_TASKS) {
        task_list[task_count].task_function = task_function;
        task_list[task_count].priority = priority;
        task_count++;
        return 0;
    }
    return -1;  // No more tasks can be created
}
