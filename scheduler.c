#include "rtos_kernel.h"

void schedule_tasks() {
    // Basic scheduling logic, for instance, round-robin
    for (int i = 0; i < task_count; i++) {
        task_list[i].task_function();
    }
}
