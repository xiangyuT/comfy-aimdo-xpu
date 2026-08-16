#include "thread-plat.h"

#include <stdlib.h>

Mutex mutex_create(void) {
    CRITICAL_SECTION *mutex = malloc(sizeof(*mutex));

    if (!mutex) {
        return NULL;
    }

    InitializeCriticalSection(mutex);
    return mutex;
}

void mutex_lock(Mutex mutex) {
    EnterCriticalSection(mutex);
}

bool mutex_try_lock(Mutex mutex) {
    return TryEnterCriticalSection(mutex) != 0;
}

void mutex_unlock(Mutex mutex) {
    LeaveCriticalSection(mutex);
}

void mutex_destroy(Mutex mutex) {
    if (!mutex) {
        return;
    }

    DeleteCriticalSection(mutex);
    free(mutex);
}

CondVar condvar_create(void) {
    CONDITION_VARIABLE *condvar = malloc(sizeof(*condvar));

    if (!condvar) {
        return NULL;
    }

    InitializeConditionVariable(condvar);
    return condvar;
}

void condvar_wait(CondVar condvar, Mutex mutex) {
    SleepConditionVariableCS(condvar, mutex, INFINITE);
}

void condvar_signal(CondVar condvar) {
    WakeConditionVariable(condvar);
}

void condvar_broadcast(CondVar condvar) {
    WakeAllConditionVariable(condvar);
}

void condvar_destroy(CondVar condvar) {
    free(condvar);
}

bool thread_create(Thread *thread, ThreadProc proc, void *arg) {
    *thread = CreateThread(NULL, 0, proc, arg, 0, NULL);
    return *thread != NULL;
}

void thread_join(Thread thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}
