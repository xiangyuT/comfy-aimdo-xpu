#include "plat.h"

#include <stdarg.h>

int log_level;

typedef void (*AimdoLogCallback)(int level, const char *message);

static AimdoLogCallback log_callback;

SHARED_EXPORT
void set_log_callback(AimdoLogCallback callback) {
    log_callback = callback;
}

void aimdo_log_message(int level, const char *message) {
    if (log_callback) {
        log_callback(level, message);
    } else {
        fputs(message, stderr);
        fflush(stderr);
    }
}

void aimdo_log(int level, const char *file, int line, const char *format, ...) {
    char message[2048];
    int prefix_length;
    size_t prefix;
    va_list args;

    prefix_length = snprintf(message, sizeof(message), "aimdo: %s:%d:%s:",
                             file, line, get_level_str(level));
    if (prefix_length < 0) {
        return;
    }
    prefix = MIN((size_t)prefix_length, sizeof(message) - 1);
    va_start(args, format);
    vsnprintf(message + prefix, sizeof(message) - prefix, format, args);
    va_end(args);

    aimdo_log_message(level, message);
}

static inline void set_log_level(int level) {
    log_level = level;
}

SHARED_EXPORT void set_log_level_none() { set_log_level(__NONE__); }
SHARED_EXPORT void set_log_level_critical() { set_log_level(CRITICAL); }
SHARED_EXPORT void set_log_level_error() { set_log_level(AIMDO_LOG_ERROR); }
SHARED_EXPORT void set_log_level_warning() { set_log_level(WARNING); }
SHARED_EXPORT void set_log_level_info() { set_log_level(INFO); }
SHARED_EXPORT void set_log_level_debug() { set_log_level(DEBUG); }
SHARED_EXPORT void set_log_level_verbose() { set_log_level(VERBOSE); }
SHARED_EXPORT void set_log_level_vverbose() { set_log_level(VVERBOSE); }

static const char *level_strs [] = {
    #define LEVEL_STR1(L) [L] = #L
    LEVEL_STR1(ALL),
    LEVEL_STR1(CRITICAL),
    [AIMDO_LOG_ERROR] = "ERROR",
    LEVEL_STR1(WARNING),
    LEVEL_STR1(INFO),
    LEVEL_STR1(DEBUG),
    LEVEL_STR1(VERBOSE),
    LEVEL_STR1(VVERBOSE),
};

const char *get_level_str(int level) {
    if (level < 0 || level > VVERBOSE) {
        return "UNKNOWN";
    }
    return level_strs[level];
}

uint64_t log_shot_counter;

void log_reset_shots() {
    log_shot_counter++;
}
