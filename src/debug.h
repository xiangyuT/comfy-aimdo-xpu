#pragma once

/* Shared by the C core and C++ backends without importing platform headers. */
#ifdef __cplusplus
extern "C" {
#endif

enum DebugLevels {
    __NONE__ = -1,
    /* Default to everything so native diagnostics work before Python init. */
    ALL = 0,
    CRITICAL,
    AIMDO_LOG_ERROR,
    WARNING,
    INFO,
    DEBUG,
    VERBOSE,
    VVERBOSE,
};

/* Deliver an already formatted record without the aimdo_log format limit.
 * Uses the registered collector, or stderr with an immediate flush. */
void aimdo_log_message(int level, const char *message);

#ifdef __cplusplus
}
#endif
