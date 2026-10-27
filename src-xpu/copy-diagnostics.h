#pragma once

#include "debug.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstring>

namespace aimdo_xpu {

/* Only bounded metadata belongs in this formatter. Exception text uses the
 * chunked path below so an arbitrarily long what() is never silently cut off.
 * No heap allocation or device query is needed to report the primary error. */
inline void log_h2d_record(int level, uint64_t call, const char *stage,
                           const char *format, ...) {
    char fields[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(fields, sizeof(fields), format, args);
    va_end(args);
    char record[640];
    std::snprintf(record, sizeof(record),
                  "[AIMDO XPU %s] op=h2d call=%llu stage=%s %s\n",
                  level == AIMDO_LOG_ERROR ? "ERROR" : "WARNING",
                  static_cast<unsigned long long>(call), stage, fields);
    aimdo_log_message(level, record);
}

inline void log_h2d_exception(int level, uint64_t call, const char *stage,
                              const char *message) {
    constexpr size_t kChunkBytes = 192;
    if (!message) message = "<no exception message>";
    size_t remaining = std::strlen(message);
    size_t part = 0;
    do {
        size_t count = std::min(remaining, kChunkBytes);
        if (count < remaining) {
            /* Keep valid UTF-8 code points together for Python's decoder. */
            while (count &&
                   (static_cast<unsigned char>(message[count]) & 0xc0) == 0x80)
                --count;
            if (!count) count = std::min(remaining, kChunkBytes);
        }
        char escaped[kChunkBytes * 2 + 1];
        size_t used = 0;
        for (size_t i = 0; i < count; ++i) {
            const char value = message[i];
            if (value == '\n' || value == '\r' || value == '\\') {
                escaped[used++] = '\\';
                escaped[used++] = value == '\n' ? 'n' : value == '\r' ? 'r' : '\\';
            } else {
                escaped[used++] = value;
            }
        }
        escaped[used] = '\0';
        log_h2d_record(level, call, stage,
                        "message_part=%zu message=%s", ++part, escaped);
        message += count;
        remaining -= count;
    } while (remaining);
    log_h2d_record(level, call, stage, "message_end parts=%zu", part);
}

}  // namespace aimdo_xpu
