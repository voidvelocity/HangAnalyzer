#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One file per process. Values are deliberately independent of msPTI enum ABI. */
enum hang_kind {
    HANG_RUNTIME_ENTER = 1, HANG_RUNTIME_EXIT = 2,
    HANG_HCCL_ENTER = 3, HANG_HCCL_EXIT = 4,
    HANG_KERNEL_DONE = 5, HANG_HCCL_DONE = 6,
    HANG_STATUS = 7, HANG_RUNTIME_API_DONE = 8
};

typedef struct __attribute__((aligned(64))) hang_event {
    uint64_t seq;                 /* published last; zero means incomplete */
    uint64_t observed_ns;         /* CLOCK_MONOTONIC, when recorder observed it */
    uint64_t activity_start_ns;   /* msPTI clock; only for activity records */
    uint64_t activity_end_ns;     /* msPTI clock; only for activity records */
    uint64_t correlation_id;
    uint32_t pid;
    uint32_t tid;
    uint32_t device_id;
    uint32_t stream_id;
    uint16_t kind;
    uint16_t flags;
    uint32_t code;                /* callback ID or msPTI error */
    char name[64];                /* truncated UTF-8/ASCII label */
} hang_event;

/* Call once in each worker process, after fork and before NPU work. */
int hang_mspti_start(const char *path, uint64_t capacity);
void hang_mspti_stop(void);
/* Flush completed Activity records. May block; call from a control thread only. */
int hang_mspti_flush(void);
/* Disable/enable collection without discarding the ring. */
int hang_mspti_set_enabled(int enabled);

#ifdef __cplusplus
}
static_assert(sizeof(hang_event) == 128, "hang_event ABI changed");
#endif
