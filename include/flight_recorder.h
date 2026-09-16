#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Every record is a host observation. END means the host API returned, not device completion. */
enum flight_type {
    FL_REQUEST_BEGIN=1, FL_REQUEST_END, FL_SCHEDULER_BEGIN, FL_SCHEDULER_END,
    FL_MODEL_BEGIN, FL_MODEL_END, FL_GRAPH_BEGIN, FL_GRAPH_END,
    FL_KV_BEGIN, FL_KV_END, FL_PP_SEND_BEGIN, FL_PP_SEND_END,
    FL_PP_RECV_BEGIN, FL_PP_RECV_END, FL_HCCL_BEGIN, FL_HCCL_END,
    FL_EVENT_RECORD, FL_EVENT_WAIT, FL_STREAM_SYNC_BEGIN, FL_STREAM_SYNC_END,
    FL_DEVICE_SYNC_BEGIN, FL_DEVICE_SYNC_END, FL_CHECKPOINT,
    FL_DEVICE_CONFIRMED
};

enum flight_checkpoint_flags {
    FL_CHECKPOINT_SUBMITTED = 1U << 0,
    FL_CHECKPOINT_POOL_FULL = 1U << 1,
    FL_CHECKPOINT_QUERY_ERROR = 1U << 2,
    FL_CHECKPOINT_NOT_READY = 1U << 3,
};

/* Fixed little-endian layout; seq is published last. 0 means an incomplete slot. */
typedef struct __attribute__((aligned(64))) flight_event {
    uint64_t timestamp_ns;
    uint64_t seq;
    uint32_t pid;
    uint32_t tid;
    uint16_t device_id;
    uint16_t rank_id;
    uint32_t stream_id;
    uint16_t type;
    uint16_t flags;
    uint64_t correlation_id;
    uint64_t arg0;
    uint64_t arg1;
} flight_event;

/* Initialize once per process, after fork. path is a unique rank/PID file. */
int flight_init(const char *path, uint16_t rank, uint16_t device, uint64_t capacity);
uint64_t flight_record(uint16_t type, uint32_t stream, uint64_t correlation,
                       uint64_t arg0, uint64_t arg1, uint16_t flags);
void flight_close(void);

#ifdef __cplusplus
}

#include <cstddef>
static_assert(sizeof(flight_event) == 64, "flight_event ABI changed");
static_assert(offsetof(flight_event, correlation_id) == 40, "flight_event padding changed");
static_assert(offsetof(flight_event, arg0) == 48, "flight_event padding changed");
static_assert(offsetof(flight_event, arg1) == 56, "flight_event padding changed");
#endif
