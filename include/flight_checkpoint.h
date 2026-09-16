#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *flight_device_event;
typedef void *flight_device_stream;

/* query: 1=complete, 0=not ready, negative=backend error. */
typedef struct flight_checkpoint_backend {
    void *user;
    int (*create)(void *user, flight_device_event *event);
    int (*destroy)(void *user, flight_device_event event);
    int (*record)(void *user, flight_device_event event, flight_device_stream stream);
    int (*query)(void *user, flight_device_event event);
} flight_checkpoint_backend;

typedef struct flight_checkpoint_manager flight_checkpoint_manager;

/* Registration is control path: it allocates a fixed event pool for one stream. */
flight_checkpoint_manager *flight_checkpoint_create(const flight_checkpoint_backend *backend,
                                                     uint32_t slots_per_stream);
int flight_checkpoint_register_stream(flight_checkpoint_manager *manager, uint32_t stream_id,
                                      flight_device_stream stream);

/* Sparse hot-path checkpoint. Returns 0 or errno-style EAGAIN/ENOENT/backend error. */
int flight_checkpoint_submit(flight_checkpoint_manager *manager, uint32_t stream_id,
                             uint64_t submitted_seq, uint64_t checkpoint_id,
                             uint64_t *generation);

/* Never synchronizes. Queries at most budget pending events; returns newly completed count. */
int flight_checkpoint_poll(flight_checkpoint_manager *manager, uint32_t budget);

/* Optional C++ poller keeps progressing even when the Python/business thread blocks. */
int flight_checkpoint_start_poller(flight_checkpoint_manager *manager,
                                   uint32_t interval_us, uint32_t budget);
void flight_checkpoint_stop_poller(flight_checkpoint_manager *manager);

/* Returns EBUSY and keeps the manager alive if any event remains pending. */
int flight_checkpoint_destroy(flight_checkpoint_manager *manager);

/* Optional CANN adapter, exported by libflightcheckpoint_cann.so. */
flight_checkpoint_manager *flight_checkpoint_create_cann(uint32_t slots_per_stream);

#ifdef __cplusplus
}
#endif
