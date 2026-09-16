#include "flight_checkpoint.h"
#include <acl/acl_rt.h>

namespace {
int create_event(void *, flight_device_event *out) {
    aclrtEvent event{};
    aclError rc = aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    *out = reinterpret_cast<flight_device_event>(event);
    return rc;
}
int destroy_event(void *, flight_device_event event) {
    return aclrtDestroyEvent(reinterpret_cast<aclrtEvent>(event));
}
int record_event(void *, flight_device_event event, flight_device_stream stream) {
    return aclrtRecordEvent(reinterpret_cast<aclrtEvent>(event),
                            reinterpret_cast<aclrtStream>(stream));
}
int query_event(void *, flight_device_event event) {
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    aclError rc = aclrtQueryEventStatus(reinterpret_cast<aclrtEvent>(event), &status);
    if (rc != ACL_SUCCESS) return -int(rc ? rc : 1);
    return status == ACL_EVENT_RECORDED_STATUS_COMPLETE ? 1 : 0;
}
}

extern "C" flight_checkpoint_manager *flight_checkpoint_create_cann(uint32_t slots_per_stream) {
    flight_checkpoint_backend backend{};
    backend.create = create_event;
    backend.destroy = destroy_event;
    backend.record = record_event;
    backend.query = query_event;
    return flight_checkpoint_create(&backend, slots_per_stream);
}
