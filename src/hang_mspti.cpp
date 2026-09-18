#include "hang_mspti.h"
#include <mspti.h>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace {
constexpr uint64_t MAGIC = 0x315450534d474e48ULL; // HNGMSPT1 little endian
struct alignas(64) Header {
    uint64_t magic;
    uint32_t version;
    uint32_t event_size;
    uint64_t capacity;
    uint32_t pid;
    uint32_t reserved;
    uint64_t write_seq;
    uint64_t started_wall_ns;
    uint64_t started_mono_ns;
    uint64_t dropped;
    char pad[64];
};
static_assert(sizeof(Header) == 128);
Header *g_header = nullptr;
hang_event *g_events = nullptr;
size_t g_size = 0;
int g_fd = -1;
pid_t g_owner = 0;
msptiSubscriberHandle g_subscriber = nullptr;
std::mutex g_lifecycle;
bool g_enabled = false;

uint64_t now(clockid_t clock) {
    timespec ts{};
    clock_gettime(clock, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}
void name_copy(char (&dst)[64], const char *src) {
    if (!src) { dst[0] = 0; return; }
    size_t n = strnlen(src, sizeof(dst) - 1);
    memcpy(dst, src, n);
    dst[n] = 0;
}
void record(uint16_t kind, uint32_t device, uint32_t stream, uint64_t corr,
            uint64_t start, uint64_t end, uint32_t code, const char *name) {
    Header *h = g_header;
    if (!h || getpid() != g_owner) return;
    uint64_t seq = __atomic_add_fetch(&h->write_seq, 1, __ATOMIC_RELAXED);
    hang_event &slot = g_events[(seq - 1) % h->capacity];
    __atomic_store_n(&slot.seq, uint64_t(0), __ATOMIC_RELEASE);
    slot.observed_ns = now(CLOCK_MONOTONIC);
    slot.activity_start_ns = start;
    slot.activity_end_ns = end;
    slot.correlation_id = corr;
    slot.pid = uint32_t(g_owner);
    slot.tid = uint32_t(syscall(SYS_gettid));
    slot.device_id = device;
    slot.stream_id = stream;
    slot.kind = kind;
    slot.flags = 0;
    slot.code = code;
    name_copy(slot.name, name);
    __atomic_store_n(&slot.seq, seq, __ATOMIC_RELEASE);
}
void callback(void *, msptiCallbackDomain domain, msptiCallbackId cbid,
              const msptiCallbackData *data) {
    if (!data) return;
    uint16_t kind;
    if (domain == MSPTI_CB_DOMAIN_RUNTIME)
        kind = data->callbackSite == MSPTI_API_ENTER ? HANG_RUNTIME_ENTER : HANG_RUNTIME_EXIT;
    else if (domain == MSPTI_CB_DOMAIN_HCCL)
        kind = data->callbackSite == MSPTI_API_ENTER ? HANG_HCCL_ENTER : HANG_HCCL_EXIT;
    else return;
    record(kind, UINT32_MAX, UINT32_MAX, data->correlationId, 0, 0, cbid, data->functionName);
}
constexpr size_t BUFFER_SIZE = 4 * 1024 * 1024;
constexpr size_t BUFFER_COUNT = 8;
uint8_t *g_pool[BUFFER_COUNT]{};
std::atomic<uint8_t> g_pool_busy[BUFFER_COUNT];
void buffer_request(uint8_t **buffer, size_t *size, size_t *max_records) {
    *buffer = nullptr;
    for (size_t i = 0; i < BUFFER_COUNT; ++i) {
        uint8_t free = 0;
        if (g_pool_busy[i].compare_exchange_strong(free, 1, std::memory_order_acq_rel)) {
            *buffer = g_pool[i];
            break;
        }
    }
    *size = *buffer ? BUFFER_SIZE : 0;
    *max_records = 0;
    if (!*buffer && g_header) __atomic_add_fetch(&g_header->dropped, 1, __ATOMIC_RELAXED);
}
void buffer_complete(uint8_t *buffer, size_t, size_t valid) {
    if (valid && buffer) {
        msptiActivity *base = nullptr;
        while (msptiActivityGetNextRecord(buffer, valid, &base) == MSPTI_SUCCESS) {
            if (base->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
                auto *k = reinterpret_cast<msptiActivityKernel *>(base);
                record(HANG_KERNEL_DONE, k->ds.deviceId, k->ds.streamId, k->correlationId,
                       k->start, k->end, 0, k->name);
            } else if (base->kind == MSPTI_ACTIVITY_KIND_HCCL) {
                auto *h = reinterpret_cast<msptiActivityHccl *>(base);
                record(HANG_HCCL_DONE, h->ds.deviceId, h->ds.streamId, 0,
                       h->start, h->end, 0, h->name);
            } else if (base->kind == MSPTI_ACTIVITY_KIND_RUNTIME_API) {
                auto *a = reinterpret_cast<msptiActivityApi *>(base);
                record(HANG_RUNTIME_API_DONE, UINT32_MAX, UINT32_MAX,
                       a->correlationId, a->start, a->end, a->pt.threadId, a->name);
            }
        }
    }
    for (size_t i = 0; i < BUFFER_COUNT; ++i) {
        if (buffer == g_pool[i]) {
            g_pool_busy[i].store(0, std::memory_order_release);
            break;
        }
    }
}
int check(msptiResult result, const char *operation) {
    if (result == MSPTI_SUCCESS) return 0;
    record(HANG_STATUS, UINT32_MAX, UINT32_MAX, 0, 0, 0,
           uint32_t(result), operation);
    return int(result);
}
int set_enabled_locked(bool enabled) {
    if (enabled == g_enabled) return 0;
    int error = 0;
    if (enabled) {
        error = check(msptiEnableDomain(1, g_subscriber, MSPTI_CB_DOMAIN_RUNTIME), "runtime_domain");
        if (!error) error = check(msptiEnableDomain(1, g_subscriber, MSPTI_CB_DOMAIN_HCCL), "hccl_domain");
        if (!error) error = check(msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL), "kernel_activity");
        if (!error) error = check(msptiActivityEnable(MSPTI_ACTIVITY_KIND_HCCL), "hccl_activity");
        if (!error) error = check(msptiActivityEnable(MSPTI_ACTIVITY_KIND_RUNTIME_API), "runtime_api_activity");
        if (!error) { g_enabled = true; record(HANG_STATUS, UINT32_MAX, UINT32_MAX, 0, 0, 0, 0, "enabled"); }
    } else {
        msptiActivityFlushAll(1);
        error = check(msptiEnableDomain(0, g_subscriber, MSPTI_CB_DOMAIN_RUNTIME), "runtime_domain_off");
        if (!error) error = check(msptiEnableDomain(0, g_subscriber, MSPTI_CB_DOMAIN_HCCL), "hccl_domain_off");
        if (!error) error = check(msptiActivityDisable(MSPTI_ACTIVITY_KIND_KERNEL), "kernel_activity_off");
        if (!error) error = check(msptiActivityDisable(MSPTI_ACTIVITY_KIND_HCCL), "hccl_activity_off");
        if (!error) error = check(msptiActivityDisable(MSPTI_ACTIVITY_KIND_RUNTIME_API), "runtime_api_activity_off");
        if (!error) { g_enabled = false; record(HANG_STATUS, UINT32_MAX, UINT32_MAX, 0, 0, 0, 0, "disabled"); }
    }
    return error;
}
void cleanup_map() {
    if (g_header) munmap(g_header, g_size);
    if (g_fd >= 0) close(g_fd);
    g_header = nullptr; g_events = nullptr; g_fd = -1; g_size = 0; g_owner = 0;
}
}

extern "C" int hang_mspti_start(const char *path, uint64_t capacity) {
    std::lock_guard<std::mutex> lock(g_lifecycle);
    if (!path || capacity < 64 || capacity > (1ULL << 24) || g_header) return EINVAL;
    if (capacity > (SIZE_MAX - sizeof(Header)) / sizeof(hang_event)) return EOVERFLOW;
    g_size = sizeof(Header) + size_t(capacity) * sizeof(hang_event);
    g_fd = open(path, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (g_fd < 0) return errno;
    if (ftruncate(g_fd, off_t(g_size)) != 0) { int e=errno; cleanup_map(); unlink(path); return e; }
    void *mapped = mmap(nullptr, g_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, 0);
    if (mapped == MAP_FAILED) { int e=errno; cleanup_map(); unlink(path); return e; }
    g_header = static_cast<Header *>(mapped);
    g_events = reinterpret_cast<hang_event *>(static_cast<char *>(mapped) + sizeof(Header));
    g_owner = getpid();
    g_header->magic = MAGIC;
    g_header->version = 1;
    g_header->event_size = sizeof(hang_event);
    g_header->capacity = capacity;
    g_header->pid = uint32_t(g_owner);
    g_header->started_wall_ns = now(CLOCK_REALTIME);
    g_header->started_mono_ns = now(CLOCK_MONOTONIC);
    for (size_t i = 0; i < BUFFER_COUNT; ++i) {
        if (!g_pool[i] && posix_memalign(reinterpret_cast<void **>(&g_pool[i]), 8, BUFFER_SIZE) != 0) {
            record(HANG_STATUS, UINT32_MAX, UINT32_MAX, 0, 0, 0, ENOMEM, "buffer_pool");
            for (size_t j = 0; j < i; ++j) { free(g_pool[j]); g_pool[j] = nullptr; }
            cleanup_map();
            return ENOMEM;
        }
        g_pool_busy[i].store(0, std::memory_order_relaxed);
    }
    int error = check(msptiSubscribe(&g_subscriber, callback, nullptr), "subscribe");
    if (!error) error = check(msptiActivityRegisterCallbacks(buffer_request, buffer_complete), "buffers");
    if (!error) error = set_enabled_locked(true);
    if (error) {
        if (g_subscriber) { msptiUnsubscribe(g_subscriber); g_subscriber = nullptr; }
        cleanup_map();
    }
    return error;
}

extern "C" int hang_mspti_flush(void) {
    if (!g_header || getpid() != g_owner) return EINVAL;
    return check(msptiActivityFlushAll(1), "flush");
}

extern "C" int hang_mspti_set_enabled(int enabled) {
    std::lock_guard<std::mutex> lock(g_lifecycle);
    if (!g_header || getpid() != g_owner) return EINVAL;
    return set_enabled_locked(enabled != 0);
}

extern "C" void hang_mspti_stop(void) {
    std::lock_guard<std::mutex> lock(g_lifecycle);
    if (!g_header || getpid() != g_owner) return;
    msptiActivityFlushAll(1);
    set_enabled_locked(false);
    if (g_subscriber) { msptiUnsubscribe(g_subscriber); g_subscriber = nullptr; }
    cleanup_map();
}
