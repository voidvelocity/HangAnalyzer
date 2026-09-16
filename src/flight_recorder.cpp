#include "flight_recorder.h"
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <pthread.h>
#endif

namespace {
constexpr uint64_t MAGIC = 0x31524647534e4341ULL; // ACNSGFR1, little endian
struct alignas(64) Header {
    uint64_t magic;
    uint32_t version;
    uint32_t event_size;
    uint64_t capacity;
    uint32_t pid;
    uint16_t rank;
    uint16_t device;
    uint64_t write_seq;
    uint64_t last_ns;
    uint64_t started_wall_ns;
    uint64_t started_mono_ns;
    char pad[64];
};
static_assert(sizeof(Header) == 128);
static_assert(sizeof(flight_event) == 64);
Header *header = nullptr;
flight_event *events = nullptr;
size_t mapped_size = 0;
int fd = -1;
pid_t owner = 0;
uint64_t ns(clockid_t id) {
    timespec t{};
    clock_gettime(id, &t); // normally vDSO; may fall back to a syscall
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}
uint32_t thread_id() {
#ifdef __APPLE__
    uint64_t value = 0;
    pthread_threadid_np(nullptr, &value);
    return uint32_t(value);
#else
    thread_local uint32_t value = uint32_t(syscall(SYS_gettid));
    return value;
#endif
}
}

extern "C" int flight_init(const char *path, uint16_t rank, uint16_t device, uint64_t capacity) {
    if (!path || capacity < 2 || capacity > (1ULL << 26) || header) return EINVAL;
    if (capacity > (SIZE_MAX - sizeof(Header)) / sizeof(flight_event)) return EOVERFLOW;
    mapped_size = sizeof(Header) + size_t(capacity) * sizeof(flight_event);
    fd = open(path, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) return errno;
    if (ftruncate(fd, off_t(mapped_size)) != 0) { int e=errno; close(fd); fd=-1; unlink(path); return e; }
    void *p = mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { int e=errno; close(fd); fd=-1; unlink(path); return e; }
    header = static_cast<Header *>(p);
    events = reinterpret_cast<flight_event *>(static_cast<char *>(p) + sizeof(Header));
    owner = getpid();
    header->magic = MAGIC;
    header->version = 1;
    header->event_size = sizeof(flight_event);
    header->capacity = capacity;
    header->pid = uint32_t(owner);
    header->rank = rank;
    header->device = device;
    header->started_wall_ns = ns(CLOCK_REALTIME);
    header->started_mono_ns = ns(CLOCK_MONOTONIC);
    __atomic_store_n(&header->last_ns, header->started_mono_ns, __ATOMIC_RELEASE);
    return 0;
}

extern "C" uint64_t flight_record(uint16_t type, uint32_t stream, uint64_t correlation,
                                   uint64_t arg0, uint64_t arg1, uint16_t flags) {
    if (!header || getpid() != owner) return 0;
    uint64_t seq = __atomic_add_fetch(&header->write_seq, 1, __ATOMIC_RELAXED);
    flight_event &slot = events[(seq - 1) % header->capacity];
    __atomic_store_n(&slot.seq, uint64_t(0), __ATOMIC_RELEASE);
    slot.timestamp_ns = ns(CLOCK_MONOTONIC);
    slot.pid = uint32_t(owner);
    slot.tid = thread_id();
    slot.device_id = header->device;
    slot.rank_id = header->rank;
    slot.stream_id = stream;
    slot.type = type;
    slot.flags = flags;
    slot.correlation_id = correlation;
    slot.arg0 = arg0;
    slot.arg1 = arg1;
    __atomic_store_n(&slot.seq, seq, __ATOMIC_RELEASE);
    __atomic_store_n(&header->last_ns, slot.timestamp_ns, __ATOMIC_RELEASE);
    return seq;
}

extern "C" void flight_close() {
    if (header) { munmap(header, mapped_size); header=nullptr; events=nullptr; }
    if (fd >= 0) { close(fd); fd=-1; }
}
