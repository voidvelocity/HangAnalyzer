#include "flight_checkpoint.h"
#include "flight_recorder.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

namespace {
enum SlotState : uint8_t { FREE=0, RESERVED=1, PENDING=2, QUERYING=3 };
struct Slot {
    flight_device_event event{};
    std::atomic<uint8_t> state{FREE};
    std::atomic<uint64_t> generation{0};
    uint64_t submitted_seq{};
    uint64_t checkpoint_id{};
    uint64_t query_error_reported_generation{};
    uint64_t not_ready_reported_generation{};
};
struct Stream {
    uint32_t id{};
    flight_device_stream handle{};
    std::vector<std::unique_ptr<Slot>> slots;
    std::atomic<uint32_t> cursor{0};
};
}

struct flight_checkpoint_manager {
    flight_checkpoint_backend backend{};
    uint32_t slots_per_stream{};
    std::mutex control;
    std::vector<std::unique_ptr<Stream>> streams;
    std::atomic_flag polling = ATOMIC_FLAG_INIT;
    std::atomic<uint64_t> poll_cursor{0};
    std::atomic<bool> stop_poller{false};
    std::atomic<bool> poller_running{false};
    std::thread poller;
};

static Stream *find_stream(flight_checkpoint_manager *m, uint32_t id) {
    // Streams are registered before traffic and never removed.
    for (auto &stream : m->streams) if (stream->id == id) return stream.get();
    return nullptr;
}

extern "C" flight_checkpoint_manager *flight_checkpoint_create(
        const flight_checkpoint_backend *backend, uint32_t slots_per_stream) {
    if (!backend || !backend->create || !backend->destroy || !backend->record ||
        !backend->query || slots_per_stream == 0 || slots_per_stream > 1024) return nullptr;
    auto *m = new (std::nothrow) flight_checkpoint_manager;
    if (!m) return nullptr;
    m->backend = *backend;
    m->slots_per_stream = slots_per_stream;
    return m;
}

extern "C" int flight_checkpoint_register_stream(flight_checkpoint_manager *m,
        uint32_t stream_id, flight_device_stream stream_handle) {
    if (!m) return EINVAL;
    std::lock_guard<std::mutex> guard(m->control);
    if (find_stream(m, stream_id)) return EEXIST;
    auto stream = std::make_unique<Stream>();
    stream->id = stream_id;
    stream->handle = stream_handle;
    for (uint32_t i=0; i<m->slots_per_stream; ++i) {
        auto slot = std::make_unique<Slot>();
        int rc = m->backend.create(m->backend.user, &slot->event);
        if (rc != 0) {
            for (auto &old : stream->slots) m->backend.destroy(m->backend.user, old->event);
            return rc > 0 ? rc : EIO;
        }
        stream->slots.push_back(std::move(slot));
    }
    m->streams.push_back(std::move(stream));
    return 0;
}

extern "C" int flight_checkpoint_submit(flight_checkpoint_manager *m, uint32_t stream_id,
        uint64_t submitted_seq, uint64_t checkpoint_id, uint64_t *generation) {
    if (!m || submitted_seq == 0) return EINVAL;
    Stream *stream = find_stream(m, stream_id);
    if (!stream) return ENOENT;
    uint32_t start = stream->cursor.fetch_add(1, std::memory_order_relaxed);
    Slot *chosen = nullptr;
    for (uint32_t n=0; n<m->slots_per_stream; ++n) {
        Slot *slot = stream->slots[(start+n) % m->slots_per_stream].get();
        uint8_t expected = FREE;
        if (slot->state.compare_exchange_strong(expected, RESERVED, std::memory_order_acq_rel)) {
            chosen = slot; break;
        }
    }
    if (!chosen) {
        flight_record(FL_CHECKPOINT, stream_id, checkpoint_id, submitted_seq, 0,
                      FL_CHECKPOINT_POOL_FULL);
        return EAGAIN;
    }
    uint64_t gen = chosen->generation.fetch_add(1, std::memory_order_relaxed) + 1;
    chosen->submitted_seq = submitted_seq;
    chosen->checkpoint_id = checkpoint_id;
    int rc = m->backend.record(m->backend.user, chosen->event, stream->handle);
    if (rc != 0) {
        chosen->state.store(FREE, std::memory_order_release);
        return rc > 0 ? rc : EIO;
    }
    flight_record(FL_CHECKPOINT, stream_id, checkpoint_id, submitted_seq, gen,
                  FL_CHECKPOINT_SUBMITTED);
    chosen->state.store(PENDING, std::memory_order_release);
    if (generation) *generation = gen;
    return 0;
}

extern "C" int flight_checkpoint_poll(flight_checkpoint_manager *m, uint32_t budget) {
    if (!m || budget == 0) return 0;
    if (m->polling.test_and_set(std::memory_order_acquire)) return 0;
    int completed = 0;
    size_t total = m->streams.size() * size_t(m->slots_per_stream);
    if (total == 0) { m->polling.clear(std::memory_order_release); return 0; }
    uint64_t start = m->poll_cursor.fetch_add(budget, std::memory_order_relaxed);
    uint32_t checked = 0;
    for (size_t offset=0; offset<total && checked<budget; ++offset) {
        size_t flat = (start + offset) % total;
        Stream *stream = m->streams[flat / m->slots_per_stream].get();
        Slot *slot = stream->slots[flat % m->slots_per_stream].get();
        uint8_t expected = PENDING;
        if (!slot->state.compare_exchange_strong(expected, QUERYING, std::memory_order_acq_rel)) continue;
        ++checked;
        uint64_t gen = slot->generation.load(std::memory_order_acquire);
        int rc = m->backend.query(m->backend.user, slot->event);
        if (rc == 1) {
            // Exact generation is persisted before releasing the reusable slot.
            flight_record(FL_DEVICE_CONFIRMED, stream->id, slot->checkpoint_id,
                          slot->submitted_seq, gen, 0);
            slot->state.store(FREE, std::memory_order_release);
            ++completed;
        } else if (rc == 0) {
            if (slot->not_ready_reported_generation != gen) {
                flight_record(FL_CHECKPOINT, stream->id, slot->checkpoint_id,
                              slot->submitted_seq, gen, FL_CHECKPOINT_NOT_READY);
                slot->not_ready_reported_generation = gen;
            }
            slot->state.store(PENDING, std::memory_order_release);
        } else {
            if (slot->query_error_reported_generation != gen) {
                flight_record(FL_CHECKPOINT, stream->id, slot->checkpoint_id,
                              slot->submitted_seq, gen, FL_CHECKPOINT_QUERY_ERROR);
                slot->query_error_reported_generation = gen;
            }
            slot->state.store(PENDING, std::memory_order_release);
        }
    }
    m->polling.clear(std::memory_order_release);
    return completed;
}

extern "C" int flight_checkpoint_start_poller(flight_checkpoint_manager *m,
        uint32_t interval_us, uint32_t budget) {
    if (!m || interval_us == 0 || budget == 0) return EINVAL;
    bool expected = false;
    if (!m->poller_running.compare_exchange_strong(expected, true)) return EALREADY;
    m->stop_poller.store(false, std::memory_order_release);
    try {
        m->poller = std::thread([m, interval_us, budget] {
            while (!m->stop_poller.load(std::memory_order_acquire)) {
                flight_checkpoint_poll(m, budget);
                std::this_thread::sleep_for(std::chrono::microseconds(interval_us));
            }
            m->poller_running.store(false, std::memory_order_release);
        });
    } catch (...) {
        m->poller_running.store(false, std::memory_order_release);
        return EAGAIN;
    }
    return 0;
}

extern "C" void flight_checkpoint_stop_poller(flight_checkpoint_manager *m) {
    if (!m) return;
    m->stop_poller.store(true, std::memory_order_release);
    if (m->poller.joinable()) m->poller.join();
}

extern "C" int flight_checkpoint_destroy(flight_checkpoint_manager *m) {
    if (!m) return 0;
    if (m->poller_running.load(std::memory_order_acquire) || m->poller.joinable()) return EBUSY;
    {
        std::lock_guard<std::mutex> guard(m->control);
        for (auto &stream : m->streams)
            for (auto &slot : stream->slots)
                if (slot->state.load(std::memory_order_acquire) != FREE) return EBUSY;
    }
    for (auto &stream : m->streams)
        for (auto &slot : stream->slots)
            m->backend.destroy(m->backend.user, slot->event);
    delete m;
    return 0;
}
