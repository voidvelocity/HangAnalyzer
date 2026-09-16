#include "flight_checkpoint.h"
#include "flight_recorder.h"
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

struct FakeEvent { std::atomic<bool> ready{false}; int records=0; };
struct Fake { std::vector<FakeEvent *> events; };
int create(void *user, flight_device_event *out) {
    auto *event = new FakeEvent;
    static_cast<Fake *>(user)->events.push_back(event);
    *out = event; return 0;
}
int destroy(void *, flight_device_event event) { delete static_cast<FakeEvent *>(event); return 0; }
int record(void *, flight_device_event event, flight_device_stream) {
    auto *e = static_cast<FakeEvent *>(event); e->ready=false; ++e->records; return 0;
}
int query(void *, flight_device_event event) {
    return static_cast<FakeEvent *>(event)->ready.load() ? 1 : 0;
}
#define REQUIRE(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "requirement failed at line %d: %s\n", __LINE__, #expr); return 1; \
} } while (0)

int main() {
    std::string path = "/tmp/checkpoint-pool-" + std::to_string(getpid()) + ".flight";
    unlink(path.c_str());
    REQUIRE(flight_init(path.c_str(), 0, 0, 128) == 0);
    Fake fake;
    flight_checkpoint_backend backend{&fake, create, destroy, record, query};
    auto *m = flight_checkpoint_create(&backend, 1);
    REQUIRE(m);
    REQUIRE(flight_checkpoint_register_stream(m, 11, reinterpret_cast<void *>(1)) == 0);
    REQUIRE(flight_checkpoint_register_stream(m, 22, reinterpret_cast<void *>(2)) == 0);
    uint64_t g11=0, g22=0;
    REQUIRE(flight_checkpoint_submit(m, 11, 100, 500, &g11) == 0 && g11 == 1);
    REQUIRE(flight_checkpoint_submit(m, 11, 101, 501, nullptr) == EAGAIN);
    REQUIRE(flight_checkpoint_submit(m, 22, 200, 600, &g22) == 0 && g22 == 1);
    REQUIRE(flight_checkpoint_poll(m, 8) == 0);
    fake.events[1]->ready = true;
    REQUIRE(flight_checkpoint_poll(m, 8) == 1);
    fake.events[0]->ready = true;
    REQUIRE(flight_checkpoint_poll(m, 8) == 1);
    uint64_t reused=0;
    REQUIRE(flight_checkpoint_submit(m, 11, 101, 501, &reused) == 0 && reused == 2);
    REQUIRE(flight_checkpoint_destroy(m) == EBUSY);
    REQUIRE(flight_checkpoint_start_poller(m, 100, 8) == 0);
    fake.events[0]->ready = true;
    for (int i=0; i<100 && fake.events[0]->records == 2; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    flight_checkpoint_stop_poller(m);
    REQUIRE(flight_checkpoint_destroy(m) == 0);
    flight_close();
    unlink(path.c_str());
    std::puts("checkpoint pool/generation/nonblocking poll passed");
}
