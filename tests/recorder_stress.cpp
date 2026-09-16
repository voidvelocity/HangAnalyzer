#include "flight_recorder.h"
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

int main() {
    constexpr int threads = 4, per_thread = 50000;
    constexpr uint64_t total = uint64_t(threads) * per_thread;
    std::string path = "/tmp/flight-stress-" + std::to_string(getpid()) + ".flight";
    unlink(path.c_str());
    if (int e = flight_init(path.c_str(), 3, 1, total + 16)) {
        std::fprintf(stderr, "flight_init failed: %d\n", e); return 1;
    }
    std::vector<std::vector<uint64_t>> sequences(threads);
    auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([t, &sequences] {
            sequences[t].reserve(per_thread);
            for (int i = 0; i < per_thread; ++i)
                sequences[t].push_back(flight_record(FL_CHECKPOINT, t, i, t, i, 0));
        });
    }
    for (auto &w : workers) w.join();
    auto end = std::chrono::steady_clock::now();
    flight_close();
    std::vector<unsigned char> seen(total + 1);
    for (auto &part : sequences) for (auto seq : part) {
        if (!seq || seq > total || seen[seq]) { std::fprintf(stderr, "duplicate/bad seq %" PRIu64 "\n", seq); return 2; }
        seen[seq] = 1;
    }
    if (std::find(seen.begin() + 1, seen.end(), 0) != seen.end()) return 3;
    std::ifstream file(path, std::ios::binary);
    file.seekg(128);
    for (uint64_t seq = 1; seq <= total; ++seq) {
        flight_event e{}; file.read(reinterpret_cast<char *>(&e), sizeof(e));
        if (!file || e.seq != seq) { std::fprintf(stderr, "bad disk slot %" PRIu64 " got %" PRIu64 "\n", seq, e.seq); return 4; }
    }
    unlink(path.c_str());
    double nanos = std::chrono::duration<double, std::nano>(end - start).count() / total;
    std::printf("threads=%d events=%" PRIu64 " elapsed_ns_per_event=%.1f\n", threads, total, nanos);
    return 0;
}
