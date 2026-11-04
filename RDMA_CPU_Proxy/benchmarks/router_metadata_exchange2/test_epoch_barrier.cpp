#include "epoch_barrier.hpp"
#include <array>
#include <stdexcept>
#include <iostream>

int main() {
    try {
        std::array<uint64_t, 8> arrivals{};
        int polls = 0;
        for (uint64_t epoch = 1; epoch <= 100; ++epoch) {
            // Last participant is deliberately delayed; old epochs must fail.
            arrivals.fill(epoch); arrivals[7] = epoch - 1;
            int before = polls;
            rme2::wait_epoch_all(8, epoch, [&](int p) { return arrivals[p]; }, [&] {
                ++polls;
                if (polls - before == 3) arrivals[7] = epoch;
            });
            if (polls - before != 3) throw std::runtime_error("barrier accepted stale arrival");
            rme2::wait_epoch_all(8, epoch, [&](int p) { return arrivals[p]; }, [] {
                throw std::runtime_error("ready barrier unexpectedly waited");
            });
        }
        bool aborted = false;
        try {
            rme2::wait_epoch_all(8, 101, [&](int p) { return arrivals[p]; }, [] {
                throw std::runtime_error("timeout/abort");
            });
        } catch (const std::runtime_error&) { aborted = true; }
        if (!aborted) throw std::runtime_error("barrier swallowed abort");
        std::cout << "epoch barrier stale arrivals, reuse, immediate release and abort: PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
