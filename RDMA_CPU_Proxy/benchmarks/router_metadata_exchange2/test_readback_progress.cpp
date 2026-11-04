#include "readback_progress.hpp"
#include <stdexcept>
#include <iostream>

void require(bool condition) {
    if (!condition) throw std::runtime_error("readback progress contract failed");
}
int main() {
    try {
        // Source zero (including the local source) stays late while source seven
        // completes. Delayed headers/payloads must not block other sources.
        std::array<rme2::ReadbackStage, 8> stages{};
        std::array<bool, 8> ready{}, headers{}, payloads{};
        std::array<int, 8> header_calls{}, payload_calls{}, installs{};
        auto step = [&] {
            return rme2::progress_sources(stages,
                [&](size_t s) { return ready[s]; },
                [&](size_t s) { ++header_calls[s]; },
                [&](size_t s) { return headers[s]; },
                [&](size_t s) { ++payload_calls[s]; },
                [&](size_t s) { return payloads[s]; },
                [&](size_t s) { ++installs[s]; });
        };
        require(!step());
        ready[7] = true; require(step());
        require(header_calls[7] == 1 && payload_calls[7] == 0);
        require(!step());
        headers[7] = true; require(step());
        require(installs[7] == 0);
        payloads[7] = true; require(step());
        require(installs[7] == 1 && header_calls[0] == 0);
        ready.fill(true); require(step());
        headers.fill(true); require(step());
        payloads.fill(true); require(step());
        require(!step());
        for (int s = 0; s < 8; ++s)
            require(header_calls[s] == 1 && payload_calls[s] == 1 && installs[s] == 1);
        // A fresh iteration cannot consume old completions before publication.
        stages.fill(rme2::ReadbackStage::waiting); ready.fill(false);
        require(!step());
        // Failed submission does not advance a source to an unrecorded event.
        bool failed = false;
        try {
            rme2::progress_sources(stages, [](size_t) { return true; },
                [](size_t) { throw std::runtime_error("submit failed"); },
                [](size_t) { return true; }, [](size_t) {},
                [](size_t) { return true; }, [](size_t) {});
        } catch (const std::runtime_error&) { failed = true; }
        require(failed && stages[0] == rme2::ReadbackStage::waiting);
        std::cout << "out-of-order readiness, deferred completion, reuse and failure: PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
