#pragma once
#include <cstdint>

namespace rme2 {
// Each participant publishes its epoch before calling. Old arrivals cannot
// satisfy a later iteration. Callers supply acquire loads and abort/timeout checks.
template<class Observe, class Pause>
void wait_epoch_all(int participants, uint64_t epoch, Observe observe, Pause pause) {
    for (int peer = 0; peer < participants; ++peer)
        while (observe(peer) < epoch) pause();
}
}
