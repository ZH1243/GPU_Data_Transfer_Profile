#pragma once
#include <array>
#include <cstddef>

namespace rme2 {
enum class ReadbackStage { waiting, header, payload, installed };

// Nonblocking sweep: a late source must not prevent ready sources from moving
// forward. Callbacks must query, never synchronously wait. Advance at most one
// stage per source per sweep; transitions commit only after successful work.
template<std::size_t N, class Ready, class Header, class HeaderDone, class Payload,
         class PayloadDone, class Install>
bool progress_sources(std::array<ReadbackStage, N>& stages, Ready ready,
                      Header header, HeaderDone header_done, Payload payload,
                      PayloadDone payload_done, Install install) {
    bool changed = false;
    for (std::size_t s = 0; s < N; ++s) {
        switch (stages[s]) {
        case ReadbackStage::waiting:
            if (ready(s)) {
                header(s);
                stages[s] = ReadbackStage::header;
                changed = true;
            }
            break;
        case ReadbackStage::header:
            if (header_done(s)) {
                payload(s);
                stages[s] = ReadbackStage::payload;
                changed = true;
            }
            break;
        case ReadbackStage::payload:
            if (payload_done(s)) {
                install(s);
                stages[s] = ReadbackStage::installed;
                changed = true;
            }
            break;
        case ReadbackStage::installed:
            break;
        }
    }
    return changed;
}
} // namespace rme2
