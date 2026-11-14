#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rdma_proxy {

struct ForwardingRoundPhase {
    enum class Kind { kLocal, kRemote };
    Kind kind{Kind::kLocal};
    int source_node_rank{-1};
};

template <typename Payload>
struct ForwardingRound {
    uint64_t iteration{0};
    std::size_t phase_index{0};
    ForwardingRoundPhase phase;
    uint64_t ordinal{0};
    // At most one data notification per source GPU, in source-GPU order.
    // Empty destinations still participate through their batch markers.
    std::vector<Payload> payloads;
    uint64_t first_completion_ns{0};
    uint64_t last_completion_ns{0};
};

// Receiver-owned, CPU-only state. It never waits for another source, performs
// visibility operations, updates scheduler readiness, or publishes descriptors.
// The caller drains every source queue into this collector and only exposes
// payloads returned by pop_ready() to the scheduler. In particular, future-round
// payloads must not leak through an intervening early-local scheduling call.
//
// Each (phase, source GPU) has an independent FIFO sequence. Different source
// nodes may interleave on a source queue; phases are *retired* in the supplied
// deterministic order, not assumed to arrive in that order. Ordinals count
// actual forwarding batches in that context and are unrelated to chunk offsets
// or a batch-start barrier. An explicit phase end proves exhaustion, including
// an empty phase. The transport may fuse data/batch-end/phase-end into one wire
// record, but must call these methods in that order before pop_ready().
template <typename Payload>
class ForwardingRoundCollector {
public:
    ForwardingRoundCollector(std::size_t participants,
                             std::vector<ForwardingRoundPhase> phases,
                             std::size_t max_pending_batches)
        : participants_(participants), phases_(std::move(phases)),
          max_pending_batches_(max_pending_batches) {
        if (participants_ == 0 || max_pending_batches_ == 0)
            throw std::runtime_error("invalid forwarding-round collector dimensions");
        for (std::size_t i = 0; i < phases_.size(); ++i) {
            if (phases_[i].source_node_rank < 0)
                throw std::runtime_error("invalid forwarding-round source node");
            for (std::size_t j = 0; j < i; ++j) {
                if (phases_[i].kind == phases_[j].kind &&
                    phases_[i].source_node_rank == phases_[j].source_node_rank)
                    throw std::runtime_error("duplicate forwarding-round phase context");
            }
        }
        reset_storage();
    }

    void begin_iteration(uint64_t iteration) {
        if (active_ && !complete())
            throw std::runtime_error("forwarding-round reset would discard pending work");
        if (have_iteration_ && iteration <= iteration_)
            throw std::runtime_error("forwarding-round iteration did not advance");
        reset_storage();
        iteration_ = iteration;
        have_iteration_ = true;
        active_ = true;
    }

    void add_data(uint64_t iteration, std::size_t phase_index,
                  std::size_t source_gpu, uint64_t ordinal, Payload payload) {
        auto& source = validate(iteration, phase_index, source_gpu, ordinal);
        if (source.open_data)
            throw std::runtime_error("duplicate forwarding-round data notification");
        require_capacity();
        source.open_data.emplace(std::move(payload));
        ++pending_batches_;
    }

    // Call only after the data transfer and required visibility handling have
    // completed. A marker is required even when this destination has no rows.
    // The optional clock samples are supplied by the caller, so diagnostics
    // disabled runs pay no clock-reading cost.
    void complete_batch(uint64_t iteration, std::size_t phase_index,
                        std::size_t source_gpu, uint64_t ordinal,
                        uint64_t completion_time_ns = 0) {
        auto& source = validate(iteration, phase_index, source_gpu, ordinal);
        if (ordinal == std::numeric_limits<uint64_t>::max())
            throw std::runtime_error("forwarding-round ordinal overflow");
        const bool has_data = source.open_data.has_value();
        if (!has_data) require_capacity();
        source.batches.push_back(Batch{ordinal, std::move(source.open_data), completion_time_ns});
        source.open_data.reset();
        ++source.next_ordinal;
        if (!has_data) ++pending_batches_;
    }

    // next_ordinal is the number of completed batches, not the last batch ID.
    // A final batch may carry this marker; a zero-batch source sends only this.
    void complete_phase(uint64_t iteration, std::size_t phase_index,
                        std::size_t source_gpu, uint64_t next_ordinal) {
        auto& source = validate(iteration, phase_index, source_gpu, next_ordinal);
        if (source.open_data)
            throw std::runtime_error("forwarding phase ended before its batch completion");
        source.exhausted = true;
    }

    bool pop_ready(ForwardingRound<Payload>& output) {
        if (!active_) return false;
        while (next_phase_ < states_.size()) {
            auto& phase = states_[next_phase_];
            bool has_batch = false;
            for (const auto& source : phase.sources) {
                if (!source.batches.empty()) {
                    if (source.batches.front().ordinal != phase.next_round)
                        throw std::runtime_error("forwarding-round internal FIFO mismatch");
                    has_batch = true;
                } else if (!source.exhausted) {
                    return false;
                }
            }
            if (!has_batch) {
                // All participants explicitly retired without another round.
                ++next_phase_;
                continue;
            }
            ForwardingRound<Payload> ready;
            ready.iteration = iteration_;
            ready.phase_index = next_phase_;
            ready.phase = phases_[next_phase_];
            ready.ordinal = phase.next_round;
            ready.payloads.reserve(participants_);
            for (auto& source : phase.sources) {
                if (source.batches.empty()) continue; // Exhausted shorter source.
                auto& batch = source.batches.front();
                if (batch.payload) ready.payloads.push_back(std::move(*batch.payload));
                if (batch.completion_ns != 0) {
                    if (ready.first_completion_ns == 0 ||
                        batch.completion_ns < ready.first_completion_ns)
                        ready.first_completion_ns = batch.completion_ns;
                    ready.last_completion_ns = std::max(
                        ready.last_completion_ns, batch.completion_ns);
                }
                source.batches.pop_front();
                --pending_batches_;
            }
            ++phase.next_round;
            output = std::move(ready);
            return true;
        }
        return false;
    }

    bool active() const { return active_; }
    std::size_t pending_batches() const { return pending_batches_; }

    // No configured phases (e.g. single-node without local staging) completes
    // immediately. Otherwise empty transport queues do not imply completion:
    // even a source with no
    // data must explicitly exhaust every context. Open data counts as pending.
    bool complete() const {
        if (!active_ || pending_batches_ != 0) return false;
        for (const auto& phase : states_)
            for (const auto& source : phase.sources)
                if (!source.exhausted) return false;
        return true;
    }

    // Only cancellation/error/shutdown may deliberately discard pending data.
    // A later iteration must still use a fresh identity.
    void cancel() {
        reset_storage();
        active_ = false;
    }

private:
    struct Batch {
        uint64_t ordinal;
        std::optional<Payload> payload;
        uint64_t completion_ns;
    };
    struct SourceState {
        uint64_t next_ordinal{0};
        bool exhausted{false};
        std::optional<Payload> open_data;
        std::deque<Batch> batches;
    };
    struct PhaseState {
        uint64_t next_round{0};
        std::vector<SourceState> sources;
    };

    SourceState& validate(uint64_t iteration, std::size_t phase_index,
                          std::size_t source_gpu, uint64_t ordinal) {
        if (!active_ || iteration != iteration_)
            throw std::runtime_error("forwarding-round notification iteration mismatch");
        if (phase_index >= states_.size() || source_gpu >= participants_)
            throw std::runtime_error("forwarding-round notification context mismatch");
        auto& source = states_[phase_index].sources[source_gpu];
        if (source.exhausted || ordinal != source.next_ordinal)
            throw std::runtime_error("late, duplicate or out-of-order forwarding-round notification");
        return source;
    }

    void require_capacity() const {
        // This is a protocol error bound, never a wait. Size the bound from
        // iteration metadata (at least one source token per nonempty batch),
        // allowing the receiver to drain queues even if another GPU is late.
        if (pending_batches_ >= max_pending_batches_)
            throw std::runtime_error("forwarding-round pending batch limit exceeded");
    }

    void reset_storage() {
        states_.clear();
        states_.resize(phases_.size());
        for (auto& phase : states_) phase.sources.resize(participants_);
        next_phase_ = 0;
        pending_batches_ = 0;
    }

    std::size_t participants_;
    std::vector<ForwardingRoundPhase> phases_;
    std::size_t max_pending_batches_;
    std::vector<PhaseState> states_;
    std::size_t next_phase_{0};
    std::size_t pending_batches_{0};
    uint64_t iteration_{0};
    bool have_iteration_{false};
    bool active_{false};
};

} // namespace rdma_proxy
