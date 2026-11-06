#include "two_node_layout.hpp"
#include "rdma_transport.hpp"
#include "router_routing.hpp"
#include "logging.hpp"
#include <iostream>
#include <vector>

void check(bool ok) { if (!ok) throw std::runtime_error("two-node contract test failed"); }
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected);
}
void test_x3_x4(const std::vector<std::size_t>& indices, const std::vector<uint8_t>& masks,
                int tokens, int source) {
    namespace two = rme2::two;
    const uint64_t iteration = (uint64_t(1) << 32) + 19;
    const int count = int(indices.size()), destination = source ^ 8;
    check(indices.size() == masks.size());
    const size_t bytes = two::x3_x4_bytes(count);
    std::vector<int32_t> storage((two::x3_x4_bytes(tokens) + 3) / 4, -1);
    auto* h = storage.data();
    std::fill(h, h + two::header_words, 0);
    h[0] = two::x3_x4_magic; h[1] = two::version;
    h[2] = int32_t(uint32_t(iteration)); h[3] = int32_t(iteration >> 32);
    h[4] = source; h[5] = destination; h[6] = count;
    auto* x3 = h + two::header_words;
    auto* x4 = reinterpret_cast<uint8_t*>(x3 + count);
    std::copy(indices.begin(), indices.end(), x3);
    std::copy(masks.begin(), masks.end(), x4);
    two::InstalledX3X4 view;
    const auto install = [&](size_t length, bool validate = true) {
        two::install_x3_x4(view, h, tokens, iteration, source, destination, length, validate);
    };
    install(bytes);
    check(view.x3 == x3 && view.x4 == x4 && view.count == count && view.iteration == iteration);
    check(std::equal(indices.begin(), indices.end(), view.x3));
    check(std::equal(masks.begin(), masks.end(), view.x4));
    rejects([&] { install(bytes - 1); });
    rejects([&] { install(bytes + 1); });
    for (int field : {0, 1, 2, 3, 4, 5, 7, 15}) {
        h[field] ^= 1;
        rejects([&] { install(bytes, false); }); // structural checks survive skip-index-validation
        h[field] ^= 1;
    }
    h[6] = tokens + 1; rejects([&] { install(bytes, false); });
    h[6] = -1; rejects([&] { install(bytes, false); });
    h[6] = count;
    if (count) {
        const auto saved_token = x3[0];
        x3[0] = tokens; rejects([&] { install(bytes); });
        install(bytes, false); x3[0] = saved_token;
        const auto saved_mask = x4[0];
        x4[0] = 0; rejects([&] { install(bytes); }); x4[0] = saved_mask;
    }
    // Reuse a full record for an empty next iteration; no stale payload exposed.
    h[6] = 0; ++h[2];
    two::install_x3_x4(view, h, tokens, iteration + 1, source, destination, two::x3_x4_bytes(0), true);
    check(view.count == 0 && view.iteration == iteration + 1 && view.x4 == reinterpret_cast<const uint8_t*>(view.x3));
}
int main() {
    try {
        namespace two = rme2::two;
        rdma_proxy::Logger::instance().set_level(rdma_proxy::LogLevel::kError);
        constexpr int tokens = 257, capacity = tokens * 8;
        for (int source = 0; source < 16; ++source) {
            rdma_proxy::ProxyConfig c;
            c.num_nodes = 2; c.num_gpus_per_node = 8; c.num_tokens = tokens;
            c.node_rank = source / 8; c.local_gpu_index = source % 8;
            c.router_num_experts = 128; c.router_top_k = 8;
            c.router_routing_enabled = true; c.mock_mode = true;
            c.router_seed = UINT64_MAX - source;
            rdma_proxy::RouterRouting routing(c); routing.initialize();
            int base = 0;
            for (int dest = 0; dest < 16; ++dest) {
                auto m = routing.expert_metadata_for_gpu(dest / 8, dest % 8);
                std::vector<int32_t> record(two::header_words + m.expert_token_indices.size());
                auto* h = record.data();
                const uint64_t iteration = (uint64_t(1) << 32) + 17;
                h[0] = two::magic; h[1] = two::version;
                h[2] = int32_t(uint32_t(iteration)); h[3] = int32_t(iteration >> 32);
                h[4] = source; h[5] = dest; h[6] = base;
                std::copy(m.expert_offsets.begin(), m.expert_offsets.end(), h + 7);
                std::copy(m.expert_token_indices.begin(), m.expert_token_indices.end(), h + two::header_words);
                size_t bytes = record.size() * 4;
                two::check_received(h, capacity, iteration, source, dest, bytes);
                rme2::Installed view;
                rme2::install(view, h + 6, h + two::header_words, tokens, iteration, true, 8);
                check(view.offsets == h + 7 && view.indices == h + two::header_words);
                check(view.count == int(m.expert_token_indices.size()));
                base += view.count;
                rejects([&] { two::check_received(h, capacity, iteration + 1, source, dest, bytes); });
                rejects([&] { two::check_received(h, capacity, iteration, source ^ 1, dest, bytes); });
                rejects([&] { two::check_received(h, capacity, iteration, source, dest ^ 8, bytes); });
                rejects([&] { two::check_received(h, capacity, iteration, source, dest, bytes - 1); });
                h[15] = capacity + 1;
                rejects([&] { two::check_record(h, capacity, iteration, source, dest); });
                h[6] = 0; std::fill(h + 7, h + 16, 0);
                view.heads.fill(9); view.tails.fill(7);
                two::check_received(h, capacity, iteration, source, dest, two::record_bytes(0));
                rme2::install(view, h + 6, h + 16, tokens, iteration, true, 8);
                check(view.count == 0 && view.heads[0] == 0 && view.tails[7] == 0);
                h[0] = 0;
                rejects([&] { two::check_record(h, capacity, iteration, source, dest); });
            }
            check(base == capacity);
            test_x3_x4(routing.token_indices_for_node(1 - c.node_rank),
                       routing.token_masks_for_node(1 - c.node_rank), tokens, source);
        }
        test_x3_x4({}, {}, tokens, 0);
        test_x3_x4({256}, {128}, tokens, 15); // odd-byte record length and partial payload
        test_x3_x4({0, 4, 1}, {128, 128, 1}, tokens, 7); // stable mask order, not globally sorted x3
        rejects([&] { test_x3_x4({4, 0}, {128, 128}, tokens, 7); });
        rejects([&] { test_x3_x4({0, 1}, {1, 128}, tokens, 7); });
        for (int s = 0; s < 16; ++s) {
            check(two::record_offset(s, capacity) + two::stride(capacity) <= two::words(capacity));
            if (s) check(two::record_offset(s, capacity) == two::record_offset(s - 1, capacity) + two::stride(capacity));
        }
        check(two::record_offset(15, capacity) + two::stride(capacity) == two::x3_x4_offset(capacity));
        check(two::x3_x4_offset(capacity) * 4 + two::x3_x4_bytes(tokens) <= two::words(capacity) * 4);
        check(two::directory_words == 17 * two::header_words);
        rejects([&] { two::send_id(0, 0); });
        rejects([&] { two::send_id(UINT64_MAX, 0); });
        rejects([&] { two::send_id(3, 9); });
        check(two::send_id(3, 7) == 55);
        check(two::send_id(3, two::x3_x4_tag) == 56);
        two::CompletionProgress progress(5);
        rejects([&] { progress.complete_send(two::send_id(4, 0)); });
        rejects([&] { progress.receive_source(4, 0); });
        rejects([&] { progress.receive_source(5, 9); });
        for (int s = 7; s >= 0; --s) {
            check(!progress.complete());
            progress.complete_send(two::send_id(5, s));
            rejects([&] { progress.complete_send(two::send_id(5, s)); });
            check(progress.receive_source(5, s) == s);
            check(!progress.complete()); // delivery is not installation
            progress.installed(s);
            rejects([&] { progress.receive_source(5, s); });
        }
        check(!progress.complete()); // all expert tables alone no longer complete an iteration
        progress.complete_send(two::send_id(5, two::x3_x4_tag));
        check(progress.receive_source(5, two::x3_x4_tag) == two::x3_x4_tag);
        check(!progress.complete());
        progress.installed(two::x3_x4_tag);
        check(progress.complete());
        rejects([&] { progress.receive_source(5, two::x3_x4_tag); });
        rejects([&] { progress.complete_send(two::send_id(5, two::x3_x4_tag)); });
        // Incoming x3/x4 can complete first; the final outgoing send still gates reuse.
        two::CompletionProgress early(6);
        early.installed(early.receive_source(6, two::x3_x4_tag));
        for (int s = 0; s < 8; ++s) {
            early.installed(early.receive_source(6, s));
            early.complete_send(two::send_id(6, s));
        }
        check(!early.complete());
        early.complete_send(two::send_id(6, two::x3_x4_tag));
        check(early.complete());
        std::cout << "two-node expert/x3/x4 routing, record identity, bounds, installation and completion: PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
