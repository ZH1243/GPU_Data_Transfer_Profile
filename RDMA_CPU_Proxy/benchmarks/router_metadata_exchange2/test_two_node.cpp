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
        }
        for (int s = 0; s < 16; ++s) {
            check(two::record_offset(s, capacity) + two::stride(capacity) <= two::words(capacity));
            if (s) check(two::record_offset(s, capacity) == two::record_offset(s - 1, capacity) + two::stride(capacity));
        }
        rejects([&] { two::send_id(0, 0); });
        rejects([&] { two::send_id(UINT64_MAX, 0); });
        check(two::send_id(3, 7) == 31);
        two::CompletionProgress progress(5);
        rejects([&] { progress.complete_send(two::send_id(4, 0)); });
        rejects([&] { progress.receive_source(4, 0); });
        rejects([&] { progress.receive_source(5, 8); });
        for (int s = 7; s >= 0; --s) {
            check(!progress.complete());
            progress.complete_send(two::send_id(5, s));
            rejects([&] { progress.complete_send(two::send_id(5, s)); });
            check(progress.receive_source(5, s) == s);
            check(!progress.complete()); // delivery is not installation
            progress.installed(s);
            rejects([&] { progress.receive_source(5, s); });
        }
        check(progress.complete());
        std::cout << "two-node production routing, record identity, bounds and borrowed installation: PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
