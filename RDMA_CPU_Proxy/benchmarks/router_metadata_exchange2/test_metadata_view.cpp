#include "metadata_view.hpp"
#include "router_routing.hpp"
#include "logging.hpp"
#include <iostream>
#include <vector>
#include <string>

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::runtime_error&) { threw = true; }
    check(threw, "invalid metadata was accepted");
}
rdma_proxy::ProxyConfig config(int tokens, uint64_t seed, int source) {
    rdma_proxy::ProxyConfig c;
    c.num_nodes = 1; c.num_gpus_per_node = 8; c.num_tokens = tokens;
    c.local_gpu_index = source; c.router_routing_enabled = true;
    c.router_num_experts = 128; c.router_top_k = 8; c.router_seed = seed; c.mock_mode = true;
    return c;
}
template<class T> void array(const T& values) {
    std::cout << '[';
    bool first = true;
    for (auto v : values) { if (!first) std::cout << ','; first = false; std::cout << +v; }
    std::cout << ']';
}
int main(int argc, char** argv) {
    try {
        rdma_proxy::Logger::instance().set_level(rdma_proxy::LogLevel::kError);
        if (argc == 5 && std::string(argv[1]) == "--dump") {
            rdma_proxy::RouterRouting routing(config(std::stoi(argv[2]), std::stoull(argv[3]), std::stoi(argv[4])));
            routing.initialize();
            std::vector<int32_t> offsets{0}, values;
            for (int dest = 0; dest < 8; ++dest) {
                const auto& m = routing.expert_metadata_for_gpu(0, dest);
                int base = values.size();
                for (int e = 1; e <= 16; ++e) offsets.push_back(base + m.expert_offsets[e]);
                values.insert(values.end(), m.expert_token_indices.begin(), m.expert_token_indices.end());
            }
            std::cout << '['; array(offsets); std::cout << ','; array(values); std::cout << ',';
            array(routing.token_indices_for_node(0)); std::cout << ',';
            array(routing.token_masks_for_node(0)); std::cout << "]\n";
            return 0;
        }
        for (int source = 0; source < 8; ++source) {
            rdma_proxy::RouterRouting routing(config(257, UINT64_MAX - source, source));
            routing.initialize();
            int base = 0;
            for (int dest = 0; dest < 8; ++dest) {
                const auto& m = routing.expert_metadata_for_gpu(0, dest);
                std::array<int32_t, rme2::header_words> header{};
                header[0] = base;
                std::copy(m.expert_offsets.begin(), m.expert_offsets.end(), header.begin() + 1);
                rme2::Installed view;
                rme2::install(view, header.data(), m.expert_token_indices.data(), 257, 1, true);
                check(view.offsets == header.data() + 1 && view.indices == m.expert_token_indices.data(),
                      "installation copied backing storage");
                check(view.count == int(m.expert_token_indices.size()) && view.iteration == 1,
                      "installed count/epoch mismatch");
                base += view.count;
                view.heads.fill(9); view.tails.fill(7);
                header.fill(0);
                int32_t dummy = 0;
                rme2::install(view, header.data(), &dummy, 257, 2, true);
                check(view.count == 0 && view.iteration == 2 && view.heads[0] == 0 && view.tails[15] == 0,
                      "empty reinstall did not reset progress");
            }
            check(base == 257 * 8, "routing partition lost assignments");
        }
        std::array<int32_t, rme2::header_words> header{};
        int32_t indices[2] = {1, 0};
        rme2::Installed view;
        header[0] = -1;
        rejects([&] { rme2::install(view, header.data(), indices, 2, 1, false); });
        header[0] = 0; header[1] = 1;
        rejects([&] { rme2::install(view, header.data(), indices, 2, 1, false); });
        header.fill(2); header[0] = 0; header[1] = 0;
        rejects([&] { rme2::install(view, header.data(), indices, 2, 1, true); }); // unsorted
        indices[0] = 0; indices[1] = 2;
        rejects([&] { rme2::install(view, header.data(), indices, 2, 1, true); }); // out of range
        header[0] = 15;
        rejects([&] { rme2::install(view, header.data(), indices, 2, 1, false); }); // capacity
        header.fill(0); header[5] = 1;
        rejects([&] { rme2::install(view, header.data(), indices, 2, 1, false); }); // decreasing offsets
        std::cout << "metadata views, production routing compatibility, malformed input: PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
