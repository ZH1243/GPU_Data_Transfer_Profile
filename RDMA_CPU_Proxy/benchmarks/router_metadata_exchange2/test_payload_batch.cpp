#include "payload_batch.hpp"
#include <iostream>

void require(bool ok) { if (!ok) throw std::runtime_error("payload batch contract failed"); }
int main() {
    try {
        rme2::PayloadBatch batch;
        std::array<int, 8> src{}, dst{};
        batch.append(nullptr, nullptr, 0, 0);
        require(batch.count == 0);
        // Sparse receive, variable lengths, and self/source identity retention.
        for (int s : {1, 3, 7}) batch.append(&dst[s], &src[s], s + 1, s);
        require(batch.count == 3 && batch.sizes[0] == 2 && batch.sizes[2] == 8);
        require(batch.sources[1] == &src[3] && batch.destinations[1] == &dst[3]);
        require(batch.source_ranks[2] == 7);
        batch.reset();
        require(batch.count == 0);
        for (int s = 0; s < 8; ++s) batch.append(&dst[s], &src[s], sizeof(int), s);
        require(batch.count == 8);
        bool rejected = false;
        try { batch.append(&dst[0], &src[0], sizeof(int), 0); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected);
        batch.reset();
        for (int s = 0; s < 8; ++s) batch.append(&dst[s], &src[s], 0, s);
        require(batch.count == 0); // all-empty iteration submits no batch
        std::cout << "payload batch sparse/full/empty descriptors and reuse: PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
