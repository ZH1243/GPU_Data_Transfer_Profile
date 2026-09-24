#include "common.hpp"
#include "shared_mapping.hpp"
#include <cstdio>
#include <new>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

void require(bool ok) { if (!ok) throw std::runtime_error("test assertion failed"); }
template<class F> void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}
int main() {
    try {
        require(bytes("16MB") == 16000000 && bytes("16MiB") == 16777216);
        for (auto s : {"0", "-1", "1.5MB", "MB", "1XB", "18446744073709551615GB"})
            rejects([&] { bytes(s); });
        rejects([] { split("1MB,"); });
        rejects([] { checked_add(SIZE_MAX, 1); });
        rejects([] { checked_mul(SIZE_MAX, 2); });
        Layout example({16000000,12000000,8000000,8000000,6000000,4000000,2000000}, 5);
        require(example.source_bytes == 280000000);
        require(example.src(4, 7) == 278000000 && example.dst(4, 7) == 8000000);
        // Exhaust every directed pair for unequal sizes and verify contiguous,
        // non-overlapping source and per-sender destination coverage.
        for (int n = 2; n <= max_ranks; ++n) {
            std::vector<size_t> sizes;
            for (int k = 1; k < n; ++k) sizes.push_back(13 * k + 3);
            Layout l(sizes, 7);
            for (int sender = 0; sender < n; ++sender) {
                size_t cursor = 0;
                for (size_t b = 0; b < l.batches; ++b) for (int k = 1; k < n; ++k) {
                    int receiver = (sender + k) % n;
                    int incoming_step = (receiver - sender + n) % n;
                    require(incoming_step == k);
                    require(l.src(b, k) == cursor);
                    cursor += sizes[k-1];
                    require(l.dst(b, k) + sizes[k-1] <= l.receive(incoming_step));
                    if (b + 1 == l.batches) require(l.dst(b, k) + sizes[k-1] == l.receive(incoming_step));
                }
                require(cursor == l.source_bytes);
            }
        }
        auto* memory = mmap(nullptr, sizeof(BarrierState), PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        require(memory != MAP_FAILED);
        auto* state = new(memory) BarrierState{};
        std::vector<pid_t> children;
        for (int rank = 0; rank < 4; ++rank) {
            auto pid = fork();
            require(pid >= 0);
            if (pid == 0) {
                try {
                    Barrier barrier(*state, rank, 4, 10);
                    for (int epoch = 1; epoch <= 10000; ++epoch) {
                        if (rank == epoch % 4 && epoch % 97 == 0) usleep(50);
                        barrier.wait();
                        for (int peer = 0; peer < 4; ++peer)
                            require(__atomic_load_n(&state->epochs[peer].value, __ATOMIC_ACQUIRE) >= static_cast<uint64_t>(epoch));
                    }
                    _exit(0);
                } catch (...) { abort_all(*state); _exit(1); }
            }
            children.push_back(pid);
        }
        for (auto pid : children) {
            int status;
            require(waitpid(pid, &status, 0) == pid);
            require(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        }
        state->~BarrierState();
        state = new(memory) BarrierState{};
        abort_all(*state);
        rejects([&] { Barrier(*state, 0, 2, 1).wait(); });
        state->~BarrierState();
        state = new(memory) BarrierState{};
        rejects([&] { Barrier(*state, 0, 2, 1).wait(); });
        require(aborted(*state));
        state->~BarrierState();
        munmap(memory, sizeof(BarrierState));
        char directory[] = "/tmp/nvlink-mapping-test.XXXXXX";
        require(mkdtemp(directory) != nullptr);
        std::string path = std::string(directory) + "/shared";
        auto peer = fork();
        require(peer >= 0);
        if (peer == 0) {
            try {
                SharedMapping<BarrierState> mapping(path, false, 5);
                Barrier barrier(mapping.get(), 1, 2, 5);
                for (int i = 0; i < 1000; ++i) barrier.wait();
                _exit(0);
            } catch (...) { _exit(1); }
        }
        usleep(20000); // Peer starts before the shared file exists.
        {
            SharedMapping<BarrierState> mapping(path, true, 5);
            Barrier barrier(mapping.get(), 0, 2, 5);
            for (int i = 0; i < 1000; ++i) barrier.wait();
            rejects([&] { SharedMapping<BarrierState> duplicate(path, true, 1); });
        }
        int peer_status;
        require(waitpid(peer, &peer_status, 0) == peer);
        require(WIFEXITED(peer_status) && WEXITSTATUS(peer_status) == 0);
        require(unlink(path.c_str()) == 0);
        rejects([&] { SharedMapping<BarrierState> missing(path, false, 1); });
        require(rmdir(directory) == 0);
        std::puts("PASS: parsing, overflow, directed layouts, 10000 process barriers, abort and timeout");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
