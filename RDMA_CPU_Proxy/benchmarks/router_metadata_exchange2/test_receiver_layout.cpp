#include "receiver_layout.hpp"
#include <stdexcept>
#include <iostream>

int main() {
    using namespace rme2;
    static_assert(sizeof(receiver::Flag) == 64 && alignof(receiver::Flag) == 64);
    static_assert(sizeof(receiver::Result) == 64);
    for (int tokens : {1, 33, 257, 20480, 10000000}) {
        const int capacity = tokens * 8;
        const auto wire_bytes = two::words(capacity) * 4;
        const auto start = receiver::flags_offset(capacity);
        if (start < wire_bytes || start % 64 || start - wire_bytes >= 64 ||
            receiver::allocation_bytes(capacity) != start + 8 * sizeof(receiver::Flag))
            throw std::runtime_error("flags overlap wire records or are misaligned");
        for (int node = 0; node < 2; ++node)
            for (int source = 0; source < 8; ++source) {
                const auto local = two::record_offset(node * 8 + source, capacity) * 4;
                const auto remote = two::record_offset((1 - node) * 8 + source, capacity) * 4;
                const auto bytes = two::record_bytes(capacity);
                if (!(local + bytes <= remote || remote + bytes <= local) ||
                    local + bytes > two::x3_x4_offset(capacity) * 4)
                    throw std::runtime_error("receiver record overlaps incoming RDMA or x3/x4");
            }
    }
    std::cout << "receiver layout passed\n";
}
