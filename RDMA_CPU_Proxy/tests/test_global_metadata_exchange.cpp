#include "rdma_connection.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

uint16_t find_free_port_base(int count) {
    for (int base = 30000; base + count < 60000; base += count) {
        std::vector<int> sockets;
        bool available = true;
        for (int offset = 0; offset < count; ++offset) {
            const int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
            if (fd < 0) {
                available = false;
                break;
            }
            sockaddr_in6 addr{};
            addr.sin6_family = AF_INET6;
            addr.sin6_addr = in6addr_any;
            addr.sin6_port = htons(static_cast<uint16_t>(base + offset));
            if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
                ::close(fd);
                available = false;
                break;
            }
            sockets.push_back(fd);
        }
        for (const int fd : sockets) ::close(fd);
        if (available && sockets.size() == static_cast<std::size_t>(count)) {
            return static_cast<uint16_t>(base);
        }
    }
    throw std::runtime_error("could not find free TCP ports for metadata test");
}

}  // namespace

int main() {
    constexpr int kGpuCount = 4;
    const uint16_t port_base = find_free_port_base(kGpuCount);
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::string first_error;
    std::vector<std::thread> proxies;

    for (int gpu = 0; gpu < kGpuCount; ++gpu) {
        proxies.emplace_back([&, gpu] {
            try {
                rdma_proxy::ProxyConfig config;
                config.node_rank = 0;
                config.num_nodes = 1;
                config.local_gpu_index = gpu;
                config.num_gpus_per_node = kGpuCount;
                config.mock_mode = false;
                config.completion_timeout_ms = 5000;
                rdma_proxy::ConnectionManager manager(config);

                std::vector<std::string> received;
                std::exception_ptr receive_error;
                std::thread receiver([&] {
                    try {
                        received = manager.receive_global_control_messages(
                            static_cast<uint16_t>(port_base + gpu),
                            kGpuCount - 1,
                            1024,
                            config.completion_timeout_ms);
                    } catch (...) {
                        receive_error = std::current_exception();
                    }
                });

                std::exception_ptr send_error;
                for (int destination = kGpuCount - 1; destination >= 0; --destination) {
                    if (destination == gpu) continue;
                    try {
                        rdma_proxy::PeerAddress endpoint;
                        endpoint.node_rank = 0;
                        endpoint.host = "::1";
                        endpoint.port = static_cast<uint16_t>(port_base + destination);
                        manager.send_global_control_message(
                            endpoint,
                            "source=" + std::to_string(gpu) +
                                " destination=" + std::to_string(destination),
                            config.completion_timeout_ms);
                    } catch (...) {
                        if (!send_error) send_error = std::current_exception();
                    }
                }
                receiver.join();
                if (send_error) std::rethrow_exception(send_error);
                if (receive_error) std::rethrow_exception(receive_error);

                std::set<std::string> actual(received.begin(), received.end());
                std::set<std::string> expected;
                for (int source = 0; source < kGpuCount; ++source) {
                    if (source == gpu) continue;
                    expected.insert(
                        "source=" + std::to_string(source) +
                        " destination=" + std::to_string(gpu));
                }
                if (actual != expected) {
                    throw std::runtime_error("global metadata all-to-all payload mismatch");
                }
            } catch (const std::exception& error) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) first_error = error.what();
            }
        });
    }
    for (auto& proxy : proxies) proxy.join();
    if (failed.load()) {
        std::cerr << first_error << '\n';
        return 1;
    }
    std::cout << "test_global_metadata_exchange passed\n";
    return 0;
}
