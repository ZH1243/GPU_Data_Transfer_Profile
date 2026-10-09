#include "proxy.hpp"
#include "logging.hpp"

#include <chrono>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace rdma_proxy;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

namespace rdma_proxy {
struct QpCalibrationTestAccess {
    static void enable_tcp_control(Proxy& proxy) {
        auto c = proxy.config_;
        c.mock_mode = false;
        proxy.connection_manager_ = ConnectionManager(c);
    }
    static bool complete(const Proxy& proxy) { return proxy.rdma_qp_calibration_complete_; }
    static std::size_t measured(const Proxy& proxy) { return proxy.rdma_iteration_bandwidth_gbps_.size(); }
    static uint64_t markers(const Proxy& proxy) { return proxy.peers_[0].workers[0]->send_marker_completions(); }
    static void recreate(Proxy& proxy) {
        const auto send = proxy.peers_[0].local_send_mr.addr;
        const auto recv = proxy.peers_[0].local_recv_mr.addr;
        proxy.calibration_deadline_ = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        proxy.recreate_peer_qps(proxy.peers_[0]);
        require(proxy.peers_[0].local_send_mr.addr == send && proxy.peers_[0].local_recv_mr.addr == recv,
                "QP recreation replaced registered buffers");
    }
    static bool reduce(Proxy& proxy, bool value, uint64_t signature = 123) {
        proxy.calibration_deadline_ = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        return proxy.calibration_local_all(value, signature);
    }
};
}

ProxyConfig config() {
    ProxyConfig c;
    c.num_nodes = 2;
    c.num_gpus_per_node = 1;
    c.num_tokens = 65;
    c.token_dimension = 8;
    c.tokens_per_chunk = 7;
    c.num_qps_per_peer = 3;
    c.max_in_flight_chunks_per_qp = 4;
    c.mock_mode = true;
    c.num_iterations = 2;
    c.rdma_bandwidth_summary_dir.clear();
    c.completion_timeout_ms = 2000;
    c.rdma_qp_calibration_enabled = true;
    c.min_bandwidth_gbps_needed = 1e-12;
    c.rdma_qp_calibration_warmup_iterations = 1;
    c.rdma_qp_calibration_sample_iterations = 2;
    c.rdma_qp_calibration_max_attempts = 2;
    c.rdma_qp_calibration_timeout_ms = 10000;
    c.peers.push_back({1, "mock", 18515});
    return c;
}

void expect_invalid(const std::function<void(ProxyConfig&)>& modify) {
    auto c = config();
    modify(c);
    try { validate_config(c); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error("invalid calibration config was accepted");
}

void test_config() {
    require(!ProxyConfig{}.rdma_qp_calibration_enabled, "calibration default changed");
    validate_config(config());
    expect_invalid([](auto& c) { c.num_nodes = 3; });
    expect_invalid([](auto& c) { c.nvlink_forwarding_enabled = true; });
    expect_invalid([](auto& c) { c.min_bandwidth_gbps_needed = 0; });
    expect_invalid([](auto& c) { c.min_bandwidth_gbps_needed = std::numeric_limits<double>::quiet_NaN(); });
    expect_invalid([](auto& c) { c.rdma_qp_calibration_warmup_iterations = -1; });
    expect_invalid([](auto& c) { c.rdma_qp_calibration_sample_iterations = 0; });
    expect_invalid([](auto& c) { c.rdma_qp_calibration_max_attempts = 0; });
    expect_invalid([](auto& c) { c.rdma_qp_calibration_timeout_ms = 0; });

    const auto path = std::filesystem::temp_directory_path() /
        ("qp_calibration_config_" + std::to_string(getpid()) + ".json");
    {
        std::ofstream out(path);
        out << R"({"num_nodes":2,"num_gpus_per_node":1,"num_tokens":65,"token_dimension":8,
          "peers":[{"node_rank":1,"host":"mock","port":18515}],"mock_mode":true,
          "rdma_qp_calibration_enabled":true,"min_bandwidth_gbps_needed":341.5,
          "rdma_qp_calibration_warmup_iterations":3,"rdma_qp_calibration_sample_iterations":7,
          "rdma_qp_calibration_max_attempts":4,"rdma_qp_calibration_timeout_ms":9000})";
    }
    auto parsed = load_config_file(path.string());
    require(parsed.rdma_qp_calibration_enabled && parsed.min_bandwidth_gbps_needed == 341.5 &&
        parsed.rdma_qp_calibration_warmup_iterations == 3 && parsed.rdma_qp_calibration_sample_iterations == 7 &&
        parsed.rdma_qp_calibration_max_attempts == 4 && parsed.rdma_qp_calibration_timeout_ms == 9000,
        "JSON calibration settings not parsed");
    std::vector<std::string> args{"test", "--config", path.string(), "--rdma_qp_calibration_enabled=false",
        "--min_bandwidth_gbps_needed=342", "--rdma_qp_calibration_warmup_iterations=0",
        "--rdma_qp_calibration_sample_iterations=6", "--rdma_qp_calibration_max_attempts=9",
        "--rdma_qp_calibration_timeout_ms=1234"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    parsed = load_config(static_cast<int>(argv.size()), argv.data());
    require(!parsed.rdma_qp_calibration_enabled && parsed.min_bandwidth_gbps_needed == 342 &&
        parsed.rdma_qp_calibration_warmup_iterations == 0 && parsed.rdma_qp_calibration_sample_iterations == 6 &&
        parsed.rdma_qp_calibration_max_attempts == 9 && parsed.rdma_qp_calibration_timeout_ms == 1234,
        "CLI calibration overrides not parsed");
    std::filesystem::remove(path);
}

void test_run(bool enabled) {
    auto c = config();
    c.rdma_qp_calibration_enabled = enabled;
    Proxy p(c);
    p.initialize();
    require(!QpCalibrationTestAccess::complete(p), "calibration ran before external buffer initialization");
    p.prepare_iteration_step(0);
    require(QpCalibrationTestAccess::complete(p) == enabled, "calibration did not run at preparation");
    require(QpCalibrationTestAccess::measured(p) == 0, "calibration polluted iteration statistics");
    const auto markers = QpCalibrationTestAccess::markers(p);
    require(markers == (enabled ? 6 : 0), "expected sampling plus confirmation, or no calibration");
    p.run_iteration_step(0);
    p.run_iteration_step(1);
    require(QpCalibrationTestAccess::markers(p) == markers + 2, "calibration repeated on later iterations");
    require(QpCalibrationTestAccess::measured(p) == 2, "incorrect real iteration count");
    if (enabled) {
        QpCalibrationTestAccess::recreate(p);
        require(QpCalibrationTestAccess::markers(p) == 0, "recreated workers kept stale completions");
    }
    p.shutdown();
}

void test_exhaustion() {
    auto c = config();
    c.min_bandwidth_gbps_needed = 1e15;
    Proxy p(c);
    p.initialize();
    bool rejected = false;
    try { p.run(); }
    catch (const std::runtime_error& e) {
        const std::string message = e.what();
        rejected = message.find("exhausted") != std::string::npos;
        require(message.find("qp_calibration_history local_rank=0 local_gpu=0 remote_rank=1") != std::string::npos,
                "exhaustion diagnostic omitted GPU pair identity");
        require(message.find("round=1 generation=1") != std::string::npos &&
                message.find("round=2 generation=2") != std::string::npos,
                "exhaustion diagnostic omitted previous QP generations");
        require(message.find("local_min_gbps=") != std::string::npos &&
                message.find("local_max_gbps=") != std::string::npos &&
                message.find("local_median_gbps=") != std::string::npos &&
                message.find("remote_median_gbps=") != std::string::npos &&
                message.find("pair_min_median_gbps=") != std::string::npos,
                "exhaustion diagnostic omitted bandwidth history");
    }
    require(rejected, "unattainable threshold did not exhaust attempts");
    require(!QpCalibrationTestAccess::complete(p) && QpCalibrationTestAccess::measured(p) == 0,
            "real run started after failed calibration");
    // Two generations were tested. Recreating workers reset the first one's counters.
    require(QpCalibrationTestAccess::markers(p) == 3, "failed QP set was not recreated");
    p.shutdown();
}

void test_local_coordination(bool mismatch) {
    auto a = config();
    a.num_gpus_per_node = 2;
    a.local_iteration_sync_run_id = "calibration_test_" + std::to_string(getpid()) + (mismatch ? "m" : "s");
    auto b = a;
    b.local_gpu_index = 1;
    Proxy p0(a), p1(b);
    p0.initialize();
    p1.initialize();
    std::exception_ptr errors[2];
    const auto work = [&](Proxy& p, int gpu) {
        try {
            if (mismatch) {
                QpCalibrationTestAccess::reduce(p, true, gpu + 1);
                throw std::logic_error("mismatched settings passed");
            }
            for (int round = 0; round < 100; ++round) {
                if ((round + gpu) % 3 == 0) std::this_thread::sleep_for(std::chrono::microseconds(100));
                const bool local = gpu == 0 || round % 2 == 0;
                require(QpCalibrationTestAccess::reduce(p, local) == (round % 2 == 0),
                        "local calibration reduction raced");
            }
            p.run();
            require(QpCalibrationTestAccess::complete(p) && QpCalibrationTestAccess::measured(p) == 2,
                    "multi-GPU calibration did not complete");
        } catch (...) { errors[gpu] = std::current_exception(); }
    };
    std::thread t0(work, std::ref(p0), 0), t1(work, std::ref(p1), 1);
    t0.join(); t1.join();
    p0.shutdown(); p1.shutdown();
    for (const auto& error : errors) {
        if (!mismatch && error) std::rethrow_exception(error);
        if (mismatch) {
            require(static_cast<bool>(error), "mismatch did not propagate");
            try { std::rethrow_exception(error); }
            catch (const std::runtime_error& e) {
                require(std::string(e.what()).find("differ") != std::string::npos, "wrong mismatch failure");
            }
        }
    }
}

void test_timeout() {
    auto c = config();
    c.rdma_qp_calibration_timeout_ms = 1;
    c.rdma_qp_calibration_sample_iterations = 10000;
    Proxy p(c);
    p.initialize();
    bool failed = false;
    try { p.run(); }
    catch (const std::runtime_error& e) { failed = std::string(e.what()).find("timed out") != std::string::npos; }
    require(failed && QpCalibrationTestAccess::measured(p) == 0, "calibration timeout did not prevent real run");
    p.shutdown();
}

uint16_t unused_port() {
    const int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
    require(fd >= 0, "test socket failed");
    sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_loopback;
    require(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0, "test bind failed");
    socklen_t size = sizeof(addr);
    require(getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &size) == 0, "test getsockname failed");
    ::close(fd);
    return ntohs(addr.sin6_port);
}

void test_paired_control(bool exhaust) {
    const uint16_t ports[] = {unused_port(), unused_port()};
    require(ports[0] != ports[1], "test selected duplicate ports");
    std::vector<std::unique_ptr<Proxy>> proxies;
    for (int rank = 0; rank < 4; ++rank) {
        auto c = config();
        c.node_rank = rank / 2;
        c.local_gpu_index = rank % 2;
        c.num_gpus_per_node = 2;
        c.listen_port = ports[rank % 2];
        c.peers[0] = {1 - c.node_rank, "::1", c.listen_port};
        // Mock completions are loopback, but a recreated MR address exchanged
        // over real TCP points at the other proxy. This hybrid harness checks
        // control/lifecycle, not mock loopback payload-pattern expectations.
        c.validate_data = false;
        c.local_iteration_sync_run_id = "paired_calibration_" + std::to_string(getpid()) +
                                       (exhaust ? "fail" : "pass");
        if (exhaust) c.min_bandwidth_gbps_needed = 1e15;
        proxies.emplace_back(new Proxy(c));
        proxies.back()->initialize();
        // Keep mock RDMA data, but exercise actual cross-node TCP decisions and
        // QP metadata exchange as well as separate node-local reductions.
        QpCalibrationTestAccess::enable_tcp_control(*proxies.back());
    }
    std::exception_ptr errors[4];
    std::vector<std::thread> threads;
    for (int rank = 0; rank < 4; ++rank) threads.emplace_back([&, rank] {
        try { proxies[rank]->run(); }
        catch (...) { errors[rank] = std::current_exception(); }
    });
    for (auto& thread : threads) thread.join();
    for (int rank = 0; rank < 4; ++rank) {
        auto& p = *proxies[rank];
        if (!exhaust) {
            if (errors[rank]) std::rethrow_exception(errors[rank]);
            require(QpCalibrationTestAccess::complete(p) && QpCalibrationTestAccess::measured(p) == 2,
                    "paired calibration did not release all real iterations");
        } else {
            require(static_cast<bool>(errors[rank]), "paired exhaustion did not fail all ranks");
            try { std::rethrow_exception(errors[rank]); }
            catch (const std::runtime_error& e) {
                require(std::string(e.what()).find("exhausted") != std::string::npos,
                        "paired retry failed before coordinated exhaustion");
                const std::string message = e.what();
                require(message.find("qp_calibration_history local_rank=" + std::to_string(rank / 2) +
                            " local_gpu=" + std::to_string(rank % 2)) != std::string::npos &&
                        message.find("round=1 generation=1") != std::string::npos &&
                        message.find("round=2 generation=2") != std::string::npos,
                        "paired exhaustion did not report each failing pair's history");
            }
            require(QpCalibrationTestAccess::measured(p) == 0 && QpCalibrationTestAccess::markers(p) == 3,
                    "paired retry did not recreate QPs or started real iterations");
        }
        p.shutdown();
    }
}

void test_large_router_metadata() {
    auto a = config();
    a.mock_mode = false;
    a.router_routing_enabled = true;
    a.num_tokens = 200000;
    a.listen_port = unused_port();
    auto b = a;
    b.node_rank = 1;
    const PeerAddress server{1, "::1", a.listen_port};
    const PeerAddress client{0, "::1", a.listen_port};
    ConnectionManager m0(a), m1(b);
    RouterX3Metadata metadata;
    metadata.source_node_rank = 0;
    metadata.destination_node_rank = 1;
    metadata.local_gpu_index = 0;
    metadata.num_nodes = 2;
    metadata.num_gpus_per_node = 2;
    metadata.num_experts = 128;
    metadata.top_k = 8;
    metadata.num_tokens = a.num_tokens;
    metadata.token_dimension = 4096;
    metadata.element_bytes = 2;
    metadata.tokens_per_chunk = 32;
    metadata.token_masks.assign(a.num_tokens, 3);
    for (std::size_t token = 0; token < a.num_tokens; ++token)
        metadata.token_indices.push_back(token);
    const auto large = serialize_router_x3_metadata(metadata);
    require(large.size() > 1024 * 1024, "router regression payload must exceed old cap");
    metadata.source_node_rank = 1;
    metadata.destination_node_rank = 0;
    metadata.token_indices.resize(3);
    metadata.token_masks.resize(3);
    const auto small = serialize_router_x3_metadata(metadata);
    std::exception_ptr errors[2];
    // Exercise both receive paths, and a large remote list with a tiny local list.
    const auto run = [&](const ConnectionManager& manager, const PeerAddress& peer, int rank) {
        try {
            for (int round = 0; round < 2; ++round) {
                const bool sending_large = rank == round;
                const auto& own = sending_large ? large : small;
                const auto& expected = sending_large ? small : large;
                const auto received = manager.exchange_control_message(peer, own, 5000);
                require(received == expected, "large asymmetric router metadata was corrupted");
                const auto parsed = deserialize_router_x3_metadata(received, a.num_tokens);
                require(parsed.token_indices.size() == (sending_large ? 3 : a.num_tokens),
                        "large router metadata failed to deserialize");
            }
        } catch (...) { errors[rank] = std::current_exception(); }
    };
    std::thread t0(run, std::cref(m0), std::cref(server), 0);
    std::thread t1(run, std::cref(m1), std::cref(client), 1);
    t0.join(); t1.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

void test_transport() {
    auto a = config();
    a.mock_mode = false;
    a.listen_port = unused_port();
    auto b = a;
    b.node_rank = 1;
    const PeerAddress server{1, "::1", a.listen_port};
    const PeerAddress client{0, "::1", a.listen_port};
    ConnectionManager m0(a), m1(b);
    std::exception_ptr errors[2];
    const auto run = [&](const ConnectionManager& manager, const PeerAddress& peer, int rank) {
        try {
            for (int round = 0; round < 40; ++round) {
                const auto own = std::to_string(rank) + " sample " + std::to_string(round);
                const auto expected = std::to_string(1 - rank) + " sample " + std::to_string(round);
                require(manager.exchange_control_message(peer, own, 2000) == expected,
                        "calibration control exchanges were reordered");
            }
        } catch (...) { errors[rank] = std::current_exception(); }
    };
    std::thread t0(run, std::cref(m0), std::cref(server), 0);
    std::thread t1(run, std::cref(m1), std::cref(client), 1);
    t0.join(); t1.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);

    // A connected endpoint that sends no payload must not block forever.
    b.listen_port = unused_port();
    ConnectionManager silent_server(b);
    bool timed_out = false;
    std::thread listener([&] {
        try { silent_server.exchange_control_message(client, "waiting", 150); }
        catch (const std::runtime_error& e) { timed_out = std::string(e.what()).find("timed out") != std::string::npos; }
    });
    int fd = -1;
    for (int attempt = 0; attempt < 100; ++attempt) {
        fd = ::socket(AF_INET6, SOCK_STREAM, 0);
        sockaddr_in6 addr{};
        addr.sin6_family = AF_INET6;
        addr.sin6_addr = in6addr_loopback;
        addr.sin6_port = htons(b.listen_port);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) break;
        ::close(fd); fd = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    listener.join();
    if (fd >= 0) ::close(fd);
    require(fd >= 0 && timed_out, "connected silent calibration peer did not time out");
}

int main(int argc, char** argv) {
    Logger::instance().set_level(LogLevel::kError);
    if (argc == 2 && std::string(argv[1]) == "--transport-only") {
        test_transport();
        test_large_router_metadata();
        test_paired_control(false);
        test_paired_control(true);
        std::cout << "Calibration TCP tests passed\n";
        return 0;
    }
    test_config();
    test_run(false);
    test_run(true);
    test_exhaustion();
    test_timeout();
    test_local_coordination(false);
    test_local_coordination(true);
    std::cout << "QP calibration tests passed (mock transport; no hardware bandwidth claim)\n";
}
