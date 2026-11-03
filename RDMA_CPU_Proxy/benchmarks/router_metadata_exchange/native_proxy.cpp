#include "native_proxy.h"
#include "router_routing.hpp"
#include "protocol.hpp"
#include "logging.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <time.h>

namespace {
using namespace rdma_proxy;
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
using Key = std::pair<int, int>;
constexpr int Expert = 0, X3 = 1, Control = 2, Aggregate = 3;
constexpr size_t FrameBytes = 16;
const std::array<const char*, 6> Fields = {"elapsed_ms", "serialize_ms", "self_install_ms",
    "communication_prepare_ms", "tcp_exchange_ms", "receive_process_ms"};
const std::array<const char*, 9> ReceiveFields = {
    "expert_deserialize_ms", "expert_validate_ms", "expert_record_install_ms",
    "expert_progress_init_ms", "x3x4_deserialize_ms", "x3x4_validate_ms",
    "chunk_build_ms", "x3x4_record_install_ms", "overhead_ms"};
// Completion timestamps share the exchange-stage origin and overlap.
const std::array<const char*, 28> DiagnosticFields = {
    "local_expert_complete_ms", "remote_expert_complete_ms", "remote_x3x4_complete_ms",
    "socket_sends_complete_ms", "poll_ms", "poll_calls", "poll_zero_returns",
    "sendmsg_ms", "sendmsg_calls", "sendmsg_would_block", "sendmsg_interrupted",
    "recv_ms", "recv_calls", "recv_would_block", "recv_interrupted",
    "shm_publish_ms", "shm_receive_copy_ms", "receive_budget_yields", "exchange_other_ms",
    "shm_scan_ms", "shm_scan_after_socket_ms", "shm_scan_passes", "shm_scan_after_socket_passes",
    "shm_sequence_checks", "shm_not_ready_checks", "remote_observation_delay_max_ms",
    "remote_consume_delay_max_ms", "socket_work_complete_ms"};
enum DiagnosticIndex : size_t {
    SendMs = 7, SendCalls, SendWouldBlock, SendInterrupted,
    RecvMs, RecvCalls, RecvWouldBlock, RecvInterrupted,
    ShmPublishMs, ShmReceiveCopyMs, ReceiveBudgetYields, ExchangeOtherMs,
    ShmScanMs, ShmScanAfterSocketMs, ShmScanPasses, ShmScanAfterSocketPasses,
    ShmSequenceChecks, ShmNotReadyChecks, RemoteObserveMaxMs, RemoteConsumeMaxMs, SocketWorkCompleteMs
};
using ExchangeDiagnostics = std::array<double, DiagnosticFields.size()>;
struct ReceiveTimes : std::array<double, ReceiveFields.size()> {
    // Nested measurement, already included in expert_deserialize_ms.
    double expert_index_bounds_ms = 0;
};
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
double ms(Time end, Time begin) { return std::chrono::duration<double, std::milli>(end - begin).count(); }
// Comparable only between processes on the same host; never subtract across nodes.
uint64_t monotonic_ns() {
    timespec now{};
    require(clock_gettime(CLOCK_MONOTONIC, &now) == 0, "monotonic clock failed");
    return uint64_t(now.tv_sec) * 1000000000ULL + uint64_t(now.tv_nsec);
}
double ns_ms(uint64_t end, uint64_t begin) {
    return end >= begin ? double(end - begin) / 1e6 : -double(begin - end) / 1e6;
}
Time deadline(double seconds) { return Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds)); }
int remaining(Time until) {
    const double value = ms(until, Clock::now());
    require(value > 0, "TCP operation timed out");
    return static_cast<int>(std::min(std::ceil(value), double(std::numeric_limits<int>::max())));
}
void append64(std::string& out, uint64_t v) {
    for (int shift = 56; shift >= 0; shift -= 8) out.push_back(static_cast<char>(v >> shift));
}
uint64_t read64(const char* p) {
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i) result = (result << 8) | static_cast<unsigned char>(p[i]);
    return result;
}
struct Socket {
    int fd = -1;
    Socket() = default;
    explicit Socket(int v) : fd(v) {}
    ~Socket() { if (fd >= 0) ::close(fd); }
    Socket(Socket&& other) noexcept : fd(other.fd) { other.fd = -1; }
    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) { if (fd >= 0) ::close(fd); fd = other.fd; other.fd = -1; }
        return *this;
    }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
};
void nonblocking(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    require(flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0, "fcntl failed");
}
void configure(int fd) {
    nonblocking(fd);
    int one = 1;
    require(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0, "TCP_NODELAY failed");
#ifdef SO_NOSIGPIPE
    require(setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) == 0, "SO_NOSIGPIPE failed");
#endif
}
void wait_fd(int fd, short events, Time until) {
    pollfd p{fd, events, 0};
    for (;;) {
        int rc = poll(&p, 1, remaining(until));
        if (rc < 0 && errno == EINTR) continue;
        require(rc > 0, rc == 0 ? "TCP operation timed out" : "poll failed");
        require(!(p.revents & POLLNVAL), "invalid socket");
        return; // send/recv or SO_ERROR handles hangup and errors.
    }
}
ssize_t try_send(int fd, const char* data, size_t size) {
#ifdef MSG_NOSIGNAL
    constexpr int flags = MSG_NOSIGNAL;
#else
    constexpr int flags = 0;
#endif
    ssize_t n = ::send(fd, data, size, flags);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return 0;
    require(n > 0, "socket send failed/peer disconnected");
    return n;
}
ssize_t try_recv(int fd, char* data, size_t size, ExchangeDiagnostics* diagnostics = nullptr) {
    auto before = diagnostics ? Clock::now() : Time{};
    ssize_t n = ::recv(fd, data, size, 0);
    const int saved_errno = errno;
    if (diagnostics) {
        (*diagnostics)[RecvMs] += ms(Clock::now(), before);
        ++(*diagnostics)[RecvCalls];
        if (n < 0 && (saved_errno == EAGAIN || saved_errno == EWOULDBLOCK)) ++(*diagnostics)[RecvWouldBlock];
        if (n < 0 && saved_errno == EINTR) ++(*diagnostics)[RecvInterrupted];
    }
    errno = saved_errno;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return 0;
    require(n > 0, "socket receive failed/peer disconnected");
    return n;
}
void send_all(int fd, const char* data, size_t size, Time until) {
    size_t sent = 0;
    while (sent < size) { wait_fd(fd, POLLOUT, until); sent += try_send(fd, data + sent, size - sent); }
}
void recv_all(int fd, char* data, size_t size, Time until) {
    size_t received = 0;
    while (received < size) { wait_fd(fd, POLLIN, until); received += try_recv(fd, data + received, size - received); }
}
void send_message(int fd, const std::string& payload, Time until) {
    std::string header;
    append64(header, payload.size());
    send_all(fd, header.data(), header.size(), until);
    send_all(fd, payload.data(), payload.size(), until);
}
std::string recv_message(int fd, Time until) {
    std::array<char, 8> header{};
    recv_all(fd, header.data(), header.size(), until);
    uint64_t length = read64(header.data());
    require(length <= 64 * 1024 * 1024, "control message exceeds limit");
    std::string result(static_cast<size_t>(length), '\0');
    recv_all(fd, result.data(), result.size(), until);
    return result;
}
using AddrPtr = std::unique_ptr<addrinfo, decltype(&freeaddrinfo)>;
AddrPtr resolve(const std::string& host, int port) {
    addrinfo hints{}, *addresses = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_INET;
    require(getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &addresses) == 0,
            "cannot resolve host " + host);
    return AddrPtr(addresses, freeaddrinfo);
}
Socket connect_to(const std::string& host, int port, Time until) {
    auto addresses = resolve(host, port);
    for (;;) {
        for (auto* address = addresses.get(); address; address = address->ai_next) {
            Socket sock(::socket(address->ai_family, address->ai_socktype, address->ai_protocol));
            require(sock.fd >= 0, "socket failed");
            configure(sock.fd);
            int rc = ::connect(sock.fd, address->ai_addr, address->ai_addrlen);
            if (rc == 0) return sock;
            if (errno == EINPROGRESS) {
                wait_fd(sock.fd, POLLOUT, until);
                int error = 0; socklen_t size = sizeof(error);
                if (getsockopt(sock.fd, SOL_SOCKET, SO_ERROR, &error, &size) == 0 && error == 0) return sock;
            }
        }
        remaining(until);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
std::string quote(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
// Format generated JSON outside all timed intervals; no JSON dependency needed.
std::string pretty(const std::string& text) {
    std::string out;
    int depth = 0;
    bool in_string = false, escaped = false;
    auto newline = [&] { out += '\n'; out.append(size_t(depth) * 2, ' '); };
    for (char ch : text) {
        if (in_string) {
            out += ch;
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') in_string = false;
        } else if (ch == '"') { in_string = true; out += ch; }
        else if (ch == '{' || ch == '[') { out += ch; ++depth; newline(); }
        else if (ch == '}' || ch == ']') { --depth; newline(); out += ch; }
        else if (ch == ',') { out += ch; newline(); }
        else if (ch == ':') out += ": ";
        else out += ch;
    }
    return out;
}
struct Config {
    RouterMetadataBenchConfig a;
    std::vector<std::string> hosts;
    explicit Config(const RouterMetadataBenchConfig& c) : a(c) {
        require(a.hosts && a.bind_host && a.output, "null config string");
        require(!a.disable_expert_index_validation || a.binary,
                "disabling expert index validation requires binary mode");
        require(!a.local_shared_memory || a.binary, "shared memory requires binary mode");
        require(a.receive_drain_max_steps >= 1 && a.receive_drain_max_steps <= 1024, "invalid receive drain step budget");
        require(!a.same_index_aggregation || (a.binary && a.local_shared_memory), "aggregation requires binary shared memory");
        const uint32_t endian_probe = 1;
        require(!a.binary || (sizeof(size_t) == 8 && *reinterpret_cast<const uint8_t*>(&endian_probe) == 1),
                "binary mode requires 64-bit size_t and little-endian hosts");
        std::istringstream input(a.hosts); std::string host;
        while (std::getline(input, host, ',')) { require(!host.empty(), "empty node address"); hosts.push_back(host); }
        require(!hosts.empty() && std::string(a.hosts).back() != ',', "invalid hosts");
        require(a.gpus_per_node >= 1 && a.gpus_per_node <= 8 && a.world > 0 &&
                static_cast<size_t>(a.world) == hosts.size() * a.gpus_per_node, "invalid topology");
        require(a.rank >= 0 && a.rank < a.world, "rank out of range");
        require(a.num_tokens > 0 && a.num_experts > 0 && a.num_experts % a.world == 0 &&
                a.top_k > 0 && a.top_k <= a.num_experts, "invalid router dimensions");
        require(int64_t(a.num_tokens) * a.top_k <= INT32_MAX, "expert index capacity exceeds int32");
        require(a.token_dimension > 0 && a.element_bytes > 0 && a.tokens_per_chunk > 0,
                "invalid chunk dimensions");
        require(a.datasets > 0 && a.warmup >= 0 && a.iterations > 0 &&
                int64_t(a.warmup) + a.iterations <= INT32_MAX, "invalid iteration counts");
        require(std::isfinite(a.timeout) && a.timeout > 0 && a.timeout <= 86400, "invalid timeout");
        require(a.port_base > 0 && a.node_port_stride >= 0 &&
                int64_t(a.port_base) + (hosts.size() - 1) * uint64_t(a.node_port_stride) + a.gpus_per_node <= 65536,
                "invalid port range");
    }
    int node(int rank) const { return rank / a.gpus_per_node; }
    int gpu(int rank) const { return rank % a.gpus_per_node; }
    int port(int rank) const { return a.port_base + node(rank) * a.node_port_stride + gpu(rank); }
    int epg() const { return a.num_experts / a.world; }
    std::string json() const {
        std::ostringstream out;
        out << "{\"backend\":\"cpp\",\"timing_schema\":8,\"wire_format\":"
            << quote(a.binary ? "binary" : "text") << ",\"receive_storage\":"
            << quote(a.same_index_aggregation ? "shared_aggregate_arenas" : (a.binary ? "direct_reusable_vectors" : "payload_buffer")) << ",\"binary_protocol_version\":1,\"hosts\":[";
        for (size_t i = 0; i < hosts.size(); ++i) { if (i) out << ','; out << quote(hosts[i]); }
        out << ']';
#define FIELD(name) out << ",\"" #name "\":" << a.name
        FIELD(world); FIELD(gpus_per_node); FIELD(port_base); FIELD(node_port_stride);
        FIELD(num_tokens); FIELD(num_experts); FIELD(top_k); FIELD(token_dimension);
        FIELD(element_bytes); FIELD(tokens_per_chunk); FIELD(datasets); FIELD(warmup);
        FIELD(iterations); FIELD(seed); FIELD(timeout);
        out << ",\"metadata_tcp_channels_per_rank\":" << (a.same_index_aggregation ? hosts.size() - 1 :
            a.world - 1 - (a.local_shared_memory ? a.gpus_per_node - 1 : 0) + hosts.size() - 1);
        out << ",\"same_index_aggregation\":" << (a.same_index_aggregation ? "true" : "false");
        out << ",\"receive_drain\":" << (a.receive_drain ? "true" : "false")
            << ",\"receive_max_steps_per_socket\":" << (a.receive_drain ? a.receive_drain_max_steps : 1);
        out << ",\"local_shared_memory\":" << (a.local_shared_memory ? "true" : "false");
#undef FIELD
        out << ",\"expert_index_validation_enabled\":" << (a.disable_expert_index_validation ? "false" : "true");
        out << ",\"preserialized\":" << (a.preserialized ? "true" : "false")
            << ",\"verify\":" << (a.verify ? "true" : "false") << '}';
        return out.str();
    }
};
std::vector<int> kinds(const Config& c, int left, int right) {
    std::vector<int> result;
    if (c.a.same_index_aggregation) {
        if (c.gpu(left) == c.gpu(right)) result.push_back(Aggregate);
    } else {
        if (!c.a.local_shared_memory || c.node(left) != c.node(right)) result.push_back(Expert);
        if (c.gpu(left) == c.gpu(right)) result.push_back(X3);
    }
    if (left == 0) result.push_back(Control);
    return result;
}
struct Mesh {
    std::map<Key, Socket> channels;
    std::map<int, Socket> controls;
    explicit Mesh(const Config& c) {
        auto address = resolve(c.a.bind_host, c.port(c.a.rank));
        Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
        require(listener.fd >= 0, "listener socket failed");
        int one = 1;
        require(setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == 0, "SO_REUSEADDR failed");
        require(::bind(listener.fd, address->ai_addr, address->ai_addrlen) == 0,
                "bind failed on metadata port " + std::to_string(c.port(c.a.rank)) + ": " + std::strerror(errno));
        require(::listen(listener.fd, std::max(128, 3 * c.a.world)) == 0, "listen failed");
        nonblocking(listener.fd);
        auto add = [&](int peer, int kind, Socket sock) {
            if (kind == Control) controls.emplace(peer, std::move(sock));
            else channels.emplace(Key{peer, kind}, std::move(sock));
        };
        for (int peer = 0; peer < c.a.rank; ++peer) {
            std::string host = c.node(peer) == c.node(c.a.rank) ? "127.0.0.1" : c.hosts[c.node(peer)];
            for (int kind : kinds(c, peer, c.a.rank)) {
                auto until = deadline(c.a.timeout);
                Socket sock = connect_to(host, c.port(peer), until);
                send_message(sock.fd, std::to_string(c.a.rank) + " " + std::to_string(kind), until);
                add(peer, kind, std::move(sock));
            }
        }
        std::set<Key> expected;
        for (int peer = c.a.rank + 1; peer < c.a.world; ++peer)
            for (int kind : kinds(c, c.a.rank, peer)) expected.emplace(peer, kind);
        auto until = deadline(c.a.timeout);
        while (!expected.empty()) {
            wait_fd(listener.fd, POLLIN, until);
            Socket sock(::accept(listener.fd, nullptr, nullptr));
            if (sock.fd < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            require(sock.fd >= 0, "accept failed"); configure(sock.fd);
            std::istringstream in(recv_message(sock.fd, until)); int peer = -1, kind = -1;
            require(bool(in >> peer >> kind) && expected.erase({peer, kind}) == 1, "invalid/duplicate handshake");
            add(peer, kind, std::move(sock));
        }
    }
    // Gather to root and release peers. All control traffic is outside timing.
    std::vector<std::string> gather(const Config& c, const std::string& tag, const std::string& value) {
        auto until = deadline(c.a.timeout);
        if (c.a.rank) {
            send_message(controls.at(0).fd, tag + "\n" + value, until);
            auto reply = recv_message(controls.at(0).fd, until);
            if (tag == "shm_run") {
                require(reply.compare(0, tag.size() + 1, tag + "\n") == 0, "control phase mismatch");
                return {reply.substr(tag.size() + 1)};
            }
            require(reply == tag, "control phase mismatch");
            return {};
        }
        std::vector<std::string> values{value};
        for (int peer = 1; peer < c.a.world; ++peer) {
            auto message = recv_message(controls.at(peer).fd, until);
            require(message.compare(0, tag.size() + 1, tag + "\n") == 0, "control phase mismatch");
            values.push_back(message.substr(tag.size() + 1));
        }
        // Validate configuration before releasing other ranks.
        if (tag == "configuration")
            for (const auto& v : values) require(v == value, "ranks have different configurations");
        for (int peer = 1; peer < c.a.world; ++peer) send_message(controls.at(peer).fd, tag == "shm_run" ? tag + "\n" + value : tag, until);
        if (tag == "shm_run") return {value};
        return values;
    }
};
// Each source owns one arena. Destination slices together are bounded by T*K,
// avoiding a worst-case T*K allocation for every source/destination pair.
// Payloads are immutable until the existing global completion barrier releases.
struct LocalMemory {
    struct alignas(64) Header {
        uint64_t sequence;
        uint64_t counts[8];
        uint64_t x3_count;
        uint64_t ingress_start_ns, receive_complete_ns, publish_ns;
    };
    std::string name;
    void* base = MAP_FAILED;
    size_t bytes = 0, stride = 0;
    bool owner = false;
    LocalMemory() = default;
    LocalMemory(const LocalMemory&) = delete;
    ~LocalMemory() {
        if (base != MAP_FAILED) munmap(base, bytes);
        if (owner) shm_unlink(name.c_str());
    }
    Header* header(int source_gpu) const {
        return reinterpret_cast<Header*>(static_cast<char*>(base) + stride * source_gpu);
    }
    int32_t* offsets(int source_gpu) const {
        return reinterpret_cast<int32_t*>(reinterpret_cast<char*>(header(source_gpu)) + sizeof(Header));
    }
    int32_t* indices(const Config& c, int source_gpu) const {
        return offsets(source_gpu) + c.a.gpus_per_node * (c.epg() + 1);
    }
    int slot(const Config& c, int source) const {
        if (c.node(source) == c.node(c.a.rank)) return c.gpu(source);
        int remote_node = c.node(source) - (c.node(source) > c.node(c.a.rank) ? 1 : 0);
        return c.a.gpus_per_node + remote_node * c.a.gpus_per_node + c.gpu(source);
    }
    void initialize(const Config& c, Mesh& mesh) {
        if (!c.a.local_shared_memory) return;
        static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr),
                      "shared sequence requires lock-free 64-bit atomics");
        std::ostringstream id;
        id << std::hex << getpid() << "_" << Clock::now().time_since_epoch().count();
        auto identity = mesh.gather(c, "shm_run", id.str()).front();
        name = "/rmeta_" + identity + "_" + std::to_string(c.node(c.a.rank));
        stride = (sizeof(Header) + sizeof(int32_t) *
            (size_t(c.a.gpus_per_node) * (c.epg() + 1) + size_t(c.a.num_tokens) * c.a.top_k) + (c.a.same_index_aggregation ? size_t(c.a.num_tokens) * 9 : 0) + 63) & ~size_t(63);
        require(stride <= std::numeric_limits<size_t>::max() / c.a.gpus_per_node, "shared memory size overflow");
        size_t slots = c.a.same_index_aggregation ? c.a.world : c.a.gpus_per_node;
        require(stride <= std::numeric_limits<size_t>::max() / slots, "aggregate memory size overflow");
        bytes = stride * slots;
        auto until = deadline(c.a.timeout);
        Socket fd;
        if (c.gpu(c.a.rank) == 0) {
            fd = Socket(shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600));
            require(fd.fd >= 0, "cannot create metadata shared memory: " + std::string(std::strerror(errno)));
            owner = true;
            require(ftruncate(fd.fd, static_cast<off_t>(bytes)) == 0, "cannot size metadata shared memory");
        } else {
            for (;;) {
                fd = Socket(shm_open(name.c_str(), O_RDWR, 0600));
                if (fd.fd >= 0) break;
                require(errno == ENOENT, "cannot open metadata shared memory");
                remaining(until);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            for (;;) {
                struct stat st{};
                require(fstat(fd.fd, &st) == 0, "cannot stat metadata shared memory");
                if (st.st_size >= static_cast<off_t>(bytes)) break;
                remaining(until);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        base = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd.fd, 0);
        require(base != MAP_FAILED, "cannot map metadata shared memory");
        // Each process prefaults only its own arena; no concurrent initialization writes.
        std::memset(header(c.gpu(c.a.rank)), 0, stride);
        if (c.a.same_index_aggregation)
            for (int source = c.gpu(c.a.rank); source < c.a.world; source += c.a.gpus_per_node)
                if (c.node(source) != c.node(c.a.rank)) std::memset(header(slot(c, source)), 0, stride);
        mesh.gather(c, "shm_mapped", "");
        // All ranks now hold mappings; unlink early so killed runs leave no named object.
        if (owner) { require(shm_unlink(name.c_str()) == 0, "cannot unlink metadata shared memory"); owner = false; }
    }
};
struct Dataset {
    std::vector<RouterExpertMetadata> experts;
    std::map<int, RouterX3Metadata> nodes;
    std::map<Key, std::string> cached;
};
Dataset prepare(const Config& c, int source, int dataset) {
    ProxyConfig config;
    config.mock_mode = true; config.router_routing_enabled = true;
    config.node_rank = c.node(source); config.local_gpu_index = c.gpu(source);
    config.num_nodes = static_cast<int>(c.hosts.size()); config.num_gpus_per_node = c.a.gpus_per_node;
    config.num_tokens = c.a.num_tokens; config.router_num_experts = c.a.num_experts;
    config.router_top_k = c.a.top_k;
    config.router_seed = c.a.seed + uint64_t(source) + uint64_t(dataset) * c.a.world;
    RouterRouting routing(config); routing.initialize();
    Dataset result;
    for (int dest = 0; dest < c.a.world; ++dest)
        result.experts.push_back(routing.expert_metadata_for_gpu(c.node(dest), c.gpu(dest)));
    for (int node = 0; node < config.num_nodes; ++node) {
        if (node == c.node(source)) continue;
        RouterX3Metadata x;
        x.source_node_rank = c.node(source); x.destination_node_rank = node; x.local_gpu_index = c.gpu(source);
        x.num_nodes = config.num_nodes; x.num_gpus_per_node = c.a.gpus_per_node;
        x.num_experts = c.a.num_experts; x.top_k = c.a.top_k; x.num_tokens = c.a.num_tokens;
        x.token_dimension = c.a.token_dimension; x.element_bytes = c.a.element_bytes;
        x.tokens_per_chunk = c.a.tokens_per_chunk;
        x.token_indices = routing.token_indices_for_node(node); x.token_masks = routing.token_masks_for_node(node);
        result.nodes.emplace(node * c.a.gpus_per_node + c.gpu(source), std::move(x));
    }
    return result;
}
std::string encode(const Dataset& d, Key key) {
    return key.second == Expert ? serialize_router_expert_metadata(d.experts.at(key.first)) :
                                 serialize_router_x3_metadata(d.nodes.at(key.first));
}
// Fixed big-endian uint64 header fields; array storage is sent directly in
// little-endian native widths (int32 offsets/indices, uint64 x3, uint8 x4).
constexpr uint64_t BinaryMagic = 0x524d455842494e01ULL; // RMEXBIN + version 1
std::string binary_header(const Dataset& d, Key key) {
    std::string header;
    append64(header, BinaryMagic); append64(header, key.second);
    if (key.second == Aggregate) {
        const auto& x = d.nodes.at(key.first);
        for (uint64_t value : {uint64_t(x.source_node_rank * x.num_gpus_per_node + x.local_gpu_index),
                uint64_t(x.destination_node_rank), uint64_t(x.num_nodes), uint64_t(x.num_gpus_per_node),
                uint64_t(x.num_experts), uint64_t(x.num_tokens), uint64_t(x.top_k),
                uint64_t(x.token_dimension), uint64_t(x.element_bytes), uint64_t(x.tokens_per_chunk)}) append64(header, value);
        for (int gpu = 0; gpu < x.num_gpus_per_node; ++gpu)
            append64(header, d.experts.at(x.destination_node_rank * x.num_gpus_per_node + gpu).expert_token_indices.size());
        append64(header, x.token_indices.size());
    } else if (key.second == Expert) {
        const auto& m = d.experts.at(key.first);
        for (uint64_t value : {uint64_t(m.source_node_rank), uint64_t(m.source_gpu_index),
                uint64_t(m.destination_node_rank), uint64_t(m.destination_gpu_index),
                uint64_t(m.num_nodes), uint64_t(m.num_gpus_per_node), uint64_t(m.num_experts),
                uint64_t(m.experts_per_gpu), uint64_t(m.first_global_expert),
                uint64_t(m.num_tokens), uint64_t(m.expert_token_indices.size())}) append64(header, value);
    } else {
        const auto& m = d.nodes.at(key.first);
        for (uint64_t value : {uint64_t(m.source_node_rank), uint64_t(m.destination_node_rank),
                uint64_t(m.local_gpu_index), uint64_t(m.num_nodes), uint64_t(m.num_gpus_per_node),
                uint64_t(m.num_experts), uint64_t(m.top_k), uint64_t(m.num_tokens),
                uint64_t(m.token_dimension), uint64_t(m.element_bytes), uint64_t(m.tokens_per_chunk),
                uint64_t(m.token_indices.size())}) append64(header, value);
    }
    return header;
}
struct BinaryReader {
    const std::string& payload;
    size_t offset = 0;
    uint64_t word() {
        require(payload.size() - offset >= 8, "truncated binary header");
        auto v = read64(payload.data() + offset); offset += 8; return v;
    }
    int integer() {
        auto v = word(); require(v <= INT32_MAX, "binary header integer out of range"); return int(v);
    }
    template<class T> void array(std::vector<T>& values, size_t count) {
        require(count <= (payload.size() - offset) / sizeof(T), "truncated binary array");
        values.resize(count);
        if (count) std::memcpy(values.data(), payload.data() + offset, count * sizeof(T));
        offset += count * sizeof(T);
    }
    void prefix(int kind) {
        require(word() == BinaryMagic, "binary protocol version mismatch");
        require(word() == uint64_t(kind), "binary metadata kind mismatch");
    }
};
size_t read_expert_header(const Config& c, const std::string& header, size_t payload_bytes, RouterExpertMetadata& m) {
    BinaryReader r{header}; r.prefix(Expert);
    m.source_node_rank = r.integer(); m.source_gpu_index = r.integer();
    m.destination_node_rank = r.integer(); m.destination_gpu_index = r.integer();
    m.num_nodes = r.integer(); m.num_gpus_per_node = r.integer(); m.num_experts = r.integer();
    m.experts_per_gpu = r.integer(); m.first_global_expert = r.integer();
    m.num_tokens = r.word(); auto count = r.word();
    require(m.num_nodes > 0 && m.num_gpus_per_node > 0 && m.num_experts > 0 &&
            m.experts_per_gpu == c.epg() && m.num_tokens <= size_t(c.a.num_tokens) &&
            count <= uint64_t(c.a.num_tokens) * c.a.top_k, "invalid binary expert dimensions");
    require(payload_bytes >= r.offset && payload_bytes - r.offset == (size_t(m.experts_per_gpu) + 1 + count) * sizeof(int32_t),
            "binary expert payload length mismatch");
    return count;
}
void validate_expert_arrays(const RouterExpertMetadata& m, double* index_bounds_ms = nullptr, bool skip_index_bounds = false) {
    auto count = m.expert_token_indices.size();
    require(m.expert_offsets.front() == 0 && m.expert_offsets.back() == int64_t(count) &&
            std::is_sorted(m.expert_offsets.begin(), m.expert_offsets.end()), "invalid binary expert offsets");
    if (skip_index_bounds) return;
    const auto before = Clock::now();
    const bool valid = std::all_of(m.expert_token_indices.begin(), m.expert_token_indices.end(),
            [&](int32_t i) { return i >= 0 && size_t(i) < m.num_tokens; });
    const auto after = Clock::now();
    if (index_bounds_ms) *index_bounds_ms += ms(after, before);
    require(valid, "binary expert index out of range");
}
[[maybe_unused]] RouterExpertMetadata decode_expert_binary(const Config& c, const std::string& payload) {
    RouterExpertMetadata m;
    auto count = read_expert_header(c, payload, payload.size(), m);
    BinaryReader r{payload, 13 * 8};
    r.array(m.expert_offsets, size_t(m.experts_per_gpu) + 1); r.array(m.expert_token_indices, count);
    validate_expert_arrays(m, nullptr, c.a.disable_expert_index_validation); return m;
}
size_t read_x3_header(const Config& c, const std::string& header, size_t payload_bytes, RouterX3Metadata& m) {
    BinaryReader r{header}; r.prefix(X3);
    m.source_node_rank = r.integer(); m.destination_node_rank = r.integer(); m.local_gpu_index = r.integer();
    m.num_nodes = r.integer(); m.num_gpus_per_node = r.integer(); m.num_experts = r.integer(); m.top_k = r.integer();
    m.num_tokens = r.word(); m.token_dimension = r.word(); m.element_bytes = r.word(); m.tokens_per_chunk = r.word();
    auto count = r.word();
    require(m.num_nodes > 0 && m.num_gpus_per_node > 0 && m.num_gpus_per_node <= 8 &&
            m.num_experts > 0 && m.top_k > 0 && m.num_tokens <= size_t(c.a.num_tokens) &&
            count <= m.num_tokens && m.token_dimension > 0 && m.element_bytes > 0 && m.tokens_per_chunk > 0,
            "invalid binary x3/x4 dimensions");
    require(payload_bytes >= r.offset && payload_bytes - r.offset == count * (sizeof(uint64_t) + sizeof(uint8_t)),
            "binary x3/x4 payload length mismatch");
    return count;
}
void validate_x3_arrays(const RouterX3Metadata& m) {
    require(std::all_of(m.token_indices.begin(), m.token_indices.end(),
            [&](size_t i) { return i < m.num_tokens; }), "binary x3 index out of range");
    require(std::all_of(m.token_masks.begin(), m.token_masks.end(),
            [&](uint8_t mask) { return mask > 0 && unsigned(mask) < (1u << m.num_gpus_per_node); }),
            "binary x4 mask out of range");
 }
[[maybe_unused]] RouterX3Metadata decode_x3_binary(const Config& c, const std::string& payload) {
    RouterX3Metadata m;
    auto count = read_x3_header(c, payload, payload.size(), m);
    BinaryReader r{payload, 14 * 8};
    r.array(m.token_indices, count); r.array(m.token_masks, count);
    validate_x3_arrays(m); return m;
}
struct Installed {
    std::map<int, RouterExpertMetadata> experts;
    std::map<int, RouterX3Metadata> nodes;
    std::map<int, std::vector<ChunkDescriptor>> chunks;
    std::map<int, std::pair<std::vector<int32_t>, std::vector<int32_t>>> progress;
};
void install(const Config& c, Key key, const std::string& payload, Installed& installed, ReceiveTimes& times) {
    auto previous = Clock::now();
    auto mark = [&](size_t field) {
        auto now = Clock::now();
        times[field] += ms(now, previous);
        previous = now;
    };
    int source = key.first;
    if (key.second == Expert) {
        RouterExpertMetadata parsed;
        if (!c.a.binary) parsed = deserialize_router_expert_metadata(payload, c.a.num_tokens, size_t(c.a.num_tokens) * c.a.top_k);
        auto& m = c.a.binary ? installed.experts.at(source) : parsed;
        if (c.a.binary) validate_expert_arrays(m, &times.expert_index_bounds_ms, c.a.disable_expert_index_validation);
        mark(0);
        require(m.source_node_rank == c.node(source) && m.source_gpu_index == c.gpu(source) &&
                m.destination_node_rank == c.node(c.a.rank) && m.destination_gpu_index == c.gpu(c.a.rank) &&
                m.num_nodes == int(c.hosts.size()) && m.num_gpus_per_node == c.a.gpus_per_node &&
                m.num_experts == c.a.num_experts && m.experts_per_gpu == c.epg() &&
                m.first_global_expert == c.a.rank * c.epg() && m.num_tokens == size_t(c.a.num_tokens),
                "expert identity/dimensions mismatch");
        for (int e = 0; !c.a.disable_expert_index_validation && e < c.epg(); ++e)
            require(std::is_sorted(m.expert_token_indices.begin() + m.expert_offsets[e],
                                   m.expert_token_indices.begin() + m.expert_offsets[e + 1]), "unsorted expert indices");
        mark(1);
        if (!c.a.binary) require(installed.experts.emplace(source, std::move(m)).second, "duplicate expert source");
        mark(2);
        if (c.a.binary) {
            auto& progress = installed.progress.at(source);
            std::fill(progress.first.begin(), progress.first.end(), 0);
            std::fill(progress.second.begin(), progress.second.end(), 0);
        } else installed.progress.emplace(source, std::make_pair(std::vector<int32_t>(c.epg()), std::vector<int32_t>(c.epg())));
        mark(3);
    } else {
        RouterX3Metadata parsed;
        if (!c.a.binary) parsed = deserialize_router_x3_metadata(payload, c.a.num_tokens);
        auto& m = c.a.binary ? installed.nodes.at(source) : parsed;
        if (c.a.binary) validate_x3_arrays(m);
        mark(4);
        require(m.source_node_rank == c.node(source) && m.destination_node_rank == c.node(c.a.rank) &&
                m.local_gpu_index == c.gpu(c.a.rank) && c.gpu(source) == c.gpu(c.a.rank) &&
                m.num_nodes == int(c.hosts.size()) && m.num_gpus_per_node == c.a.gpus_per_node &&
                m.num_experts == c.a.num_experts && m.top_k == c.a.top_k && m.num_tokens == size_t(c.a.num_tokens) &&
                m.token_dimension == size_t(c.a.token_dimension) && m.element_bytes == size_t(c.a.element_bytes) &&
                m.tokens_per_chunk == size_t(c.a.tokens_per_chunk), "x3/x4 identity/dimensions mismatch");
        mark(5);
        auto chunks = compute_chunks_from_token_indices(m.token_indices, m.token_dimension, m.element_bytes, m.tokens_per_chunk, 1);
        mark(6);
        if (!c.a.binary) require(installed.nodes.emplace(source, std::move(m)).second, "duplicate x3/x4 source");
        installed.chunks.insert_or_assign(source, std::move(chunks));
        mark(7);
    }
}
struct SharedObservation {
    int source = -1, ingress_rank = -1;
    uint64_t ingress_start = 0, received = 0, published = 0, observed = 0, installed = 0;
    std::string json(uint64_t origin) const {
        std::ostringstream out; out << std::setprecision(17);
        out << "{\"source_rank\":" << source << ",\"ingress_rank\":" << ingress_rank
            << ",\"ingress_exchange_start_ns\":" << ingress_start
            << ",\"receive_complete_ns\":" << received
            << ",\"publish_ns\":" << published << ",\"observe_ns\":" << observed
            << ",\"install_complete_ns\":" << installed
            << ",\"publish_from_consumer_start_ms\":" << ns_ms(published, origin)
            << ",\"observe_from_consumer_start_ms\":" << ns_ms(observed, origin)
            << ",\"install_from_consumer_start_ms\":" << ns_ms(installed, origin)
            << ",\"observation_delay_ms\":" << ns_ms(observed, published)
            << ",\"consume_delay_ms\":" << ns_ms(installed, observed);
        if (received) out << ",\"ingress_receive_complete_ms\":" << ns_ms(received, ingress_start)
                         << ",\"receive_to_publish_ms\":" << ns_ms(published, received);
        out << '}'; return out.str();
    }
};
struct Record {
    int rank = 0;
    std::array<double, 6> times{};
    ReceiveTimes receive_times{};
    ExchangeDiagnostics diagnostics{};
    uint64_t exchange_start_ns = 0;
    std::vector<SharedObservation> shared_sources;
    uint64_t sent = 0, received = 0, experts = 0, nodes = 0, shm_sent = 0, shm_received = 0;
    std::string wire() const {
        std::ostringstream out; out << std::setprecision(17) << rank;
        for (double t : times) out << ' ' << t;
        for (double t : receive_times) out << ' ' << t;
        out << ' ' << receive_times.expert_index_bounds_ms;
        for (double value : diagnostics) out << ' ' << value;
        out << ' ' << sent << ' ' << received << ' ' << experts << ' ' << nodes << ' ' << shm_sent << ' ' << shm_received;
        out << ' ' << exchange_start_ns << ' ' << shared_sources.size();
        for (const auto& source : shared_sources)
            out << ' ' << source.source << ' ' << source.ingress_rank << ' ' << source.ingress_start
                << ' ' << source.received << ' ' << source.published << ' ' << source.observed << ' ' << source.installed;
        return out.str();
    }
    static Record parse(const std::string& text) {
        Record r; std::istringstream in(text); in >> r.rank;
        for (double& t : r.times) in >> t;
        for (double& t : r.receive_times) in >> t;
        in >> r.receive_times.expert_index_bounds_ms;
        for (double& value : r.diagnostics) in >> value;
        require(bool(in >> r.sent >> r.received >> r.experts >> r.nodes >> r.shm_sent >> r.shm_received), "invalid timing record");
        size_t count = 0;
        require(bool(in >> r.exchange_start_ns >> count) && count <= text.size(), "invalid shared observation count");
        r.shared_sources.resize(count);
        for (auto& source : r.shared_sources)
            require(bool(in >> source.source >> source.ingress_rank >> source.ingress_start >> source.received
                            >> source.published >> source.observed >> source.installed), "invalid shared observation");
        return r;
    }
    std::string json() const {
        std::ostringstream out; out << std::setprecision(17) << "{\"rank\":" << rank;
        for (size_t i = 0; i < times.size(); ++i) out << ',' << quote(Fields[i]) << ':' << times[i];
        out << ",\"exchange_diagnostics\":{";
        for (size_t i = 0; i < diagnostics.size(); ++i) {
            if (i) out << ',';
            out << quote(DiagnosticFields[i]) << ':' << diagnostics[i];
        }
        out << '}';
        out << ",\"exchange_start_ns\":" << exchange_start_ns << ",\"shared_source_observations\":[";
        for (size_t i = 0; i < shared_sources.size(); ++i) {
            if (i) out << ',';
            out << shared_sources[i].json(exchange_start_ns);
        }
        out << ']';
        out << ",\"receive_process_breakdown_ms\":{";
        for (size_t i = 0; i < receive_times.size(); ++i) {
            if (i) out << ',';
            out << quote(ReceiveFields[i]) << ':' << receive_times[i];
        }
        out << '}';
        out << ",\"expert_index_bounds_ms\":" << receive_times.expert_index_bounds_ms;
        out << ",\"install_ms\":" << times[2] + times[5] << ",\"sent_bytes\":" << sent
            << ",\"received_bytes\":" << received << ",\"shm_sent_bytes\":" << shm_sent << ",\"shm_received_bytes\":" << shm_received
            << ",\"exchange_ms\":" << times[4]
            << ",\"installed_expert_sources\":" << experts
            << ",\"installed_x3x4_sources\":" << nodes << '}';
        return out.str();
    }
};
struct Transfer {
    Key key;
    std::string output, input;
    std::vector<iovec> segments;
    std::array<iovec, 2> receive_arrays{};
    size_t payload_bytes = 0, array_index = 0, array_offset = 0;
    bool arrays_ready = false;
    size_t output_bytes = 0;
    std::array<char, FrameBytes> header{};
    size_t sent = 0, header_received = 0, received = 0;
    bool done = false;
};
// sendmsg references the existing arrays; no packed copy or decimal conversion.
size_t send_segments(int fd, const Transfer& t, ExchangeDiagnostics* diagnostics = nullptr) {
    std::array<iovec, 19> remaining_segments{};
    size_t skip = t.sent, count = 0;
    require(t.segments.size() <= remaining_segments.size(), "too many send segments");
    for (auto segment : t.segments) {
        if (skip >= segment.iov_len) { skip -= segment.iov_len; continue; }
        segment.iov_base = static_cast<char*>(segment.iov_base) + skip;
        segment.iov_len -= skip; skip = 0;
        remaining_segments[count++] = segment;
    }
    msghdr message{};
    message.msg_iov = remaining_segments.data(); message.msg_iovlen = count;
#ifdef MSG_NOSIGNAL
    constexpr int flags = MSG_NOSIGNAL;
#else
    constexpr int flags = 0;
#endif
    auto before = diagnostics ? Clock::now() : Time{};
    auto n = ::sendmsg(fd, &message, flags);
    const int saved_errno = errno;
    if (diagnostics) {
        (*diagnostics)[SendMs] += ms(Clock::now(), before);
        ++(*diagnostics)[SendCalls];
        if (n < 0 && (saved_errno == EAGAIN || saved_errno == EWOULDBLOCK)) ++(*diagnostics)[SendWouldBlock];
        if (n < 0 && saved_errno == EINTR) ++(*diagnostics)[SendInterrupted];
    }
    errno = saved_errno;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return 0;
    require(n > 0, "sendmsg failed/peer disconnected");
    return size_t(n);
}
Installed reserve_binary_receive_storage(const Config& c) {
    Installed storage;
    for (int source = 0; source < c.a.world; ++source) {
        auto& m = storage.experts[source];
        m.expert_offsets.reserve(size_t(c.epg()) + 1);
        m.expert_token_indices.reserve(size_t(c.a.num_tokens) * c.a.top_k);
        storage.progress.emplace(source, std::make_pair(std::vector<int32_t>(c.epg()), std::vector<int32_t>(c.epg())));
        if (source != c.a.rank && c.gpu(source) == c.gpu(c.a.rank)) {
            auto& x = storage.nodes[source];
            x.token_indices.reserve(c.a.num_tokens); x.token_masks.reserve(c.a.num_tokens);
        }
    }
    return storage;
}
// Capacity is reserved once before warmup. Changing lengths can zero-initialize
// newly exposed elements, but cannot allocate or invalidate the array pointers.
template<class T> void resize_reserved(std::vector<T>& v, size_t count) {
    require(count <= v.capacity(), "binary receive capacity exceeded");
    v.resize(count);
}
void prepare_direct_receive(const Config& c, Transfer& t, Installed& storage) {
    if (t.key.second == Expert) {
        auto& m = storage.experts.at(t.key.first);
        auto count = read_expert_header(c, t.input, t.payload_bytes, m);
        resize_reserved(m.expert_offsets, size_t(c.epg()) + 1);
        resize_reserved(m.expert_token_indices, count);
        t.receive_arrays = {iovec{m.expert_offsets.data(), m.expert_offsets.size() * sizeof(int32_t)},
                            iovec{m.expert_token_indices.data(), m.expert_token_indices.size() * sizeof(int32_t)}};
    } else {
        auto& m = storage.nodes.at(t.key.first);
        auto count = read_x3_header(c, t.input, t.payload_bytes, m);
        resize_reserved(m.token_indices, count); resize_reserved(m.token_masks, count);
        t.receive_arrays = {iovec{m.token_indices.data(), m.token_indices.size() * sizeof(size_t)},
                            iovec{m.token_masks.data(), m.token_masks.size()}};
    }
    t.arrays_ready = true;
}
void prepare_aggregate_receive(const Config& c, Transfer& t, LocalMemory& local) {
    BinaryReader r{t.input}; r.prefix(Aggregate);
    for (uint64_t expected : {uint64_t(t.key.first), uint64_t(c.node(c.a.rank)),
            uint64_t(c.hosts.size()), uint64_t(c.a.gpus_per_node), uint64_t(c.a.num_experts),
            uint64_t(c.a.num_tokens), uint64_t(c.a.top_k), uint64_t(c.a.token_dimension),
            uint64_t(c.a.element_bytes), uint64_t(c.a.tokens_per_chunk)})
        require(r.word() == expected, "aggregate identity/dimensions mismatch");
    require(c.gpu(t.key.first) == c.gpu(c.a.rank) && c.node(t.key.first) != c.node(c.a.rank), "invalid aggregate peer");
    int slot = local.slot(c, t.key.first);
    auto* h = local.header(slot);
    size_t total = 0;
    for (int gpu = 0; gpu < c.a.gpus_per_node; ++gpu) {
        auto count = r.word();
        require(count <= size_t(c.a.num_tokens) * c.a.top_k - total, "aggregate index count exceeds capacity");
        h->counts[gpu] = count; total += count;
    }
    h->x3_count = r.word();
    require(h->x3_count <= size_t(c.a.num_tokens), "aggregate x3 count exceeds capacity");
    size_t bytes = (size_t(c.a.gpus_per_node) * (c.epg() + 1) + total) * 4 + h->x3_count * 9;
    require(t.payload_bytes == t.input.size() + bytes, "aggregate payload length mismatch");
    t.receive_arrays = {iovec{local.offsets(slot), bytes}, iovec{nullptr, 0}};
    t.arrays_ready = true;
}
bool direct_arrays_complete(Transfer& t) {
    while (t.array_index < t.receive_arrays.size() && t.array_offset == t.receive_arrays[t.array_index].iov_len) {
        ++t.array_index; t.array_offset = 0;
    }
    return t.array_index == t.receive_arrays.size();
}
std::pair<Record, Installed> exchange(const Config& c, Mesh& mesh, LocalMemory& local, const Dataset& data, uint64_t iteration, Installed installed) {
    auto started = Clock::now();
    std::map<Key, std::string> serialized_payloads;
    if (!c.a.preserialized)
        for (const auto& pair : mesh.channels) serialized_payloads.emplace(pair.first, c.a.binary ? binary_header(data, pair.first) : encode(data, pair.first));
    const auto& payloads = c.a.preserialized ? data.cached : serialized_payloads;
    auto serialized = Clock::now();
    installed.experts.insert_or_assign(c.a.rank, data.experts.at(c.a.rank)); // native vector copy, baseline semantics
    installed.progress.emplace(c.a.rank, std::make_pair(std::vector<int32_t>(c.epg()), std::vector<int32_t>(c.epg())));
    auto self_installed = Clock::now();
    std::vector<Transfer> transfers;
    transfers.reserve(mesh.channels.size()); // keep segment pointers stable
    std::vector<pollfd> descriptors;
    for (const auto& [key, socket] : mesh.channels) {
        transfers.emplace_back();
        auto& t = transfers.back(); t.key = key;
        const auto& payload = payloads.at(key);
        size_t array_bytes = 0;
        std::vector<iovec> arrays;
        if (c.a.binary) {
            if (key.second == Aggregate) {
                int first = c.node(key.first) * c.a.gpus_per_node;
                for (int gpu = 0; gpu < c.a.gpus_per_node; ++gpu) {
                    const auto& m = data.experts.at(first + gpu);
                    arrays.push_back({const_cast<int32_t*>(m.expert_offsets.data()), m.expert_offsets.size() * 4});
                }
                for (int gpu = 0; gpu < c.a.gpus_per_node; ++gpu) {
                    const auto& m = data.experts.at(first + gpu);
                    arrays.push_back({const_cast<int32_t*>(m.expert_token_indices.data()), m.expert_token_indices.size() * 4});
                }
                const auto& x = data.nodes.at(key.first);
                arrays.push_back({const_cast<size_t*>(x.token_indices.data()), x.token_indices.size() * 8});
                arrays.push_back({const_cast<uint8_t*>(x.token_masks.data()), x.token_masks.size()});
            } else if (key.second == Expert) {
                const auto& m = data.experts.at(key.first);
                arrays = {iovec{const_cast<int32_t*>(m.expert_offsets.data()), m.expert_offsets.size() * sizeof(int32_t)},
                          iovec{const_cast<int32_t*>(m.expert_token_indices.data()), m.expert_token_indices.size() * sizeof(int32_t)}};
            } else {
                const auto& m = data.nodes.at(key.first);
                arrays = {iovec{const_cast<size_t*>(m.token_indices.data()), m.token_indices.size() * sizeof(uint64_t)},
                          iovec{const_cast<uint8_t*>(m.token_masks.data()), m.token_masks.size()}};
            }
            for (const auto& array : arrays) array_bytes += array.iov_len;
        }
        t.output.reserve(FrameBytes + payload.size());
        append64(t.output, payload.size() + array_bytes); append64(t.output, iteration);
        t.output += payload;
        t.segments.push_back(iovec{t.output.data(), t.output.size()});
        if (c.a.binary) for (const auto& array : arrays) t.segments.push_back(array);
        t.output_bytes = t.output.size() + array_bytes;
        if (c.a.binary) t.input.resize(key.second == Aggregate ? (13 + c.a.gpus_per_node) * 8 : (key.second == Expert ? 13 * 8 : 14 * 8));
        descriptors.push_back(pollfd{socket.fd, POLLIN | POLLOUT, 0});
    }
    auto until = deadline(c.a.timeout);
    size_t pending = transfers.size();
    const uint64_t max_payload = 4096 + (uint64_t(c.a.num_tokens) * c.a.top_k + c.epg() + 1) * 32;
    double receive_ms = 0;
    ReceiveTimes receive_times{};
    std::vector<bool> local_done(c.a.world, false);
    size_t local_pending = c.a.same_index_aggregation ? c.a.world - 1 : (c.a.local_shared_memory ? c.a.gpus_per_node - 1 : 0);
    uint64_t shm_sent = 0, shm_received = 0;
    ExchangeDiagnostics diagnostics{};
    std::array<size_t, 3> incoming_pending{c.a.local_shared_memory ? size_t(c.a.gpus_per_node - 1) : 0,
        c.a.same_index_aggregation ? size_t(c.a.world - c.a.gpus_per_node) : 0, 0};
    auto receive_path = [&](Key key) -> size_t {
        return (key.second == X3 || key.second == Aggregate) ? 2 : (c.node(key.first) == c.node(c.a.rank) ? 0 : 1);
    };
    for (const auto& t : transfers) ++incoming_pending[receive_path(t.key)];
    size_t sends_pending = transfers.size();
    std::vector<SharedObservation> observations;
    observations.reserve(local_pending);
    auto prepared = Clock::now();
    const uint64_t exchange_start_ns = monotonic_ns();
    auto received_one = [&](size_t path) {
        require(incoming_pending[path] > 0, "duplicate path completion");
        if (--incoming_pending[path] == 0) diagnostics[path] = ms(Clock::now(), prepared);
    };
    if (c.a.local_shared_memory) {
        auto publish_started = Clock::now();
        int gpu = c.gpu(c.a.rank);
        auto* h = local.header(gpu);
        size_t cursor = 0;
        for (int dest = 0; dest < c.a.gpus_per_node; ++dest) {
            h->counts[dest] = 0;
            if (dest == gpu) continue;
            const auto& m = data.experts.at(c.node(c.a.rank) * c.a.gpus_per_node + dest);
            require(m.expert_token_indices.size() <= size_t(c.a.num_tokens) * c.a.top_k - cursor,
                    "shared-memory producer capacity exceeded");
            require(m.expert_offsets.size() == size_t(c.epg() + 1), "invalid producer offsets");
            h->counts[dest] = m.expert_token_indices.size();
            std::copy(m.expert_offsets.begin(), m.expert_offsets.end(), local.offsets(gpu) + dest * (c.epg() + 1));
            std::copy(m.expert_token_indices.begin(), m.expert_token_indices.end(), local.indices(c, gpu) + cursor);
            cursor += m.expert_token_indices.size();
            shm_sent += (m.expert_offsets.size() + m.expert_token_indices.size()) * sizeof(int32_t);
        }
        h->ingress_start_ns = exchange_start_ns;
        h->receive_complete_ns = 0; // local generation, no incoming aggregate
        h->publish_ns = monotonic_ns();
        __atomic_store_n(&h->sequence, iteration + 1, __ATOMIC_RELEASE);
        diagnostics[ShmPublishMs] += ms(Clock::now(), publish_started);
    }
    while (pending || local_pending) {
        // Never block on sockets while a shared-memory producer may publish.
        int wait_ms = remaining(until);
        if (local_pending) {
            const bool sockets_done = pending == 0;
            auto scan_started = Clock::now();
            double scan_consume_ms = 0;
            ++diagnostics[ShmScanPasses];
            if (sockets_done) ++diagnostics[ShmScanAfterSocketPasses];
            for (int source = 0; source < c.a.world; ++source) {
                if (source == c.a.rank || local_done[source] ||
                    (!c.a.same_index_aggregation && c.node(source) != c.node(c.a.rank))) continue;
                int gpu = c.gpu(source);
                int slot = local.slot(c, source);
                auto* h = local.header(slot);
                ++diagnostics[ShmSequenceChecks];
                auto sequence = __atomic_load_n(&h->sequence, __ATOMIC_ACQUIRE);
                require(sequence <= iteration + 1, "future shared-memory iteration");
                if (sequence != iteration + 1) { ++diagnostics[ShmNotReadyChecks]; continue; }
                auto consume_started = Clock::now();
                SharedObservation observation;
                observation.source = source;
                observation.ingress_rank = c.node(c.a.rank) * c.a.gpus_per_node + gpu;
                observation.observed = monotonic_ns();
                observation.ingress_start = h->ingress_start_ns;
                observation.received = h->receive_complete_ns;
                observation.published = h->publish_ns;
                size_t cursor = 0, total = 0;
                for (int d = 0; d < c.a.gpus_per_node; ++d) {
                    require(h->counts[d] <= size_t(c.a.num_tokens) * c.a.top_k - total, "invalid shared-memory count");
                    total += h->counts[d];
                    if (d < c.gpu(c.a.rank)) cursor += h->counts[d];
                }
                auto before = Clock::now();
                auto& m = installed.experts.at(source);
                m.source_node_rank = c.node(source); m.source_gpu_index = gpu;
                m.destination_node_rank = c.node(c.a.rank); m.destination_gpu_index = c.gpu(c.a.rank);
                m.num_nodes = c.hosts.size(); m.num_gpus_per_node = c.a.gpus_per_node;
                m.num_experts = c.a.num_experts; m.experts_per_gpu = c.epg();
                m.first_global_expert = c.a.rank * c.epg(); m.num_tokens = c.a.num_tokens;
                resize_reserved(m.expert_offsets, c.epg() + 1);
                resize_reserved(m.expert_token_indices, h->counts[c.gpu(c.a.rank)]);
                double decode_ms = ms(Clock::now(), before);
                receive_times[0] += decode_ms; receive_ms += decode_ms;
                auto* offsets = local.offsets(slot) + c.gpu(c.a.rank) * (c.epg() + 1);
                auto copy_started = Clock::now();
                std::copy_n(offsets, m.expert_offsets.size(), m.expert_offsets.data());
                std::copy_n(local.indices(c, slot) + cursor, m.expert_token_indices.size(), m.expert_token_indices.data());
                diagnostics[ShmReceiveCopyMs] += ms(Clock::now(), copy_started);
                shm_received += (m.expert_offsets.size() + m.expert_token_indices.size()) * sizeof(int32_t);
                before = Clock::now();
                install(c, {source, Expert}, "", installed, receive_times);
                receive_ms += ms(Clock::now(), before);
                observation.installed = monotonic_ns();
                observations.push_back(observation);
                if (c.node(source) != c.node(c.a.rank)) {
                    diagnostics[RemoteObserveMaxMs] = std::max(diagnostics[RemoteObserveMaxMs], ns_ms(observation.observed, observation.published));
                    diagnostics[RemoteConsumeMaxMs] = std::max(diagnostics[RemoteConsumeMaxMs], ns_ms(observation.installed, observation.observed));
                }
                scan_consume_ms += ms(Clock::now(), consume_started);
                local_done[source] = true; --local_pending;
                received_one(c.node(source) == c.node(c.a.rank) ? 0 : 1);
            }
            double scan_ms = std::max(0.0, ms(Clock::now(), scan_started) - scan_consume_ms);
            diagnostics[ShmScanMs] += scan_ms;
            if (sockets_done) diagnostics[ShmScanAfterSocketMs] += scan_ms;
            wait_ms = 0;
        }
        if (!pending) continue;
        auto poll_started = Clock::now();
        int rc = poll(descriptors.data(), descriptors.size(), wait_ms);
        const int poll_errno = errno;
        diagnostics[4] += ms(Clock::now(), poll_started);
        ++diagnostics[5];
        if (rc == 0) ++diagnostics[6];
        errno = poll_errno;
        if (rc < 0 && errno == EINTR) continue;
        if (rc == 0 && wait_ms == 0) continue;
        require(rc > 0, rc == 0 ? "metadata exchange timed out" : "exchange poll failed");
        for (size_t i = 0; i < transfers.size(); ++i) {
            auto& p = descriptors[i]; auto& t = transfers[i];
            if (p.fd < 0 || !p.revents) continue;
            require(!(p.revents & (POLLERR | POLLNVAL)), "metadata socket error");
            if (p.revents & POLLOUT) {
                t.sent += send_segments(p.fd, t, &diagnostics);
                if (t.sent == t.output_bytes && --sends_pending == 0)
                    diagnostics[3] = ms(Clock::now(), prepared);
            }
            const bool was_done = t.done;
            const int receive_steps = c.a.receive_drain ? c.a.receive_drain_max_steps : 1;
            for (int step = 0; step < receive_steps &&
                 (p.revents & (POLLIN | POLLHUP)) && !t.done; ++step) {
                bool no_progress = false;
                auto receive = [&](int fd, char* destination, size_t bytes) {
                    auto count = try_recv(fd, destination, bytes, &diagnostics);
                    no_progress = count == 0; // would-block or EINTR: return to polling
                    return count;
                };
                if (t.header_received < FrameBytes) {
                    t.header_received += receive(p.fd, t.header.data() + t.header_received, FrameBytes - t.header_received);
                    if (t.header_received == FrameBytes) {
                        uint64_t size = read64(t.header.data());
                        require(read64(t.header.data() + 8) == iteration, "stale/mixed iteration");
                        require(size > 0 && size <= max_payload, "invalid payload size");
                        t.payload_bytes = size;
                        if (c.a.binary) require(size >= t.input.size(), "truncated binary protocol header");
                        else t.input.resize(static_cast<size_t>(size));
                    }
                } else if (c.a.binary) {
                    if (!t.arrays_ready) {
                        t.received += receive(p.fd, &t.input[t.received], t.input.size() - t.received);
                        if (t.received == t.input.size()) {
                            auto before = Clock::now();
                            if (t.key.second == Aggregate) prepare_aggregate_receive(c, t, local);
                            else prepare_direct_receive(c, t, installed);
                            double elapsed = ms(Clock::now(), before);
                            receive_times[t.key.second == Expert ? 0 : 4] += elapsed;
                            receive_ms += elapsed;
                        }
                    } else if (!direct_arrays_complete(t)) {
                        auto& array = t.receive_arrays[t.array_index];
                        t.array_offset += receive(p.fd, static_cast<char*>(array.iov_base) + t.array_offset,
                            std::min(size_t(262144), array.iov_len - t.array_offset));
                    }
                    if (t.arrays_ready && direct_arrays_complete(t)) {
                        // Publish expert slices before x3/x4 chunk construction.
                        if (t.key.second == Aggregate) {
                            int slot = local.slot(c, t.key.first);
                            auto* h = local.header(slot);
                            h->receive_complete_ns = monotonic_ns();
                            h->ingress_start_ns = exchange_start_ns;
                            size_t count = 0;
                            for (int gpu = 0; gpu < c.a.gpus_per_node; ++gpu) count += h->counts[gpu];
                            shm_sent += (size_t(c.a.gpus_per_node) * (c.epg() + 1) + count) * 4;
                            h->publish_ns = monotonic_ns();
                            __atomic_store_n(&h->sequence, iteration + 1, __ATOMIC_RELEASE);
                            auto before = Clock::now();
                            auto& x = installed.nodes.at(t.key.first);
                            x.source_node_rank = c.node(t.key.first); x.destination_node_rank = c.node(c.a.rank);
                            x.local_gpu_index = c.gpu(c.a.rank); x.num_nodes = c.hosts.size();
                            x.num_gpus_per_node = c.a.gpus_per_node; x.num_experts = c.a.num_experts;
                            x.top_k = c.a.top_k; x.num_tokens = c.a.num_tokens;
                            x.token_dimension = c.a.token_dimension; x.element_bytes = c.a.element_bytes;
                            x.tokens_per_chunk = c.a.tokens_per_chunk;
                            resize_reserved(x.token_indices, h->x3_count); resize_reserved(x.token_masks, h->x3_count);
                            auto* bytes = reinterpret_cast<char*>(local.indices(c, slot) + count);
                            if (h->x3_count) {
                                std::memcpy(x.token_indices.data(), bytes, h->x3_count * 8);
                                std::memcpy(x.token_masks.data(), bytes + h->x3_count * 8, h->x3_count);
                            }
                            double elapsed = ms(Clock::now(), before);
                            receive_times[4] += elapsed; receive_ms += elapsed;
                            before = Clock::now(); install(c, {t.key.first, X3}, "", installed, receive_times);
                            receive_ms += ms(Clock::now(), before);
                        } else {
                            auto before = Clock::now(); install(c, t.key, t.input, installed, receive_times);
                            receive_ms += ms(Clock::now(), before);
                        }
                        t.done = true;
                    }
                } else {
                    t.received += receive(p.fd, &t.input[t.received], std::min(size_t(262144), t.input.size() - t.received));
                    if (t.received == t.input.size()) {
                        auto before = Clock::now(); install(c, t.key, t.input, installed, receive_times);
                        receive_ms += ms(Clock::now(), before); t.done = true;
                    }
                }
                if (no_progress) break;
                if (c.a.receive_drain && step + 1 == receive_steps && !t.done)
                    ++diagnostics[ReceiveBudgetYields];
            }
            if (t.done && !was_done) received_one(receive_path(t.key));
            p.events = static_cast<short>((t.done ? 0 : POLLIN) | (t.sent == t.output_bytes ? 0 : POLLOUT));
            if (!p.events) {
                p.fd = -1;
                if (--pending == 0) diagnostics[SocketWorkCompleteMs] = ms(Clock::now(), prepared);
            }
            else require(!(p.revents & POLLHUP), "peer disconnected during metadata exchange");
        }
    }
    require(installed.experts.size() == size_t(c.a.world) && installed.nodes.size() == c.hosts.size() - 1,
            "incomplete installed metadata");
    auto finished = Clock::now();
    // Function entry/exit, temporary destruction, and instrumentation bookkeeping
    // are retained as a residual so the receive substages sum to the parent.
    // Roundoff can make an otherwise zero residual a tiny negative number.
    receive_times.back() = std::max(0.0, receive_ms - std::accumulate(receive_times.begin(), receive_times.end() - 1, 0.0));
    Record r; r.rank = c.a.rank;
    r.receive_times = receive_times;
    r.exchange_start_ns = exchange_start_ns;
    r.shared_sources = std::move(observations);
    diagnostics[ExchangeOtherMs] = ms(finished, prepared) - receive_ms - diagnostics[4]
        - diagnostics[SendMs] - diagnostics[RecvMs] - diagnostics[ShmPublishMs] - diagnostics[ShmReceiveCopyMs];
    r.diagnostics = diagnostics;
    r.shm_sent = shm_sent; r.shm_received = shm_received;
    r.times = {ms(finished, started), ms(serialized, started), ms(self_installed, serialized),
               ms(prepared, self_installed), ms(finished, prepared) - receive_ms, receive_ms};
    r.experts = installed.experts.size(); r.nodes = installed.nodes.size();
    for (const auto& t : transfers) { r.sent += t.output_bytes; r.received += t.payload_bytes + FrameBytes; }
    return {r, std::move(installed)};
}
std::string summary(std::vector<double> values) {
    require(!values.empty(), "empty statistics");
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end()); size_t n = values.size();
    double median = n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
    std::ostringstream out; out << std::setprecision(17)
        << "{\"min\":" << values.front() << ",\"median\":" << median << ",\"mean\":" << mean
        << ",\"p95\":" << values[size_t(std::ceil(n * .95)) - 1]
        << ",\"p99\":" << values[size_t(std::ceil(n * .99)) - 1] << ",\"max\":" << values.back() << '}';
    return out.str();
}
std::string index_bounds_summary(const std::vector<Record>& rows) {
    std::vector<double> values;
    for (const auto& row : rows) values.push_back(row.receive_times.expert_index_bounds_ms);
    return summary(values);
}
std::string receive_summary(const std::vector<Record>& rows) {
    std::ostringstream out; out << '{';
    for (size_t field = 0; field < ReceiveFields.size(); ++field) {
        if (field) out << ',';
        std::vector<double> values;
        for (const auto& row : rows) values.push_back(row.receive_times[field]);
        out << quote(ReceiveFields[field]) << ':' << summary(values);
    }
    out << '}'; return out.str();
}
std::string diagnostic_summary(const std::vector<Record>& rows) {
    std::ostringstream out; out << '{';
    for (size_t field = 0; field < DiagnosticFields.size(); ++field) {
        if (field) out << ',';
        std::vector<double> values;
        for (const auto& row : rows) values.push_back(row.diagnostics[field]);
        out << quote(DiagnosticFields[field]) << ':' << summary(values);
    }
    out << '}'; return out.str();
}
std::string stage_summary(const std::vector<Record>& rows, bool full) {
    std::ostringstream out; out << '{';
    for (size_t field = full ? 0 : 1; field < Fields.size(); ++field) {
        if (field != (full ? 0u : 1u)) out << ',';
        std::vector<double> values; for (const auto& row : rows) values.push_back(row.times[field]);
        out << quote(Fields[field]) << ':' << summary(values);
    }
    if (full) {
        std::vector<double> values; for (const auto& row : rows) values.push_back(row.times[2] + row.times[5]);
        out << ",\"install_ms\":" << summary(values);
        out << ",\"exchange_diagnostics\":" << diagnostic_summary(rows);
        out << ",\"receive_process_breakdown_ms\":" << receive_summary(rows);
        out << ",\"expert_index_bounds_ms\":" << index_bounds_summary(rows);
    }
    out << '}'; return out.str();
}
void report(const Config& c, const std::vector<std::vector<Record>>& records) {
    std::vector<Record> critical;
    for (const auto& row : records) critical.push_back(*std::max_element(row.begin(), row.end(),
        [](const auto& x, const auto& y) { return x.times[0] < y.times[0]; }));
    std::vector<double> totals; for (const auto& row : critical) totals.push_back(row.times[0]);
    std::string common = "\"critical_rank_ms\":" + summary(totals) +
        ",\"critical_rank_stage_ms\":" + stage_summary(critical, false) +
        ",\"critical_rank_exchange_diagnostics\":" + diagnostic_summary(critical) +
        ",\"critical_rank_receive_process_ms\":" + receive_summary(critical) +
        ",\"critical_rank_expert_index_bounds_ms\":" + index_bounds_summary(critical) +
        ",\"expert_index_validation_enabled\":" + (c.a.disable_expert_index_validation ? "false" : "true") +
        ",\"same_index_aggregation\":" + (c.a.same_index_aggregation ? "true" : "false") +
        ",\"receive_drain\":" + (c.a.receive_drain ? "true" : "false") +
        ",\"local_shared_memory\":" + (c.a.local_shared_memory ? "true" : "false") +
        ",\"backend\":\"cpp\",\"wire_format\":" + quote(c.a.binary ? "binary" : "text") + ",\"mode\":" + quote(c.a.preserialized ? "preserialized" : "serialize-in-timer");
    std::cout << pretty('{' + common + '}') << std::endl;
    if (!*c.a.output) return;
    std::filesystem::path path(c.a.output);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path); require(bool(out), "cannot open report " + path.string());
    out << '{' << common << ",\"configuration\":" << c.json() << ",\"rank_summaries\":[";
    for (int rank = 0; rank < c.a.world; ++rank) {
        if (rank) out << ',';
        std::vector<Record> rows; for (const auto& iteration : records) rows.push_back(iteration.at(rank));
        auto stats = stage_summary(rows, true);
        out << "{\"rank\":" << rank << ',' << stats.substr(1);
    }
    out << "],\"iterations\":[";
    for (size_t i = 0; i < records.size(); ++i) {
        if (i) out << ',';
        out << '[';
        for (size_t r = 0; r < records[i].size(); ++r) { if (r) out << ','; out << records[i][r].json(); }
        out << ']';
    }
    out << "]}\n"; out.close(); require(bool(out), "report write failed");
    std::cout << "Saved " << path << std::endl;
}
int run(const RouterMetadataBenchConfig& raw) {
    Config c(raw);
    Logger::instance().set_level(LogLevel::kError);
    Mesh mesh(c);
    mesh.gather(c, "configuration", c.json());
    LocalMemory local;
    local.initialize(c, mesh);
    if (!c.a.rank) std::cout << "Connected " << c.a.world << " C++ CPU proxies; preparing "
                            << c.a.datasets << " routing datasets per rank" << std::endl;
    std::vector<Dataset> datasets;
    // Optional exact-content verification uses production CPU routing to build
    // source-specific oracles before timing, then compares arrays after timing.
    std::vector<std::map<Key, std::string>> expected(c.a.verify ? c.a.datasets : 0);
    for (int d = 0; d < c.a.datasets; ++d) {
        auto data = prepare(c, c.a.rank, d);
        if (c.a.preserialized)
            for (const auto& entry : mesh.channels) data.cached.emplace(entry.first, c.a.binary ? binary_header(data, entry.first) : encode(data, entry.first));
        datasets.push_back(std::move(data));
        if (c.a.verify) {
            for (int source = 0; source < c.a.world; ++source) {
                auto oracle = prepare(c, source, d);
                expected[d][{source, Expert}] = encode(oracle, {c.a.rank, Expert});
                if (source != c.a.rank && c.gpu(source) == c.gpu(c.a.rank))
                    expected[d][{source, X3}] = encode(oracle, {c.a.rank, X3});
            }
        }
    }
    Installed reusable = c.a.binary ? reserve_binary_receive_storage(c) : Installed{};
    std::vector<std::vector<Record>> records;
    for (int iteration = 0; iteration < c.a.warmup + c.a.iterations; ++iteration) {
        int d = iteration % c.a.datasets;
        mesh.gather(c, "start:" + std::to_string(iteration), "");
        auto result = exchange(c, mesh, local, datasets[d], iteration, std::move(reusable));
        if (c.a.verify) {
            for (const auto& [source, metadata] : result.second.experts)
                require(serialize_router_expert_metadata(metadata) == expected[d].at({source, Expert}), "expert content verification failed");
            for (const auto& [source, metadata] : result.second.nodes)
                require(serialize_router_x3_metadata(metadata) == expected[d].at({source, X3}), "x3/x4 content verification failed");
        }
        if (c.a.binary) reusable = std::move(result.second);
        auto gathered = mesh.gather(c, "complete:" + std::to_string(iteration), result.first.wire());
        if (!c.a.rank && iteration >= c.a.warmup) {
            std::vector<Record> row;
            for (const auto& text : gathered) row.push_back(Record::parse(text));
            records.push_back(std::move(row));
        }
    }
    if (!c.a.rank) report(c, records);
    return 0;
}
} // namespace

extern "C" uint32_t router_metadata_bench_abi_version() { return 6; }
extern "C" int router_metadata_bench_run(const RouterMetadataBenchConfig* config, char* error, size_t capacity) {
    if (error && capacity) error[0] = '\0';
    try {
        require(config != nullptr, "null benchmark configuration");
        return run(*config);
    } catch (const std::exception& e) {
        if (error && capacity) { std::strncpy(error, e.what(), capacity - 1); error[capacity - 1] = '\0'; }
    } catch (...) {
        if (error && capacity) { std::strncpy(error, "unknown native benchmark failure", capacity - 1); error[capacity - 1] = '\0'; }
    }
    return 1;
}

#ifdef ROUTER_METADATA_BENCH_CODEC_TEST
int main() {
    try {
        RouterMetadataBenchConfig raw{};
        raw.receive_drain_max_steps = 16;
        raw.rank = 0; raw.world = 4; raw.gpus_per_node = 2;
        raw.port_base = 20515; raw.num_tokens = 137; raw.num_experts = 16; raw.top_k = 4;
        raw.token_dimension = 4096; raw.element_bytes = 2; raw.tokens_per_chunk = 7;
        raw.datasets = 1; raw.iterations = 1; raw.timeout = 5; raw.seed = 1234; raw.binary = 1;
        raw.hosts = "127.0.0.1,127.0.0.1"; raw.bind_host = "0.0.0.0"; raw.output = "";
        Config c(raw);
        Logger::instance().set_level(LogLevel::kError);
        auto data = prepare(c, 0, 0);
        auto pack_for_test = [&](Key key) {
            auto payload = binary_header(data, key);
            auto append = [&](const auto& array) {
                if (!array.empty()) payload.append(reinterpret_cast<const char*>(array.data()),
                                                   array.size() * sizeof(array[0]));
            };
            if (key.second == Expert) {
                append(data.experts.at(key.first).expert_offsets);
                append(data.experts.at(key.first).expert_token_indices);
            } else {
                append(data.nodes.at(key.first).token_indices);
                append(data.nodes.at(key.first).token_masks);
            }
            return payload;
        };
        auto reject = [&](const std::string& payload, int kind) {
            bool failed = false;
            try {
                if (kind == Expert) (void)decode_expert_binary(c, payload);
                else (void)decode_x3_binary(c, payload);
            } catch (const std::exception&) { failed = true; }
            require(failed, "malformed binary payload was accepted");
        };
        for (int dest = 0; dest < c.a.world; ++dest) {
            auto bytes = pack_for_test({dest, Expert});
            require(serialize_router_expert_metadata(decode_expert_binary(c, bytes)) == encode(data, {dest, Expert}),
                    "binary/text expert roundtrip mismatch");
            reject(bytes.substr(0, bytes.size() - 1), Expert);
            auto corrupt = bytes; corrupt[0] ^= 1; reject(corrupt, Expert);
            corrupt = bytes; corrupt[13 * 8] = 1; reject(corrupt, Expert); // nonzero first offset
            reject(bytes + 'x', Expert);
        }
        auto bytes = pack_for_test({2, X3});
        require(serialize_router_x3_metadata(decode_x3_binary(c, bytes)) == encode(data, {2, X3}),
                "binary/text x3/x4 roundtrip mismatch");
        reject(bytes.substr(0, 20), X3);
        auto corrupt = bytes; corrupt.back() = 0; reject(corrupt, X3);

        // Header preparation must retain the same typed-array storage across
        // changing routing tables, including empty input lists.
        c.a.rank = 2;
        // Aggregate headers must be validated before direct shared-memory reception.
        LocalMemory arena;
        arena.stride = 4096; arena.bytes = arena.stride * c.a.world;
        arena.base = mmap(nullptr, arena.bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
        require(arena.base != MAP_FAILED, "test aggregate arena mapping failed");
        Transfer aggregate; aggregate.key = {0, Aggregate};
        aggregate.input = binary_header(data, {2, Aggregate});
        size_t aggregate_bytes = 0;
        for (int dest = 2; dest < 4; ++dest)
            aggregate_bytes += (data.experts.at(dest).expert_offsets.size() + data.experts.at(dest).expert_token_indices.size()) * 4;
        aggregate_bytes += data.nodes.at(2).token_indices.size() * 9;
        aggregate.payload_bytes = aggregate.input.size() + aggregate_bytes;
        prepare_aggregate_receive(c, aggregate, arena);
        require(aggregate.receive_arrays[0].iov_base == arena.offsets(arena.slot(c, 0)) &&
                aggregate.receive_arrays[0].iov_len == aggregate_bytes, "aggregate receive does not target shared arena");
        auto reject_aggregate = [&](Transfer bad) {
            bool failed = false;
            try { prepare_aggregate_receive(c, bad, arena); } catch (const std::exception&) { failed = true; }
            require(failed, "malformed aggregate accepted");
            require(arena.header(arena.slot(c, 0))->sequence == 0, "incomplete aggregate published");
        };
        auto bad = aggregate; --bad.payload_bytes; reject_aggregate(bad);
        bad = aggregate; bad.input[0] ^= 1; reject_aggregate(bad);
        bad = aggregate; bad.input[2 * 8 + 7] = 1; reject_aggregate(bad); // wrong source
        bad = aggregate; bad.input.replace(12 * 8, 8, 8, char(0xff)); reject_aggregate(bad);
        bad = aggregate; bad.input.replace(bad.input.size() - 8, 8, 8, char(0xff)); reject_aggregate(bad);
        bad = aggregate; bad.input.resize(16); reject_aggregate(bad);

        auto storage = reserve_binary_receive_storage(c);
        auto* expert_base = storage.experts.at(0).expert_token_indices.data();
        auto* x3_base = storage.nodes.at(0).token_indices.data();
        auto check_validation_switch = [&](bool disabled, bool expect_failure) {
            c.a.disable_expert_index_validation = disabled;
            ReceiveTimes times{};
            bool failed = false;
            try { install(c, {0, Expert}, "", storage, times); }
            catch (const std::exception&) { failed = true; }
            require(failed == expect_failure, "expert validation switch behaved incorrectly");
            if (disabled) require(times.expert_index_bounds_ms == 0, "disabled bounds scan was timed");
        };
        auto& malformed = storage.experts.at(0);
        malformed = data.experts.at(2);
        malformed.expert_offsets.assign(c.epg() + 1, 2);
        malformed.expert_offsets.front() = 0;
        malformed.expert_token_indices = {1, 0}; // valid bounds, unsorted first expert
        check_validation_switch(false, true);
        check_validation_switch(true, false);
        malformed.expert_token_indices = {0, c.a.num_tokens}; // sorted, out of bounds
        check_validation_switch(false, true);
        check_validation_switch(true, false);
        malformed.expert_offsets.front() = 1; // structural check must remain active
        check_validation_switch(true, true);
        c.a.disable_expert_index_validation = false;
        for (int dataset = 0; dataset < 4; ++dataset) {
            data = prepare(c, 0, dataset);
            for (int kind : {Expert, X3}) {
                auto payload = pack_for_test({2, kind});
                Transfer t; t.key = {0, kind}; t.payload_bytes = payload.size();
                size_t header_size = kind == Expert ? 13 * 8 : 14 * 8;
                t.input = payload.substr(0, header_size);
                prepare_direct_receive(c, t, storage);
                size_t offset = header_size;
                for (const auto& array : t.receive_arrays) {
                    if (array.iov_len) std::memcpy(array.iov_base, payload.data() + offset, array.iov_len);
                    offset += array.iov_len;
                }
                ReceiveTimes times{};
                install(c, t.key, t.input, storage, times);
                require(offset == payload.size(), "direct receive size mismatch");
                const auto actual = kind == Expert ? serialize_router_expert_metadata(storage.experts.at(0)) :
                                                     serialize_router_x3_metadata(storage.nodes.at(0));
                require(actual == encode(data, {2, kind}), "direct receive changed metadata");
                require(expert_base == storage.experts.at(0).expert_token_indices.data() &&
                        x3_base == storage.nodes.at(0).token_indices.data(), "receive storage reallocated");
            }
        }

        // Force short scatter/gather writes: advance correctly across all segment
        // boundaries, including zero-length arrays, on a nonblocking socket.
        int fds[2]; require(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");
        Socket sender(fds[0]), receiver(fds[1]); nonblocking(sender.fd); nonblocking(receiver.fd);
        int small = 1024; setsockopt(sender.fd, SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
        std::string a(100003, 'a'), b(70001, 'b'), received;
        Transfer t;
        auto add_slices = [&](std::string& value) {
            size_t begin = 0;
            for (size_t slice = 1; slice <= 9; ++slice) {
                size_t end = value.size() * slice / 9;
                t.segments.push_back({value.data() + begin, end - begin}); begin = end;
            }
        };
        add_slices(a); t.segments.push_back({nullptr, 0}); add_slices(b); // aggregate maximum: 19 segments
        t.output_bytes = a.size() + b.size();
        ExchangeDiagnostics io{};
        char probe;
        require(try_recv(receiver.fd, &probe, 1, &io) == 0 && io[RecvWouldBlock] == 1,
                "empty nonblocking receive was not counted");
        auto until = deadline(5);
        while (received.size() < t.output_bytes) {
            remaining(until);
            if (t.sent < t.output_bytes) t.sent += send_segments(sender.fd, t, &io);
            char buffer[113];
            auto n = try_recv(receiver.fd, buffer, sizeof(buffer), &io);
            received.append(buffer, size_t(n));
        }
        require(t.sent == t.output_bytes && received == a + b, "partial sendmsg corrupted array bytes");
        require(io[SendCalls] > 1 && io[RecvCalls] > 1 && io[SendWouldBlock] > 0,
                "partial I/O instrumentation did not count calls/backpressure");
        std::cout << "Binary codec and partial-send tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
#endif
