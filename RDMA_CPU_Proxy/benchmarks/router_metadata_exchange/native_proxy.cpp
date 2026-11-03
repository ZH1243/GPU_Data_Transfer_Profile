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
#include <sys/uio.h>
#include <unistd.h>

namespace {
using namespace rdma_proxy;
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
using Key = std::pair<int, int>;
constexpr int Expert = 0, X3 = 1, Control = 2;
constexpr size_t FrameBytes = 16;
const std::array<const char*, 6> Fields = {"elapsed_ms", "serialize_ms", "self_install_ms",
    "communication_prepare_ms", "tcp_exchange_ms", "receive_process_ms"};
const std::array<const char*, 9> ReceiveFields = {
    "expert_deserialize_ms", "expert_validate_ms", "expert_record_install_ms",
    "expert_progress_init_ms", "x3x4_deserialize_ms", "x3x4_validate_ms",
    "chunk_build_ms", "x3x4_record_install_ms", "overhead_ms"};
using ReceiveTimes = std::array<double, ReceiveFields.size()>;
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
double ms(Time end, Time begin) { return std::chrono::duration<double, std::milli>(end - begin).count(); }
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
ssize_t try_recv(int fd, char* data, size_t size) {
    ssize_t n = ::recv(fd, data, size, 0);
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
        out << "{\"backend\":\"cpp\",\"timing_schema\":3,\"wire_format\":"
            << quote(a.binary ? "binary" : "text") << ",\"receive_storage\":"
            << quote(a.binary ? "direct_reusable_vectors" : "payload_buffer") << ",\"binary_protocol_version\":1,\"hosts\":[";
        for (size_t i = 0; i < hosts.size(); ++i) { if (i) out << ','; out << quote(hosts[i]); }
        out << ']';
#define FIELD(name) out << ",\"" #name "\":" << a.name
        FIELD(world); FIELD(gpus_per_node); FIELD(port_base); FIELD(node_port_stride);
        FIELD(num_tokens); FIELD(num_experts); FIELD(top_k); FIELD(token_dimension);
        FIELD(element_bytes); FIELD(tokens_per_chunk); FIELD(datasets); FIELD(warmup);
        FIELD(iterations); FIELD(seed); FIELD(timeout);
#undef FIELD
        out << ",\"preserialized\":" << (a.preserialized ? "true" : "false")
            << ",\"verify\":" << (a.verify ? "true" : "false") << '}';
        return out.str();
    }
};
std::vector<int> kinds(const Config& c, int left, int right) {
    std::vector<int> result{Expert};
    if (c.gpu(left) == c.gpu(right)) result.push_back(X3);
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
            require(recv_message(controls.at(0).fd, until) == tag, "control phase mismatch");
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
        for (int peer = 1; peer < c.a.world; ++peer) send_message(controls.at(peer).fd, tag, until);
        return values;
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
    if (key.second == Expert) {
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
void validate_expert_arrays(const RouterExpertMetadata& m) {
    auto count = m.expert_token_indices.size();
    require(m.expert_offsets.front() == 0 && m.expert_offsets.back() == int64_t(count) &&
            std::is_sorted(m.expert_offsets.begin(), m.expert_offsets.end()), "invalid binary expert offsets");
    require(std::all_of(m.expert_token_indices.begin(), m.expert_token_indices.end(),
            [&](int32_t i) { return i >= 0 && size_t(i) < m.num_tokens; }), "binary expert index out of range");
}
[[maybe_unused]] RouterExpertMetadata decode_expert_binary(const Config& c, const std::string& payload) {
    RouterExpertMetadata m;
    auto count = read_expert_header(c, payload, payload.size(), m);
    BinaryReader r{payload, 13 * 8};
    r.array(m.expert_offsets, size_t(m.experts_per_gpu) + 1); r.array(m.expert_token_indices, count);
    validate_expert_arrays(m); return m;
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
        if (c.a.binary) validate_expert_arrays(m);
        mark(0);
        require(m.source_node_rank == c.node(source) && m.source_gpu_index == c.gpu(source) &&
                m.destination_node_rank == c.node(c.a.rank) && m.destination_gpu_index == c.gpu(c.a.rank) &&
                m.num_nodes == int(c.hosts.size()) && m.num_gpus_per_node == c.a.gpus_per_node &&
                m.num_experts == c.a.num_experts && m.experts_per_gpu == c.epg() &&
                m.first_global_expert == c.a.rank * c.epg() && m.num_tokens == size_t(c.a.num_tokens),
                "expert identity/dimensions mismatch");
        for (int e = 0; e < c.epg(); ++e)
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
struct Record {
    int rank = 0;
    std::array<double, 6> times{};
    ReceiveTimes receive_times{};
    uint64_t sent = 0, received = 0, experts = 0, nodes = 0;
    std::string wire() const {
        std::ostringstream out; out << std::setprecision(17) << rank;
        for (double t : times) out << ' ' << t;
        for (double t : receive_times) out << ' ' << t;
        out << ' ' << sent << ' ' << received << ' ' << experts << ' ' << nodes;
        return out.str();
    }
    static Record parse(const std::string& text) {
        Record r; std::istringstream in(text); in >> r.rank;
        for (double& t : r.times) in >> t;
        for (double& t : r.receive_times) in >> t;
        require(bool(in >> r.sent >> r.received >> r.experts >> r.nodes), "invalid timing record");
        return r;
    }
    std::string json() const {
        std::ostringstream out; out << std::setprecision(17) << "{\"rank\":" << rank;
        for (size_t i = 0; i < times.size(); ++i) out << ',' << quote(Fields[i]) << ':' << times[i];
        out << ",\"receive_process_breakdown_ms\":{";
        for (size_t i = 0; i < receive_times.size(); ++i) {
            if (i) out << ',';
            out << quote(ReceiveFields[i]) << ':' << receive_times[i];
        }
        out << '}';
        out << ",\"install_ms\":" << times[2] + times[5] << ",\"sent_bytes\":" << sent
            << ",\"received_bytes\":" << received << ",\"installed_expert_sources\":" << experts
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
size_t send_segments(int fd, const Transfer& t) {
    std::array<iovec, 3> remaining_segments{};
    size_t skip = t.sent, count = 0;
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
    auto n = ::sendmsg(fd, &message, flags);
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
bool direct_arrays_complete(Transfer& t) {
    while (t.array_index < t.receive_arrays.size() && t.array_offset == t.receive_arrays[t.array_index].iov_len) {
        ++t.array_index; t.array_offset = 0;
    }
    return t.array_index == t.receive_arrays.size();
}
std::pair<Record, Installed> exchange(const Config& c, Mesh& mesh, const Dataset& data, uint64_t iteration, Installed installed) {
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
        std::array<iovec, 2> arrays{};
        if (c.a.binary) {
            if (key.second == Expert) {
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
        if (c.a.binary) t.input.resize(key.second == Expert ? 13 * 8 : 14 * 8);
        descriptors.push_back(pollfd{socket.fd, POLLIN | POLLOUT, 0});
    }
    auto until = deadline(c.a.timeout);
    size_t pending = transfers.size();
    const uint64_t max_payload = 4096 + (uint64_t(c.a.num_tokens) * c.a.top_k + c.epg() + 1) * 32;
    double receive_ms = 0;
    ReceiveTimes receive_times{};
    auto prepared = Clock::now();
    while (pending) {
        int rc = poll(descriptors.data(), descriptors.size(), remaining(until));
        if (rc < 0 && errno == EINTR) continue;
        require(rc > 0, rc == 0 ? "metadata exchange timed out" : "exchange poll failed");
        for (size_t i = 0; i < transfers.size(); ++i) {
            auto& p = descriptors[i]; auto& t = transfers[i];
            if (p.fd < 0 || !p.revents) continue;
            require(!(p.revents & (POLLERR | POLLNVAL)), "metadata socket error");
            if (p.revents & POLLOUT) t.sent += send_segments(p.fd, t);
            if ((p.revents & (POLLIN | POLLHUP)) && !t.done) {
                if (t.header_received < FrameBytes) {
                    t.header_received += try_recv(p.fd, t.header.data() + t.header_received, FrameBytes - t.header_received);
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
                        t.received += try_recv(p.fd, &t.input[t.received], t.input.size() - t.received);
                        if (t.received == t.input.size()) {
                            auto before = Clock::now();
                            prepare_direct_receive(c, t, installed);
                            double elapsed = ms(Clock::now(), before);
                            receive_times[t.key.second == Expert ? 0 : 4] += elapsed;
                            receive_ms += elapsed;
                        }
                    } else if (!direct_arrays_complete(t)) {
                        auto& array = t.receive_arrays[t.array_index];
                        t.array_offset += try_recv(p.fd, static_cast<char*>(array.iov_base) + t.array_offset,
                            std::min(size_t(262144), array.iov_len - t.array_offset));
                    }
                    if (t.arrays_ready && direct_arrays_complete(t)) {
                        auto before = Clock::now(); install(c, t.key, t.input, installed, receive_times);
                        receive_ms += ms(Clock::now(), before); t.done = true;
                    }
                } else {
                    t.received += try_recv(p.fd, &t.input[t.received], std::min(size_t(262144), t.input.size() - t.received));
                    if (t.received == t.input.size()) {
                        auto before = Clock::now(); install(c, t.key, t.input, installed, receive_times);
                        receive_ms += ms(Clock::now(), before); t.done = true;
                    }
                }
            }
            p.events = static_cast<short>((t.done ? 0 : POLLIN) | (t.sent == t.output_bytes ? 0 : POLLOUT));
            if (!p.events) { p.fd = -1; --pending; }
            else require(!(p.revents & POLLHUP), "peer disconnected during metadata exchange");
        }
    }
    require(installed.experts.size() == size_t(c.a.world) && installed.nodes.size() == c.hosts.size() - 1,
            "incomplete installed metadata");
    auto finished = Clock::now();
    // Function entry/exit, temporary destruction, and instrumentation bookkeeping
    // are retained as a residual so the receive substages sum to the parent.
    receive_times.back() = receive_ms - std::accumulate(receive_times.begin(), receive_times.end() - 1, 0.0);
    Record r; r.rank = c.a.rank;
    r.receive_times = receive_times;
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
        out << ",\"receive_process_breakdown_ms\":" << receive_summary(rows);
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
        ",\"critical_rank_receive_process_ms\":" + receive_summary(critical) +
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
        auto result = exchange(c, mesh, datasets[d], iteration, std::move(reusable));
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

extern "C" uint32_t router_metadata_bench_abi_version() { return 2; }
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
        auto storage = reserve_binary_receive_storage(c);
        auto* expert_base = storage.experts.at(0).expert_token_indices.data();
        auto* x3_base = storage.nodes.at(0).token_indices.data();
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
        t.segments = {{a.data(), a.size()}, {nullptr, 0}, {b.data(), b.size()}};
        t.output_bytes = a.size() + b.size();
        auto until = deadline(5);
        while (received.size() < t.output_bytes) {
            remaining(until);
            if (t.sent < t.output_bytes) t.sent += send_segments(sender.fd, t);
            char buffer[113];
            auto n = try_recv(receiver.fd, buffer, sizeof(buffer));
            received.append(buffer, size_t(n));
        }
        require(t.sent == t.output_bytes && received == a + b, "partial sendmsg corrupted array bytes");
        std::cout << "Binary codec and partial-send tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
#endif
