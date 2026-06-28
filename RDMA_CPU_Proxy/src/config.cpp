#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace rdma_proxy {
namespace {

struct Json {
    using Object = std::map<std::string, Json>;
    using Array = std::vector<Json>;
    std::variant<std::nullptr_t, bool, double, std::string, Object, Array> value;
};

class JsonParser {
public:
    explicit JsonParser(std::string input) : input_(std::move(input)) {}

    Json parse() {
        auto json = parse_value();
        skip_ws();
        if (pos_ != input_.size()) {
            throw std::runtime_error("unexpected trailing characters in JSON config");
        }
        return json;
    }

private:
    Json parse_value() {
        skip_ws();
        if (pos_ >= input_.size()) throw std::runtime_error("unexpected end of JSON");
        const char c = input_[pos_];
        if (c == '{') return Json{parse_object()};
        if (c == '[') return Json{parse_array()};
        if (c == '"') return Json{parse_string()};
        if (c == 't') return parse_literal("true", Json{true});
        if (c == 'f') return parse_literal("false", Json{false});
        if (c == 'n') return parse_literal("null", Json{nullptr});
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return Json{parse_number()};
        throw std::runtime_error("invalid JSON value");
    }

    Json::Object parse_object() {
        expect('{');
        Json::Object object;
        skip_ws();
        if (peek('}')) {
            ++pos_;
            return object;
        }
        while (true) {
            skip_ws();
            const auto key = parse_string();
            skip_ws();
            expect(':');
            object.emplace(key, parse_value());
            skip_ws();
            if (peek('}')) {
                ++pos_;
                break;
            }
            expect(',');
        }
        return object;
    }

    Json::Array parse_array() {
        expect('[');
        Json::Array array;
        skip_ws();
        if (peek(']')) {
            ++pos_;
            return array;
        }
        while (true) {
            array.push_back(parse_value());
            skip_ws();
            if (peek(']')) {
                ++pos_;
                break;
            }
            expect(',');
        }
        return array;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos_ < input_.size()) {
            const char c = input_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= input_.size()) throw std::runtime_error("invalid JSON escape");
                const char e = input_[pos_++];
                switch (e) {
                    case '"':
                    case '\\':
                    case '/':
                        out.push_back(e);
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    default:
                        throw std::runtime_error("unsupported JSON escape sequence");
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    double parse_number() {
        const std::size_t begin = pos_;
        if (peek('-')) ++pos_;
        while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        if (peek('.')) {
            ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        }
        if (peek('e') || peek('E')) {
            ++pos_;
            if (peek('+') || peek('-')) ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        }
        return std::stod(input_.substr(begin, pos_ - begin));
    }

    Json parse_literal(const char* text, Json value) {
        const std::string literal(text);
        if (input_.compare(pos_, literal.size(), literal) != 0) {
            throw std::runtime_error("invalid JSON literal");
        }
        pos_ += literal.size();
        return value;
    }

    void skip_ws() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) ++pos_;
    }

    bool peek(char c) const {
        return pos_ < input_.size() && input_[pos_] == c;
    }

    void expect(char c) {
        skip_ws();
        if (!peek(c)) {
            throw std::runtime_error(std::string("expected JSON character '") + c + "'");
        }
        ++pos_;
    }

    std::string input_;
    std::size_t pos_{0};
};

const Json::Object& as_object(const Json& json, const std::string& name) {
    if (!std::holds_alternative<Json::Object>(json.value)) {
        throw std::runtime_error(name + " must be a JSON object");
    }
    return std::get<Json::Object>(json.value);
}

const Json::Array& as_array(const Json& json, const std::string& name) {
    if (!std::holds_alternative<Json::Array>(json.value)) {
        throw std::runtime_error(name + " must be a JSON array");
    }
    return std::get<Json::Array>(json.value);
}

bool has(const Json::Object& object, const std::string& key) {
    return object.find(key) != object.end();
}

const Json& get_required(const Json::Object& object, const std::string& key) {
    auto it = object.find(key);
    if (it == object.end()) throw std::runtime_error("missing required config key: " + key);
    return it->second;
}

std::string get_string(const Json::Object& object, const std::string& key, const std::string& fallback = {}) {
    if (!has(object, key)) return fallback;
    const auto& json = get_required(object, key);
    if (!std::holds_alternative<std::string>(json.value)) throw std::runtime_error(key + " must be a string");
    return std::get<std::string>(json.value);
}

double get_number(const Json::Object& object, const std::string& key, double fallback = 0) {
    if (!has(object, key)) return fallback;
    const auto& json = get_required(object, key);
    if (!std::holds_alternative<double>(json.value)) throw std::runtime_error(key + " must be a number");
    return std::get<double>(json.value);
}

bool get_bool(const Json::Object& object, const std::string& key, bool fallback = false) {
    if (!has(object, key)) return fallback;
    const auto& json = get_required(object, key);
    if (!std::holds_alternative<bool>(json.value)) throw std::runtime_error(key + " must be a boolean");
    return std::get<bool>(json.value);
}

template <typename T>
T number_as(const Json::Object& object, const std::string& key, T fallback = {}) {
    return static_cast<T>(get_number(object, key, static_cast<double>(fallback)));
}

void apply_arg(ProxyConfig& config, const std::string& key, const std::string& value) {
    if (key == "node_rank") config.node_rank = std::stoi(value);
    else if (key == "num_nodes") config.num_nodes = std::stoi(value);
    else if (key == "local_gpu_index") config.local_gpu_index = std::stoi(value);
    else if (key == "num_gpus_per_node") config.num_gpus_per_node = std::stoi(value);
    else if (key == "num_tokens") config.num_tokens = static_cast<std::size_t>(std::stoull(value));
    else if (key == "token_dimension") config.token_dimension = static_cast<std::size_t>(std::stoull(value));
    else if (key == "tokens_per_chunk") config.tokens_per_chunk = static_cast<std::size_t>(std::stoull(value));
    else if (key == "num_qps_per_peer") config.num_qps_per_peer = std::stoi(value);
    else if (key == "rdma_device_name") config.rdma_device_name = value;
    else if (key == "rdma_port") config.rdma_port = static_cast<uint8_t>(std::stoi(value));
    else if (key == "gid_index") config.gid_index = std::stoi(value);
    else if (key == "cuda_device_id") config.cuda_device_id = std::stoi(value);
    else if (key == "listen_port") {
        config.listen_port = static_cast<uint16_t>(std::stoi(value));
        for (auto& peer : config.peers) {
            peer.port = config.listen_port;
        }
    }
    else if (key == "connection_manager_port") config.connection_manager_port = static_cast<uint16_t>(std::stoi(value));
    else if (key == "peer_port") {
        const auto port = static_cast<uint16_t>(std::stoi(value));
        for (auto& peer : config.peers) {
            peer.port = port;
        }
    }
    else if (key == "peer_host") {
        if (config.peers.size() != 1) {
            throw std::runtime_error("--peer_host is only supported when the config has exactly one peer");
        }
        config.peers[0].host = value;
    }
    else if (key == "completion_poll_batch_size") config.completion_poll_batch_size = std::stoi(value);
    else if (key == "send_queue_depth") config.send_queue_depth = std::stoi(value);
    else if (key == "recv_queue_depth") config.recv_queue_depth = std::stoi(value);
    else if (key == "cq_depth") config.cq_depth = std::stoi(value);
    else if (key == "num_iterations") config.num_iterations = static_cast<std::size_t>(std::stoull(value));
    else if (key == "completion_timeout_ms") config.completion_timeout_ms = static_cast<uint64_t>(std::stoull(value));
    else if (key == "dtype") config.dtype = dtype_from_string(value);
    else if (key == "mock_mode") config.mock_mode = (value == "1" || value == "true" || value == "yes");
    else if (key == "fill_test_data") config.fill_test_data = (value == "1" || value == "true" || value == "yes");
    else if (key == "validate_data") config.validate_data = (value == "1" || value == "true" || value == "yes");
    else if (key == "log_level") config.log_level = value;
}

}  // namespace

std::size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::kBF16:
        case DataType::kFP16:
            return 2;
        case DataType::kFP32:
            return 4;
    }
    return 2;
}

std::string to_string(DataType dtype) {
    switch (dtype) {
        case DataType::kBF16:
            return "bf16";
        case DataType::kFP16:
            return "fp16";
        case DataType::kFP32:
            return "fp32";
    }
    return "bf16";
}

DataType dtype_from_string(const std::string& value) {
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "bf16" || v == "bfloat16") return DataType::kBF16;
    if (v == "fp16" || v == "half") return DataType::kFP16;
    if (v == "fp32" || v == "float") return DataType::kFP32;
    throw std::runtime_error("unsupported dtype: " + value);
}

ProxyConfig load_config_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to open config file: " + path);
    std::ostringstream ss;
    ss << input.rdbuf();

    const auto root = JsonParser(ss.str()).parse();
    const auto& object = as_object(root, "config root");

    ProxyConfig config;
    config.node_rank = number_as<int>(object, "node_rank", config.node_rank);
    config.num_nodes = number_as<int>(object, "num_nodes", config.num_nodes);
    config.local_gpu_index = number_as<int>(object, "local_gpu_index", config.local_gpu_index);
    config.num_gpus_per_node = number_as<int>(object, "num_gpus_per_node", config.num_gpus_per_node);
    config.num_tokens = number_as<std::size_t>(object, "num_tokens", config.num_tokens);
    config.token_dimension = number_as<std::size_t>(object, "token_dimension", config.token_dimension);
    config.tokens_per_chunk = number_as<std::size_t>(object, "tokens_per_chunk", config.tokens_per_chunk);
    config.num_qps_per_peer = number_as<int>(object, "num_qps_per_peer", config.num_qps_per_peer);
    config.rdma_device_name = get_string(object, "rdma_device_name", config.rdma_device_name);
    config.rdma_port = number_as<uint8_t>(object, "rdma_port", config.rdma_port);
    config.gid_index = number_as<int>(object, "gid_index", config.gid_index);
    config.cuda_device_id = number_as<int>(object, "cuda_device_id", config.cuda_device_id);
    config.listen_port = number_as<uint16_t>(object, "listen_port", config.listen_port);
    config.connection_manager_port = number_as<uint16_t>(object, "connection_manager_port", config.connection_manager_port);
    config.completion_poll_batch_size = number_as<int>(object, "completion_poll_batch_size", config.completion_poll_batch_size);
    config.send_queue_depth = number_as<int>(object, "send_queue_depth", config.send_queue_depth);
    config.recv_queue_depth = number_as<int>(object, "recv_queue_depth", config.recv_queue_depth);
    config.cq_depth = number_as<int>(object, "cq_depth", config.cq_depth);
    config.num_iterations = number_as<std::size_t>(object, "num_iterations", config.num_iterations);
    config.completion_timeout_ms = number_as<uint64_t>(
        object, "completion_timeout_ms", config.completion_timeout_ms);
    config.dtype = dtype_from_string(get_string(object, "dtype", to_string(config.dtype)));
    config.mock_mode = get_bool(object, "mock_mode", config.mock_mode);
    config.fill_test_data = get_bool(object, "fill_test_data", config.fill_test_data);
    config.validate_data = get_bool(object, "validate_data", config.validate_data);
    config.log_level = get_string(object, "log_level", config.log_level);

    if (has(object, "peers")) {
        for (const auto& entry : as_array(get_required(object, "peers"), "peers")) {
            const auto& peer_obj = as_object(entry, "peer");
            PeerAddress peer;
            peer.node_rank = number_as<int>(peer_obj, "node_rank", -1);
            peer.host = get_string(peer_obj, "host");
            peer.port = number_as<uint16_t>(peer_obj, "port", config.connection_manager_port);
            config.peers.push_back(peer);
        }
    }

    validate_config(config);
    return config;
}

ProxyConfig load_config(int argc, char** argv) {
    std::string path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            path = argv[++i];
        }
    }
    if (path.empty()) {
        throw std::runtime_error("usage: rdma_cpu_proxy --config config/example_config.json [--key=value ...]");
    }

    auto config = load_config_file(path);
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--config" || arg == "-c") {
            ++i;
            continue;
        }
        if (arg.rfind("--", 0) == 0) {
            arg = arg.substr(2);
            const auto eq = arg.find('=');
            if (eq == std::string::npos) throw std::runtime_error("expected --key=value argument: --" + arg);
            apply_arg(config, arg.substr(0, eq), arg.substr(eq + 1));
        }
    }
    validate_config(config);
    return config;
}

void validate_config(const ProxyConfig& config) {
    if (config.num_nodes < 1) throw std::runtime_error("num_nodes must be >= 1");
    if (config.node_rank < 0 || config.node_rank >= config.num_nodes) throw std::runtime_error("node_rank out of range");
    if (config.local_gpu_index < 0 || config.local_gpu_index >= config.num_gpus_per_node) {
        throw std::runtime_error("local_gpu_index out of range");
    }
    if (config.num_tokens == 0) throw std::runtime_error("num_tokens must be > 0");
    if (config.token_dimension == 0) throw std::runtime_error("token_dimension must be > 0");
    if (config.tokens_per_chunk == 0) throw std::runtime_error("tokens_per_chunk must be > 0");
    if (config.num_qps_per_peer <= 0) throw std::runtime_error("num_qps_per_peer must be > 0");
    if (config.completion_poll_batch_size <= 0) throw std::runtime_error("completion_poll_batch_size must be > 0");
    if (config.send_queue_depth <= 0 || config.recv_queue_depth <= 0 || config.cq_depth <= 0) {
        throw std::runtime_error("queue and CQ depths must be > 0");
    }
    if (config.completion_timeout_ms == 0) throw std::runtime_error("completion_timeout_ms must be > 0");
    if (config.num_nodes > 1 && static_cast<int>(config.peers.size()) != config.num_nodes - 1) {
        throw std::runtime_error("peers must contain exactly num_nodes - 1 entries");
    }
}

std::string config_summary(const ProxyConfig& config) {
    std::ostringstream out;
    out << "rank=" << config.node_rank << "/" << config.num_nodes
        << " gpu=" << config.local_gpu_index
        << " tokens=" << config.num_tokens
        << " dim=" << config.token_dimension
        << " chunk_tokens=" << config.tokens_per_chunk
        << " qps_per_peer=" << config.num_qps_per_peer
        << " iterations=" << config.num_iterations
        << " dtype=" << to_string(config.dtype)
        << " mock_mode=" << (config.mock_mode ? "true" : "false");
    return out.str();
}

}  // namespace rdma_proxy
