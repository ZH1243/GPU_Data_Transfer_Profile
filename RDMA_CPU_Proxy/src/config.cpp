#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
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

uint64_t parse_u64(const std::string& value, const std::string& key) {
    std::size_t consumed = 0;
    const int base = value.rfind("0x", 0) == 0 || value.rfind("0X", 0) == 0 ? 16 : 10;
    const auto parsed = std::stoull(value, &consumed, base);
    if (consumed != value.size()) {
        throw std::runtime_error(key + " must be an unsigned integer string");
    }
    return static_cast<uint64_t>(parsed);
}

uint64_t get_u64_string_or_number(const Json::Object& object, const std::string& key, uint64_t fallback = 0) {
    if (!has(object, key)) return fallback;
    const auto& json = get_required(object, key);
    if (std::holds_alternative<std::string>(json.value)) {
        return parse_u64(std::get<std::string>(json.value), key);
    }
    if (std::holds_alternative<double>(json.value)) {
        const double value = std::get<double>(json.value);
        if (value < 0 || value > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
            throw std::runtime_error(key + " out of uint64 range");
        }
        return static_cast<uint64_t>(value);
    }
    throw std::runtime_error(key + " must be a string or number");
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
    else if (key == "data_signal_interval") config.data_signal_interval = std::stoi(value);
    else if (key == "max_in_flight_chunks_per_qp") config.max_in_flight_chunks_per_qp = std::stoi(value);
    else if (key == "rdma_chunk_per_token_sge_enabled") {
        config.rdma_chunk_per_token_sge_enabled = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "rdma_discontinuous_token_payload_enabled") {
        config.rdma_discontinuous_token_payload_enabled = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "send_queue_depth") config.send_queue_depth = std::stoi(value);
    else if (key == "recv_queue_depth") config.recv_queue_depth = std::stoi(value);
    else if (key == "cq_depth") config.cq_depth = std::stoi(value);
    else if (key == "num_iterations") config.num_iterations = static_cast<std::size_t>(std::stoull(value));
    else if (key == "completion_timeout_ms") config.completion_timeout_ms = static_cast<uint64_t>(std::stoull(value));
    else if (key == "dtype") config.dtype = dtype_from_string(value);
    else if (key == "mock_mode") config.mock_mode = (value == "1" || value == "true" || value == "yes");
    else if (key == "fill_test_data") config.fill_test_data = (value == "1" || value == "true" || value == "yes");
    else if (key == "validate_data") config.validate_data = (value == "1" || value == "true" || value == "yes");
    else if (key == "sequential_peer_transfers") {
        config.sequential_peer_transfers = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forwarding_enabled" || key == "nvlink_forwarding_enable") {
        config.nvlink_forwarding_enabled = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_threshold_tokens") {
        config.nvlink_forward_threshold_tokens = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_threshold_chunks") {
        config.nvlink_forward_threshold_chunks = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_min_threshold_chunks") {
        config.nvlink_forward_min_threshold_chunks = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_max_threshold_chunks") {
        config.nvlink_forward_max_threshold_chunks = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_out_of_order_chunks_enabled") {
        config.nvlink_forward_out_of_order_chunks_enabled = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_chunk_tokens") {
        config.nvlink_forward_chunk_tokens = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_use_batch_api") {
        config.nvlink_forward_use_batch_api = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_stream_nonblocking") {
        config.nvlink_forward_stream_nonblocking = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_synchronize_batches") {
        config.nvlink_forward_synchronize_batches = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_completion_notifications_enabled") {
        config.nvlink_forward_completion_notifications_enabled =
            (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_notification_queue_depth") {
        config.nvlink_forward_notification_queue_depth = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_notification_log_enabled") {
        config.nvlink_forward_notification_log_enabled =
            (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_notification_log_dir") {
        config.nvlink_forward_notification_log_dir = value;
    }
    else if (key == "nvlink_forward_computation_enabled") {
        config.nvlink_forward_computation_enabled = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_computation_output_dim") {
        config.nvlink_forward_computation_output_dim = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_computation_tile_m") {
        config.nvlink_forward_computation_tile_m = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_computation_tile_n") {
        config.nvlink_forward_computation_tile_n = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_computation_num_queues") {
        config.nvlink_forward_computation_num_queues = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_computation_queue_depth") {
        config.nvlink_forward_computation_queue_depth = static_cast<std::size_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_computation_load_only_enabled") {
        config.nvlink_forward_computation_load_only_enabled =
            (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_computation_log_enabled") {
        config.nvlink_forward_computation_log_enabled = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_local_batch_sync_enabled") {
        config.nvlink_forward_local_batch_sync_enabled = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_synchronize_iteration") {
        config.nvlink_forward_synchronize_iteration = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_log_batches") {
        config.nvlink_forward_log_batches = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "log_qp_reports") {
        config.log_qp_reports = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "log_marker_wait_reports") {
        config.log_marker_wait_reports = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_forward_use_round_robin") {
        config.nvlink_forward_use_round_robin = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "nvlink_routing_probability") {
        config.nvlink_routing_probability = std::stod(value);
    }
    else if (key == "nvlink_routing_seed") {
        config.nvlink_routing_seed = static_cast<uint64_t>(std::stoull(value));
    }
    else if (key == "nvlink_forward_exchange_dir") config.nvlink_forward_exchange_dir = value;
    else if (key == "local_iteration_sync_enabled") {
        config.local_iteration_sync_enabled = (value == "1" || value == "true" || value == "yes");
    }
    else if (key == "local_iteration_sync_dir") config.local_iteration_sync_dir = value;
    else if (key == "local_iteration_sync_run_id") config.local_iteration_sync_run_id = value;
    else if (key == "rdma_bandwidth_summary_dir") config.rdma_bandwidth_summary_dir = value;
    else if (key == "cpu_affinity") config.cpu_affinity = value;
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

bool nvlink_forward_dynamic_threshold_enabled(const ProxyConfig& config) {
    return config.nvlink_forward_min_threshold_chunks != 0 ||
           config.nvlink_forward_max_threshold_chunks != 0;
}

std::size_t effective_nvlink_forward_threshold_tokens(const ProxyConfig& config) {
    if (config.nvlink_forward_threshold_chunks == 0) {
        return config.nvlink_forward_threshold_tokens;
    }
    if (config.tokens_per_chunk != 0 &&
        config.nvlink_forward_threshold_chunks >
            std::numeric_limits<std::size_t>::max() / config.tokens_per_chunk) {
        throw std::runtime_error("nvlink_forward_threshold_chunks * tokens_per_chunk overflows size_t");
    }
    return config.nvlink_forward_threshold_chunks * config.tokens_per_chunk;
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
    config.data_signal_interval = number_as<int>(object, "data_signal_interval", config.data_signal_interval);
    config.max_in_flight_chunks_per_qp = number_as<int>(
        object, "max_in_flight_chunks_per_qp", config.max_in_flight_chunks_per_qp);
    config.rdma_chunk_per_token_sge_enabled = get_bool(
        object, "rdma_chunk_per_token_sge_enabled", config.rdma_chunk_per_token_sge_enabled);
    config.rdma_discontinuous_token_payload_enabled = get_bool(
        object, "rdma_discontinuous_token_payload_enabled", config.rdma_discontinuous_token_payload_enabled);
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
    config.sequential_peer_transfers = get_bool(
        object, "sequential_peer_transfers", config.sequential_peer_transfers);
    config.nvlink_forwarding_enabled = get_bool(
        object, "nvlink_forwarding_enabled", config.nvlink_forwarding_enabled);
    config.nvlink_forward_threshold_tokens = number_as<std::size_t>(
        object, "nvlink_forward_threshold_tokens", config.nvlink_forward_threshold_tokens);
    config.nvlink_forward_threshold_chunks = number_as<std::size_t>(
        object, "nvlink_forward_threshold_chunks", config.nvlink_forward_threshold_chunks);
    config.nvlink_forward_min_threshold_chunks = number_as<std::size_t>(
        object, "nvlink_forward_min_threshold_chunks", config.nvlink_forward_min_threshold_chunks);
    config.nvlink_forward_max_threshold_chunks = number_as<std::size_t>(
        object, "nvlink_forward_max_threshold_chunks", config.nvlink_forward_max_threshold_chunks);
    config.nvlink_forward_out_of_order_chunks_enabled = get_bool(
        object,
        "nvlink_forward_out_of_order_chunks_enabled",
        config.nvlink_forward_out_of_order_chunks_enabled);
    config.nvlink_forward_chunk_tokens = number_as<std::size_t>(
        object, "nvlink_forward_chunk_tokens", config.nvlink_forward_chunk_tokens);
    config.nvlink_forward_use_batch_api = get_bool(
        object, "nvlink_forward_use_batch_api", config.nvlink_forward_use_batch_api);
    config.nvlink_forward_stream_nonblocking = get_bool(
        object, "nvlink_forward_stream_nonblocking", config.nvlink_forward_stream_nonblocking);
    config.nvlink_forward_synchronize_batches = get_bool(
        object, "nvlink_forward_synchronize_batches", config.nvlink_forward_synchronize_batches);
    config.nvlink_forward_completion_notifications_enabled = get_bool(
        object,
        "nvlink_forward_completion_notifications_enabled",
        config.nvlink_forward_completion_notifications_enabled);
    config.nvlink_forward_notification_queue_depth = number_as<std::size_t>(
        object,
        "nvlink_forward_notification_queue_depth",
        config.nvlink_forward_notification_queue_depth);
    config.nvlink_forward_notification_log_enabled = get_bool(
        object,
        "nvlink_forward_notification_log_enabled",
        config.nvlink_forward_notification_log_enabled);
    config.nvlink_forward_notification_log_dir = get_string(
        object,
        "nvlink_forward_notification_log_dir",
        config.nvlink_forward_notification_log_dir);
    config.nvlink_forward_computation_enabled = get_bool(
        object, "nvlink_forward_computation_enabled", config.nvlink_forward_computation_enabled);
    config.nvlink_forward_computation_output_dim = number_as<std::size_t>(
        object, "nvlink_forward_computation_output_dim", config.nvlink_forward_computation_output_dim);
    config.nvlink_forward_computation_tile_m = number_as<std::size_t>(
        object, "nvlink_forward_computation_tile_m", config.nvlink_forward_computation_tile_m);
    config.nvlink_forward_computation_tile_n = number_as<std::size_t>(
        object, "nvlink_forward_computation_tile_n", config.nvlink_forward_computation_tile_n);
    config.nvlink_forward_computation_num_queues = number_as<std::size_t>(
        object, "nvlink_forward_computation_num_queues", config.nvlink_forward_computation_num_queues);
    config.nvlink_forward_computation_queue_depth = number_as<std::size_t>(
        object, "nvlink_forward_computation_queue_depth", config.nvlink_forward_computation_queue_depth);
    config.nvlink_forward_computation_load_only_enabled = get_bool(
        object,
        "nvlink_forward_computation_load_only_enabled",
        config.nvlink_forward_computation_load_only_enabled);
    config.nvlink_forward_computation_log_enabled = get_bool(
        object, "nvlink_forward_computation_log_enabled", config.nvlink_forward_computation_log_enabled);
    config.nvlink_forward_local_batch_sync_enabled = get_bool(
        object, "nvlink_forward_local_batch_sync_enabled", config.nvlink_forward_local_batch_sync_enabled);
    config.nvlink_forward_synchronize_iteration = get_bool(
        object, "nvlink_forward_synchronize_iteration", config.nvlink_forward_synchronize_iteration);
    config.nvlink_forward_log_batches = get_bool(
        object, "nvlink_forward_log_batches", config.nvlink_forward_log_batches);
    config.log_qp_reports = get_bool(object, "log_qp_reports", config.log_qp_reports);
    config.log_marker_wait_reports = get_bool(
        object, "log_marker_wait_reports", config.log_marker_wait_reports);
    config.nvlink_forward_use_round_robin = get_bool(
        object, "nvlink_forward_use_round_robin", config.nvlink_forward_use_round_robin);
    config.nvlink_routing_probability = get_number(
        object, "nvlink_routing_probability", config.nvlink_routing_probability);
    config.nvlink_routing_seed = number_as<uint64_t>(
        object, "nvlink_routing_seed", config.nvlink_routing_seed);
    config.nvlink_forward_exchange_dir = get_string(
        object, "nvlink_forward_exchange_dir", config.nvlink_forward_exchange_dir);
    config.local_iteration_sync_enabled = get_bool(
        object, "local_iteration_sync_enabled", config.local_iteration_sync_enabled);
    config.local_iteration_sync_dir = get_string(
        object, "local_iteration_sync_dir", config.local_iteration_sync_dir);
    config.local_iteration_sync_run_id = get_string(
        object, "local_iteration_sync_run_id", config.local_iteration_sync_run_id);
    config.rdma_bandwidth_summary_dir = get_string(
        object, "rdma_bandwidth_summary_dir", config.rdma_bandwidth_summary_dir);
    config.cpu_affinity = get_string(object, "cpu_affinity", config.cpu_affinity);
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
    if (has(object, "nvlink_forward_destinations")) {
        for (const auto& entry : as_array(
                 get_required(object, "nvlink_forward_destinations"), "nvlink_forward_destinations")) {
            const auto& dst_obj = as_object(entry, "nvlink_forward_destination");
            NvlinkForwardDestination dst;
            dst.gpu_index = number_as<int>(dst_obj, "gpu_index", -1);
            dst.cuda_device_id = number_as<int>(dst_obj, "cuda_device_id", dst.gpu_index);
            dst.buffer_addr = get_u64_string_or_number(dst_obj, "buffer_addr", 0);
            dst.buffer_bytes = number_as<std::size_t>(dst_obj, "buffer_bytes", 0);
            config.nvlink_forward_destinations.push_back(dst);
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
        throw std::runtime_error(config_help());
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
    if (config.data_signal_interval < 0) throw std::runtime_error("data_signal_interval must be >= 0");
    if (config.max_in_flight_chunks_per_qp <= 0) {
        throw std::runtime_error("max_in_flight_chunks_per_qp must be > 0");
    }
    if (config.rdma_chunk_per_token_sge_enabled &&
        config.tokens_per_chunk > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("tokens_per_chunk exceeds verbs num_sge range for per-token SGE mode");
    }
    if (config.rdma_discontinuous_token_payload_enabled && !config.rdma_chunk_per_token_sge_enabled) {
        throw std::runtime_error(
            "rdma_discontinuous_token_payload_enabled requires rdma_chunk_per_token_sge_enabled=true");
    }
    if (config.rdma_discontinuous_token_payload_enabled && config.nvlink_forwarding_enabled) {
        throw std::runtime_error(
            "rdma_discontinuous_token_payload_enabled is only supported when NVLink forwarding is disabled");
    }
    if (config.send_queue_depth <= 0 || config.recv_queue_depth <= 0 || config.cq_depth <= 0) {
        throw std::runtime_error("queue and CQ depths must be > 0");
    }
    if (config.max_in_flight_chunks_per_qp > config.send_queue_depth) {
        throw std::runtime_error("max_in_flight_chunks_per_qp must be <= send_queue_depth");
    }
    if (config.completion_timeout_ms == 0) throw std::runtime_error("completion_timeout_ms must be > 0");
    if (config.num_nodes > 1 && static_cast<int>(config.peers.size()) != config.num_nodes - 1) {
        throw std::runtime_error("peers must contain exactly num_nodes - 1 entries");
    }
    if (config.local_iteration_sync_enabled && config.local_iteration_sync_dir.empty()) {
        throw std::runtime_error(
            "local_iteration_sync_dir must be non-empty when local iteration synchronization is enabled");
    }
    if (config.nvlink_forward_local_batch_sync_enabled) {
        if (!config.nvlink_forwarding_enabled) {
            throw std::runtime_error(
                "nvlink_forward_local_batch_sync_enabled requires nvlink_forwarding_enabled=true");
        }
        if (!config.nvlink_forward_synchronize_batches) {
            throw std::runtime_error(
                "nvlink_forward_local_batch_sync_enabled requires nvlink_forward_synchronize_batches=true");
        }
        if (config.local_iteration_sync_dir.empty()) {
            throw std::runtime_error(
                "local_iteration_sync_dir must be non-empty when NVLink local batch synchronization is enabled");
        }
    }
    if (config.nvlink_forward_completion_notifications_enabled) {
        if (!config.nvlink_forwarding_enabled) {
            throw std::runtime_error(
                "nvlink_forward_completion_notifications_enabled requires nvlink_forwarding_enabled=true");
        }
        if (!config.nvlink_forward_synchronize_batches) {
            throw std::runtime_error(
                "nvlink_forward_completion_notifications_enabled requires nvlink_forward_synchronize_batches=true");
        }
        if (config.nvlink_forward_notification_queue_depth == 0) {
            throw std::runtime_error(
                "nvlink_forward_notification_queue_depth must be > 0 when NVLink completion notifications are enabled");
        }
        if (config.nvlink_forward_notification_queue_depth >
            static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("nvlink_forward_notification_queue_depth exceeds uint32 range");
        }
    }
    if (config.nvlink_forward_notification_log_enabled) {
        if (!config.nvlink_forward_completion_notifications_enabled) {
            throw std::runtime_error(
                "nvlink_forward_notification_log_enabled requires "
                "nvlink_forward_completion_notifications_enabled=true");
        }
        if (config.nvlink_forward_notification_log_dir.empty()) {
            throw std::runtime_error(
                "nvlink_forward_notification_log_dir must be non-empty when notification logging is enabled");
        }
    }
    if (config.nvlink_forward_computation_load_only_enabled &&
        !config.nvlink_forward_computation_enabled) {
        throw std::runtime_error(
            "nvlink_forward_computation_load_only_enabled requires "
            "nvlink_forward_computation_enabled=true");
    }
    if (config.nvlink_forward_computation_enabled) {
        if (!config.nvlink_forward_completion_notifications_enabled) {
            throw std::runtime_error(
                "nvlink_forward_computation_enabled requires "
                "nvlink_forward_completion_notifications_enabled=true");
        }
        if (config.dtype != DataType::kBF16 && config.dtype != DataType::kFP16) {
            throw std::runtime_error(
                "NVLink forwarding computation supports only bf16 and fp16 inputs/outputs");
        }
        if (config.peers.empty()) {
            throw std::runtime_error(
                "nvlink_forward_computation_enabled requires at least one remote peer receive-buffer slot");
        }
        if (config.token_dimension == 0 || config.token_dimension % 16 != 0) {
            throw std::runtime_error(
                "token_dimension must be a non-zero multiple of 16 for NVLink forwarding computation");
        }
        if (config.nvlink_forward_computation_output_dim == 0 ||
            config.nvlink_forward_computation_output_dim % 16 != 0) {
            throw std::runtime_error(
                "nvlink_forward_computation_output_dim must be a non-zero multiple of 16");
        }
        if (config.nvlink_forward_computation_tile_m == 0 ||
            config.nvlink_forward_computation_tile_m % 16 != 0) {
            throw std::runtime_error(
                "nvlink_forward_computation_tile_m must be a non-zero multiple of 16");
        }
        if (config.nvlink_forward_computation_tile_n == 0 ||
            config.nvlink_forward_computation_tile_n % 16 != 0) {
            throw std::runtime_error(
                "nvlink_forward_computation_tile_n must be a non-zero multiple of 16");
        }
        const auto max_task_dimension = static_cast<std::size_t>(std::numeric_limits<uint32_t>::max());
        if (config.token_dimension > max_task_dimension ||
            config.nvlink_forward_computation_output_dim > max_task_dimension ||
            config.nvlink_forward_computation_tile_m > max_task_dimension ||
            config.nvlink_forward_computation_tile_n > max_task_dimension) {
            throw std::runtime_error("NVLink forwarding computation matrix/tile metadata exceeds uint32 range");
        }
        if (config.nvlink_forward_computation_num_queues == 0) {
            throw std::runtime_error("nvlink_forward_computation_num_queues must be > 0");
        }
        if (config.nvlink_forward_computation_queue_depth == 0) {
            throw std::runtime_error("nvlink_forward_computation_queue_depth must be > 0");
        }
        if (config.nvlink_forward_computation_num_queues >
                static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
            config.nvlink_forward_computation_queue_depth >
                static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("NVLink forwarding computation queue count/depth exceeds uint32 range");
        }
        const auto max_size = std::numeric_limits<std::size_t>::max();
        if (config.token_dimension > max_size / config.nvlink_forward_computation_output_dim ||
            config.token_dimension * config.nvlink_forward_computation_output_dim >
                max_size / dtype_size(config.dtype)) {
            throw std::runtime_error("NVLink forwarding computation weight allocation size overflows size_t");
        }
        if (config.num_tokens > max_size / config.peers.size()) {
            throw std::runtime_error("NVLink forwarding computation row capacity overflows size_t");
        }
        const auto rows = config.num_tokens * config.peers.size();
        if (rows > max_task_dimension) {
            throw std::runtime_error("NVLink forwarding computation row capacity exceeds uint32 range");
        }
        if (config.nvlink_forward_computation_output_dim > max_size / std::max<std::size_t>(rows, 1) ||
            rows * config.nvlink_forward_computation_output_dim > max_size / dtype_size(config.dtype)) {
            throw std::runtime_error("NVLink forwarding computation output allocation size overflows size_t");
        }
    }
    if (config.nvlink_forwarding_enabled) {
        const auto dynamic_threshold = nvlink_forward_dynamic_threshold_enabled(config);
        const auto forward_threshold_tokens = effective_nvlink_forward_threshold_tokens(config);
        if (config.num_gpus_per_node <= 1) {
            throw std::runtime_error("nvlink_forwarding_enabled requires num_gpus_per_node > 1");
        }
        if (config.num_gpus_per_node > 8) {
            throw std::runtime_error("nvlink_forwarding_enabled supports at most 8 local GPUs");
        }
        if (dynamic_threshold) {
            if (config.nvlink_forward_min_threshold_chunks == 0 ||
                config.nvlink_forward_max_threshold_chunks == 0) {
                throw std::runtime_error(
                    "dynamic NVLink forwarding requires both "
                    "nvlink_forward_min_threshold_chunks and nvlink_forward_max_threshold_chunks to be > 0");
            }
            if (config.nvlink_forward_min_threshold_chunks > config.nvlink_forward_max_threshold_chunks) {
                throw std::runtime_error(
                    "nvlink_forward_min_threshold_chunks must be <= nvlink_forward_max_threshold_chunks");
            }
            if (config.nvlink_forward_use_round_robin) {
                throw std::runtime_error(
                    "dynamic NVLink forwarding chunk thresholds do not support round-robin forwarding");
            }
            if (config.nvlink_forward_out_of_order_chunks_enabled && config.num_iterations == 0) {
                throw std::runtime_error(
                    "nvlink_forward_out_of_order_chunks_enabled requires finite num_iterations");
            }
        } else if (forward_threshold_tokens == 0) {
            throw std::runtime_error(
                "nvlink_forward_threshold_tokens or nvlink_forward_threshold_chunks must be > 0 when "
                "NVLink forwarding is enabled");
        }
        if (config.nvlink_forward_out_of_order_chunks_enabled && !dynamic_threshold) {
            throw std::runtime_error(
                "nvlink_forward_out_of_order_chunks_enabled requires dynamic chunk thresholds");
        }
        if (!dynamic_threshold &&
            config.nvlink_forward_threshold_tokens != 0 &&
            config.nvlink_forward_threshold_chunks != 0 &&
            config.nvlink_forward_threshold_tokens != forward_threshold_tokens) {
            throw std::runtime_error(
                "nvlink_forward_threshold_tokens must equal "
                "nvlink_forward_threshold_chunks * tokens_per_chunk when both are set");
        }
        if (config.nvlink_forward_chunk_tokens == 0) {
            throw std::runtime_error("nvlink_forward_chunk_tokens must be > 0 when NVLink forwarding is enabled");
        }
        if (!dynamic_threshold && config.nvlink_forward_use_round_robin) {
            const auto expected_threshold =
                config.nvlink_forward_chunk_tokens * static_cast<std::size_t>(config.num_gpus_per_node - 1);
            if (forward_threshold_tokens != expected_threshold) {
                throw std::runtime_error(
                    "round-robin NVLink forwarding requires the effective forwarding threshold to equal "
                    "nvlink_forward_chunk_tokens * (num_gpus_per_node - 1)");
            }
        }
        if (!dynamic_threshold && forward_threshold_tokens > config.num_tokens) {
            throw std::runtime_error(
                "effective NVLink forwarding threshold must be <= num_tokens when NVLink forwarding is enabled");
        }
        if (!dynamic_threshold && config.num_tokens % forward_threshold_tokens != 0) {
            throw std::runtime_error(
                "unsupported NVLink forwarding configuration: num_tokens must be an exact multiple of "
                "the effective NVLink forwarding threshold");
        }
        if (config.nvlink_routing_probability < 0.0 || config.nvlink_routing_probability > 1.0) {
            throw std::runtime_error("nvlink_routing_probability must be in [0, 1]");
        }
        if (config.nvlink_forward_exchange_dir.empty()) {
            throw std::runtime_error("nvlink_forward_exchange_dir must be non-empty when NVLink forwarding is enabled");
        }
        if (!config.nvlink_forward_destinations.empty() &&
            config.nvlink_forward_destinations.size() != static_cast<std::size_t>(config.num_gpus_per_node - 1)) {
            throw std::runtime_error(
                "manual nvlink_forward_destinations must contain exactly one entry for each non-source local GPU");
        }
        std::vector<bool> seen(static_cast<std::size_t>(config.num_gpus_per_node), false);
        const auto required_bytes =
            config.num_tokens * config.token_dimension * dtype_size(config.dtype) * config.peers.size();
        for (const auto& dst : config.nvlink_forward_destinations) {
            if (dst.gpu_index < 0 || dst.gpu_index >= config.num_gpus_per_node) {
                throw std::runtime_error("NVLink forwarding destination gpu_index out of range");
            }
            if (dst.gpu_index == config.local_gpu_index) {
                throw std::runtime_error("NVLink forwarding destinations must not include the source GPU");
            }
            if (seen[static_cast<std::size_t>(dst.gpu_index)]) {
                throw std::runtime_error("duplicate NVLink forwarding destination gpu_index");
            }
            seen[static_cast<std::size_t>(dst.gpu_index)] = true;
            if (dst.cuda_device_id < 0) {
                throw std::runtime_error("NVLink forwarding destination cuda_device_id must be >= 0");
            }
            if (dst.buffer_addr == 0) {
                throw std::runtime_error("NVLink forwarding destination buffer_addr must be non-zero");
            }
            if (dst.buffer_bytes < required_bytes) {
                throw std::runtime_error("NVLink forwarding destination buffer_bytes is smaller than forwarding buffer");
            }
        }
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
        << " data_signal_interval=" << config.data_signal_interval
        << " max_in_flight_chunks_per_qp=" << config.max_in_flight_chunks_per_qp
        << " rdma_chunk_per_token_sge_enabled="
        << (config.rdma_chunk_per_token_sge_enabled ? "true" : "false")
        << " rdma_discontinuous_token_payload_enabled="
        << (config.rdma_discontinuous_token_payload_enabled ? "true" : "false")
        << " iterations=" << config.num_iterations
        << " dtype=" << to_string(config.dtype)
        << " sequential_peer_transfers=" << (config.sequential_peer_transfers ? "true" : "false")
        << " nvlink_forwarding_enabled=" << (config.nvlink_forwarding_enabled ? "true" : "false")
        << " nvlink_forward_threshold_tokens=" << config.nvlink_forward_threshold_tokens
        << " nvlink_forward_threshold_chunks=" << config.nvlink_forward_threshold_chunks
        << " nvlink_forward_min_threshold_chunks=" << config.nvlink_forward_min_threshold_chunks
        << " nvlink_forward_max_threshold_chunks=" << config.nvlink_forward_max_threshold_chunks
        << " nvlink_forward_out_of_order_chunks_enabled="
        << (config.nvlink_forward_out_of_order_chunks_enabled ? "true" : "false")
        << " nvlink_forward_effective_threshold_tokens="
        << effective_nvlink_forward_threshold_tokens(config)
        << " nvlink_forward_chunk_tokens=" << config.nvlink_forward_chunk_tokens
        << " nvlink_forward_use_batch_api=" << (config.nvlink_forward_use_batch_api ? "true" : "false")
        << " nvlink_forward_synchronize_batches="
        << (config.nvlink_forward_synchronize_batches ? "true" : "false")
        << " nvlink_forward_completion_notifications_enabled="
        << (config.nvlink_forward_completion_notifications_enabled ? "true" : "false")
        << " nvlink_forward_notification_queue_depth="
        << config.nvlink_forward_notification_queue_depth
        << " nvlink_forward_notification_log_enabled="
        << (config.nvlink_forward_notification_log_enabled ? "true" : "false")
        << " nvlink_forward_notification_log_dir="
        << config.nvlink_forward_notification_log_dir
        << " nvlink_forward_computation_enabled="
        << (config.nvlink_forward_computation_enabled ? "true" : "false")
        << " nvlink_forward_computation_output_dim="
        << config.nvlink_forward_computation_output_dim
        << " nvlink_forward_computation_tile_m="
        << config.nvlink_forward_computation_tile_m
        << " nvlink_forward_computation_tile_n="
        << config.nvlink_forward_computation_tile_n
        << " nvlink_forward_computation_num_queues="
        << config.nvlink_forward_computation_num_queues
        << " nvlink_forward_computation_queue_depth="
        << config.nvlink_forward_computation_queue_depth
        << " nvlink_forward_computation_load_only_enabled="
        << (config.nvlink_forward_computation_load_only_enabled ? "true" : "false")
        << " nvlink_forward_computation_log_enabled="
        << (config.nvlink_forward_computation_log_enabled ? "true" : "false")
        << " nvlink_forward_local_batch_sync_enabled="
        << (config.nvlink_forward_local_batch_sync_enabled ? "true" : "false")
        << " nvlink_forward_synchronize_iteration="
        << (config.nvlink_forward_synchronize_iteration ? "true" : "false")
        << " nvlink_forward_log_batches=" << (config.nvlink_forward_log_batches ? "true" : "false")
        << " log_qp_reports=" << (config.log_qp_reports ? "true" : "false")
        << " log_marker_wait_reports=" << (config.log_marker_wait_reports ? "true" : "false")
        << " nvlink_forward_use_round_robin=" << (config.nvlink_forward_use_round_robin ? "true" : "false")
        << " nvlink_routing_probability=" << config.nvlink_routing_probability
        << " nvlink_routing_seed=" << config.nvlink_routing_seed
        << " nvlink_forward_exchange_dir=" << config.nvlink_forward_exchange_dir
        << " local_iteration_sync_enabled=" << (config.local_iteration_sync_enabled ? "true" : "false")
        << " local_iteration_sync_dir=" << config.local_iteration_sync_dir
        << " local_iteration_sync_run_id="
        << (config.local_iteration_sync_run_id.empty() ? "default" : config.local_iteration_sync_run_id)
        << " rdma_bandwidth_summary_dir=" << config.rdma_bandwidth_summary_dir
        << " cpu_affinity=" << (config.cpu_affinity.empty() ? "none" : config.cpu_affinity)
        << " mock_mode=" << (config.mock_mode ? "true" : "false");
    return out.str();
}

std::string config_help() {
    return
        "usage: rdma_cpu_proxy --config FILE [--key=value ...]\n"
        "\n"
        "Persistent NVLink forwarding computation options:\n"
        "  --nvlink_forward_computation_enabled=BOOL       consume ready notifications with a persistent GEMM\n"
        "  --nvlink_forward_computation_output_dim=N       output feature dimension (multiple of 16)\n"
        "  --nvlink_forward_computation_tile_m=M           output-tile row dimension (multiple of 16)\n"
        "  --nvlink_forward_computation_tile_n=N           output-tile column dimension (multiple of 16)\n"
        "  --nvlink_forward_computation_num_queues=Q       device-resident CPU-to-GPU queue count\n"
        "  --nvlink_forward_computation_queue_depth=D      entries per queue (default 1024)\n"
        "  --nvlink_forward_computation_load_only_enabled=BOOL stage A/B in shared memory without GEMM/output stores\n"
        "  --nvlink_forward_computation_log_enabled=BOOL   enable per-notification/per-queue diagnostics\n"
        "\n"
        "The computation mode also requires nvlink_forwarding_enabled,\n"
        "nvlink_forward_synchronize_batches, and\n"
        "nvlink_forward_completion_notifications_enabled. Input K is token_dimension;\n"
        "BF16 and FP16 are supported on Hopper (compute capability 9.0+).";
}

}  // namespace rdma_proxy
