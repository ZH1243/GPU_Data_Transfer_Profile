#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

constexpr std::size_t kDefaultExperts = 16;
constexpr std::size_t kDefaultTokens = 1024;
constexpr std::size_t kTokenSizeBytes = 8 * 1024;
constexpr double kDefaultProbability = 0.5;
constexpr std::size_t kDefaultThreads = 4;
constexpr std::uint64_t kDefaultSeed = 12345;

using TokenAddress = std::byte*;

struct Options {
    std::size_t experts = kDefaultExperts;
    std::size_t tokens = kDefaultTokens;
    double probability = kDefaultProbability;
    std::size_t threads = kDefaultThreads;
    std::uint64_t seed = kDefaultSeed;
};

// Owns y (the array of expert-list addresses), every expert list, and the
// corresponding list lengths.  raw_y() is the address array requested by the
// benchmark; ownership remains in this object so all allocations are freed.
class ExpertTokenLists {
public:
    explicit ExpertTokenLists(std::vector<std::size_t> counts)
        : expert_count_(counts.size()), counts_(std::move(counts)),
          y_(std::make_unique<TokenAddress*[]>(expert_count_)) {
        for (std::size_t expert = 0; expert < expert_count_; ++expert) {
            y_[expert] = nullptr;
        }
        try {
            for (std::size_t expert = 0; expert < expert_count_; ++expert) {
                if (counts_[expert] != 0) {
                    y_[expert] = new TokenAddress[counts_[expert]];
                }
            }
        } catch (...) {
            for (std::size_t expert = 0; expert < expert_count_; ++expert) {
                delete[] y_[expert];
            }
            throw;
        }
    }

    ~ExpertTokenLists() {
        for (std::size_t expert = 0; expert < expert_count_; ++expert) {
            delete[] y_[expert];
        }
    }

    ExpertTokenLists(const ExpertTokenLists&) = delete;
    ExpertTokenLists& operator=(const ExpertTokenLists&) = delete;
    ExpertTokenLists(ExpertTokenLists&&) = delete;
    ExpertTokenLists& operator=(ExpertTokenLists&&) = delete;

    TokenAddress** raw_y() const { return y_.get(); }
    const std::vector<std::size_t>& counts() const { return counts_; }

private:
    std::size_t expert_count_;
    std::vector<std::size_t> counts_;
    std::unique_ptr<TokenAddress*[]> y_;
};

template <typename Mask>
std::vector<Mask> make_random_masks(std::size_t token_count,
                                    std::size_t expert_count,
                                    double probability,
                                    std::uint64_t seed) {
    static_assert(std::is_unsigned<Mask>::value, "Mask must be unsigned");

    std::mt19937_64 generator(seed);
    std::bernoulli_distribution is_routed(probability);
    std::vector<Mask> masks(token_count, Mask{0});

    for (Mask& mask : masks) {
        for (std::size_t expert = 0; expert < expert_count; ++expert) {
            if (is_routed(generator)) {
                mask |= static_cast<Mask>(Mask{1} << expert);
            }
        }
    }
    return masks;
}

// The caller (the main thread) executes the counting and allocation portions.
// Each launched worker owns a disjoint range of experts, so workers never write
// to the same expert token list and no locks or atomics are needed.
template <typename Mask>
std::unique_ptr<ExpertTokenLists> get_expert_token_address_list(
    std::byte* token_array,
    std::size_t token_count,
    std::size_t token_size_bytes,
    const std::vector<Mask>& expert_mask_array,
    std::size_t expert_count,
    std::size_t worker_count) {
    std::vector<std::size_t> counts(expert_count, 0);

    for (Mask mask : expert_mask_array) {
        for (std::size_t expert = 0; expert < expert_count; ++expert) {
            if ((mask & static_cast<Mask>(Mask{1} << expert)) != 0) {
                ++counts[expert];
            }
        }
    }

    auto result = std::make_unique<ExpertTokenLists>(std::move(counts));
    TokenAddress** const y = result->raw_y();
    const std::size_t experts_per_worker = expert_count / worker_count;

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    try {
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            const std::size_t first_expert = worker * experts_per_worker;
            const std::size_t last_expert = first_expert + experts_per_worker;

            workers.emplace_back([=, &expert_mask_array]() {
                std::array<std::size_t, 64> write_positions{};

                for (std::size_t token = 0; token < token_count; ++token) {
                    const Mask mask = expert_mask_array[token];
                    for (std::size_t expert = first_expert;
                         expert < last_expert; ++expert) {
                        if ((mask & static_cast<Mask>(Mask{1} << expert)) != 0) {
                            y[expert][write_positions[expert]++] =
                                token_array + token * token_size_bytes;
                        }
                    }
                }
            });
        }
    } catch (...) {
        for (std::thread& worker : workers) {
            worker.join();
        }
        throw;
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
    return result;
}

std::size_t parse_size(const char* text, const char* option) {
    std::size_t parsed_characters = 0;
    const std::string value(text);
    const unsigned long long parsed = std::stoull(value, &parsed_characters);
    if (parsed_characters != value.size() ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return static_cast<std::size_t>(parsed);
}

double parse_probability(const char* text) {
    std::size_t parsed_characters = 0;
    const std::string value(text);
    const double parsed = std::stod(value, &parsed_characters);
    if (parsed_characters != value.size() || parsed < 0.0 || parsed > 1.0) {
        throw std::invalid_argument("probability must be between 0 and 1");
    }
    return parsed;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--experts 8|16|32|64] [--tokens N] [--probability P]"
                 " [--threads N] [--seed N]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }

        const char* value = argv[++i];
        if (argument == "--experts") {
            options.experts = parse_size(value, "--experts");
        } else if (argument == "--tokens") {
            options.tokens = parse_size(value, "--tokens");
        } else if (argument == "--probability") {
            options.probability = parse_probability(value);
        } else if (argument == "--threads") {
            options.threads = parse_size(value, "--threads");
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint64_t>(parse_size(value, "--seed"));
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (options.experts != 8 && options.experts != 16 &&
        options.experts != 32 && options.experts != 64) {
        throw std::invalid_argument("--experts must be 8, 16, 32, or 64");
    }
    if (options.tokens == 0) {
        throw std::invalid_argument("--tokens must be greater than zero");
    }
    if (options.threads == 0 || options.experts % options.threads != 0) {
        throw std::invalid_argument(
            "--threads must be greater than zero and divide --experts exactly");
    }
    if (options.tokens >
        std::numeric_limits<std::size_t>::max() / kTokenSizeBytes) {
        throw std::invalid_argument("token array size overflows size_t");
    }
    return options;
}

template <typename Mask>
void run_benchmark(const Options& options) {
    const std::size_t token_array_bytes = options.tokens * kTokenSizeBytes;
    auto token_array = std::make_unique<std::byte[]>(token_array_bytes);
    const auto expert_mask_array = make_random_masks<Mask>(
        options.tokens, options.experts, options.probability, options.seed);

    // Touch every token so the token allocation is physically backed before
    // timing the address-list construction.
    for (std::size_t token = 0; token < options.tokens; ++token) {
        token_array[token * kTokenSizeBytes] = std::byte{0};
    }

    const auto start = std::chrono::steady_clock::now();
    auto lists = get_expert_token_address_list(
        token_array.get(), options.tokens, kTokenSizeBytes, expert_mask_array,
        options.experts, options.threads);
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::size_t total_addresses = 0;
    for (std::size_t count : lists->counts()) {
        total_addresses += count;
    }

    std::cout << "experts=" << options.experts
              << ", tokens=" << options.tokens
              << ", token_size_bytes=" << kTokenSizeBytes
              << ", probability=" << options.probability
              << ", threads=" << options.threads
              << ", mask_bytes=" << sizeof(Mask) << '\n'
              << "routed_token_addresses=" << total_addresses << '\n'
              << "get_expert_token_address_list_us=" << elapsed_us << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        switch (options.experts) {
            case 8:
                run_benchmark<std::uint8_t>(options);
                break;
            case 16:
                run_benchmark<std::uint16_t>(options);
                break;
            case 32:
                run_benchmark<std::uint32_t>(options);
                break;
            case 64:
                run_benchmark<std::uint64_t>(options);
                break;
            default:
                throw std::logic_error("unreachable expert count");
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
