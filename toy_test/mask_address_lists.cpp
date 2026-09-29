#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::size_t x = 0;
    unsigned int y = 0;
    double probability = 0.0;
    std::size_t worker_count = 0;
    std::uint64_t seed = 0;
};

[[noreturn]] void fail(const char* program, std::string_view message) {
    std::cerr << "Error: " << message << '\n'
              << "Usage: " << program << " <x> <y> <p> <z> [seed]\n"
              << "  x     Number of integers/masks (x > 0)\n"
              << "  y     Number of address lists and mask bits (8, 16, or 24)\n"
              << "  p     Probability that each mask bit is set (0.0 to 1.0)\n"
              << "  z     Number of worker threads (1 <= z <= y)\n"
              << "  seed  Optional random seed (default: random_device)\n";
    std::exit(EXIT_FAILURE);
}

template <typename T>
T parse_unsigned(const char* program, const char* text, std::string_view name) {
    T value{};
    const std::string_view input(text);
    const auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    if (result.ec != std::errc{} || result.ptr != input.data() + input.size()) {
        fail(program, std::string(name) + " must be a non-negative integer");
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        fail(argv[0], "expected four required arguments and one optional argument");
    }

    Options options;
    options.x = parse_unsigned<std::size_t>(argv[0], argv[1], "x");
    options.y = parse_unsigned<unsigned int>(argv[0], argv[2], "y");
    options.worker_count = parse_unsigned<std::size_t>(argv[0], argv[4], "z");

    char* end = nullptr;
    options.probability = std::strtod(argv[3], &end);
    if (end == argv[3] || *end != '\0' || !std::isfinite(options.probability) ||
        options.probability < 0.0 || options.probability > 1.0) {
        fail(argv[0], "p must be a finite number between 0.0 and 1.0");
    }

    if (options.x == 0) {
        fail(argv[0], "x must be greater than zero");
    }
    if (options.y != 8 && options.y != 16 && options.y != 24) {
        fail(argv[0], "y must be 8, 16, or 24");
    }
    if (options.worker_count == 0 || options.worker_count > options.y) {
        fail(argv[0], "z must be between 1 and y");
    }

    if (argc == 6) {
        options.seed = parse_unsigned<std::uint64_t>(argv[0], argv[5], "seed");
    } else {
        std::random_device random_device;
        options.seed = (static_cast<std::uint64_t>(random_device()) << 32U) ^
                       static_cast<std::uint64_t>(random_device());
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    // The integer list owns the objects whose addresses are placed in the output lists.
    std::vector<int> integers(options.x);
    for (std::size_t i = 0; i < options.x; ++i) {
        integers[i] = static_cast<int>(i %
            static_cast<std::size_t>(std::numeric_limits<int>::max()));
    }

    // uint32_t is sufficient for all supported mask widths (up to 24 bits).
    std::vector<std::uint32_t> masks(options.x, 0U);
    std::mt19937_64 generator(options.seed);
    std::bernoulli_distribution bit_is_set(options.probability);
    for (std::uint32_t& mask : masks) {
        for (unsigned int bit = 0; bit < options.y; ++bit) {
            if (bit_is_set(generator)) {
                mask |= (std::uint32_t{1} << bit);
            }
        }
    }

    std::vector<std::vector<const int*>> address_lists(options.y);
    const long double expected_per_list =
        static_cast<long double>(options.x) * options.probability;
    if (expected_per_list <= static_cast<long double>(
                                 std::numeric_limits<std::size_t>::max())) {
        const std::size_t reserve_size = static_cast<std::size_t>(expected_per_list);
        for (auto& address_list : address_lists) {
            address_list.reserve(reserve_size);
        }
    }

    // Divide the experts (mask bits/address lists) into disjoint ranges. Each address
    // list has exactly one writer, so appends require no locks.
    std::vector<std::thread> workers;
    workers.reserve(options.worker_count);
    std::mutex start_mutex;
    std::condition_variable start_cv;
    std::size_t ready_workers = 0;
    bool start_requested = false;
    std::vector<std::chrono::steady_clock::time_point> finish_times(
        options.worker_count);

    const std::size_t base_experts_per_worker = options.y / options.worker_count;
    const std::size_t extra_experts = options.y % options.worker_count;

    for (std::size_t worker_id = 0; worker_id < options.worker_count; ++worker_id) {
        const std::size_t expert_begin = worker_id * base_experts_per_worker +
            (worker_id < extra_experts ? worker_id : extra_experts);
        const std::size_t expert_count = base_experts_per_worker +
            (worker_id < extra_experts ? 1U : 0U);
        const std::size_t expert_end = expert_begin + expert_count;

        workers.emplace_back([&, worker_id, expert_begin, expert_end] {
            // Wait until all worker threads have been created and reached this gate.
            {
                std::unique_lock<std::mutex> lock(start_mutex);
                ++ready_workers;
                start_cv.notify_all();
                start_cv.wait(lock, [&] { return start_requested; });
            }

            for (std::size_t expert = expert_begin; expert < expert_end; ++expert) {
                const std::uint32_t expert_bit = std::uint32_t{1} << expert;
                auto& address_list = address_lists[expert];
                for (std::size_t i = 0; i < masks.size(); ++i) {
                    if ((masks[i] & expert_bit) != 0U) {
                        address_list.push_back(&integers[i]);
                    }
                }
            }
            finish_times[worker_id] = std::chrono::steady_clock::now();
        });
    }

    // Start timing only after every pre-created worker is ready. The timestamp from
    // the last worker to finish is captured inside that worker, before thread teardown.
    std::unique_lock<std::mutex> start_lock(start_mutex);
    start_cv.wait(start_lock, [&] { return ready_workers == options.worker_count; });
    const auto start = std::chrono::steady_clock::now();
    start_requested = true;
    start_lock.unlock();
    start_cv.notify_all();

    for (std::thread& worker : workers) {
        worker.join();
    }
    const auto stop = *std::max_element(finish_times.begin(), finish_times.end());

    std::size_t total_addresses = 0;
    std::uint64_t checksum = 0;
    for (const auto& address_list : address_lists) {
        total_addresses += address_list.size();
        for (const int* address : address_list) {
            checksum += static_cast<std::uint64_t>(*address);
        }
    }

    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    std::cout << "x: " << options.x << '\n'
              << "y: " << options.y << '\n'
              << "p: " << options.probability << '\n'
              << "worker threads: " << options.worker_count << '\n'
              << "seed: " << options.seed << '\n'
              << "addresses added: " << total_addresses << '\n'
              << "checksum: " << checksum << '\n'
              << std::fixed << std::setprecision(3)
              << "iteration time: " << elapsed.count() << " ms\n";

    return EXIT_SUCCESS;
}
