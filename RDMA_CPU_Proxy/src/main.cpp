#include "config.hpp"
#include "cpu_affinity.hpp"
#include "logging.hpp"
#include "proxy.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            if (argument == "--help" || argument == "-h") {
                std::cout << rdma_proxy::config_help() << '\n';
                return 0;
            }
        }
        auto config = rdma_proxy::load_config(argc, argv);
        rdma_proxy::Logger::instance().set_level(rdma_proxy::log_level_from_string(config.log_level));
        rdma_proxy::apply_cpu_affinity(config);

        rdma_proxy::Proxy proxy(config);
        proxy.initialize();
        proxy.run();
        proxy.shutdown();
        return 0;
    } catch (const std::exception& e) {
        RDMA_PROXY_LOG_ERROR(e.what());
        std::cerr << "rdma_cpu_proxy failed: " << e.what() << '\n';
        return 1;
    }
}
