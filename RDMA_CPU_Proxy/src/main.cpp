#include "config.hpp"
#include "logging.hpp"
#include "proxy.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        auto config = rdma_proxy::load_config(argc, argv);
        rdma_proxy::Logger::instance().set_level(rdma_proxy::log_level_from_string(config.log_level));

        rdma_proxy::Proxy proxy(config);
        proxy.initialize();
        proxy.run_once();
        proxy.shutdown();
        return 0;
    } catch (const std::exception& e) {
        RDMA_PROXY_LOG_ERROR(e.what());
        std::cerr << "rdma_cpu_proxy failed: " << e.what() << '\n';
        return 1;
    }
}
