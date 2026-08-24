#include "rdma_proxy_c_api.h"

#include "config.hpp"
#include "cpu_affinity.hpp"
#include "logging.hpp"
#include "proxy.hpp"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct RdmaProxyHandle {
    explicit RdmaProxyHandle(rdma_proxy::ProxyConfig config)
        : proxy(std::move(config)) {}

    rdma_proxy::Proxy proxy;
    bool initialized{false};
    bool shut_down{false};
};

namespace {

thread_local std::string last_error;

void clear_error() {
    last_error.clear();
}

int record_current_exception() noexcept {
    try {
        throw;
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception";
    }
    return -1;
}

std::vector<char*> make_mutable_argv(
    int argc,
    const char* const* argv,
    std::vector<std::string>& storage) {
    if (argc < 0) {
        throw std::runtime_error("argc must be non-negative");
    }
    if (argc != 0 && argv == nullptr) {
        throw std::runtime_error("argv must be non-null when argc is nonzero");
    }

    storage.clear();
    storage.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        if (argv[i] == nullptr) {
            throw std::runtime_error("argv contains a null entry");
        }
        storage.emplace_back(argv[i]);
    }

    std::vector<char*> mutable_argv;
    mutable_argv.reserve(storage.size());
    for (auto& argument : storage) {
        mutable_argv.push_back(argument.data());
    }
    return mutable_argv;
}

}  // namespace

extern "C" int rdma_proxy_abi_version(void) {
    return 1;
}

extern "C" RdmaProxyHandle* rdma_proxy_create(
    int argc,
    const char* const* argv) {
    clear_error();
    try {
        std::vector<std::string> argument_storage;
        auto mutable_argv = make_mutable_argv(argc, argv, argument_storage);
        auto config = rdma_proxy::load_config(argc, mutable_argv.data());
        rdma_proxy::Logger::instance().set_level(
            rdma_proxy::log_level_from_string(config.log_level));
        rdma_proxy::apply_cpu_affinity(config);
        return new RdmaProxyHandle(std::move(config));
    } catch (...) {
        record_current_exception();
        return nullptr;
    }
}

extern "C" int rdma_proxy_initialize(RdmaProxyHandle* handle) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_initialize received a null handle";
        return -1;
    }
    try {
        handle->proxy.initialize();
        handle->initialized = true;
        handle->shut_down = false;
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_run(RdmaProxyHandle* handle) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_run received a null handle";
        return -1;
    }
    if (!handle->initialized || handle->shut_down) {
        last_error = "rdma_proxy_run requires an initialized proxy";
        return -1;
    }
    try {
        handle->proxy.run();
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" int rdma_proxy_shutdown(RdmaProxyHandle* handle) {
    clear_error();
    if (handle == nullptr) {
        last_error = "rdma_proxy_shutdown received a null handle";
        return -1;
    }
    try {
        handle->proxy.shutdown();
        handle->shut_down = true;
        return 0;
    } catch (...) {
        return record_current_exception();
    }
}

extern "C" void rdma_proxy_destroy(RdmaProxyHandle* handle) {
    delete handle;
}

extern "C" int rdma_proxy_run_argv(
    int argc,
    const char* const* argv) {
    clear_error();
    RdmaProxyHandle* handle = rdma_proxy_create(argc, argv);
    if (handle == nullptr) return -1;

    int status = rdma_proxy_initialize(handle);
    if (status == 0) status = rdma_proxy_run(handle);

    const std::string operation_error = last_error;
    const int shutdown_status = rdma_proxy_shutdown(handle);
    const std::string shutdown_error = last_error;
    rdma_proxy_destroy(handle);

    if (status != 0) {
        last_error = operation_error;
        return status;
    }
    if (shutdown_status != 0) {
        last_error = shutdown_error;
        return shutdown_status;
    }
    clear_error();
    return 0;
}

extern "C" const char* rdma_proxy_last_error(void) {
    return last_error.c_str();
}

