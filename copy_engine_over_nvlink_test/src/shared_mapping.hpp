#pragma once
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Publish an initialized MAP_SHARED file by rename. Other processes cannot map
// partially initialized storage. The launcher owns the containing directory.
template<class T> class SharedMapping {
    T* ptr = nullptr;
public:
    SharedMapping(const std::string& path, bool creator, unsigned timeout) {
        int fd = -1;
        const std::string initial = path + ".initial";
        if (creator) {
            struct stat existing{};
            if (lstat(path.c_str(), &existing) == 0)
                throw std::runtime_error("shared file already exists; use a fresh run directory");
            fd = open(initial.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
            if (fd < 0) throw std::runtime_error("cannot create shared file: " + std::string(std::strerror(errno)));
            if (ftruncate(fd, sizeof(T)) != 0) {
                close(fd); unlink(initial.c_str());
                throw std::runtime_error("cannot size shared file");
            }
        } else {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout);
            while ((fd = open(path.c_str(), O_RDWR)) < 0) {
                if (errno != ENOENT) throw std::runtime_error("cannot open shared file");
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("timeout waiting for rank 0 shared-memory setup");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size != static_cast<off_t>(sizeof(T))) {
            close(fd); throw std::runtime_error("shared file size mismatch");
        }
        void* memory = mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (memory == MAP_FAILED) throw std::runtime_error("shared mmap failed");
        ptr = static_cast<T*>(memory);
        if (creator) {
            new(memory) T{};
            if (rename(initial.c_str(), path.c_str()) != 0) {
                munmap(ptr, sizeof(T)); ptr = nullptr;
                throw std::runtime_error("cannot publish shared file");
            }
        }
    }
    SharedMapping(const SharedMapping&) = delete;
    SharedMapping& operator=(const SharedMapping&) = delete;
    ~SharedMapping() { if (ptr) munmap(ptr, sizeof(T)); }
    T& get() { return *ptr; }
};
