#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H
#include <assert.h>
#include <list>

#include <cuda_runtime_api.h>

#include "context.h"
#include "dock.h"
#include "utils.h"

namespace dock {
class CudaContext : public Context {
public:
    explicit CudaContext(int device) : device_id_(device) {
        q_ = std::make_shared<BlockQueue<RequestSP>>();
        cudaDeviceProp prop;
        auto err = cudaGetDeviceProperties(&prop, device);
        if (err == cudaSuccess) {
            sm_max_size_ = prop.sharedMemPerBlock;
        }
    }
    ~CudaContext()  override {
        if (t_) {
            commit(nullptr);
            t_->join();
            t_.reset();
        }
    }
    bool create() {
        if (!t_) {
            t_ = std::make_shared<std::thread>([=]() { this->run(); });
            e_.wait();
            e_.reset();
            if (running_) {
                // std::cerr << "running now " << std::endl;
                return true;
            }
            // std::cerr << "NOT RUNNING" << std::endl;
            t_->join();
            t_.reset();
            return false;
        }
        return true;
    }
    void commit(RequestSP req) override {
        q_->push(req);
        if (req) {
            req->wait();
        }
    }

    void *hostMemory(int &sz) {
        sz = host_size_;
        return host_;
    }
    cudaStream_t stream() {
        return stream_;
    }
    int smsize() {
        return sm_max_size_;
    }
    cudaError_t sync() {
        auto err = cudaStreamSynchronize(stream_);
        while (cuda_memories_.size() > 1) {
            cuda_memories_.pop_front();
        }
        while (pin_memories_.size() > 1) {
            pin_memories_.pop_front();
        }
        while (host_memories_.size() > 1) {
            host_memories_.pop_front();
        }
        return err;
    }
    void *requireHostMemory(int size) {
        if (size == 0) {
            return nullptr;
        }

        // auto p = requirePinMemory(size);
        // if (!p) {
            auto p = requireCpuMemory(size);
            assert(p);
        // }
        return p;
    }
    void *requireDeviceMemory(int size) {
        if (device_size_ >= size) {
            assert(!cuda_memories_.empty());
            return cuda_memories_.back().get();
        }
        void *p = nullptr;
        size    = (size + 4095) / 4096 * 4096;  // 4K aligned
        auto err = cudaMalloc(&p, size);
        if (err != cudaSuccess) {
            std::cerr << "malloc cuda memory fail, size " << size << " err " << err << " msg "
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        auto sp = std::shared_ptr<void>(p, [](void *p1) {
            if (p1) {
                cudaFree(p1);
            }
        });
        cuda_memories_.push_back(sp);
        device_size_ = size;
        return p;
    }

private:
    void *requireCpuMemory(int size) {
        if (host_size_ >= size) {
            assert(!host_memories_.empty());
            return host_memories_.back().get();
        }
        size    = (size + 4095) / 4096 * 4096;  // 4K aligned
        void *p = malloc(size);
        auto sp = std::shared_ptr<void>(p, [](void *p1) {
            if (p1) {
                free(p1);
            }
        });
        host_memories_.push_back(sp);
        host_size_ = size;
        return p;

    }
    void *requirePinMemory(int size) {
        if (pin_size_ >= size) {
            assert(!pin_memories_.empty());
            return pin_memories_.back().get();
        }
        void *p = nullptr;
        size    = (size + 4095) / 4096 * 4096;  // 4K aligned
        auto err = cudaHostAlloc(&p, size, 0);
        if (err != cudaSuccess) {
            std::cerr << "malloc cuda memory fail, size " << size << " err " << err << " msg "
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        // std::cerr << "Alloc pin buffer " << size << " ptr " << p << std::endl;
        auto sp = std::shared_ptr<void>(p, [](void *p1) {
            if (p1) {
                auto e = cudaFreeHost(p1);
                if (e != cudaSuccess) {
                    std::cerr << "Cuda free fail " << e << " " << cudaGetErrorString(e) << std::endl;
                }
            }
        });
        pin_memories_.push_back(sp);
        pin_size_ = size;
        return p;

    }
    void run() {
        std::vector<std::function<void()>> cleanups;
        cleanups.push_back([this]() { this->e_.set(); });

        auto err = cudaSetDevice(device_id_);
        if (err != cudaSuccess) {
            std::cerr << "set cuda device " << device_id_ << " fail, err: " << err << ", "
                      << cudaGetErrorString(err) << std::endl;
            return;
        }
        err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) {
            std::cerr << "create stream at device " << device_id_ << " fail, err: " << err << ", "
                      << cudaGetErrorString(err) << std::endl;
            return;
        }
        cleanups.push_back([this]() { cudaStreamDestroy(this->stream_); });

        running_ = true;
        e_.set();
        while (true) {
            auto req = q_->pop();
            if (!req) {
                break;
            }
            req->run(this);
            req->signal();
        }
        cuda_memories_.clear();
        pin_memories_.clear();
        host_memories_.clear();
        running_ = false;
    }

public:
    int device_id_ = -1;
    std::shared_ptr<BlockQueue<RequestSP>> q_;
    std::shared_ptr<std::thread> t_;
    Event e_;
    bool running_        = false;
    void *host_          = nullptr;
    int host_size_       = 0;
    int pin_size_       = 0;
    int device_size_     = 0;
    int sm_max_size_     = 0;  // max sm size on this device
    cudaStream_t stream_ = nullptr;
    std::list<std::shared_ptr<void>> cuda_memories_;
    std::list<std::shared_ptr<void>> host_memories_;
    std::list<std::shared_ptr<void>> pin_memories_;
};
};
#endif