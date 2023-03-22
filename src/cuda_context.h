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
                return true;
            }
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
            auto p = cuda_memories_.front();
            cuda_memories_.pop_front();
            cudaFree(p);
        }
        return err;
    }
    void *requireDeviceMemory(int size) {
        cuda_memories_.clear();
        if (device_size_ >= size) {
            assert(!cuda_memories_.empty());
            return cuda_memories_.back();
        }
        void *p = nullptr;
        size    = (size + 4095) / 4096 * 4096;  // 4K aligned
        auto err = cudaMalloc(&p, size);
        if (err != cudaSuccess) {
            std::cerr << "malloc cuda memory fail, size " << size << " err " << err << " msg "
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        cuda_memories_.push_back(p);
        device_size_ = size;
        return p;
    }

private:
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

        err = cudaHostAlloc(&host_, host_size_, 0);
        if (err != cudaSuccess) {
            std::cerr << "create host memory at device " << device_id_ << " fail, err: " << err << ", "
                      << cudaGetErrorString(err) << std::endl;
            return;
        }
        cleanups.push_back([this]() { cudaFree(this->host_); });

        e_.set();
        running_ = true;
        while (true) {
            auto req = q_->pop();
            if (!req) {
                break;
            }
            req->run(this);
            req->signal();
        }
        for (auto p : cuda_memories_) {
            cudaFree(p);
        }
        running_ = false;
    }

public:
    int device_id_ = -1;
    std::shared_ptr<BlockQueue<RequestSP>> q_;
    std::shared_ptr<std::thread> t_;
    Event e_;
    bool running_        = false;
    void *host_          = nullptr;
    int host_size_ = 512*1024;
    int device_size_     = 0;
    int sm_max_size_     = 0;  // max sm size on this device
    cudaStream_t stream_ = nullptr;
    std::list<void *> cuda_memories_;
};
};
#endif