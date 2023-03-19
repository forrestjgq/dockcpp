#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H
#include <cuda_runtime_api.h>

#include "context.h"
#include "utils.h"
#include "dock.h"

namespace dock {
class CudaContext : public Context {
public:
    CudaContext(int device) : device_id_(device) {
        q_ = std::make_shared<BlockQueue<RequestSP>>();
    }
    ~CudaContext()  override {
        if (t_) {
            commit(nullptr);
            t_->join();
            t_.reset();
        }
    }
    bool create() {
        if(!t_) {
            t_ = std::make_shared<std::thread>([this]() { this->run(); });
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
    void *deviceMemory(int &sz) {
        sz = device_size_;
        return device_;
    }
    cudaStream_t stream() {
        return stream_;
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

        err = cudaMalloc(&device_, device_size_);
        if (err != cudaSuccess) {
            std::cerr << "create host memory at device " << device_id_ << " fail, err: " << err << ", "
                      << cudaGetErrorString(err) << std::endl;
            return;
        }
        cleanups.push_back([this]() { cudaFree(this->device_); });

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
        running_ = false;
    }
private:
    int device_id_          = -1;
    std::shared_ptr<BlockQueue<RequestSP>> q_;
    std::shared_ptr<std::thread> t_;
    Event e_;
    bool running_ = false;
    void *host_   = nullptr;
    void *device_ = nullptr;
    int host_size_ = 512*1024;
    int device_size_ = 2*1024*1024;
    cudaStream_t stream_ = nullptr;
};
std::shared_ptr<Context> createCudaContext(int device) {
    auto ctx = std::make_shared<CudaContext>(device);
    if (ctx->create()) {
        return ctx;
    }
    return nullptr;
}
};
#endif