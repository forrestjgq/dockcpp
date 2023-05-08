#include "cuda_context.h"
#include <map>
#include <memory>
#include <mutex>
#include "vinasrv.h"
#if 1
namespace dock {

#pragma GCC push_options
#pragma GCC optimize("O0")
class StreamRequest : public Request {
public:
    StreamRequest(VinaCallback &callback) : callback_(std::move(callback)) {
    }
    ~StreamRequest() override = default;
    void run(Context *ctx) override {
        auto cudaCtx = (CudaContext *)ctx;
        callback_(cudaCtx->stream());
    }

private:
    std::function<void(cudaStream_t )> callback_;
};
class VinaSrv {
public:
    VinaSrv() = default;
    ~VinaSrv() = default;

    bool run(VinaCallback &callback) {
#if 0
        int id = 0;
        std::uint64_t cnt = 9999999999;
        {
            std::unique_lock<std::mutex> lock(mt_);
            for (auto it = running_.begin(); it != running_.end(); ++it) {
                if (it->second < cnt) {
                    cnt = it->second;
                    id = it->first;
                }
            }
        }
        if (id == 0) {
            return false;
        }
         auto ctx = insts_[id];
         auto req = std::make_shared<StreamRequest>(callback);
         {
            std::unique_lock<std::mutex> lock(mt_);
            running_[id]++;
         }
         ctx->commit(req);
         {
            std::unique_lock<std::mutex> lock(mt_);
            running_[id]--;
         }
#endif
         return true;

    }
    bool init(int device, int nrinst) {
        device_ = device;
        nrinst_ = nrinst;
#if 0
        for(auto i = 0; i < nrinst_; i++) {
            auto ctx = std::make_shared<CudaContext>(device_);
            if (ctx->create()) {
                seq_++;
                insts_[seq_] = ctx;
                running_[seq_] = 0;
            } else {
                return false;
            }
        }
#endif
        return true;
    }
private:
#if 0
    std::map<int, std::shared_ptr<CudaContext>> insts_;
    std::map<int, std::uint64_t> running_;
    std::uint64_t seq_ = 0;
    std::mutex mt_;
#endif
    int device_;
    int nrinst_;
};

static std::shared_ptr<VinaSrv> instance_;
bool create_vina_server(int device, int nrinst) {
    // if (instance_) {
    //     std::cerr << "vina server already created" << std::endl;
    //     return false;
    // }
    instance_ = std::make_shared<VinaSrv>();
    if (!instance_->init(device, nrinst)) {
        instance_.reset();
        return false;
    }
    return true;
}
void destroy_vina_server() {
    // instance_.reset();
}
#pragma GCC pop_options
bool submit_vina_server(VinaCallback callback) {
    if (instance_) {
        return instance_->run(callback);
    }
    return false;
}
// void vina_test() {
//     if(create_vina_server(1, 2)) {
//         destroy_vina_server();
//     }
// }
};
#endif