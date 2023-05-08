#include "cuda_context.h"
#include <map>
#include <memory>
#include <utils.h>
#include <mutex>
#include "vinasrv.h"

namespace dock {

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
private:
    explicit VinaSrv(int device, int nrinst) : device_(device), nrinst_(nrinst) { }
public:
    static bool create(int device, int nrinst) {
        if (instance_) {
            std::cerr << "vina server already created" << std::endl;
            return false;
        }
        instance_ = std::shared_ptr<VinaSrv>(new VinaSrv(device, nrinst));
        if (!instance_->init()) {
            instance_.reset();
            return false;
        }
        return true;
    }
    static void destroy() {
        instance_.reset();
    }
    static bool submit(VinaCallback &callback) {
        auto p = get();
        if (p) {
            return p->run(callback);
        }
        return false;
    }

private:
    static VinaSrv *get() {
        return instance_.get();
    }
    bool run(VinaCallback &callback) {

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
         return true;

    }
    bool init() {
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
        return true;
    }
private:
    static std::shared_ptr<VinaSrv> instance_;
    std::map<int, std::shared_ptr<CudaContext>> insts_;
    std::map<int, std::uint64_t> running_;
    std::uint64_t seq_ = 0;
    std::mutex mt_;
    int device_;
    int nrinst_;
};

bool create_vina_server(int device, int nrinst) {
    return VinaSrv::create(device, nrinst);
}
void destroy_vina_server() {
    VinaSrv::destroy();
}
bool submit_vina_server(VinaCallback callback) {
    return VinaSrv::submit(callback);
}

std::shared_ptr<VinaSrv> VinaSrv::instance_;
};