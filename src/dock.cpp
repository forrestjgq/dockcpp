#include "dock.h"
#include <assert.h>
#include <chrono>
#include <iostream>
#include <string.h>

#include "cuda_context.h"
#include "cuda_runtime_api.h"

namespace dock {

#if 0
static void dumparr(int m, int n, uint8_t *p) {
    printf(" ====== host(%dx%d) ======\n", m, n);
    for (int i = 0; i < m; i++) {
        printf("%d: ", i);
        uint8_t *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%d ", int(p1[j]));
        }
        printf("\n");
    }
}
static void dumparr(int m, int n, float *p) {
    printf(" ====== host(%dx%d) ======\n", m, n);
    for (int i = 0; i < m; i++) {
        printf("%d: ", i);
        float *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%f ", p1[j]);
        }
        printf("\n");
    }
}
#endif
extern void dock_gpu(float *init_coord,       // npred *3
                     float *pocket,           // npocket * 3
                     float *pred_cross_dist,  // npred x npocket
                     float *pred_holo_dist,   // npred x npred
                     float *values,           // nval
                     int *torsions,           // ntorsion x 2
                     uint8_t *masks,          // ntorsion x npred
                     int npred, int npocket, int nval, int ntorsion,
                     float *loss,   // output, scalar
                     float *dev,    // input device mem
                     int &devSize,  // dev size in byte
                     cudaStream_t stream, int smsize);
void dock_grad_gpu(float *init_coord, float *pocket, float *pred_cross_dist, float *pred_holo_dist,
                   float *values, int *torsions, uint8_t *masks, int npred, int npocket, int nval,
                   int ngval, int ntorsion,
                   float *loss,  // ngval float array
                   float *dev, int &devSize, cudaStream_t stream, int smsize);
static std::uint64_t now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

int dock_cpu_async(float *init_coord, float *pocket, float *pred_cross_dist, float *pred_holo_dist,
                   float *values, int *torsions, uint8_t *masks, int npred, int npocket, int nval,
                   int ntorsion, void *dev, int devsize, void *host, int &hostsize,
                   cudaStream_t stream, int smsize, float *outloss) {
    bool eval        = dev == nullptr;
    uint8_t *tmp     = (uint8_t *)dev;
    uint8_t *hosttmp = (uint8_t *)host;
    size_t tsz, asz;
    int hsize = 0;

#define COPY(val, tp, blksz) \
 tp *gpu_##val  = (tp *)tmp; \
 tp *host_##val = (tp *)hosttmp; \
 tsz            = sizeof(tp) * blksz; \
 asz            = (tsz + 3) / 4 * 4; \
 hosttmp += asz; \
 tmp += asz; \
 hsize += asz; \
 if (!eval) \
  memcpy(host_##val, val, tsz);

    COPY(init_coord, float, npred * 3);
    COPY(pocket, float, npocket * 3);
    COPY(pred_cross_dist, float, npred *npocket);
    COPY(pred_holo_dist, float, npred *npred);
    COPY(values, float, nval);
    COPY(torsions, int, ntorsion * 2);
    COPY(masks, uint8_t, ntorsion * npred);
#undef COPY

    if (!eval) {
        assert(hsize <= hostsize);
        cudaMemcpyAsync(dev, host, hsize, cudaMemcpyHostToDevice, stream);
    } else {
        hostsize = hsize;
    }

    float *loss;
    loss        = (float *)tmp, tmp += sizeof(float);
    int tmpsz   = 0;
    int cudahdr = int(tmp - (uint8_t *)dev);
    if (!eval) {
        tmpsz = devsize - cudahdr;
    }

    // must be last one to copy for alignment
    dock_gpu(gpu_init_coord, gpu_pocket, gpu_pred_cross_dist, gpu_pred_holo_dist, gpu_values,
             gpu_torsions, gpu_masks, npred, npocket, nval, ntorsion, loss,
             eval ? nullptr : (float *)tmp, tmpsz, stream, smsize);

    if (!eval) {
        cudaMemcpyAsync(outloss, loss, sizeof(float), cudaMemcpyDeviceToHost, stream);
        return 0;
    }
    return tmpsz + cudahdr;
}
template<typename T>
int copy_data(T *&val, uint8_t *&tmp, uint8_t *&hosttmp, int blksz, bool eval) {
    int tsz = sizeof(T) * blksz; // actual size
    int asize = (tsz + 3)/4*4; // offset size, for 4 bytes alignment

    if (!eval) {
        memcpy(hosttmp, val, tsz);
        val = (T *)tmp;
        tmp += asize;
        hosttmp += asize;
    }

    return asize;
}
// if device == nullptr, enter eval mode, this will evaluate:
//     - required cuda memory in total, output to cudasz
//     - how much cuda memory should be reserved for request header, output to hdrsz
//     - required pin memory, include header/values/loss, and output to hostsz
// else, start copy mode:
//     - copy input header data to host
//     - async copy pin header to cuda header
//     - set input pointers to cuda pointers
void dock_grad_session(float *&init_coord,       // npred * 3 floats
                       float *&pocket,           // npocket * 3 floats
                       float *&pred_cross_dist,  // npred * npocket floats
                       float *&pred_holo_dist,   // npred * npred floats
                       int *&torsions,           // ntorsion * 2 ints
                       uint8_t *&masks,          // npred * ntorsion masks
                       int npred, int npocket, int nval, int ntorsion,
                       void *host,  // cuda host memory
                       int &hostsz, void *device, int &cudasz, int &hdrsz, cudaStream_t stream) {
    bool eval        = device == nullptr;
    uint8_t *devtmp  = (uint8_t *)device;
    uint8_t *hosttmp = (uint8_t *)host;
    int hsize = 0; // hdr size 
#define COPY(val, tp, blksz) hsize += copy_data<tp>(val, devtmp, hosttmp, blksz, eval)
    // copy hdr to pin memory
    COPY(init_coord, float, npred * 3);
    COPY(pocket, float, npocket * 3);
    COPY(pred_cross_dist, float, npred *npocket);
    COPY(pred_holo_dist, float, npred *npred);
    COPY(torsions, int, ntorsion * 2);
    COPY(masks, uint8_t, ntorsion * npred);
#undef COPY
    // now devtmp is the first place for future writing, and set hdr size

    // try to copy hdr data if not in eval mode
    if (!eval) {
        cudaMemcpyAsync(device, host, hsize, cudaMemcpyHostToDevice, stream);
        return;
    }

    hdrsz      = hsize;
    int smsize = 0; // eval how much tmp memory required for dock_grad_gpu
    dock_grad_gpu(nullptr, nullptr, nullptr, nullptr,
                  nullptr, nullptr, nullptr, npred, npocket, nval, nval + 1, ntorsion,
                  nullptr,  nullptr, smsize, nullptr, 0);
    int valuesz = nval * (nval + 1) * sizeof(float); // in bytes, for values
    int losssz = (nval + 1) * sizeof(float); // for output loss
    hostsz = hdrsz + valuesz + losssz;
    cudasz = smsize + hostsz;
    return;
}
// header already be copied, now copy values to cuda memory and start kernel, then copy losses out
void dock_grad_exec(float *init_coord,       // npred * 3 floats
                        float *pocket,           // npocket * 3 floats
                        float *pred_cross_dist,  // npred * npocket floats
                        float *pred_holo_dist,   // npred * npred floats
                        float *values,           // nval float, as x in f(x), host memory
                        int *torsions,           // ntorsion * 2 ints
                        uint8_t *masks,          // npred * ntorsion masks
                        int npred, int npocket, int nval, int ntorsion, float eps,
                        float *&outloss,  // output losses on pin memory
                        void *host,     // cuda host memory
                        void *device, int cudasz, cudaStream_t stream, int smsize) {
    // std::cout << "cudasz in exec " << cudasz << std::endl;
    cudaMemsetAsync(device, 0, cudasz, stream);
    uint8_t *tmp     = (uint8_t *)device;
    uint8_t *hosttmp = (uint8_t *)host;

    // copy values
    int valuelen = nval * sizeof(float);
    float *valueBatch = (float *)hosttmp;
    memcpy(valueBatch, values, valuelen);
    valueBatch += nval;
    for (int i = 0; i < nval; i++, valueBatch += nval) {
        memcpy(valueBatch, values, valuelen);
        valueBatch[i] += eps;
    }
    int cpsize            = (nval + 1) * valuelen;
    cudaMemcpyAsync(tmp, hosttmp, cpsize, cudaMemcpyHostToDevice, stream);
    valueBatch = (float *)tmp; // set to device memory
    hosttmp += cpsize, tmp += cpsize;

    float *loss; // output
    int losssz = (nval + 1) * sizeof(float);
    loss = (float *)tmp, tmp += losssz;

    // how much left for kernel tmp cuda mem
    int devsize = cudasz - cpsize - losssz;

    // must be last one to copy for alignment
    dock_grad_gpu(init_coord, pocket, pred_cross_dist, pred_holo_dist,
                  valueBatch, torsions, masks, npred, npocket, nval, nval + 1, ntorsion,
                  loss, (float *)tmp, devsize, stream, smsize);
    
    outloss = (float *)hosttmp;
    cudaMemcpyAsync(outloss, loss, (nval + 1) * sizeof(float), cudaMemcpyDeviceToHost);
}
// return 1 for lacking of host memory
int dock_grad_cpu_async(float *init_coord,       // npred * 3 floats
                        float *pocket,           // npocket * 3 floats
                        float *pred_cross_dist,  // npred * npocket floats
                        float *pred_holo_dist,   // npred * npred floats
                        float *values,           // nval float, as x in f(x)
                        int *torsions,           // ntorsion * 2 ints
                        uint8_t *masks,          // npred * ntorsion masks
                        int npred, int npocket, int nval, int ntorsion, float eps,
                        float *losses,  // should be nval+1 floats, output
                        void *host,     // cuda host memory
                        int &hostsz, void *device, int cudasz, cudaStream_t stream, int smsize) {
    bool eval        = device == nullptr;
    uint8_t *tmp     = (uint8_t *)device;
    uint8_t *hosttmp = (uint8_t *)host;
    size_t tsz;
    int hsize = 0, asize = 0;

#define COPY(val, tp, blksz) \
 tp *gpu_##val  = (tp *)tmp; \
 tp *host_##val = (tp *)hosttmp; \
 tsz            = sizeof(tp) * blksz; \
 asize          = (tsz + 3) / 4 * 4; \
 hosttmp += asize; \
 tmp += asize; \
 hsize += asize; \
 if (!eval) \
  memcpy(host_##val, val, tsz);

    COPY(init_coord, float, npred * 3);
    COPY(pocket, float, npocket * 3);
    COPY(pred_cross_dist, float, npred *npocket);
    COPY(pred_holo_dist, float, npred *npred);

    float *valueBatch = (float *)hosttmp;
    if (!eval) {
        memcpy(valueBatch, values, nval * sizeof(float));
        for (int i = 0; i < nval; i++) {
            float *start = valueBatch + (i + 1) * nval;
            memcpy(start, values, nval * sizeof(float));
            start[i] += eps;
        }
    }
    int cpsize            = (nval + 1) * nval * sizeof(float);
    float *gpu_valueBatch = (float *)tmp;
    hosttmp += cpsize;
    tmp += cpsize;
    hsize += cpsize;

    COPY(torsions, int, ntorsion * 2);
    COPY(masks, uint8_t, ntorsion * npred);
#undef COPY

    if (!eval) {
        cudaMemcpyAsync(device, host, hsize, cudaMemcpyHostToDevice, stream);
    }

    float *loss;
    loss = (float *)tmp, tmp += (nval + 1) * sizeof(float);
    // how much left for kernel tmp cuda mem
    int devsize = 0;
    if (!eval) {
        devsize = cudasz - hsize - (nval + 1) * sizeof(float);
    }

    // must be last one to copy for alignment
    dock_grad_gpu(gpu_init_coord, gpu_pocket, gpu_pred_cross_dist, gpu_pred_holo_dist,
                  gpu_valueBatch, gpu_torsions, gpu_masks, npred, npocket, nval, nval + 1, ntorsion,
                  loss, eval ? nullptr : (float *)tmp, devsize, stream, smsize);
    if (!eval) {
        cudaMemcpyAsync(losses, loss, (nval + 1) * sizeof(float), cudaMemcpyDeviceToHost);
        return 0;
    }
    int valuesz = nval * (nval + 1) * sizeof(float); // in bytes, for values
    int losssz = (nval + 1) * sizeof(float); // for output loss
    hostsz = hsize + valuesz + losssz;
    return devsize + hostsz;
}

class CudaDockRequest : public Request {
public:
    explicit CudaDockRequest(float *init_coord,       // npred * 3 floats
                             float *pocket,           // npocket * 3 floats
                             float *pred_cross_dist,  // npred * npocket floats
                             float *pred_holo_dist,   // npred * npred floats
                             float *values,           // nval float, as x in f(x)
                             int *torsions,           // ntorsion * 2 ints
                             uint8_t *masks,          // npred * ntorsion masks
                             int npred, int npocket, int nval, int ntorsion,
                             float *loss  // should be 1 floats, output

    ) {
        init_coord_      = init_coord;
        pocket_          = pocket;
        pred_cross_dist_ = pred_cross_dist;
        pred_holo_dist_  = pred_holo_dist;
        values_          = values;
        torsions_        = torsions;
        masks_           = masks;
        npred_           = npred;
        npocket_         = npocket;
        nval_            = nval;
        ntorsion_        = ntorsion;
        loss_            = loss;
    }
    ~CudaDockRequest() override = default;
    void run(Context *ctx) override {
        auto cudaCtx = dynamic_cast<CudaContext *>(ctx);
        if (cudaCtx) {
            int hostsz = 0, cudasz = 0;
            auto stream = cudaCtx->stream();
            auto smsize = cudaCtx->smsize();
            cudasz      = dock_cpu_async(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_,
                                         values_, torsions_, masks_, npred_, npocket_, nval_, ntorsion_,
                                         nullptr, 0, nullptr, hostsz, stream, smsize, loss_);
            assert(cudasz > 0);
            auto device = cudaCtx->requireDeviceMemory(cudasz);
            auto host = cudaCtx->requireHostMemory(hostsz);
            if (device == nullptr || host == nullptr) {
                result_ = RequestResult::Fail;
                return;
            }
            dock_cpu_async(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_, values_,
                           torsions_, masks_, npred_, npocket_, nval_, ntorsion_, device, cudasz,
                           host, hostsz, stream, smsize, loss_);
            auto err = cudaCtx->sync();
            if (cudaSuccess != err) {
                std::cerr << "cuda err " << err << " msg: " << cudaGetErrorString(err) << std::endl;
                result_ = RequestResult::Fail;
            }
        }
    }

private:
    float *init_coord_;
    float *pocket_;
    float *pred_cross_dist_;
    float *pred_holo_dist_;
    float *values_;
    int *torsions_;
    uint8_t *masks_;
    int npred_;
    int npocket_;
    int nval_;
    int ntorsion_;
    float *loss_;
};
std::shared_ptr<Request> createCudaDockRequest(float *init_coord,       // npred * 3 floats
                                               float *pocket,           // npocket * 3 floats
                                               float *pred_cross_dist,  // npred * npocket floats
                                               float *pred_holo_dist,   // npred * npred floats
                                               float *values,           // nval float, as x in f(x)
                                               int *torsions,           // ntorsion * 2 ints
                                               uint8_t *masks,          // npred * ntorsion masks
                                               int npred, int npocket, int nval, int ntorsion,
                                               float *loss  // should be 1 floats, output
) {
    return std::make_shared<CudaDockRequest>(init_coord, pocket, pred_cross_dist, pred_holo_dist,
                                             values, torsions, masks, npred, npocket, nval,
                                             ntorsion, loss);
}
class CudaDockGradSessionRequest : public Request {
public:
    explicit CudaDockGradSessionRequest(float *init_coord,       // npred * 3 floats
                                 float *pocket,           // npocket * 3 floats
                                 float *pred_cross_dist,  // npred * npocket floats
                                 float *pred_holo_dist,   // npred * npred floats
                                 int *torsions,           // ntorsion * 2 ints
                                 uint8_t *masks,          // npred * ntorsion masks
                                 int npred, int npocket, int nval, int ntorsion, float eps

    ) {
        init_coord_      = init_coord;
        pocket_          = pocket;
        pred_cross_dist_ = pred_cross_dist;
        pred_holo_dist_  = pred_holo_dist;
        torsions_        = torsions;
        masks_           = masks;
        npred_           = npred;
        npocket_         = npocket;
        nval_            = nval;
        ntorsion_        = ntorsion;
        eps_             = eps;
        
    }
    ~CudaDockGradSessionRequest() override = default;
    void run(Context *ctx) override {
        auto cudaCtx = dynamic_cast<CudaContext *>(ctx);
        if (cudaCtx) {
            // void *host  = cudaCtx->hostMemory(hostsz);
            auto stream = cudaCtx->stream();
            dock_grad_session(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_, torsions_,
                          masks_, npred_, npocket_, nval_, ntorsion_, nullptr, host_size_, nullptr,
                          cuda_size_, hdr_size_, stream);

            device_ = cudaCtx->requireDeviceMemory(cuda_size_);
            host_ = cudaCtx->requireHostMemory(host_size_);
            if (device_ == nullptr || host_ == nullptr) {
                std::cerr << "Session commit failed" << std::endl;
                result_ = RequestResult::Fail;
                return;
            }
            dock_grad_session(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_, torsions_,
                              masks_, npred_, npocket_, nval_, ntorsion_, host_, host_size_,
                              device_, cuda_size_, hdr_size_, stream);
            auto err = cudaCtx->sync();
            if (cudaSuccess != err) {
                std::cerr << "cuda err " << err << " msg: " << cudaGetErrorString(err) << std::endl;
                result_ = RequestResult::Fail;
            }
        }
    }

public:
    float *init_coord_;
    float *pocket_;
    float *pred_cross_dist_;
    float *pred_holo_dist_;
    int *torsions_;
    uint8_t *masks_;
    int npred_;
    int npocket_;
    int nval_;
    int ntorsion_;
    float eps_;
    void *device_ = nullptr;
    void *host_ = nullptr;
    int hdr_size_ = 0;
    int cuda_size_ = 0;
    int host_size_ = 0;
};
std::shared_ptr<Request> createCudaDockGradSessionRequest(
    float *init_coord,       // npred * 3 floats
    float *pocket,           // npocket * 3 floats
    float *pred_cross_dist,  // npred * npocket floats
    float *pred_holo_dist,   // npred * npred floats
    int *torsions,           // ntorsion * 2 ints
    uint8_t *masks,          // npred * ntorsion masks
    int npred, int npocket, int nval, int ntorsion, float eps
) {
    return std::make_shared<CudaDockGradSessionRequest>(init_coord, pocket, pred_cross_dist,
                                                 pred_holo_dist,  torsions, masks, npred,
                                                 npocket, nval, ntorsion, eps);
}

#define OFFSETPTR(p, n)  ((void *)(((uint8_t*)(p)) + (n)))
class CudaDockGradSubmitRequest : public Request {
public:
    explicit CudaDockGradSubmitRequest(std::shared_ptr<Request> session, float *values, float *losses) {
        session_ = std::dynamic_pointer_cast<CudaDockGradSessionRequest>(session);
        values_ = values;
        losses_ = losses;
    }
    ~CudaDockGradSubmitRequest() override = default;
    void run(Context *ctx) override {
        if (!session_) {
            std::cerr << "invalid session request" << std::endl;
            result_ = RequestResult::Fail;
            return;
        }
        auto cudaCtx = dynamic_cast<CudaContext *>(ctx);
        if (cudaCtx) {
            // void *host  = cudaCtx->hostMemory(hostsz);
            auto stream = cudaCtx->stream();
            int smsize  = cudaCtx->smsize();
            float *outloss = nullptr;
            dock_grad_exec(session_->init_coord_, session_->pocket_, session_->pred_cross_dist_,
                           session_->pred_holo_dist_, values_, session_->torsions_,
                           session_->masks_, session_->npred_, session_->npocket_, session_->nval_,
                           session_->ntorsion_, session_->eps_, outloss,
                           OFFSETPTR(session_->host_, session_->hdr_size_),
                           OFFSETPTR(session_->device_, session_->hdr_size_),
                           session_->cuda_size_ - session_->hdr_size_, stream, smsize);

            auto err = cudaCtx->sync();
            if (cudaSuccess != err) {
                std::cerr << "cuda err " << err << " msg: " << cudaGetErrorString(err) << std::endl;
                result_ = RequestResult::Fail;
            } else {
                auto eps = session_->eps_;
                for (auto i = 1; i < session_->nval_+1; i++) {
                    outloss[i] = (outloss[i] - outloss[0]) / eps;
                }
                memcpy(losses_, outloss, (session_->nval_+1) * sizeof(float));
            }

        }
    }

private:
    float *values_ = nullptr;
    float *losses_ = nullptr;
    std::shared_ptr<CudaDockGradSessionRequest> session_;
};
std::shared_ptr<Request> createCudaDockGradSubmitRequest(std::shared_ptr<Request> request,
                                                         float *values, float *losses) {
    return std::make_shared<CudaDockGradSubmitRequest>(request, values, losses);
}
class CudaDockGradRequest : public Request {
public:
    explicit CudaDockGradRequest(float *init_coord,       // npred * 3 floats
                                 float *pocket,           // npocket * 3 floats
                                 float *pred_cross_dist,  // npred * npocket floats
                                 float *pred_holo_dist,   // npred * npred floats
                                 float *values,           // nval float, as x in f(x)
                                 int *torsions,           // ntorsion * 2 ints
                                 uint8_t *masks,          // npred * ntorsion masks
                                 int npred, int npocket, int nval, int ntorsion, float eps,
                                 float *losses  // should be nval+1 floats, output

    ) {
        init_coord_      = init_coord;
        pocket_          = pocket;
        pred_cross_dist_ = pred_cross_dist;
        pred_holo_dist_  = pred_holo_dist;
        values_          = values;
        torsions_        = torsions;
        masks_           = masks;
        npred_           = npred;
        npocket_         = npocket;
        nval_            = nval;
        ntorsion_        = ntorsion;
        eps_             = eps;
        losses_          = losses;
    }
    ~CudaDockGradRequest() override = default;
    void run(Context *ctx) override {
        auto cudaCtx = dynamic_cast<CudaContext *>(ctx);
        if (cudaCtx) {
            int hostsz = 0, cudasz = 0;
            auto stream = cudaCtx->stream();
            int smsize  = cudaCtx->smsize();
            cudasz = dock_grad_cpu_async(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_,
                                         values_, torsions_, masks_, npred_, npocket_, nval_,
                                         ntorsion_, eps_, losses_, nullptr, hostsz, nullptr, 0, stream,
                                         smsize);
            if (cudasz < 10) {
                std::cerr << "dock grad sz " << cudasz << std::endl;
                result_ = RequestResult::Fail;
                return;
            }
            auto device = cudaCtx->requireDeviceMemory(cudasz);
            void *host  = cudaCtx->requireHostMemory(hostsz);
            if (device == nullptr||host == nullptr) {
                result_ = RequestResult::Fail;
                return;
            }
            cudasz   = dock_grad_cpu_async(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_,
                                           values_, torsions_, masks_, npred_, npocket_, nval_,
                                           ntorsion_, eps_, losses_, host, hostsz, device, cudasz,
                                           stream, smsize);
            auto err = cudaCtx->sync();
            if (cudaSuccess != err) {
                std::cerr << "cuda err " << err << " msg: " << cudaGetErrorString(err) << std::endl;
                result_ = RequestResult::Fail;
            } else {
                for (auto i = 1; i < nval_+1; i++) {
                    losses_[i] = (losses_[i] - losses_[0]) / eps_;
                }
            }
        }
    }

private:
    float *init_coord_;
    float *pocket_;
    float *pred_cross_dist_;
    float *pred_holo_dist_;
    float *values_;
    int *torsions_;
    uint8_t *masks_;
    int npred_;
    int npocket_;
    int nval_;
    int ntorsion_;
    float eps_;
    float *losses_;
};
std::shared_ptr<Request> createCudaDockGradRequest(
    float *init_coord,       // npred * 3 floats
    float *pocket,           // npocket * 3 floats
    float *pred_cross_dist,  // npred * npocket floats
    float *pred_holo_dist,   // npred * npred floats
    float *values,           // nval float, as x in f(x)
    int *torsions,           // ntorsion * 2 ints
    uint8_t *masks,          // npred * ntorsion masks
    int npred, int npocket, int nval, int ntorsion, float eps,
    float *losses  // should be nval+1 floats, output

) {
    return std::make_shared<CudaDockGradRequest>(init_coord, pocket, pred_cross_dist,
                                                 pred_holo_dist, values, torsions, masks, npred,
                                                 npocket, nval, ntorsion, eps, losses);
}
class CudaDockGradPerfRequest : public Request {
public:
    explicit CudaDockGradPerfRequest(float *init_coord,       // npred * 3 floats
                                     float *pocket,           // npocket * 3 floats
                                     float *pred_cross_dist,  // npred * npocket floats
                                     float *pred_holo_dist,   // npred * npred floats
                                     float *values,           // nval float, as x in f(x)
                                     int *torsions,           // ntorsion * 2 ints
                                     uint8_t *masks,          // npred * ntorsion masks
                                     int npred, int npocket, int nval, int ntorsion, int loop

    ) {
        init_coord_      = init_coord;
        pocket_          = pocket;
        pred_cross_dist_ = pred_cross_dist;
        pred_holo_dist_  = pred_holo_dist;
        values_          = values;
        torsions_        = torsions;
        masks_           = masks;
        npred_           = npred;
        npocket_         = npocket;
        nval_            = nval;
        ntorsion_        = ntorsion;
        losses_          = new float[nval + 1];
        loop_            = loop;
    }
    ~CudaDockGradPerfRequest() override = default;
    void run(Context *ctx) override {
        auto cudaCtx = dynamic_cast<CudaContext *>(ctx);
        if (cudaCtx) {
            int hostsz = 0, cudasz = 0;
            void *host  = cudaCtx->hostMemory(hostsz);
            auto stream = cudaCtx->stream();
            int smsize  = cudaCtx->smsize();
            cudasz = dock_grad_cpu_async(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_,
                                         values_, torsions_, masks_, npred_, npocket_, nval_,
                                         ntorsion_, 0.01, losses_, host, hostsz, nullptr, 0, stream,
                                         smsize);
            assert(cudasz >= 10);
            auto device = cudaCtx->requireDeviceMemory(cudasz);
            assert(device != nullptr);
            auto start = now();
            for (auto i = 0; i < loop_; i++) {
                dock_grad_cpu_async(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_,
                                    values_, torsions_, masks_, npred_, npocket_, nval_, ntorsion_,
                                    0.01, losses_, host, hostsz, device, cudasz, stream, smsize);
            }

            auto err = cudaCtx->sync();
            assert(err == cudaSuccess);
            auto end = now();
            auto du  = end - start;
            qps_     = loop_ * 1000 / du;
            std::cout << "start " << start << " end " << end << " du " << du << " loop " << loop_
                      << " QPS " << qps_ << std::endl;
        }
    }
    std::string getProp(const std::string &key) override {
        if (key == "qps") {
            return std::to_string(qps_);
        }
        return Request::getProp(key);
    }

private:
    float *init_coord_;
    float *pocket_;
    float *pred_cross_dist_;
    float *pred_holo_dist_;
    float *values_;
    int *torsions_;
    uint8_t *masks_;
    int npred_;
    int npocket_;
    int nval_;
    int ntorsion_;
    float *losses_;
    int loop_ = 0;
    int qps_  = 0;
};
std::shared_ptr<Request> createCudaDockGradPerfRequest(
    float *init_coord,       // npred * 3 floats
    float *pocket,           // npocket * 3 floats
    float *pred_cross_dist,  // npred * npocket floats
    float *pred_holo_dist,   // npred * npred floats
    float *values,           // nval float, as x in f(x)
    int *torsions,           // ntorsion * 2 ints
    uint8_t *masks,          // npred * ntorsion masks
    int npred, int npocket, int nval, int ntorsion, int loop

) {
    return std::make_shared<CudaDockGradPerfRequest>(init_coord, pocket, pred_cross_dist,
                                                     pred_holo_dist, values, torsions, masks, npred,
                                                     npocket, nval, ntorsion, loop);
}
}  // namespace dock