#include "dock.h"
#include <assert.h>
#include <chrono>
#include <iostream>
#include <string.h>

#include "cuda_context.h"
#include "cuda_runtime_api.h"
#include "dtype.h"
#include "cu/dockcu.h"

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
static void dumparr(int m, int n, dtype *p) {
    printf(" ====== host(%dx%d) ======\n", m, n);
    for (int i = 0; i < m; i++) {
        printf("%d: ", i);
        dtype *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%f ", p1[j]);
        }
        printf("\n");
    }
}
#endif

template<typename T>
int copy_data(T *&val, uint8_t *&tmp, uint8_t *&hosttmp, int blksz, bool eval) {
    int tsz = sizeof(T) * blksz; // actual size
    int asize = (tsz + sizeof(dtype) - 1)/sizeof(dtype)*sizeof(dtype); // offset size, for 4 bytes alignment

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
void dock_grad_session(dtype *&init_coord,       // npred * 3 dtypes
                       dtype *&pocket,           // npocket * 3 dtypes
                       dtype *&pred_cross_dist,  // npred * npocket dtypes
                       dtype *&pred_holo_dist,   // npred * npred dtypes
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
    COPY(init_coord, dtype, npred * 3);
    COPY(pocket, dtype, npocket * 3);
    COPY(pred_cross_dist, dtype, npred *npocket);
    COPY(pred_holo_dist, dtype, npred *npred);
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
    int devsz = dock_grad_gpu_mem_size(npred, npocket, nval, ntorsion);
    int valuesz = nval  * sizeof(dtype); // in bytes, for values
    int losssz = (nval + 1) * sizeof(dtype); // for output loss
    hostsz = hdrsz + valuesz + losssz;
    cudasz = devsz + hostsz;
    return;
}

class CudaDockGradSessionRequest : public Request {
public:
    explicit CudaDockGradSessionRequest(dtype *init_coord,       // npred * 3 dtypes
                                 dtype *pocket,           // npocket * 3 dtypes
                                 dtype *pred_cross_dist,  // npred * npocket dtypes
                                 dtype *pred_holo_dist,   // npred * npred dtypes
                                 int *torsions,           // ntorsion * 2 ints
                                 uint8_t *masks,          // npred * ntorsion masks
                                 int npred, int npocket, int nval, int ntorsion, dtype eps

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
                              masks_, npred_, npocket_, nval_, ntorsion_, nullptr, host_size_,
                              nullptr, cuda_size_, hdr_size_, stream);

            device_ = cudaCtx->requireDeviceMemory(cuda_size_);
            host_   = cudaCtx->requireHostMemory(host_size_);
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
    dtype *init_coord_;
    dtype *pocket_;
    dtype *pred_cross_dist_;
    dtype *pred_holo_dist_;
    int *torsions_;
    uint8_t *masks_;
    int npred_;
    int npocket_;
    int nval_;
    int ntorsion_;
    dtype eps_;
    void *device_ = nullptr;
    void *host_ = nullptr;
    int hdr_size_ = 0;
    int cuda_size_ = 0;
    int host_size_ = 0;
};
std::shared_ptr<Request> createCudaDockGradSessionRequest(
    dtype *init_coord,       // npred * 3 dtypes
    dtype *pocket,           // npocket * 3 dtypes
    dtype *pred_cross_dist,  // npred * npocket dtypes
    dtype *pred_holo_dist,   // npred * npred dtypes
    int *torsions,           // ntorsion * 2 ints
    uint8_t *masks,          // npred * ntorsion masks
    int npred, int npocket, int nval, int ntorsion, dtype eps
) {
    return std::make_shared<CudaDockGradSessionRequest>(init_coord, pocket, pred_cross_dist,
                                                 pred_holo_dist,  torsions, masks, npred,
                                                 npocket, nval, ntorsion, eps);
}

#define OFFSETPTR(p, n)  ((void *)(((uint8_t*)(p)) + (n)))
class CudaDockGradSubmitRequest : public Request {
public:
    explicit CudaDockGradSubmitRequest(std::shared_ptr<Request> session, dtype *values, dtype *losses) {
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
            auto eps    = session_->eps_;
            auto stream = cudaCtx->stream();
            int smsize  = cudaCtx->smsize();
            // copy host values to device

            dtype *devValues, *devLosses;
            int sz = 0, nval = session_->nval_;

            dtype *dev = (dtype *)OFFSETPTR(session_->device_, session_->hdr_size_);
            devValues  = dev, sz += nval, dev += nval;
            devLosses  = dev, sz += nval + 1, dev += nval + 1;
            sz         = sz * sizeof(dtype) + session_->hdr_size_;  // bytes used in device_

            cudaMemcpyAsync(devValues, values_, nval * sizeof(dtype), cudaMemcpyHostToDevice,
                            stream);

            dock_grad_gpu(session_->init_coord_, session_->pocket_, session_->pred_cross_dist_,
                          session_->pred_holo_dist_, devValues, session_->torsions_,
                          session_->masks_, session_->npred_, session_->npocket_, session_->nval_,
                          session_->ntorsion_, devLosses, dev, session_->cuda_size_ - sz, stream,
                          smsize, eps);

            cudaMemcpyAsync(losses_, devLosses, (nval + 1) * sizeof(dtype), cudaMemcpyDeviceToHost,
                            stream);
            auto err = cudaCtx->sync();
            if (cudaSuccess != err) {
                std::cerr << "cuda err " << err << " msg: " << cudaGetErrorString(err) << std::endl;
                result_ = RequestResult::Fail;
            } else {
                for (auto i = 1; i < session_->nval_ + 1; i++) {
                    losses_[i] = (losses_[i] - losses_[0]) / eps;
                }
            }
        }
    }

private:
    dtype *values_ = nullptr;
    dtype *losses_ = nullptr;
    std::shared_ptr<CudaDockGradSessionRequest> session_;
};
std::shared_ptr<Request> createCudaDockGradSubmitRequest(std::shared_ptr<Request> request,
                                                         dtype *values, dtype *losses) {
    return std::make_shared<CudaDockGradSubmitRequest>(request, values, losses);
}
}  // namespace dock