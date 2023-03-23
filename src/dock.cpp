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
                     int npred,
                     int npocket,
                     int nval,
                     int ntorsion,
                     float *loss,     // output, scalar
                     float *dev,      // input device mem
                     int &devSize,  // dev size in byte
                     cudaStream_t stream,
                     int smsize);
void dock_grad_gpu(float *init_coord,
                   float *pocket,
                   float *pred_cross_dist,
                   float *pred_holo_dist,
                   float *values,
                   int *torsions,
                   uint8_t *masks,
                   int npred,
                   int npocket,
                   int nval,
                   int ngval,
                   int ntorsion,
                   float *loss,  // ngval float array
                   float *dev,
                   int &devSize,
                   cudaStream_t stream,
                   int smsize);
static std::uint64_t now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}


int dock_cpu_async(float *init_coord,
                   float *pocket,
                   float *pred_cross_dist,
                   float *pred_holo_dist,
                   float *values,
                   int *torsions,
                   uint8_t *masks,
                   int npred,
                   int npocket,
                   int nval,
                   int ntorsion,
                   void *dev,
                   int devsize,
                   void *host,
                   int hostsize,
                   cudaStream_t stream,
                   int smsize,
                   float *outloss) {
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
 if (hsize >= hostsize) \
  return 1; \
 if (!eval) memcpy(host_##val, val, tsz);

    COPY(init_coord, float, npred * 3);
    COPY(pocket, float, npocket * 3);
    COPY(pred_cross_dist, float, npred *npocket);
    COPY(pred_holo_dist, float, npred *npred);
    COPY(values, float, nval);
    COPY(torsions, int, ntorsion * 2);
    COPY(masks, uint8_t, ntorsion * npred);
#undef COPY
    if(!eval) {
        cudaMemcpyAsync(dev, host, hsize, cudaMemcpyHostToDevice, stream);
    }

    float *loss;
    loss      = (float *)tmp, tmp += sizeof(float);
    int tmpsz = 0;
    int cudahdr = int(tmp - (uint8_t *)dev);
    if (!eval) {
        tmpsz = devsize - cudahdr;
    }

    // must be last one to copy for alignment
    dock_gpu(gpu_init_coord,
             gpu_pocket,
             gpu_pred_cross_dist,
             gpu_pred_holo_dist,
             gpu_values,
             gpu_torsions,
             gpu_masks,
             npred,
             npocket,
             nval,
             ntorsion,
             loss,
             eval ? nullptr : (float *)tmp,
             tmpsz,
             stream,
             smsize);

    if(!eval) {
        cudaMemcpyAsync(outloss, loss, sizeof(float), cudaMemcpyDeviceToHost, stream);
        return 0;
    }
    return tmpsz + cudahdr;
}
// return 1 for lacking of host memory
int dock_grad_cpu_async(float *init_coord,       // npred * 3 floats
                        float *pocket,           // npocket * 3 floats
                        float *pred_cross_dist,  // npred * npocket floats
                        float *pred_holo_dist,   // npred * npred floats
                        float *values,           // nval float, as x in f(x)
                        int *torsions,           // ntorsion * 2 ints
                        uint8_t *masks,          // npred * ntorsion masks
                        int npred,
                        int npocket,
                        int nval,
                        int ntorsion,
                        float eps,
                        float *losses,  // should be nval+1 floats, output
                        void *host,     // cuda host memory
                        int hostsz,
                        void *device,
                        int cudasz,
                        cudaStream_t stream,
                        int smsize) {
    bool eval        = device == nullptr;
    uint8_t *tmp     = (uint8_t *)device;
    uint8_t *hosttmp = (uint8_t *)host;
    size_t tsz;
    int hsize = 0, asize = 0;

#define COPY(val, tp, blksz) \
 tp *gpu_##val  = (tp *)tmp; \
 tp *host_##val = (tp *)hosttmp; \
 tsz            = sizeof(tp) * blksz; \
 asize = (tsz + 3)/4*4;\
 hosttmp += asize; \
 tmp += asize; \
 hsize += asize; \
 if (hsize >= hostsz) { \
  return 1; \
 } \
 if(!eval) memcpy(host_##val, val, tsz);

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
    int cpsize = (nval + 1) * nval * sizeof(float);
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
    dock_grad_gpu(gpu_init_coord,
                  gpu_pocket,
                  gpu_pred_cross_dist,
                  gpu_pred_holo_dist,
                  gpu_valueBatch,
                  gpu_torsions,
                  gpu_masks,
                  npred,
                  npocket,
                  nval,
                  nval + 1,
                  ntorsion,
                  loss,
                  eval ? nullptr : (float *)tmp,
                  devsize,
                  stream,
                  smsize);
    if (!eval) {
        cudaMemcpyAsync(losses, loss, (nval + 1) * sizeof(float), cudaMemcpyDeviceToHost);
        return 0;
    }
    return devsize + hsize + (nval + 1) * sizeof(float);
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
                             int npred,
                             int npocket,
                             int nval,
                             int ntorsion,
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
            void *host   = cudaCtx->hostMemory(hostsz);
            auto stream  = cudaCtx->stream();
            auto smsize  = cudaCtx->smsize();
            cudasz = dock_cpu_async(init_coord_,
                           pocket_,
                           pred_cross_dist_,
                           pred_holo_dist_,
                           values_,
                           torsions_,
                           masks_,
                           npred_,
                           npocket_,
                           nval_,
                           ntorsion_,
                           nullptr,
                           0,
                           host,
                           hostsz,
                           stream,
                           smsize,
                           loss_);
            if (cudasz < 10) {
                result_ = RequestResult::Fail;
                return;
            }
            auto device  = cudaCtx->requireDeviceMemory(cudasz);
            if (device == nullptr) {
                result_ = RequestResult::Fail;
                return;
            }
            dock_cpu_async(init_coord_,
                           pocket_,
                           pred_cross_dist_,
                           pred_holo_dist_,
                           values_,
                           torsions_,
                           masks_,
                           npred_,
                           npocket_,
                           nval_,
                           ntorsion_,
                           device,
                           cudasz,
                           host,
                           hostsz,
                           stream,
                           smsize,
                           loss_);
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
                                                  float *values,   // nval float, as x in f(x)
                                                  int *torsions,   // ntorsion * 2 ints
                                                  uint8_t *masks,  // npred * ntorsion masks
                                                  int npred,
                                                  int npocket,
                                                  int nval,
                                                  int ntorsion,
                                                  float *loss  // should be 1 floats, output
) {
    return std::make_shared<CudaDockRequest>(init_coord,
                                             pocket,
                                             pred_cross_dist,
                                             pred_holo_dist,
                                             values,
                                             torsions,
                                             masks,
                                             npred,
                                             npocket,
                                             nval,
                                             ntorsion,
                                             loss);
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
                                 int npred,
                                 int npocket,
                                 int nval,
                                 int ntorsion,
                                 float eps,
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
            void *host   = cudaCtx->hostMemory(hostsz);
            auto stream  = cudaCtx->stream();
            int smsize   = cudaCtx->smsize();
            cudasz = dock_grad_cpu_async(init_coord_,
                                         pocket_,
                                         pred_cross_dist_,
                                         pred_holo_dist_,
                                         values_,
                                         torsions_,
                                         masks_,
                                         npred_,
                                         npocket_,
                                         nval_,
                                         ntorsion_,
                                         eps_,
                                         losses_,
                                         host,
                                         hostsz,
                                         nullptr,
                                         0,
                                         stream,
                                         smsize);
            if (cudasz < 10) {
                result_ = RequestResult::Fail;
                return;
            }
            auto device  = cudaCtx->requireDeviceMemory(cudasz);
            if (device == nullptr) {
                result_ = RequestResult::Fail;
                return;
            }
            cudasz = dock_grad_cpu_async(init_coord_,
                                pocket_,
                                pred_cross_dist_,
                                pred_holo_dist_,
                                values_,
                                torsions_,
                                masks_,
                                npred_,
                                npocket_,
                                nval_,
                                ntorsion_,
                                eps_,
                                losses_,
                                host,
                                hostsz,
                                device,
                                cudasz,
                                stream,
                                smsize);
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
  int npred,
  int npocket,
  int nval,
  int ntorsion,
  float eps,
  float *losses  // should be nval+1 floats, output

) {
    return std::make_shared<CudaDockGradRequest>(init_coord,
                                                 pocket,
                                                 pred_cross_dist,
                                                 pred_holo_dist,
                                                 values,
                                                 torsions,
                                                 masks,
                                                 npred,
                                                 npocket,
                                                 nval,
                                                 ntorsion,
                                                 eps,
                                                 losses);
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
                                 int npred,
                                 int npocket,
                                 int nval,
                                 int ntorsion,
                                 int loop

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
        losses_          = new float[nval+1];
        loop_            = loop;
    }
    ~CudaDockGradPerfRequest() override = default;
    void run(Context *ctx) override {
        auto cudaCtx = dynamic_cast<CudaContext *>(ctx);
        if (cudaCtx) {
            int hostsz = 0, cudasz = 0;
            void *host   = cudaCtx->hostMemory(hostsz);
            auto stream  = cudaCtx->stream();
            int smsize   = cudaCtx->smsize();
            cudasz       = dock_grad_cpu_async(init_coord_,
                                         pocket_,
                                         pred_cross_dist_,
                                         pred_holo_dist_,
                                         values_,
                                         torsions_,
                                         masks_,
                                         npred_,
                                         npocket_,
                                         nval_,
                                         ntorsion_,
                                         0.001,
                                         losses_,
                                         host,
                                         hostsz,
                                         nullptr,
                                         0,
                                         stream,
                                         smsize);
            assert(cudasz >= 10);
            auto device  = cudaCtx->requireDeviceMemory(cudasz);
            assert(device != nullptr);
            auto start   = now();
            for (auto i = 0; i < loop_; i++) {
                dock_grad_cpu_async(init_coord_,
                                    pocket_,
                                    pred_cross_dist_,
                                    pred_holo_dist_,
                                    values_,
                                    torsions_,
                                    masks_,
                                    npred_,
                                    npocket_,
                                    nval_,
                                    ntorsion_,
                                    0.001,
                                    losses_,
                                    host,
                                    hostsz,
                                    device,
                                    cudasz,
                                    stream,
                                    smsize);

            }
            
            auto err = cudaCtx->sync();
            assert(err == cudaSuccess);
            auto end = now();
            auto du  = end - start;
            qps_ = loop_ * 1000 / du;
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
  int npred,
  int npocket,
  int nval,
  int ntorsion,
  int loop

) {
    return std::make_shared<CudaDockGradPerfRequest>(init_coord,
                                                 pocket,
                                                 pred_cross_dist,
                                                 pred_holo_dist,
                                                 values,
                                                 torsions,
                                                 masks,
                                                 npred,
                                                 npocket,
                                                 nval,
                                                 ntorsion,
                                                 loop);
}
}  // namespace dock