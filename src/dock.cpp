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
                     size_t devSize,  // dev size in byte
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
                   size_t devSize,
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
 memcpy(host_##val, val, tsz);

    COPY(init_coord, float, npred * 3);
    COPY(pocket, float, npocket * 3);
    COPY(pred_cross_dist, float, npred *npocket);
    COPY(pred_holo_dist, float, npred *npred);
    COPY(values, float, nval);
    COPY(torsions, int, ntorsion * 2);
    COPY(masks, uint8_t, ntorsion * npred);
    cudaMemcpyAsync(dev, host, hsize, cudaMemcpyHostToDevice, stream);

    float *loss;
    loss      = (float *)tmp, tmp += sizeof(float);
    int tmpsz = int(tmp - (uint8_t *)dev);

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
             (float *)tmp,
             tmpsz,
             stream,
             smsize);
    cudaMemcpyAsync(outloss, loss, sizeof(float), cudaMemcpyDeviceToHost, stream);
#undef COPY
    return 0;
}
#if 0
float dock_cpu(float *init_coord,
               float *pocket,
               float *pred_cross_dist,
               float *pred_holo_dist,
               float *values,
               int *torsions,
               uint8_t *masks,
               int npred,
               int npocket,
               int nval,
               int ntorsion) {
    int sz    = 2 * 1024 * 1024;
    void *mem = nullptr;
    auto err  = cudaMalloc(&mem, sz);
    if (err != cudaSuccess) {
        return 0.f;
    }
    uint8_t *tmp = (uint8_t *)mem;
    size_t tsz;

#define COPY(val, tp, sz) \
 tp *gpu_##val = (tp *)tmp; \
 tsz           = sizeof(tp) * sz; \
 tmp += tsz; \
 cudaMemcpy(gpu_##val, val, tsz, cudaMemcpyHostToDevice);

    COPY(init_coord, float, npred * 3);
    COPY(pocket, float, npocket * 3);
    COPY(pred_cross_dist, float, npred *npocket);
    COPY(pred_holo_dist, float, npred *npred);
    COPY(values, float, nval);
    COPY(torsions, int, ntorsion * 2);

    float *loss;
    loss = (float *)tmp, tmp += sizeof(float);

    int tmpsz = int(tmp - (uint8_t *)mem);

    assert(uint64_t(tmp) % 4 == 0);

    // must be last one to copy for alignment
    COPY(masks, uint8_t, ntorsion * npred);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
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
             (float *)tmp,
             tmpsz,
             stream);
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "dock gpu fail, err: " << err << " " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    cudaStreamDestroy(stream);

    float ret;
    cudaMemcpy(&ret, loss, sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "memcpy fail, err: " << err << " " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    cudaFree(mem);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -1.0;
    }
#undef COPY
    return ret;
}
#endif
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
    // printf("input loss %f %f %f %f %f %f %f %f %f\n",
    //        losses[0],
    //        losses[1],
    //        losses[2],
    //        losses[3],
    //        losses[4],
    //        losses[5],
    //        losses[6],
    //        losses[7],
    //        losses[8]);
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
 memcpy(host_##val, val, tsz);

    COPY(init_coord, float, npred * 3);
    COPY(pocket, float, npocket * 3);
    COPY(pred_cross_dist, float, npred *npocket);
    COPY(pred_holo_dist, float, npred *npred);

    float *valueBatch = (float *)hosttmp;
    memcpy(valueBatch, values, nval * sizeof(float));
    for (int i = 0; i < nval; i++) {
        float *start = valueBatch + (i + 1) * nval;
        memcpy(start, values, nval * sizeof(float));
        start[i] += eps;
    }
    int cpsize = (nval + 1) * nval * sizeof(float);
    float *gpu_valueBatch = (float *)tmp;
    hosttmp += cpsize;
    tmp += cpsize;
    hsize += cpsize;

    COPY(torsions, int, ntorsion * 2);
    COPY(masks, uint8_t, ntorsion * npred);
    // printf("host masks: ");
    // dumparr(2, npred, masks);
    cudaMemcpyAsync(device, host, hsize, cudaMemcpyHostToDevice, stream);

    float *loss;
    loss = (float *)tmp, tmp += (nval + 1) * sizeof(float);
    // how much left for kernel tmp cuda mem
    int tmpsz = cudasz - hsize - (nval + 1) * sizeof(float);

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
                  (float *)tmp,
                  tmpsz,
                  stream,
                  smsize);
    cudaMemcpyAsync(losses, loss, (nval + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaStreamSynchronize(stream);
    // printf("output loss %f %f %f %f %f %f %f %f %f\n",
    //        losses[0],
    //        losses[1],
    //        losses[2],
    //        losses[3],
    //        losses[4],
    //        losses[5],
    //        losses[6],
    //        losses[7],
    //        losses[8]);
#undef COPY
    return 0;
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
            void *device = cudaCtx->deviceMemory(cudasz);
            auto stream  = cudaCtx->stream();
            auto smsize  = cudaCtx->smsize();
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
            auto err = cudaStreamSynchronize(stream);
            if(cudaSuccess != err) {
                std::cerr << "cuda err " << err << " msg: " << cudaGetErrorString(err) << std::endl;
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
            void *device = cudaCtx->deviceMemory(cudasz);
            auto stream  = cudaCtx->stream();
            int smsize   = cudaCtx->smsize();
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
                                eps_,
                                losses_,
                                host,
                                hostsz,
                                device,
                                cudasz,
                                stream,
                                smsize);
            cudaStreamSynchronize(stream);
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
            void *device = cudaCtx->deviceMemory(cudasz);
            auto stream  = cudaCtx->stream();
            int smsize   = cudaCtx->smsize();
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
            auto err = cudaStreamSynchronize(stream);
            assert(err == cudaSuccess);
            auto end = now();
            auto du  = end - start;
            qps_ = loop_ * 1000 / du;
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