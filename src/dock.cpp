#include "cuda_runtime_api.h"
#include "dock.h"
#include <assert.h>
#include <iostream>
#include <string.h>
#include <chrono>

namespace dock {

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
                     cudaStream_t stream);
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
                   cudaStream_t stream);
static std::uint64_t now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}
class Context {
public:
    Context(int device) : device_(device) {

    }

private:
    int device_;
    int max_sm_;
};

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
    auto err = cudaMalloc(&mem, sz);
    if (err != cudaSuccess) {
        return 0.f;
    }
    uint8_t *tmp = (uint8_t*)mem;
    size_t tsz;

#define COPY(val, tp, sz) \
 tp *gpu_##val = (tp *)tmp; \
 tsz           = sizeof(tp) * sz; \
 tmp += tsz; \
 cudaMemcpy(gpu_##val, val, tsz, cudaMemcpyHostToDevice);

    COPY(init_coord, float, npred * 3);
    COPY(pocket, float, npocket * 3);
    COPY(pred_cross_dist, float, npred * npocket);
    COPY(pred_holo_dist, float, npred * npred);
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
    return ret;
}
int dock_grad_cpu(float *init_coord,
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
              float *losses) {
    int sz    = 2 * 1024 * 1024;
    void *mem = nullptr;
    auto err = cudaMalloc(&mem, sz);
    if (err != cudaSuccess) {
        return 1;
    }
    uint8_t *tmp = (uint8_t*)mem;
    size_t tsz;

#define COPY(val, tp, sz) \
 tp *gpu_##val = (tp *)tmp; \
 tsz           = sizeof(tp) * sz; \
 tmp += tsz; \
 cudaMemcpy(gpu_##val, val, tsz, cudaMemcpyHostToDevice);

    COPY(init_coord, float, npred * 3);
    COPY(pocket, float, npocket * 3);
    COPY(pred_cross_dist, float, npred * npocket);
    COPY(pred_holo_dist, float, npred * npred);

    float *valueBatch = new float[nval * (nval + 1)];
    memcpy(valueBatch, values, nval * sizeof(float));
    for (int i = 0; i < nval; i++) {
        float *start = valueBatch + (i + 1) * nval;
        memcpy(start, values, nval * sizeof(float));
        start[i] += 0.01;
    }
    COPY(valueBatch, float, nval * (nval+1));
    COPY(torsions, int, ntorsion * 2);

    float *loss;
    loss = (float *)tmp, tmp += (nval+1)*sizeof(float);

    int tmpsz = int(tmp - (uint8_t *)mem);

    assert(uint64_t(tmp) % 4 == 0);

    // must be last one to copy for alignment
    COPY(masks, uint8_t, ntorsion * npred);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
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
             nval+1,
             ntorsion,
             loss,
             (float *)tmp,
             tmpsz,
             stream);
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "dock gpu fail, err: " << err << " " << cudaGetErrorString(err) << std::endl;
        return 4;
    }
    cudaStreamDestroy(stream);

    cudaMemcpy(&losses, loss, (nval+1)*sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "memcpy fail, err: " << err << " " << cudaGetErrorString(err) << std::endl;
        return 3;
    }
    cudaFree(mem);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 2;
    }
#undef COPY
    return 0;
}
int dock_grad_cpu_perf(float *init_coord,
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
              int n) {
    int sz    = 2 * 1024 * 1024;
    void *mem = nullptr;
    auto err = cudaMalloc(&mem, sz);
    if (err != cudaSuccess) {
        return -1;
    }
    float *output = new float[nval + 1];
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto start = now();
#define COPY(val, tp, sz) \
 tp *gpu_##val = (tp *)tmp; \
 tsz           = sizeof(tp) * sz; \
 tmp += tsz; \
 cudaMemcpyAsync(gpu_##val, val, tsz, cudaMemcpyHostToDevice, stream);
    for (int i = 0; i < n; i++) {
        uint8_t *tmp = (uint8_t *)mem;
        size_t tsz;


        COPY(init_coord, float, npred * 3);
        COPY(pocket, float, npocket * 3);
        COPY(pred_cross_dist, float, npred *npocket);
        COPY(pred_holo_dist, float, npred *npred);

        float *valueBatch = new float[nval * (nval + 1)];
        memcpy(valueBatch, values, nval * sizeof(float));
        for (int i = 0; i < nval; i++) {
            float *start = valueBatch + (i + 1) * nval;
            memcpy(start, values, nval * sizeof(float));
            start[i] += 0.01;
        }
        COPY(valueBatch, float, nval *(nval + 1));
        COPY(torsions, int, ntorsion * 2);

        float *loss;
        loss = (float *)tmp, tmp += (nval + 1) * sizeof(float);

        int tmpsz = int(tmp - (uint8_t *)mem);

        assert(uint64_t(tmp) % 4 == 0);

        // must be last one to copy for alignment
        COPY(masks, uint8_t, ntorsion * npred);
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
                      stream);
        cudaMemcpyAsync(output, loss, (nval + 1) * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);
    auto end = now();
    auto du  = end - start;
    auto qps = n * 1000 / du;
    cudaStreamDestroy(stream);
    cudaFree(mem);
#undef COPY
    return qps;
}
}