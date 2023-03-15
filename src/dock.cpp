#include "cuda_runtime_api.h"
#include "dock.h"

namespace dock {

extern void dock_gpu(float *init_coord, // npred * 3
                     float *pocket, // npocket x 3
                     float *pred_cross_dist, // npred x npocket
                     float *pred_holo_dist, // npred x npred
                     float *values, // nval
                     int *torsions, // ntorsion x 2
                     uint8_t *masks, // ntorsion * npred
                     int npred,
                     int npocket,
                     int nval,
                     int ntorsion,
                     float *loss,
                     cudaStream_t stream);

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
    size_t sz = (npred * 3 + npocket * 3 + npred * npocket + npred * npred + nval) * sizeof(float)
                 + nval * sizeof(int) + ntorsion * 2 * sizeof(int) + sizeof(float) /*loss*/
                 + ntorsion * npred;
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

    // must be last one to copy for alignment
    COPY(masks, uint8_t, ntorsion * npred);
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
             nullptr);

    float ret;
    cudaMemcpy(&ret, loss, sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}
}