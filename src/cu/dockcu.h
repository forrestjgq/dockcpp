#include <cuda_runtime.h>
#include <stdint.h>
#include "dtype.h"

#ifndef DOCK_CU_H_
#define DOCK_CU_H_
namespace dock {

extern int dock_grad_gpu_mem_size(int npred, int npocket, int nval, int ntorsion);

extern void dock_grad_gpu(dtype *init_coord, dtype *pocket, dtype *pred_cross_dist,
                          dtype *pred_holo_dist, dtype *values, int *torsions, uint8_t *masks,
                          int npred, int npocket, int nval, int ntorsion,
                          dtype *loss,  // ngval dtype array
                          dtype *dev,
                          int devSize,  // in bytes
                          cudaStream_t stream, int smMaxSize, dtype eps);
};
#endif