#ifndef _DOCK_H_
#define _DOCK_H_
#include <stdint.h>
#include <stddef.h>
#include "context.h"

namespace dock {
extern std::shared_ptr<dock::Request> createCudaDockRequest(
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
  float *loss  // should be 1 floats, output
);
extern std::shared_ptr<dock::Request> createCudaDockGradRequest(
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
);
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
);
std::shared_ptr<Request> createCudaDockGradSessionRequest(
    float *init_coord,       // npred * 3 floats
    float *pocket,           // npocket * 3 floats
    float *pred_cross_dist,  // npred * npocket floats
    float *pred_holo_dist,   // npred * npred floats
    int *torsions,           // ntorsion * 2 ints
    uint8_t *masks,          // npred * ntorsion masks
    int npred, int npocket, int nval, int ntorsion, float eps
);
std::shared_ptr<Request> createCudaDockGradSubmitRequest(std::shared_ptr<Request> request,
                                                         float *values, float *losses, float *svds);
}  // namespace dock
#endif