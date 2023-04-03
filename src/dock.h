#ifndef _DOCK_H_
#define _DOCK_H_
#include <stdint.h>
#include <stddef.h>
#include "context.h"
#include "dtype.h"

namespace dock {
extern std::shared_ptr<dock::Request> createCudaDockRequest(
  dtype *init_coord,       // npred * 3 dtypes
  dtype *pocket,           // npocket * 3 dtypes
  dtype *pred_cross_dist,  // npred * npocket dtypes
  dtype *pred_holo_dist,   // npred * npred dtypes
  dtype *values,           // nval dtype, as x in f(x)
  int *torsions,           // ntorsion * 2 ints
  uint8_t *masks,          // npred * ntorsion masks
  int npred,
  int npocket,
  int nval,
  int ntorsion,
  dtype *loss  // should be 1 dtypes, output
);
extern std::shared_ptr<dock::Request> createCudaDockGradRequest(
  dtype *init_coord,       // npred * 3 dtypes
  dtype *pocket,           // npocket * 3 dtypes
  dtype *pred_cross_dist,  // npred * npocket dtypes
  dtype *pred_holo_dist,   // npred * npred dtypes
  dtype *values,           // nval dtype, as x in f(x)
  int *torsions,           // ntorsion * 2 ints
  uint8_t *masks,          // npred * ntorsion masks
  int npred,
  int npocket,
  int nval,
  int ntorsion,
  dtype eps,
  dtype *losses  // should be nval+1 dtypes, output
);
std::shared_ptr<Request> createCudaDockGradPerfRequest(
  dtype *init_coord,       // npred * 3 dtypes
  dtype *pocket,           // npocket * 3 dtypes
  dtype *pred_cross_dist,  // npred * npocket dtypes
  dtype *pred_holo_dist,   // npred * npred dtypes
  dtype *values,           // nval dtype, as x in f(x)
  int *torsions,           // ntorsion * 2 ints
  uint8_t *masks,          // npred * ntorsion masks
  int npred,
  int npocket,
  int nval,
  int ntorsion,
  int loop
);
std::shared_ptr<Request> createCudaDockGradSessionRequest(
    dtype *init_coord,       // npred * 3 dtypes
    dtype *pocket,           // npocket * 3 dtypes
    dtype *pred_cross_dist,  // npred * npocket dtypes
    dtype *pred_holo_dist,   // npred * npred dtypes
    int *torsions,           // ntorsion * 2 ints
    uint8_t *masks,          // npred * ntorsion masks
    int npred, int npocket, int nval, int ntorsion, dtype eps
);
std::shared_ptr<Request> createCudaDockGradSubmitRequest(std::shared_ptr<Request> request,
                                                         dtype *values, dtype *losses);
}  // namespace dock
#endif