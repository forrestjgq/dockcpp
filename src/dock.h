#ifndef _DOCK_H_
#define _DOCK_H_
#include <stdint.h>

namespace dock {
extern float dock_cpu(float *init_coord,
              float *pocket,
              float *pred_cross_dist,
              float *pred_holo_dist,
              float *values,
              int *torsions,
              uint8_t *masks,
              int npred,
              int npocket,
              int nval,
              int ntorsion);

}
#endif