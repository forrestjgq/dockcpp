#ifndef DOCK_OPTIMIZER_H_
#define DOCK_OPTIMIZER_H_

#include <memory>
#include "dtype.h"

namespace dock {

class Optimizer {
public:
    Optimizer()          = default;
    virtual ~Optimizer() = default;
    virtual int run(dtype *init_values, dtype *out, dtype *best_loss, dtype eps, int nval) {
        return -1;
    }
};

std::shared_ptr<Optimizer> create_lbfgsb_dock(dtype *init_coord,       // npred * 3 dtypes
                    dtype *pocket,           // npocket * 3 dtypes
                    dtype *pred_cross_dist,  // npred * npocket dtypes
                    dtype *pred_holo_dist,   // npred * npred dtypes
                    int *torsions,           // ntorsion * 2 ints
                    uint8_t *masks,          // npred * ntorsion masks
                    int npred, int npocket, int ntorsion
);
}

#endif