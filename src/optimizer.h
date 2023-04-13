#ifndef DOCK_OPTIMIZER_H_
#define DOCK_OPTIMIZER_H_

#include <memory>
#include "dtype.h"
#include "context.h"

namespace dock {

class Optimizer {
public:
    Optimizer()          = default;
    virtual ~Optimizer() = default;
    virtual int run(dtype *init_values, dtype *out, dtype *best_loss, dtype eps, int nval) {
        return -1;
    }
    virtual int post(std::shared_ptr<Request> req) {
        return -1;
    }
};

class OptimizerServer {
public:
    OptimizerServer()          = default;
    virtual ~OptimizerServer() = default;

    virtual bool submit(std::shared_ptr<Request> req, std::uint64_t &seq) {
        return false;
    }
    virtual bool poll(std::uint64_t &seq, dtype &loss, dtype *&values, int &nval) {
        return false;
    }
};

extern std::shared_ptr<OptimizerServer> create_lbfgsb_server(int device, int n);

extern std::shared_ptr<Optimizer> create_lbfgsb_dock(
    int device,
    dtype *init_coord,       // npred * 3 dtypes
    dtype *pocket,           // npocket * 3 dtypes
    dtype *pred_cross_dist,  // npred * npocket dtypes
    dtype *pred_holo_dist,   // npred * npred dtypes
    int *torsions,           // ntorsion * 2 ints
    uint8_t *masks,          // npred * ntorsion masks
    int npred, int npocket, int ntorsion);
using dtypesp = std::shared_ptr<dtype>;
extern std::shared_ptr<Request> create_lbfgsb_dock_request(dtypesp &init_values, dtypesp &init_coord,       // npred * 3 dtypes
                    dtypesp &pocket,           // npocket * 3 dtypes
                    dtypesp &pred_cross_dist,  // npred * npocket dtypes
                    dtypesp &pred_holo_dist,   // npred * npred dtypes
                    std::shared_ptr<int> &torsions,           // ntorsion * 2 ints
                    std::shared_ptr<uint8_t> &masks,          // npred * ntorsion masks
                    int nval,
                    int npred, int npocket, int ntorsion, dtype eps, void *userdata
);
}

#endif