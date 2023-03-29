
#include "cases.h"
#include "dock.h"
#include "context.h"
#include <math.h>
#include <iostream>

#include <assert.h>
int cuda_device_id_ = 6;
int main() {
    auto ctx  = dock::createCudaContext(cuda_device_id_);
    assert(ctx);
    float ret = 0;
    auto req   = dock::createCudaDockRequest(init_coord,
                                           pocket,
                                           pred_cross_dist,
                                           pred_holo_dist,
                                           values,
                                           torsions,
                                           masks,
                                           NR_PRED,
                                           NR_POCKET,
                                           sizeof(values) / sizeof(values[0]),
                                           sizeof(torsions) / (2 * sizeof(torsions[0])),
                                           &ret);

    ctx->commit(req);
    auto diff = abs(loss - ret);
    std::cout << "Expect " << loss << " Got " << ret << " Diff " << diff << std::endl;
    return 0;
}
int main2() {
    auto ctx  = dock::createCudaContext(cuda_device_id_);
    assert(ctx);
    auto req  = dock::createCudaDockGradPerfRequest(init_coord,
                                                   pocket,
                                                   pred_cross_dist,
                                                   pred_holo_dist,
                                                   values,
                                                   torsions,
                                                   masks,
                                                   NR_PRED,
                                                   NR_POCKET,
                                                   sizeof(values) / sizeof(values[0]),
                                                   sizeof(torsions) / (2 * sizeof(torsions[0])),
                                                   100000);

    ctx->commit(req);
    std::cout << "QPS: " << req->getProp("qps") << std::endl;
    return 0;
}
int main3() {
    auto ctx  = dock::createCudaContext(cuda_device_id_);
    assert(ctx);
    int nval = sizeof(values) / sizeof(values[0]);
    auto session  = dock::createCudaDockGradSessionRequest(init_coord,
                                                   pocket,
                                                   pred_cross_dist,
                                                   pred_holo_dist,
                                                   torsions,
                                                   masks,
                                                   NR_PRED,
                                                   NR_POCKET,
                                                   nval,
                                                   sizeof(torsions) / (2 * sizeof(torsions[0])),
                                                   0.05);

    ctx->commit(session);
    float *losses = new float[nval+1];
    auto req = dock::createCudaDockGradSubmitRequest(session, values, losses);
    ctx->commit(req);
    for (auto i = 0; i < nval+1; i++) {
        std::cout << i << ": " << losses[i] << std::endl;
    }
    delete[] losses;

    return 0;
}