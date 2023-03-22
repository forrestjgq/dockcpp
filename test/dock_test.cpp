
#include "cases.h"
#include "dock.h"
#include "context.h"
#include <math.h>
#include <iostream>

#include <assert.h>
int cuda_device_id_ = 0;
int main1() {
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
int main() {
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
                                                   /*100000*/1);

    ctx->commit(req);
    std::cout << "QPS: " << req->getProp("qps") << std::endl;
    return 0;
}