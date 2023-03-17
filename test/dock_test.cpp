
#include "cases.h"
#include "dock.h"
#include <math.h>
#include <iostream>

int main1() {
    auto ret  = dock::dock_cpu(init_coord,
                              pocket,
                              pred_cross_dist,
                              pred_holo_dist,
                              values,
                              torsions,
                              masks,
                              NR_PRED,
                              NR_POCKET,
                              sizeof(values) / sizeof(values[0]),
                              sizeof(torsions) / (2 * sizeof(torsions[0])));
    auto diff = abs(loss - ret);
    std::cout << "Expect " << loss << " Got " << ret << " Diff " << diff << std::endl;
    return 0;
}
int main() {
    auto ret = dock::dock_grad_cpu_perf(init_coord,
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
    std::cout << "QPS " << ret << std::endl;
    return 0;
}