#include "pybind11/numpy.h"
// #include "ATen/Tensor.h"
// #include "torch/torch.h"
#include "torch/extension.h"
#include <iostream>
#include "context.h"
#include "dock.h"

template <typename T>
using Ptr = std::shared_ptr<T>;
template <typename T, typename... Args>
Ptr<T> MakePtr(Args &&...args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}
#define CLASS(cls, doc)            py::class_<cls, Ptr<cls>>(m, #cls, doc)
#define SUBCLASS(cls, parent, doc) py::class_<cls, parent, Ptr<cls>>(m, #cls, doc)

namespace cudock {

using Tensor = torch::Tensor;

float dock(std::shared_ptr<dock::Context> ctx,
                 Tensor init_coord,
                 Tensor pocket,
                 Tensor pred_cross_dist,
                 Tensor pred_holo_dist,
                 Tensor values,
                 Tensor torsions,
                 Tensor masks) {
    int npred = init_coord.sizes()[0];
    int npocket = pocket.sizes()[0];
    int nval    = values.sizes()[0];
    int ntorsions = torsions.sizes()[0];

    float loss = -1;
    auto req = dock::createCudaDockRequest((float *)init_coord.data_ptr(),
                                                   (float *)pocket.data_ptr(),
                                                   (float *)pred_cross_dist.data_ptr(),
                                                   (float *)pred_holo_dist.data_ptr(),
                                                   (float *)values.data_ptr(),
                                                   (int *)torsions.data_ptr(),
                                                   (uint8_t *)masks.data_ptr(),
                                                   npred,
                                                   npocket,
                                                   nval,
                                                   ntorsions,
                                                   &loss);
    ctx->commit(req);
    return loss;
}
Tensor dock_grad(std::shared_ptr<dock::Context> ctx,
                 Tensor init_coord,
                 Tensor pocket,
                 Tensor pred_cross_dist,
                 Tensor pred_holo_dist,
                 Tensor values,
                 Tensor torsions,
                 Tensor masks,
                 float eps) {
    int npred = init_coord.sizes()[0];
    int npocket = pocket.sizes()[0];
    int nval    = values.sizes()[0];
    int ntorsions = torsions.sizes()[0];

    auto losses = std::shared_ptr<float>(new float[nval + 1]);
    auto req    = dock::createCudaDockGradRequest((float *)init_coord.data_ptr(),
                                               (float *)pocket.data_ptr(),
                                               (float *)pred_cross_dist.data_ptr(),
                                               (float *)pred_holo_dist.data_ptr(),
                                               (float *)values.data_ptr(),
                                               (int *)torsions.data_ptr(),
                                               (uint8_t *)masks.data_ptr(),
                                               npred,
                                               npocket,
                                               nval,
                                               ntorsions,
                                               eps,
                                               losses.get());
    ctx->commit(req);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto ploss   = losses.get();
    for (auto i = 1; i < nval + 1; i++) {
        ploss[i] = (ploss[i] - ploss[0]) / eps;
    }
    return torch::from_blob(losses.get(), { nval + 1 }, options).clone();
}

PYBIND11_MODULE(cudock, m) {
    CLASS(dock::Context, "Dock context")
      .def(py::init<>());

    m.def("create_cuda_context", &dock::createCudaContext, "create a cuda context");
    m.def("dock_grad", &dock_grad, "dock grad calculation");
    m.def("dock", &dock, "dock calculation");
}
};  // namespace cudock