// #include "pybind11/numpy.h"
// #include "pybind11/stl.h"
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

std::tuple<float, bool> dock(std::shared_ptr<dock::Context> ctx, Tensor init_coord, Tensor pocket,
                             Tensor pred_cross_dist, Tensor pred_holo_dist, Tensor values,
                             Tensor torsions, Tensor masks) {
    int npred = init_coord.sizes()[0];
    int npocket = pocket.sizes()[0];
    int nval    = values.sizes()[0];
    int ntorsions = torsions.sizes()[0];

    float loss = -1;
    auto req   = dock::createCudaDockRequest(
        (float *)init_coord.data_ptr(), (float *)pocket.data_ptr(),
        (float *)pred_cross_dist.data_ptr(), (float *)pred_holo_dist.data_ptr(),
        (float *)values.data_ptr(), (int *)torsions.data_ptr(), (uint8_t *)masks.data_ptr(), npred,
        npocket, nval, ntorsions, &loss);
    ctx->commit(req);
    auto ok = req->result() == dock::RequestResult::Success;
    return std::tuple<float, bool>(loss, ok);
}
std::shared_ptr<dock::Request> dock_create_session(std::shared_ptr<dock::Context> ctx, Tensor init_coord,
                                   Tensor pocket, Tensor pred_cross_dist, Tensor pred_holo_dist,
                                    Tensor torsions, Tensor masks, float eps, int nval) {
    int npred     = init_coord.sizes()[0];
    int npocket = pocket.sizes()[0];
    int ntorsions = torsions.sizes()[0];

    auto losses = std::shared_ptr<float>(new float[nval + 1]);
    auto req    = dock::createCudaDockGradSessionRequest(
        (float *)init_coord.data_ptr(), (float *)pocket.data_ptr(),
        (float *)pred_cross_dist.data_ptr(), (float *)pred_holo_dist.data_ptr(),
         (int *)torsions.data_ptr(), (uint8_t *)masks.data_ptr(), npred,
        npocket, nval, ntorsions, eps);

    ctx->commit(req);
    if (req->ok()) {
        return req;
    }

    std::cerr << "Session create failed" << std::endl;
    return nullptr;
}
std::tuple<Tensor, bool> dock_submit(std::shared_ptr<dock::Context> ctx,
                                   std::shared_ptr<dock::Request> session, Tensor values, Tensor svds) {
    

    int nval    = values.sizes()[0];
    auto losses = std::shared_ptr<float>(new float[nval + 1]);
    memset(losses.get(), 0, (nval + 1)*sizeof(float));
    auto req    = dock::createCudaDockGradSubmitRequest(session, (float *)values.data_ptr(), losses.get(), (float *)svds.data_ptr());
    ctx->commit(req);

    torch::Tensor t;
    if (req->ok()) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        t = torch::from_blob(losses.get(), { nval + 1 }, options).clone();
        return std::tuple<torch::Tensor, bool>(t, true);
    }
    std::cerr << "Request submit failed" << std::endl;
    return std::tuple<torch::Tensor, bool>(t, false);
}
std::tuple<Tensor, bool> dock_grad(std::shared_ptr<dock::Context> ctx, Tensor init_coord,
                                   Tensor pocket, Tensor pred_cross_dist, Tensor pred_holo_dist,
                                   Tensor values, Tensor torsions, Tensor masks, float eps) {
    int npred     = init_coord.sizes()[0];
    int npocket = pocket.sizes()[0];
    int nval    = values.sizes()[0];
    int ntorsions = torsions.sizes()[0];

    auto losses = std::shared_ptr<float>(new float[nval + 1]);
    auto req    = dock::createCudaDockGradRequest(
        (float *)init_coord.data_ptr(), (float *)pocket.data_ptr(),
        (float *)pred_cross_dist.data_ptr(), (float *)pred_holo_dist.data_ptr(),
        (float *)values.data_ptr(), (int *)torsions.data_ptr(), (uint8_t *)masks.data_ptr(), npred,
        npocket, nval, ntorsions, eps, losses.get());

    ctx->commit(req);

    torch::Tensor t;
    if (req->ok()) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        t = torch::from_blob(losses.get(), { nval + 1 }, options).clone();
        return std::tuple<torch::Tensor, bool>(t, true);
    }
    return std::tuple<torch::Tensor, bool>(t, false);
}

PYBIND11_MODULE(cudock, m) {
    CLASS(dock::Context, "Dock context")
      .def(py::init<>());
    CLASS(dock::Request, "Dock request")
      .def(py::init<>());

    m.def("create_cuda_context", &dock::createCudaContext, "create a cuda context on specified device");
    m.def("create_dock_session", &dock_create_session, "create a cuda dock grad session, for subsequent submittion by dock_submit");
    m.def("dock_submit", &dock_submit, "submit a request on a session and get loss value and grads");
    m.def("dock_grad", &dock_grad, "dock grad calculation, get loss value and grads");
    m.def("dock", &dock, "dock calculation");
}
};  // namespace cudock