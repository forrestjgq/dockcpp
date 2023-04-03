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

#if USING_DOUBLE
std::shared_ptr<dtype> copy_float_tensor(Tensor &t, int &sz) {
    sz = 1;
    auto szs = t.sizes();
    for (auto i = 0u; i < szs.size(); i++) {
        sz *= szs[i];
    }
    // printf("%s:%d: copy sz %d\n", __FILE__, __LINE__, sz);
    dtype *out = new dtype[sz];
    float *p = (float *)t.data_ptr();
    for (auto i = 0; i < sz; i++) {
        out[i] = (dtype)(p[i]);
    }
    return std::shared_ptr<dtype>(out);
}
#else
std::shared_ptr<dtype> copy_float_tensor(Tensor &t, int &sz) {
    sz = t.sizes()[0];
    dtype *out = new dtype[sz];
    float *p = (float *)t.data_ptr();
    return std::shared_ptr<dtype>(p, [](void *) {});
}
#endif

void dumparr_bind(const char *hdr, int m, int n, const double * p) {
    printf(" ====== %s (%dx%d) ======\n",hdr, m, n);
    for (int i = 0; i < m; i++) {
        printf("\t%d:\t" , i);
        const double *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%f ", p1[j]);
        }
        printf("\n");
    }
    printf("\n");
}
std::shared_ptr<dock::Request> dock_create_session(std::shared_ptr<dock::Context> ctx, Tensor init_coord,
                                   Tensor pocket, Tensor pred_cross_dist, Tensor pred_holo_dist,
                                    Tensor torsions, Tensor masks, float eps, int nval) {
    int npred, npocket, itmp, ntorsions;

    auto sp_init_coord = copy_float_tensor(init_coord, npred);
    auto sp_pocket = copy_float_tensor(pocket, npocket);
    auto sp_pred_cross_dist = copy_float_tensor(pred_cross_dist, itmp);
    auto sp_pred_holo_dist = copy_float_tensor(pred_holo_dist, itmp);

    npred /= 3;
    npocket /= 3;
    ntorsions = torsions.sizes()[0];

    
    // dumparr_bind("init corrd", npred, 3, sp_init_coord.get());

    auto req    = dock::createCudaDockGradSessionRequest(
        sp_init_coord.get(), sp_pocket.get(),
        sp_pred_cross_dist.get(), sp_pred_holo_dist.get(),
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
                                   std::shared_ptr<dock::Request> session, Tensor values) {
    

    int nval;
    auto sp_values = copy_float_tensor(values, nval);
    auto losses = std::shared_ptr<dtype>(new dtype[nval + 1]);
    memset(losses.get(), 0, (nval + 1)*sizeof(dtype));
    auto req    = dock::createCudaDockGradSubmitRequest(session, sp_values.get(), losses.get());
    ctx->commit(req);

    torch::Tensor t;
    if (req->ok()) {
        auto flosses = std::shared_ptr<float>(new float[nval+1]);
        auto pf = flosses.get();
        auto pd = losses.get();
        for (auto i = 0; i < nval+1; i++) {
            pf[i] = (float)pd[i];
        }
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        t = torch::from_blob(pf, { nval + 1 }, options).clone();
        return std::tuple<torch::Tensor, bool>(t, true);
    }
    std::cerr << "Request submit failed" << std::endl;
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
}
};  // namespace cudock