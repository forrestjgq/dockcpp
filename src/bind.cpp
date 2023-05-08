// #include "pybind11/numpy.h"
// #include "pybind11/stl.h"
// #include "ATen/Tensor.h"
// #include "torch/torch.h"
#include "torch/extension.h"
#include <iostream>
#include "context.h"
#include "dock.h"
#include "optimizer.h"

template <typename T>
using Ptr = std::shared_ptr<T>;
template <typename T, typename... Args>
Ptr<T> MakePtr(Args &&...args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}
#define CLASS(cls, doc)            py::class_<cls, Ptr<cls>>(m, #cls, doc)
#define SUBCLASS(cls, parent, doc) py::class_<cls, parent, Ptr<cls>>(m, #cls, doc)

extern int run_vina(int argc, const char* argv[]);
namespace dock {
extern bool create_vina_server(int device, int nrinst);
extern void destroy_vina_server();

};
namespace cudock {

using Tensor = torch::Tensor;

#if USE_DOUBLE
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
template<typename T>
std::shared_ptr<T> copy_tensor(Tensor &t, int &sz) {
    sz = 1;
    auto szs = t.sizes();
    for (auto i = 0u; i < szs.size(); i++) {
        sz *= szs[i];
    }
    // printf("%s:%d: copy sz %d\n", __FILE__, __LINE__, sz);
    T *out = new T[sz];
    T *p = (T *)t.data_ptr();
    memcpy(out, p, sizeof(T) * sz);
    return std::shared_ptr<T>(out);
}

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
std::tuple<Tensor, float, bool> lbfgsb(int device, Tensor init_coord, Tensor pocket, Tensor pred_cross_dist,
                                       Tensor pred_holo_dist, Tensor torsions, Tensor masks,
                                       Tensor values, float eps) {
    int npred, npocket, itmp, ntorsions, nval;
    auto sp_values = copy_float_tensor(values, nval);
    auto sp_init_coord = copy_float_tensor(init_coord, npred);
    auto sp_pocket = copy_float_tensor(pocket, npocket);
    auto sp_pred_cross_dist = copy_float_tensor(pred_cross_dist, itmp);
    auto sp_pred_holo_dist = copy_float_tensor(pred_holo_dist, itmp);

    npred /= 3;
    npocket /= 3;
    ntorsions = torsions.sizes()[0];

    dtype best = 0;
    auto out = std::shared_ptr<dtype>(new dtype[nval]);
    auto outf = std::shared_ptr<float>(new float[nval]);
    auto o     = dock::create_lbfgsb_dock(device, sp_init_coord.get(), sp_pocket.get(),
                                          sp_pred_cross_dist.get(), sp_pred_holo_dist.get(),
                                          (int *)torsions.data_ptr(), (uint8_t *)masks.data_ptr(),
                                          npred, npocket, ntorsions);

    torch::Tensor t;
    auto ret = o->run(sp_values.get(), out.get(), &best, eps, nval);
    if (ret == 0) {
        auto p = out.get();
        auto pf = outf.get();
        for (auto i = 0; i < nval; i++) {
            pf[i] = (float)p[i];
        }
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        t = torch::from_blob(pf, nval, options).clone();
        return std::tuple<torch::Tensor, float, bool>(t, (float)best, true);
    }
    return std::tuple<torch::Tensor, float, bool>(t, 0, false);
}
dock::RequestSP create_lbfgsb_dock_request(Tensor init_coord, Tensor pocket, Tensor pred_cross_dist,
                                       Tensor pred_holo_dist, Tensor torsions, Tensor masks,
                                       Tensor values, float eps) {
    int npred, npocket, itmp, ntorsions, nval;
    auto sp_values = copy_float_tensor(values, nval);
    auto sp_init_coord = copy_float_tensor(init_coord, npred);
    auto sp_pocket = copy_float_tensor(pocket, npocket);
    auto sp_pred_cross_dist = copy_float_tensor(pred_cross_dist, itmp);
    auto sp_pred_holo_dist = copy_float_tensor(pred_holo_dist, itmp);
    auto sp_masks = copy_tensor<uint8_t>(masks, itmp);
    auto sp_torsions = copy_tensor<int>(torsions, itmp);

    npred /= 3;
    npocket /= 3;
    ntorsions = torsions.sizes()[0];

    return dock::create_lbfgsb_dock_request(sp_values, sp_init_coord, sp_pocket, sp_pred_cross_dist,
                                            sp_pred_holo_dist, sp_torsions, sp_masks, nval, npred,
                                            npocket, ntorsions, eps, nullptr);
}

std::tuple<std::uint64_t, bool> post_lbfgsb_request(
    std::shared_ptr<dock::OptimizerServer> server, std::shared_ptr<dock::Request> request) {
    std::uint64_t seq;
    auto ok = server->submit(request, seq);
    return std::tuple<std::uint64_t, bool>(seq, ok);
}
std::tuple<Tensor, float, std::uint64_t, bool> poll_lbfgsb_response(
    std::shared_ptr<dock::OptimizerServer> server) {
    std::uint64_t seq = 0;
    dtype loss = 0;
    dtype *values = nullptr;
    int nval = 0;
    auto ok = server->poll(seq, loss, values, nval);
    Tensor t;
    if (ok) {
        auto outf = std::shared_ptr<float>(new float[nval]);
        auto pf = outf.get();
        for(int i = 0; i < nval; i++) {
            pf[i] = (float)values[i];
        }
        delete[] values;
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        t = torch::from_blob(pf, nval, options).clone();
    }
    return std::tuple<torch::Tensor, float, std::uint64_t, bool>(t, loss, seq, ok);
}
int run_cuvina(const std::vector<std::string> &args) {
    int argc = args.size()+1;
    const char *argv[argc];
    argv[0] = "cuvina";
    for (auto i = 0u; i < args.size(); i++) {
        argv[i+1] = args[i].c_str();
    }
    return run_vina(argc, argv);
}
PYBIND11_MODULE(cudock, m) {
    CLASS(dock::Context, "Dock context")
      .def(py::init<>());
    CLASS(dock::Request, "Dock request")
      .def(py::init<>());
    CLASS(dock::OptimizerServer, "optimizer server")
      .def(py::init<>());

    m.def("create_cuda_context", &dock::createCudaContext, "create a cuda context on specified device");
    m.def("create_dock_session", &dock_create_session, "create a cuda dock grad session, for subsequent submittion by dock_submit");
    m.def("dock_submit", &dock_submit, "submit a request on a session and get loss value and grads");

        CLASS(dock::Optimizer, "optimizer algs")
      .def(py::init<>())
      .def("run", &dock::Optimizer::run, "run a optimizing session");
    m.def("lbfgsb", &lbfgsb, "run lbfgsb optmizer");
    m.def("vina", &run_cuvina, "run cuda based vina");
    m.def("start_vina", &dock::create_vina_server, "create vina server for CUDA, param(device id, num of instance)");
    m.def("stop_vina", &dock::destroy_vina_server, "destroy vina server");
    m.def("create_lbfgsb_server", &dock::create_lbfgsb_server, "create a server with (cudaDeviceId, nrInstance)");
    m.def("create_lbfgsb_dock_request", &create_lbfgsb_dock_request, "create an lbfgsb dock request for lbfgsb server");
    m.def("post_lbfgsb_request", &post_lbfgsb_request, "send lbfgsb request to server");
    m.def("poll_lbfgsb_response", &poll_lbfgsb_response, "poll one lbfgsb response, block if no response is avaialble");
}
};  // namespace cudock