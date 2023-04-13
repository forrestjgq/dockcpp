#include <assert.h>
#include <iostream>
#include <map>

#include <cuda_runtime.h>

#include "cu/dockcu.h"
#include "cuda_context.h"
#include "culbfgsb/culbfgsb.h"
#include "dtype.h"
#include "mempool.h"
#include "optimizer.h"
#include "utils.h"

namespace dock {
void checkGpuMem(std::string file, int line) {
    #if 0
    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    total /= (1024*1024);
    avail /= (1024*1024);
    used /= (1024*1024);
    std::cout << file << ":" << line << " total " << total << " used " << used << " free " << avail << std::endl;
    #endif
}

#define CHECK_GPU_MEM() checkGpuMem(__FILE__, __LINE__)

#define OFFSETPTR(p, n) ((void *)(((uint8_t *)(p)) + (n)))
void dock_grad_session(dtype *&init_coord,       // npred * 3 dtypes
                       dtype *&pocket,           // npocket * 3 dtypes
                       dtype *&pred_cross_dist,  // npred * npocket dtypes
                       dtype *&pred_holo_dist,   // npred * npred dtypes
                       int *&torsions,           // ntorsion * 2 ints
                       uint8_t *&masks,          // npred * ntorsion masks
                       int npred, int npocket, int nval, int ntorsion,
                       void *host,  // cuda host memory
                       int &hostsz, void *device, int &cudasz, int &hdrsz, cudaStream_t stream);


class LBFGSBRequest : public Request {
public:
    explicit LBFGSBRequest(dtypesp &values, int nval, void *userdata) {
        init_values_ = values;
        best_values_ = new dtype[nval];
        memcpy(best_values_, values.get(), sizeof(dtype) * nval);
        nval_ = nval;
        user_data_ = userdata;
    }
    ~LBFGSBRequest() override {
        if (best_values_) {
            delete[] best_values_;
        }
    }

    void setCudaHandlers(MemPool *pool, MemPool *mapPool, cublasContext* cublas) {
        cublas_ = cublas;
        cuda_mem_ = pool;
        host_map_mem_ = mapPool;
    }
    void run(Context *ctx) override {
        auto cudaCtx = dynamic_cast<CudaContext *>(ctx);
        if (!cudaCtx) {
            done(false);
            return;
        }
        cuda_mem_->reset();
        host_map_mem_->reset();

        LBFGSB_CUDA_STATE<dtype> state;
        state.m_pool.m_stream = cudaCtx->stream();
        state.m_cublas_handle = cublas_;
        state.m_cuda_mem = cuda_mem_;
        state.m_host_mem = host_map_mem_;
        state.m_funcgrad_callback = [=](dtype *x, dtype &f, dtype *g, const cudaStream_t &stream,
                                        const LBFGSB_CUDA_SUMMARY<dtype> &summary) {
            return this->lbfgsb_callback(x, f, g, stream);
        };

        LBFGSB_CUDA_OPTION<dtype> lbfgsb_options;

        lbfgsbcuda::lbfgsbdefaultoption<dtype>(lbfgsb_options);
        lbfgsb_options.mode          = LCM_CUDA;
        lbfgsb_options.eps_f         = static_cast<dtype>(1e-6);
        lbfgsb_options.eps_g         = static_cast<dtype>(1e-6);
        lbfgsb_options.eps_x         = static_cast<dtype>(1e-6);
        lbfgsb_options.max_iteration = 1000;

        LBFGSB_CUDA_SUMMARY<dtype> summary;
        memset(&summary, 0, sizeof(summary));

        //
        int total = nval_ * 6 + 1;
        dtype *tmp = alloc<dtype>(total);
        cudaMemsetAsync(tmp, 0, total * sizeof(dtype), cudaCtx->stream());
        x_ = tmp, tmp += nval_;
        g_ = tmp, tmp += nval_;
        xl_ = tmp, tmp += nval_;
        xu_ = tmp, tmp += nval_;
        losses_ = tmp, tmp += nval_+1;
        nbd_ = (int *)tmp, tmp += nval_; // treat as dtype even for int
        cudaMemcpyAsync(x_, init_values_.get(), sizeof(dtype) * nval_, cudaMemcpyHostToDevice, cudaCtx->stream());

        prepare(cudaCtx->stream(), cuda_mem_);
        lbfgsbcuda::lbfgsbminimize<dtype>(nval_, state, lbfgsb_options, x_, nbd_, xl_, xu_, summary);
        after(cudaCtx->stream(), cuda_mem_);
        done(true);
    }

    int valSize() {
        return nval_;
    }
    void *userdata() {
        return user_data_;
    }
    dtype * takeValues() {
        auto v = best_values_;
        best_values_ = nullptr;
        return v;
    }
    dtype loss() {
        return best_loss_;
    }
    void setId(std::uint64_t id) {
        id_ = id;
    }
    std::uint64_t getId() {
        return id_;
    }
protected:
    virtual void prepare(cudaStream_t stream, MemPool *cudaMem) { }
    virtual void after(cudaStream_t stream, MemPool *cudaMem) { }
    void report(dtype loss, dtype *vals) {
        if (loss < best_loss_) {
            best_loss_ = loss;
            memcpy(best_values_, vals, sizeof(dtype) * nval_);
        }
    }
    template <typename T>
    T * alloc(int n) {
        return (T *)cuda_mem_->alloc(n * sizeof(T));
    }
    virtual int lbfgsb_callback(dtype *x, dtype &f, dtype *g, const cudaStream_t &stream) {
        return 1;
    }
    void done(bool ok) {
        if (!ok) {
            std::cerr << "request failed" << std::endl;
            result_ = RequestResult::Fail;
        } else {
            result_ = RequestResult::Success;
        }
    }

protected:
    dtype best_loss_ = 10000000;
    dtype *best_values_ = nullptr;

    int nval_           = 0;
    dtypesp init_values_;
    void *user_data_ = nullptr;

    std::uint64_t id_ = 0;

    dtype *x_               = nullptr;
    dtype *g_               = nullptr;
    dtype *xl_              = nullptr;
    dtype *xu_              = nullptr;
    int *nbd_               = nullptr;
    dtype *losses_          = nullptr;

    cublasContext *cublas_ = nullptr;
    MemPool *cuda_mem_     = nullptr;
    MemPool *host_map_mem_ = nullptr;
};

class LBFGSBReceiver {
public:
    LBFGSBReceiver()                        = default;
    virtual ~LBFGSBReceiver()               = default;
    virtual void notify(int id, LBFGSBRequest *req) = 0;
};
class LBFGSB : public Optimizer {
public:
    explicit LBFGSB(int id) {
        id_ = id;
    }
    ~LBFGSB() {
        if (ctx_) {
            auto deinitReq = std::make_shared<CallableRequest>([=](Context *_) {
                this->deinit();
            });
            ctx_->commit(deinitReq);
            ctx_.reset();
        }
    }
    int id() {
        return id_;
    }
    bool create(int deviceId, LBFGSBReceiver *receiver) {
        if (ctx_) {
            std::cerr << "duplicate cuda context creation" << std::endl;
            return false;
        }
        ctx_ = std::make_shared<CudaContext>(deviceId);
        auto ret = ctx_->create();
        if (!ret) {
            ctx_.reset();
            std::cerr << "cuda context create failure" << std::endl;
            return false;
        }

        bool ok = true;
        bool *pok = &ok;
        auto initReq = std::make_shared<CallableRequest>([=](Context *_) {
            *pok = this->init();
        });
        ctx_->commit(initReq);
        receiver_ = receiver;
        return ok;
    }

    int post(std::shared_ptr<Request> request) override {
        auto lbreq = std::dynamic_pointer_cast<LBFGSBRequest>(request);
        if (!lbreq) {
            std::cerr << "LBFGSB optmizer requires LBFGSBRequest" << std::endl;
            return 1;
        }
        lbreq->setCudaHandlers(cuda_mem_.get(), host_map_mem_.get(), cublas_);
        lbreq->setCallback([this](Request *r){
            auto req = (LBFGSBRequest *)r;
            assert(r->ok());
            this->receiver_->notify(id_, req);
        });
        ctx_->commit(request);
        cnt_++;
        return 0;
    }
    size_t count() {
        return cnt_;
    }

protected:
    bool init() {
        cublasContext* handle;
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != 0) {
            std::cerr << "cublas handle create fail: " << stat << std::endl;
            return false;
        }
        cublas_ = handle;
        cuda_mem_ = std::make_shared<MemPool>(
            [](int sz, void **dp) {
                void *p  = nullptr;
                auto err = cudaMalloc(&p, sz);
                if (err != cudaSuccess) {
                    return (void *)nullptr;
                }
                if (dp) *dp = p;
                return p;
            },
            [](void *p, void *dp) {
                if (p) {
                    cudaFree(p);
                }
            },
            CUDA_BLK_SIZE_, CUDA_ALIGN_);
        host_map_mem_ = std::make_shared<MemPool>(
            [](int sz, void **dp) {
                void *p  = nullptr;
                auto err = cudaHostAlloc(&p, sz, cudaHostAllocMapped);
                if (err != cudaSuccess) {
                    return (void *)nullptr;
                }
                if (dp) {
                    cudaHostGetDevicePointer((void**)dp, p, 0);
                }
                return p;
            },
            [](void *p, void *dp) {
                if (p) {
                    cudaFreeHost(p);
                }
            },
            CUDA_HOST_BLK_SIZE_, CUDA_ALIGN_);
        return true;
    }
    void deinit() {
        if (cublas_) {
            auto err = cublasDestroy(cublas_);
            if (err != 0) {
                std::cerr << "cublas destroy failed: " << err << std::endl;
            }
        }
        cuda_mem_.reset();
        host_map_mem_.reset();
    }

public:
    int id_ = 0;
    size_t cnt_ = 0;
    LBFGSBReceiver *receiver_ = nullptr;
    cublasContext* cublas_ = nullptr;
    std::shared_ptr<CudaContext> ctx_;
    std::shared_ptr<MemPool> cuda_mem_;
    std::shared_ptr<MemPool> host_map_mem_;
    const int CUDA_BLK_SIZE_      = 2 * 1024 * 1024;
    const int CUDA_HOST_BLK_SIZE_ = 4 * 1024;
    const int CUDA_ALIGN_         = sizeof(double);
};
class LBFGSBServer : public LBFGSBReceiver, public OptimizerServer {
protected:
    struct Result {
        Result() {}
        ~Result() {
            if (values_) {
                delete[] values_;
            }
        }
        std::uint64_t id_ = 0;
        int nval_ = 0;
        dtype *values_ = nullptr;
        dtype loss_ = 0;
        bool success_ = true;
    };
public:
    LBFGSBServer() {
    }
    ~LBFGSBServer() {
        for (auto it = optimizers_.begin(); it != optimizers_.end(); ++it) {
            std::cout << "op " << it->first << ": " << it->second->count() << std::endl;
        }
    }
    int create(int n, int deviceId) {
        int cnt = 0;
        for (int i = 1; i <= n; i ++) {
            auto op = std::make_shared<LBFGSB>(i);
            auto ret = op->create(deviceId, this);
            if (ret) {
                cnt++;
                optimizers_[i] = op;
                running_[i] = 0;
            } else {
                return cnt;
            }
        }
        return cnt;
    }
    void notify(int id, LBFGSBRequest *req) override {
        auto r = std::make_shared<Result>();
        r->id_ = req->getId();
        if (req->ok()) {
            r->values_ = req->takeValues();
            r->loss_ = req->loss();
            r->success_ = true;
            r->nval_ = req->valSize();
        } else {
            std::cerr << "optimizer " << id << " notify response failure for request " << r->id_ << std::endl;
            r->success_ = false;
        }
        {
            std::unique_lock<std::mutex> lock(mt_);
            running_[id]--;
        }
        q_.push(r);
    }
    bool submit(std::shared_ptr<Request> req, std::uint64_t &seq) override {
        auto lbreq = std::dynamic_pointer_cast<LBFGSBRequest>(req);
        if (!lbreq) {
            return false;
        }
        int id = 0;
        std::uint64_t cnt = 9999999999;
        {
            std::unique_lock<std::mutex> lock(mt_);
            for (auto it = running_.begin(); it != running_.end(); ++it) {
                if (it->second < cnt) {
                    cnt = it->second;
                    id = it->first;
                }
            }
        }
        if (id == 0) {
            return false;
        }
        auto ret = optimizers_[id]->post(req); 
        if (ret == 0) {
            std::unique_lock<std::mutex> lock(mt_);
            running_[id]++;
            seq = ++seq_;
            lbreq->setId(seq);
            return true;
        }
        return false;
    }
    bool poll(std::uint64_t &id, dtype &loss, dtype *&values, int &nval) override {
        auto r = q_.pop();
        id = r->id_;
        if (r->success_) {
            // move to skip copy
            values = r->values_;
            r->values_ = nullptr;
            loss = r->loss_;
            nval = r->nval_;
            return true;
        }
        return false;
    }
    


protected:
    std::map<int, std::shared_ptr<LBFGSB>> optimizers_;
    std::map<int, std::uint64_t> running_;
    std::uint64_t seq_ = 0;
    std::mutex mt_;

    BlockQueue<std::shared_ptr<Result>> q_;
};
class LBFGSBDockRequest : public LBFGSBRequest {
public:
    LBFGSBDockRequest(dtypesp &values, int nval, void *userdata,
                      dtypesp &init_coord,       // npred * 3 dtypes
                      dtypesp &pocket,           // npocket * 3 dtypes
                      dtypesp &pred_cross_dist,  // npred * npocket dtypes
                      dtypesp &pred_holo_dist,   // npred * npred dtypes
                      std::shared_ptr<int> &torsions,           // ntorsion * 2 ints
                      std::shared_ptr<uint8_t> &masks,          // npred * ntorsion masks
                      int npred, int npocket, int ntorsion, dtype eps)
        : LBFGSBRequest(values, nval, userdata) {
        init_coord_      = init_coord;
        pocket_          = pocket;
        pred_cross_dist_ = pred_cross_dist;
        pred_holo_dist_  = pred_holo_dist;
        torsions_        = torsions;
        masks_           = masks;
        npred_           = npred;
        npocket_         = npocket;
        ntorsion_        = ntorsion;
        eps_             = eps;
    }
    ~LBFGSBDockRequest() override  {
        if (host_) {
            free(host_);
        }
    }

protected:
    void after(cudaStream_t stream, MemPool *cudaMem) override {
        auto tmp = new dtype[nval_+1];
        cudaMemcpy(tmp, cuda_best_loss_, sizeof(dtype)*(nval_+1), cudaMemcpyDeviceToHost);
        report(tmp[0], tmp+1);
        delete[] tmp;
    }
    void prepare(cudaStream_t stream, MemPool *cudaMem) override {
        dtype bestLoss = 99999999;
        cuda_best_loss_ = (dtype *)cudaMem->alloc(sizeof(dtype) * (nval_+1));
        cuda_best_values_ = cuda_best_loss_ + 1;
        cudaMemcpyAsync(cuda_best_loss_, &bestLoss, sizeof(dtype), cudaMemcpyHostToDevice, stream);

        cuda_init_coord_ = init_coord_.get();
        cuda_pocket_ = pocket_.get();
        cuda_pred_cross_dist_ = pred_cross_dist_.get();
        cuda_pred_holo_dist_ = pred_holo_dist_.get();
        cuda_torsions_ = torsions_.get();
        cuda_masks_ = masks_.get();
        dock_grad_session(cuda_init_coord_, cuda_pocket_, cuda_pred_cross_dist_, cuda_pred_holo_dist_, cuda_torsions_,
                          cuda_masks_, npred_, npocket_, nval_, ntorsion_, nullptr, host_size_, nullptr,
                          cuda_size_, hdr_size_, stream);
        device_ = cudaMem->alloc(cuda_size_);
        assert(device_ != nullptr);
        uint8_t *tmp = (uint8_t *)device_;

        host_size_ += (nval_ + 1) * sizeof(dtype);
        host_            = malloc(host_size_);
        uint8_t *hosttmp = (uint8_t *)host_;
        outlosses_       = (dtype *)hosttmp, hosttmp += (nval_ + 1) * sizeof(dtype);

        int cudasz = cuda_size_;
        int hostsz = host_size_ - (nval_ + 1) * sizeof(dtype);
        dock_grad_session(cuda_init_coord_, cuda_pocket_, cuda_pred_cross_dist_, cuda_pred_holo_dist_, cuda_torsions_,
                          cuda_masks_, npred_, npocket_, nval_, ntorsion_, hosttmp, hostsz, tmp, cudasz,
                          hdr_size_, stream);
        grad_device_ = tmp + hdr_size_;
        grad_size_ = cudasz - hdr_size_;
    }
    int lbfgsb_callback(dtype *x, dtype &f, dtype *g, const cudaStream_t &stream) override {
        int smsize  = 0;
        dock_grad_gpu(cuda_init_coord_, cuda_pocket_, cuda_pred_cross_dist_, cuda_pred_holo_dist_, x, cuda_torsions_, cuda_masks_,
                      npred_, npocket_, nval_, ntorsion_, losses_, (dtype *)grad_device_, grad_size_, stream,
                      smsize, eps_);

        cudaMemcpyAsync(&f, losses_, sizeof(dtype), cudaMemcpyDeviceToHost, stream);
        collect_best_dock(losses_, x, g, cuda_best_loss_, cuda_best_values_, nval_, eps_, stream);
        auto err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "cb memcpy h2d failure: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        return 0;
    }

protected:
    dtypesp init_coord_;
    dtypesp pocket_;
    dtypesp pred_cross_dist_;
    dtypesp pred_holo_dist_;
    std::shared_ptr<int> torsions_;
    std::shared_ptr<uint8_t> masks_;

    dtype *cuda_init_coord_      = nullptr;
    dtype *cuda_pocket_          = nullptr;
    dtype *cuda_pred_cross_dist_ = nullptr;
    dtype *cuda_pred_holo_dist_  = nullptr;
    int *cuda_torsions_          = nullptr;
    uint8_t *cuda_masks_         = nullptr;

    dtype *cuda_best_values_ = nullptr;
    dtype *cuda_best_loss_   = nullptr;

    dtype *x_               = nullptr;
    dtype *g_               = nullptr;
    dtype *xl_              = nullptr;
    dtype *xu_              = nullptr;
    int *nbd_               = nullptr;
    dtype *outlosses_       = nullptr;  // host
    int npred_;
    int npocket_;
    int ntorsion_;
    dtype eps_ = 0.01;
    void *grad_device_ = nullptr;
    void *device_      = nullptr;
    void *host_        = nullptr;
    int hdr_size_      = 0;
    int cuda_size_     = 0;
    int host_size_     = 0;
    int grad_size_     = 0;
};
class LBFGSBDock : public Optimizer {
public:
    explicit LBFGSBDock(int device,
                        dtype *init_coord,       // npred * 3 dtypes
                        dtype *pocket,           // npocket * 3 dtypes
                        dtype *pred_cross_dist,  // npred * npocket dtypes
                        dtype *pred_holo_dist,   // npred * npred dtypes
                        int *torsions,           // ntorsion * 2 ints
                        uint8_t *masks,          // npred * ntorsion masks
                        int npred, int npocket, int ntorsion

    ) {
        init_coord_      = init_coord;
        pocket_          = pocket;
        pred_cross_dist_ = pred_cross_dist;
        pred_holo_dist_  = pred_holo_dist;
        torsions_        = torsions;
        masks_           = masks;
        npred_           = npred;
        npocket_         = npocket;
        nval_            = 0;
        ntorsion_        = ntorsion;
        eps_             = 0.01;

        // std::cout << "npred " << npred << " npocket " << npocket << " ntorsion " << ntorsion << std::endl;
        cudaSetDevice(device);
        CHECK_GPU_MEM();
    }
    ~LBFGSBDock() {
        if (device_) {
            auto err = cudaFree(device_);
            if (err != cudaSuccess) {
                printf("free %d failed\n", cuda_size_);
            }

        }
        if (host_) {
            free(host_);
        }
        if (tmp_values_) {
            delete[] tmp_values_;
        }
        CHECK_GPU_MEM();
    }

    int run(dtype *init_values, dtype *out, dtype *best_loss, dtype eps, int nval) override {
        CHECK_GPU_MEM();
        assert(sizeof(dtype) == 8);
        eps_ = eps;
        nval_ = nval;
        // std::cout << "nval " << nval << " eps " << eps << std::endl;

        prepare(init_values, nullptr);
        CHECK_GPU_MEM();

        LBFGSB_CUDA_OPTION<dtype> lbfgsb_options;

        lbfgsbcuda::lbfgsbdefaultoption<dtype>(lbfgsb_options);
        lbfgsb_options.mode          = LCM_CUDA;
        lbfgsb_options.eps_f         = static_cast<dtype>(1e-8);
        lbfgsb_options.eps_g         = static_cast<dtype>(1e-8);
        lbfgsb_options.eps_x         = static_cast<dtype>(1e-8);
        lbfgsb_options.max_iteration = 1000;

        LBFGSB_CUDA_STATE<dtype> state;
        int k = 0;
        cublasContext* handle;
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != 0) {
            std::cerr << "cublas handle create fail: " << stat << std::endl;
            return 1;
        }
        state.m_cublas_handle = handle;
        std::shared_ptr<int> to_free_(&k, [=](void *){
            auto err = cublasDestroy(handle);
            if (err != 0) {
                std::cerr << "cublas destroy failed: " << err << std::endl;
            }
            CHECK_GPU_MEM();
        });
        cudaStream_t stream;
        auto err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            std::cerr << "create cuda stream fail: " << cudaGetErrorString(err) << std::endl;
            return 2;
        }
        // todo
        state.m_pool.m_stream = stream;
        std::shared_ptr<int> to_free_1_(&k, [=](void *){
            auto err = cudaStreamDestroy(stream);
            if (err != 0) {
                std::cerr << "cublas destroy failed: " << err << std::endl;
            }
            CHECK_GPU_MEM();
        });


        *best_loss = 1000000;

        // auto cb = std::bind(&LBFGSB::lbfgsb_callback, this);
        state.m_funcgrad_callback = [=](dtype *x, dtype &f, dtype *g, const cudaStream_t &stream,
                                        const LBFGSB_CUDA_SUMMARY<dtype> &summary) {
            auto ret = this->lbfgsb_callback(x, f, g, stream, summary);
            if (f < *best_loss) {
                *best_loss = f;
                cudaMemcpy(out, x_, nval_*sizeof(dtype), cudaMemcpyDeviceToHost);
                // printf("update best loss %f\n", *best_loss);
                // printf("update best values %f %f %f %f %f %f %f\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6]);
            }
            return ret;
        };

        LBFGSB_CUDA_SUMMARY<dtype> summary;
        memset(&summary, 0, sizeof(summary));

        CHECK_GPU_MEM();
        lbfgsbcuda::lbfgsbminimize<dtype>(nval_, state, lbfgsb_options, x_, nbd_, xl_, xu_,
                                          summary);
        CHECK_GPU_MEM();
        return 0;
    }

protected:
    int lbfgsb_callback(dtype *x, dtype &f, dtype *g, const cudaStream_t &stream,
                        const LBFGSB_CUDA_SUMMARY<dtype> &summary) {
        int smsize  = 0;
        cudaMemcpyAsync(tmp_values_, x, nval_ * sizeof(dtype), cudaMemcpyDeviceToHost, stream);
        dock_grad_gpu(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_, x, torsions_, masks_,
                      npred_, npocket_, nval_, ntorsion_, losses_, (dtype *)grad_device_, grad_size_, stream,
                      smsize, eps_);

        cudaMemcpyAsync(outlosses_, losses_, (nval_ + 1) * sizeof(dtype), cudaMemcpyDeviceToHost,
                        stream);
        auto err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "cb grad gpu + memcpy d2h failure: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        f = outlosses_[0];
        #if 0
        auto p = tmp_values_;
        printf("\nval %f %f %f %f %f %f %f\n",  p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
        p = outlosses_;
        printf("losses %f %f %f %f %f %f %f %f\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
        #endif
        // auto p = outlosses_;
        // printf("losses %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", p[0], p[1], p[2], p[3],
        //        p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13]);
        // p = input;
        // printf("val %f %f %f %f %f %f %f %f %f %f %f %f %f\n",  p[0], p[1], p[2], p[3],
        //        p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12]);
        for (auto i = 1; i < nval_ + 1; i++) {
            outlosses_[i] = (outlosses_[i] - outlosses_[0]) / eps_;
        }
        #if 0
        printf("grads %f %f %f %f %f %f %f %f\n\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
        #endif
        cudaMemcpyAsync(g, &outlosses_[1], nval_ * sizeof(dtype), cudaMemcpyHostToDevice, stream);
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "cb memcpy h2d failure: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        return 0;
    }
    template <typename T>
    T * alloc(uint8_t *&dev, int n, int align) {
        int sz = n * sizeof(T);
        T * ret = (T *)dev;
        dev += (sz + align -  1) / align * align;
        return ret;
    }
    int prepare(dtype *init_values, cudaStream_t stream) {
        tmp_values_ = new dtype[nval_];
        dock_grad_session(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_, torsions_,
                          masks_, npred_, npocket_, nval_, ntorsion_, nullptr, host_size_, nullptr,
                          cuda_size_, hdr_size_, stream);

        // for x_, g_, xl_ , xu_ , losses_, nbd_
        int sz = nval_ * sizeof(dtype) * 5 + (nval_ + 1) * sizeof(dtype);
        // std::cout << "cuda size " << cuda_size_ << " sz " << sz << std::endl;
        cuda_size_ += sz;

        // printf("malloc %d\n", cuda_size_);
        auto err = cudaMalloc(&device_, cuda_size_);
        if (err != cudaSuccess) {
            std::cerr << "malloc cuda memory fail, size " << cuda_size_ << ", err: " << err
                      << ", msg: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        uint8_t *tmp = (uint8_t *)device_;
        int align    = sizeof(dtype);
        x_           = alloc<dtype>(tmp, nval_, align);
        g_           = alloc<dtype>(tmp, nval_, align);
        xl_          = alloc<dtype>(tmp, nval_, align);
        xu_          = alloc<dtype>(tmp, nval_, align);
        losses_      = alloc<dtype>(tmp, nval_+1, align);
        nbd_         = alloc<int>(tmp, nval_, align);
        cudaMemsetAsync(device_, 0, sz, stream);
        if (init_values) {
            cudaMemcpyAsync(x_, init_values, sizeof(dtype) * nval_, cudaMemcpyHostToDevice, stream);
        }

        host_size_ += (nval_ + 1) * sizeof(dtype);
        host_            = malloc(host_size_);
        uint8_t *hosttmp = (uint8_t *)host_;
        outlosses_       = (dtype *)hosttmp, hosttmp += (nval_ + 1) * sizeof(dtype);

        int cudasz = cuda_size_ - sz;
        int hostsz = host_size_ - (nval_ + 1) * sizeof(dtype);
        dock_grad_session(init_coord_, pocket_, pred_cross_dist_, pred_holo_dist_, torsions_,
                          masks_, npred_, npocket_, nval_, ntorsion_, hosttmp, hostsz, tmp, cudasz,
                          hdr_size_, stream);
        grad_device_ = tmp + hdr_size_;
        grad_size_ = cudasz - hdr_size_;

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "lbfgsb prepare fail, size " << cuda_size_ << ", err: " << err
                      << ", msg: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        return 0;
    }

public:
    dtype *init_coord_      = nullptr;
    dtype *pocket_          = nullptr;
    dtype *pred_cross_dist_ = nullptr;
    dtype *pred_holo_dist_  = nullptr;
    int *torsions_          = nullptr;
    uint8_t *masks_         = nullptr;
    dtype *tmp_values_      = nullptr;
    dtype *x_               = nullptr;
    dtype *g_               = nullptr;
    dtype *xl_              = nullptr;
    dtype *xu_              = nullptr;
    int *nbd_               = nullptr;
    dtype *losses_          = nullptr;
    dtype *outlosses_       = nullptr;  // host
    int npred_;
    int npocket_;
    int nval_;
    int ntorsion_;
    dtype eps_;
    void *grad_device_  = nullptr;
    void *device_  = nullptr;
    void *host_    = nullptr;
    int hdr_size_  = 0;
    int cuda_size_ = 0;
    int host_size_ = 0;
    int grad_size_ = 0;
};
std::shared_ptr<Optimizer> create_lbfgsb_dock(int device,
                                              dtype *init_coord,       // npred * 3 dtypes
                                              dtype *pocket,           // npocket * 3 dtypes
                                              dtype *pred_cross_dist,  // npred * npocket dtypes
                                              dtype *pred_holo_dist,   // npred * npred dtypes
                                              int *torsions,           // ntorsion * 2 ints
                                              uint8_t *masks,          // npred * ntorsion masks
                                              int npred, int npocket, int ntorsion) {
    return std::make_shared<LBFGSBDock>(device, init_coord, pocket, pred_cross_dist, pred_holo_dist, torsions, masks, npred, npocket, ntorsion);
}

std::shared_ptr<OptimizerServer> create_lbfgsb_server(int device, int n) {
    auto s = std::make_shared<LBFGSBServer>();
    auto ret = s->create(n, device);
    if (ret == 0) {
        return nullptr;
    }
    if (ret < n) {
        std::cerr << "create " << ret << "/" << n << " optimizers" << std::endl;
    }
    return s;
}
std::shared_ptr<Request> create_lbfgsb_dock_request(dtypesp &init_values, dtypesp &init_coord,       // npred * 3 dtypes
                    dtypesp &pocket,           // npocket * 3 dtypes
                    dtypesp &pred_cross_dist,  // npred * npocket dtypes
                    dtypesp &pred_holo_dist,   // npred * npred dtypes
                    std::shared_ptr<int> &torsions,           // ntorsion * 2 ints
                    std::shared_ptr<uint8_t> &masks,          // npred * ntorsion masks
                    int nval,
                    int npred, int npocket, int ntorsion, dtype eps, void *userdata
) {
    return std::make_shared<LBFGSBDockRequest>(init_values, nval, userdata, init_coord, pocket,
                                               pred_cross_dist, pred_holo_dist, torsions, masks,
                                               npred, npocket, ntorsion, eps);
}
};  // namespace dock