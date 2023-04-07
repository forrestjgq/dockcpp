#include <assert.h>
#include <iostream>

#include <cuda_runtime.h>
#include "cu/dockcu.h"
#include "culbfgsb/culbfgsb.h"
#include "optimizer.h"
#include "dtype.h"

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


class LBFGSB : public Optimizer {
public:
    explicit LBFGSB(dtype *init_coord,       // npred * 3 dtypes
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
        cudaSetDevice(4);
        CHECK_GPU_MEM();
    }
    ~LBFGSB() {
        if (device_) {
            auto err = cudaFree(device_);
            if (err != cudaSuccess) {
                printf("free %d failed\n", cuda_size_);
            }

        }
        if (host_) {
            free(host_);
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
        cublasStatus_t stat = cublasCreate(&(state.m_cublas_handle));
        if (stat != 0) {
            std::cerr << "cublas handle create fail: " << stat << std::endl;
            return 1;
        }
        std::shared_ptr<cublasContext> to_free_(state.m_cublas_handle, [](void *p){
            auto err = cublasDestroy((cublasContext *)p);
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
        dtype input[13];
        cudaMemcpyAsync(input, x, 13 * sizeof(dtype), cudaMemcpyDeviceToHost, stream);
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
        auto p = input;
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
std::shared_ptr<Optimizer> create_lbfgsb_dock(dtype *init_coord,       // npred * 3 dtypes
                    dtype *pocket,           // npocket * 3 dtypes
                    dtype *pred_cross_dist,  // npred * npocket dtypes
                    dtype *pred_holo_dist,   // npred * npred dtypes
                    int *torsions,           // ntorsion * 2 ints
                    uint8_t *masks,          // npred * ntorsion masks
                    int npred, int npocket, int ntorsion
) {
    return std::make_shared<LBFGSB>(init_coord, pocket, pred_cross_dist, pred_holo_dist, torsions, masks, npred, npocket, ntorsion);
}
};  // namespace dock