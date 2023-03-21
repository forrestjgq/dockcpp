#include "cuda_context.h"


namespace dock {

std::shared_ptr<Context> createCudaContext(int device) {
    auto ctx = std::make_shared<CudaContext>(device);
    if (ctx->create()) {
        return ctx;
    }
    return nullptr;
}

};  // namespace dock