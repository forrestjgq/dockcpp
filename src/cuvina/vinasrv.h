
#ifndef VINASRV_H
#define VINASRV_H
#include <functional>
#include <cuda_runtime_api.h>

namespace dock {
using VinaCallback = std::function<void(cudaStream_t )>;
bool submit_vina_server(VinaCallback callback);
};
#endif