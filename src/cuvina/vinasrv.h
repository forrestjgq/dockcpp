
#ifndef VINASRV_H
#define VINASRV_H
#include <functional>
#include <cuda_runtime_api.h>

namespace dock {
extern bool submit_vina_server(StreamCallback callback);
extern bool create_vina_server(int device, int nrinst);
extern void destroy_vina_server() ;
};
#endif