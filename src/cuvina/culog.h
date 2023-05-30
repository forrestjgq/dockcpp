#ifndef CULOG_H
#define CULOG_H
#include <cstring>
#include <cstdio>
__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}
#define PERFDEBUG 0
#if PERFDEBUG
#define MCUDBG(fmt, ...) do{ printf("%d [%d:%d:%d] [%d:%d:%d]\t" fmt "\n",  __LINE__, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,  __VA_ARGS__);}while(0)
// #define MCUVDUMP(hdr, v) CUDBG(hdr ": %f %f %f", v.x, v.y, v.z)
#define MCUVDUMP(hdr, v) CUDBG(hdr ": %f %f %f", v.d[0], v.d[1], v.d[2])
// #define VECVDUMP(hdr, vv) dump_vecv(hdr, vv, fileOf(__FILE__), __LINE__)
#define MCUVECPDUMP(hdr, vp) do{ printf("%d [%d:%d:%d] [%d:%d:%d]\t" hdr " (%f %f %f) (%f %f %f)\n", \
    __LINE__,blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,  \
    vp.first.d[0], vp.first.d[1], vp.first.d[2], vp.second.d[0], vp.second.d[1], vp.second.d[2]); }while(0)
#else
#define MCUDBG(fmt, ...) 
// #define MCUVDUMP(hdr, v) CUDBG(hdr ": %f %f %f", v.x, v.y, v.z)
#define MCUVDUMP(hdr, v) 
// #define VECVDUMP(hdr, vv) dump_vecv(hdr, vv, fileOf(__FILE__), __LINE__)
#define MCUVECPDUMP(hdr, vp) 
#endif
#define CUDEBUG 0
#if CUDEBUG
#if USE_CUDA_VINA
// extern void dump_vecv(const char *s, const vecv& vv, const char* file, int line);
// extern void dump_flv(const char *s, const flv& vv, const char *file, int line) ;
// extern void dump_vecpv(const char *s, const std::vector<vecp>& vv, const char *file, int line) ;

#define CUDBG(fmt, ...) do{ printf("%d [%d:%d] [%d:%d:%d]\t" fmt "\n",  __LINE__, lane_id(), warp_id(), threadIdx.x, threadIdx.y, threadIdx.z,  __VA_ARGS__);}while(0)
// #define CUVDUMP(hdr, v) CUDBG(hdr ": %f %f %f", v.x, v.y, v.z)
#define CUVDUMP(hdr, v) CUDBG(hdr ": %f %f %f", (v).d[0], (v).d[1], (v).d[2])
// #define VECVDUMP(hdr, vv) dump_vecv(hdr, vv, fileOf(__FILE__), __LINE__)
// #define CUVECPDUMP(hdr, vp) do{ printf("%d [%d:%d:%d] [%d:%d:%d]\t" hdr " (%f %f %f) (%f %f %f)\n", \
//     __LINE__,blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,  \
//     vp.first.x, vp.first.y, vp.first.z, vp.second.x, vp.second.y, vp.second.z); }while(0)
#define CUVECPDUMP(hdr, vp) do{ printf("%d [%d:%d:%d] [%d:%d:%d]\t" hdr " (%f %f %f) (%f %f %f)\n", \
    __LINE__,blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,  \
    vp.first.d[0], vp.first.d[1], vp.first.d[2], vp.second.d[0], vp.second.d[1], vp.second.d[2]); }while(0)
// #define FLVDUMP(hdr, vv) dump_flv(hdr, vv, fileOf(__FILE__), __LINE__)
#else
static inline const char *fileOf(const char *path) {
    auto sz = strlen(path);
    auto p = path + sz;
    while(p != path) {
        if(*p == '/') {
            return p+1;
        }
        p--;
    }
    return path;
}
// extern void dump_vecv(const char *s, const vecv& vv, const char* file, int line);
// extern void dump_flv(const char *s, const flv& vv, const char *file, int line) ;
// extern void dump_vecpv(const char *s, const std::vector<vecp>& vv, const char *file, int line) ;

#define CUDBG(fmt, ...) printf("%s:%d " fmt "\n", fileOf(__FILE__), __LINE__,  __VA_ARGS__)
#define CUVDUMP(hdr, v) CUDBG(hdr ": %f %f %f", v.x, v.y, v.z)
// #define VECVDUMP(hdr, vv) dump_vecv(hdr, vv, fileOf(__FILE__), __LINE__)
#define CUVECPDUMP(hdr, vp) printf("%s:%d " hdr " (%f %f %f) (%f %f %f)\n", fileOf(__FILE__), __LINE__, vp.first.x, vp.first.y, vp.first.z, vp.second.x, vp.second.y, vp.second.z)
// #define FLVDUMP(hdr, vv) dump_flv(hdr, vv, fileOf(__FILE__), __LINE__)
#endif
#else
#define CUDBG(fmt, ...)
#define CUVDUMP(hdr, v)
// #define VECVDUMP(hdr, vv) dump_vecv(hdr, vv, fileOf(__FILE__), __LINE__)
#define CUVECPDUMP(hdr, vp)
// #define FLVDUMP(hdr, vv)
#endif
#endif