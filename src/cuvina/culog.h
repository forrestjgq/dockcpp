#ifndef CULOG_H
#define CULOG_H
#include <cstring>
#include <cstdio>

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
#define CUDBGFL(f, l, fmt, ...) printf("%s:%d " fmt "\n", f, l, __VA_ARGS__)
#define CUVDUMP(hdr, v) CUDBG(hdr ": %f %f %f", v.x, v.y, v.z)
// #define VECVDUMP(hdr, vv) dump_vecv(hdr, vv, fileOf(__FILE__), __LINE__)
#define CUVECPDUMP(hdr, vp) printf("%s:%d " hdr " (%f %f %f) (%f %f %f)\n", fileOf(__FILE__), __LINE__, vp.first.x, vp.first.y, vp.first.z, vp.second.x, vp.second.y, vp.second.z)
// #define FLVDUMP(hdr, vv) dump_flv(hdr, vv, fileOf(__FILE__), __LINE__)
#endif