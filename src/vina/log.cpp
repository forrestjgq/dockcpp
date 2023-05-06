#include "log.h"

#define SEG(f, l, hdr) do{ printf("\n%s:%d %s\n====================\n", f, l, hdr);}while(0)
const char *getFile(const char *path) {
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
void dump_vecv(const char *s, const vecv& vv, const char* file, int line) {
    if (vv.empty()) {
        DBGFL(file, line, "%s: empty", s);
        return;
    }
    SEG(file, line, s);
    for (auto i = 0u; i < vv.size(); i++) {
        printf("%d: %f %f %f\n", i, vv[i].data[0], vv[i].data[1], vv[i].data[2]);
    }
}
void dump_flv(const char *s, const flv& vv, const char *file, int line) {
    if (vv.empty()) {
        DBGFL(file, line, "%s: empty", s);
        return;
    }
    const size_t sep = 10;
    auto lines = (vv.size() + sep - 1) / sep * sep;
    SEG(file, line, s);
    for (auto i = 0u; i < lines; i++) {
        printf("%d:\t", i);
        auto k = i * sep;
        if(i == lines-1) {
            for (auto j = k; j < vv.size(); j++) {
                printf("%f ", vv[j]);
            }
            printf("\n");
        } else {
            printf("%f %f %f %f %f %f %f %f %f %f\n", 
                vv[k], vv[k+1], vv[k+2], vv[k+3], vv[k+4], 
                vv[k+5], vv[k+6], vv[k+7], vv[k+8], vv[k+9]
            );
        }
    }
}
void dump_vecp(const char *s, const vecp& vv, const char *file, int line) {
    printf("%s:%d %s (%f %f %f) (%f %f %f)\n", file, line, s, 
    vv.first.data[0], vv.first.data[1], vv.first.data[2],
    vv.second.data[0], vv.second.data[1], vv.second.data[2]
    );
}