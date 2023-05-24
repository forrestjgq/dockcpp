#ifndef VINA_LOG_H
#define VINA_LOG_H


#include "common.h"

#define VINADEBUG 0
#define CPUDBG(fmt, ...) printf("%d " fmt "\n", __LINE__,  __VA_ARGS__)
#if VINADEBUG
extern void dump_vecv(const char *s, const vecv& vv, const char* file, int line);
extern void dump_flv(const char *s, const flv& vv, const char *file, int line) ;
extern void dump_vecp(const char *s, const vecp& vv, const char *file, int line) ;
extern const char *getFile(const char *path);

#define PRINT(fmt) printf("%s:%d " fmt "\n", getFile(__FILE__), __LINE__)
#define DBG(fmt, ...) printf("%s:%d " fmt "\n", getFile(__FILE__), __LINE__,  __VA_ARGS__)
#define DBGFL(f, l, fmt, ...) printf("%s:%d " fmt "\n", f, l, __VA_ARGS__)
#define VDUMP(hdr, v) DBG(hdr ": %f %f %f", v.data[0], v.data[1], v.data[2])
#define VECVDUMP(hdr, vv) dump_vecv(hdr, vv, getFile(__FILE__), __LINE__)
#define VECPDUMP(hdr, vv) dump_vecp(hdr, vv, getFile(__FILE__), __LINE__)
#define FLVDUMP(hdr, vv) dump_flv(hdr, vv, getFile(__FILE__), __LINE__)
#else
#define PRINT(fmt)
#define DBG(fmt, ...)
#define DBGFL(f, l, fmt, ...)
#define VDUMP(hdr, v)
#define VECVDUMP(hdr, vv)
#define VECPDUMP(hdr, vv)
#define FLVDUMP(hdr, vv)
#endif
#endif