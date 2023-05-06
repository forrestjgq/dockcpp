
#ifndef VINA_UTIL_H
#define VINA_UTIL_H

#include "cuvina/vinadef.h"

namespace dock {

#define FOR(i, n) for(int i = 0; i < n; i++)
#define FOR_RANGE(i, start, end) for(int i = start; i < end; i++)

#if USE_CUDA_VINA
    #define FORCE_INLINE __forceinline__ __device__
    #define COULD_INLINE __forceinline__ __device__
    #define GLOBAL __global__ 
    #define CU_FOR(i, n) for (int i = threadIdx.x; i < n; i += blockDim.x)
    #define SYNC() __syncthreads()
    #define IS_MAIN_THREAD() (threadIdx.x == 0)
    #define IS_SUB_THREAD() (threadIdx.y == 1)
#else
    #define FORCE_INLINE 
    #define COULD_INLINE
    #define GLOBAL
    #define CU_FOR(i, n) FOR(i, n)
    #define SYNC() 
    #define IS_MAIN_THREAD() (true)
    #define IS_SUB_THREAD() (true)
#endif
    FORCE_INLINE void make_vec(Vec &v, Flt a, Flt b, Flt c) {
        v = make_double3(a, b, c);
    }
    FORCE_INLINE Flt vec_get(const Vec &v, int i) {
        Flt *f = (Flt *)&v;
        return *(f+i);
    }
    FORCE_INLINE void vec_set(Vec &v, int i, Flt val) {
        Flt *f = (Flt *)&v;
        *(f+i) = val;
    }
    FORCE_INLINE void vec_set(Vec &dst, const Vec &src) {
        dst = src;
    }

    FORCE_INLINE void vec_add(const Vec &v1, const Vec &v2, Vec &out) {
        make_vec(out, vec_get(v1, 0) + vec_get(v2, 0), vec_get(v1, 1) + vec_get(v2, 1),vec_get(v1, 2) + vec_get(v2, 2));
    }
    // v1 += v2
    FORCE_INLINE void vec_add(Vec &v1, const Vec &v2) {
        make_vec(v1, vec_get(v1, 0) + vec_get(v2, 0), vec_get(v1, 1) + vec_get(v2, 1),vec_get(v1, 2) + vec_get(v2, 2));
    }
    FORCE_INLINE void vec_add(const Vec &v1, Flt f, Vec &out) {
        make_vec(out, vec_get(v1, 0) + f, vec_get(v1, 1) + f,vec_get(v1, 2) - f);
    }
    FORCE_INLINE Flt vec_sum(const Vec &v) {
        return v.x + v.y + v.z;
    }
    FORCE_INLINE Flt vec_sqr_sum(const Vec &v) {
        return v.x * v.x + v.y *v.y + v.z * v.z;
    }
    FORCE_INLINE void vec_sub(const Vec &v1, const Vec &v2, Vec &out) {
        make_vec(out, vec_get(v1, 0) - vec_get(v2, 0), vec_get(v1, 1) - vec_get(v2, 1),vec_get(v1, 2) - vec_get(v2, 2));
    }
    FORCE_INLINE void vec_sub(Vec &v1, const Vec &v2) {
        make_vec(v1, vec_get(v1, 0) - vec_get(v2, 0), vec_get(v1, 1) - vec_get(v2, 1),vec_get(v1, 2) - vec_get(v2, 2));
    }
    FORCE_INLINE void vec_sub(const Vec &v1, Flt f, Vec &out) {
        make_vec(out, vec_get(v1, 0) - f, vec_get(v1, 1) - f,vec_get(v1, 2) - f);
    }
    FORCE_INLINE void vec_product(const Vec &v1, const Vec &v2, Vec &out) {
        make_vec(out, vec_get(v1, 0) * vec_get(v2, 0), vec_get(v1, 1) * vec_get(v2, 1),vec_get(v1, 2) * vec_get(v2, 2));
    }
    FORCE_INLINE void vec_product(const Vec &v1, Flt f, Vec &out) {
        make_vec(out, vec_get(v1, 0) * f, vec_get(v1, 1) * f,vec_get(v1, 2) * f);
    }
    FORCE_INLINE Flt vec_product_sum(const Vec &v1, const Vec &v2) {
        return vec_get(v1, 0) * vec_get(v2, 0)+ vec_get(v1, 1) * vec_get(v2, 1)+vec_get(v1, 2) * vec_get(v2, 2);
    }
    FORCE_INLINE void cross_product(const Vec &a, const Vec &b, Vec &out) {
        return make_vec(out, vec_get(a, 1) * vec_get(b, 2) - vec_get(a, 2) * vec_get(b, 1),
                        vec_get(a, 2) * vec_get(b, 0) - vec_get(a, 0) * vec_get(b, 2),
                        vec_get(a, 0) * vec_get(b, 1) - vec_get(a, 1) * vec_get(b, 0));
    }
    FORCE_INLINE void vecp_clear(Vecp &p) {
        make_vec(p.first, 0, 0, 0);
        make_vec(p.second, 0, 0, 0);
    }
    FORCE_INLINE void vecp_add(Vecp &p1, const Vecp &p2) {
        vec_add(p1.first, p2.first);
        vec_add(p1.second, p2.second);
    }
    template <typename T>
    FORCE_INLINE T arr3d_get(struct Arr3d<T> *a, int i, int j, int k) {
        return a->data[i + a->dim[0] * (j + a->dim[1] * k)];
    }
    FORCE_INLINE Grid *cache_get_grid(Cache &c, int idx) {
        return &c.grids[idx];
    }

#define FLT_EPSILON       1.19209e-07
#define DBL_EPSILON       2.22045e-16
#define FLT_MAX           3.40282e+38
#define DBL_MAX           1.79769e+308

#define FLT_NOT_MAX(f) ((f) < (FLT_MAX * 0.1))

FORCE_INLINE void curl(Flt& e, Flt& deriv, Flt v) {
	if(e > 0 && FLT_NOT_MAX(v)) { // FIXME authentic_v can be gotten rid of everywhere now
		Flt tmp = (v < FLT_EPSILON) ? 0 : (v / (v + e));
		e *= tmp;
		deriv *= tmp * tmp;
	}
}
FORCE_INLINE void curl(Flt& e, Vec& deriv, Flt v) {
	if(e > 0 && FLT_NOT_MAX(v)) { // FIXME authentic_v can be gotten rid of everywhere now
		Flt tmp = (v < FLT_EPSILON) ? 0 : (v / (v + e));
		e *= tmp;
        tmp *= tmp;
        vec_product(deriv, tmp, deriv);
	}
}

FORCE_INLINE void curl(Flt& e, Flt v) {
	if(e > 0 && FLT_NOT_MAX(v)) {
		Flt tmp = (v < FLT_EPSILON) ? 0 : (v / (v + e));
		e *= tmp;
	}
}
};
#endif