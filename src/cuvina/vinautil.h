
#ifndef VINA_UTIL_H
#define VINA_UTIL_H

#include "cuvina/vinadef.h"
#include "cuvina/culog.h"
#if USE_CUDA_VINA == 0
#include <cmath>
#endif

namespace dock {

#define FOR(i, n) for(int i = 0; i < n; i++)
#define FOR_RANGE(i, start, end) for(int i = start; i < end; i++)

#define SQR(x) ((x) * (x))
#define EVAL_IN_WARP 1
#if USE_CUDA_VINA
    #define THREADID (blockDim.x *blockDim.y * threadIdx.z +blockDim.x * threadIdx.y + threadIdx.x)
    #define BLOCKSZ (blockDim.x * blockDim.y * blockDim.z)
    #define FORCE_INLINE __forceinline__ __device__
    // #define FORCE_INLINE static __device__
    #define COULD_INLINE __forceinline__ __device__
    // #define COULD_INLINE static __device__
    #define GLOBAL __global__ 
    #define CU_FOR(i, n) for (int i = threadIdx.x; i < n; i += blockDim.x)
    #define CU_FORY(i, n) for (int i = threadIdx.y; i < n; i += blockDim.y)
    #define SYNC() __syncthreads()
#if EVAL_IN_WARP
    #define WARPSYNC() 
#else
    #define WARPSYNC() SYNC()
#endif
    #define IS_MAIN_THREAD() (threadIdx.x == 0)
    #define IS_SUB_THREAD() (threadIdx.x == 1)
    #define IS_2DMAIN() (threadIdx.x == 0 && threadIdx.y == 0)
    #define IS_2DSUB() (threadIdx.x == 1 && threadIdx.y == 0)
    #define ZIS(n) (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == n)
    #define XY0() (threadIdx.x == 0 && threadIdx.y == 0)
    #define CU_FOR2(i, n) for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < n; i += blockDim.x * blockDim.y)
    #define CU_FOR3(i, n) for (int i = THREADID; i < n; i += BLOCKSZ)
    #define CU_FORZ(i, n) for (int i = threadIdx.z; i < n; i += blockDim.z)
    #define CU_FORYZ(i, n) for (int i = threadIdx.y + threadIdx.z * blockDim.y; i < n; i += blockDim.y * blockDim.z)
    #define IS_GRID(n) (blockIdx.x == n)
    #define CEIL(x) ceil(x)
    #define SIN(x) sin(x)
    #define COS(x) cos(x)
    #define QTNORM(qt) norm4d(qt.x, qt.y, qt.z, qt.w)
    #define QTRNORM(qt) rnorm4d(qt.x, qt.y, qt.z, qt.w)
    #define ABS(x) abs(x)
    #define SQRT(x) sqrt(x)
    #define RSQRT(x) rsqrt(x)
    #define reciprocal(x) __drcp_rn(x)
#else
    #define FORCE_INLINE inline
    #define COULD_INLINE inline
    #define GLOBAL
    #define CU_FOR(i, n) FOR(i, n)
    #define SYNC() 
    #define IS_MAIN_THREAD() (true)
    #define IS_SUB_THREAD() (true)
    #define IS_GRID(n) (true)
    #define CEIL(x) std::ceil(x)
    #define SIN(x) std::sin(x)
    #define COS(x) std::cos(x)
    #define QTNORM(qt) std::sqrt(SQR(qt.x), SQR(qt.y), SQR(qt.z), SQR(qt.w))
    #define QTRNORM(qt) (1.0/std::sqrt(SQR(qt.x), SQR(qt.y), SQR(qt.z), SQR(qt.w)))
    #define ABS(x) std::abs(x)
    #define SQRT(x) std::sqrt(x)
    #define RSQRT(x) (1.0 / std::sqrt(x))
    #define reciprocal(x) (1.0/x)
#endif

    FORCE_INLINE void make_vec(Vec &v, Flt a, Flt b, Flt c) {
        // v = make_double3(a, b, c);
        v.d[0] = a, v.d[1] = b, v.d[2] = c;
    }
    FORCE_INLINE Flt vec_get(const Vec &v, int i) {
        // Flt *f = (Flt *)&v;
        // return *(f+i);
        return v.d[i];
    }
    FORCE_INLINE void vec_set(Vec &v, int i, Flt val) {
        // Flt *f = (Flt *)&v;
        // *(f+i) = val;
        v.d[i] = val;
    }
    FORCE_INLINE void vec_set(Vec &dst, const Vec &src) {
        dst = src;
    }
    FORCE_INLINE void vec_copy_to(const Vec &src, Flt *dst) {
        // dst[0] = src.x, dst[1] = src.y, dst[2] = src.z;
        dst[0] = src.d[0], dst[1] = src.d[1], dst[2] = src.d[2];
    }
    FORCE_INLINE void vec_copy_to(const int idx, const Vec &src, Flt *dst) {
        if (idx < 3) dst[idx] = src.d[idx];
    }

    FORCE_INLINE void vec_add(const Vec &v1, const Vec &v2, Vec &out) {
        // make_vec(out, vec_get(v1, 0) + vec_get(v2, 0), vec_get(v1, 1) + vec_get(v2, 1),vec_get(v1, 2) + vec_get(v2, 2));
        out.d[0] = v1.d[0] + v2.d[0];
        out.d[1] = v1.d[1] + v2.d[1];
        out.d[2] = v1.d[2] + v2.d[2];
    }
    // v1 += v2
    FORCE_INLINE void vec_add(Vec &v1, const Vec &v2) {
        // make_vec(v1, vec_get(v1, 0) + vec_get(v2, 0), vec_get(v1, 1) + vec_get(v2, 1),vec_get(v1, 2) + vec_get(v2, 2));
        v1.d[0] += v2.d[0];
        v1.d[1] += v2.d[1];
        v1.d[2] += v2.d[2];
    }
    FORCE_INLINE void vec_add_c(int idx, Vec &v1, const Vec &v2) {
        // make_vec(v1, vec_get(v1, 0) + vec_get(v2, 0), vec_get(v1, 1) + vec_get(v2, 1),vec_get(v1, 2) + vec_get(v2, 2));
        if (idx < 3) v1.d[idx] += v2.d[idx];
    }
    FORCE_INLINE void vec_add(const Vec &v1, Flt f, Vec &out) {
        // make_vec(out, vec_get(v1, 0) + f, vec_get(v1, 1) + f,vec_get(v1, 2) - f);
        out.d[0] = v1.d[0] + f;
        out.d[1] = v1.d[1] + f;
        out.d[2] = v1.d[2] + f;
    }
    FORCE_INLINE Flt vec_sum(const Vec &v) {
        return v.d[0] + v.d[1] + v.d[2];
    }
    FORCE_INLINE Flt vec_sqr_sum(const Vec &v) {
        // return v.x * v.x + v.y *v.y + v.z * v.z;
        return v.d[0]*v.d[0] + v.d[1]*v.d[1] + v.d[2]*v.d[2];
    }
    FORCE_INLINE void vec_sub(const Vec &v1, const Vec &v2, Vec &out) {
        // make_vec(out, vec_get(v1, 0) - vec_get(v2, 0), vec_get(v1, 1) - vec_get(v2, 1),vec_get(v1, 2) - vec_get(v2, 2));
        out.d[0] = v1.d[0] - v2.d[0];
        out.d[1] = v1.d[1] - v2.d[1];
        out.d[2] = v1.d[2] - v2.d[2];
    }
    FORCE_INLINE void vec_sub_c(int idx, const Vec &v1, const Vec &v2, Vec &out) {
        // make_vec(out, vec_get(v1, 0) - vec_get(v2, 0), vec_get(v1, 1) - vec_get(v2, 1),vec_get(v1, 2) - vec_get(v2, 2));
        if (idx < 3) {
            out.d[idx] = v1.d[idx] - v2.d[idx];
        }
    }
    FORCE_INLINE void vec_sub_c(int idx, const Vec &v1, const Vec &v2, Flt *out) {
        // make_vec(out, vec_get(v1, 0) - vec_get(v2, 0), vec_get(v1, 1) - vec_get(v2, 1),vec_get(v1, 2) - vec_get(v2, 2));
        if (idx < 3) {
            out[idx] = v1.d[idx] - v2.d[idx];
        }
    }
    FORCE_INLINE void vec_sub(Vec &v1, const Vec &v2) {
        // make_vec(v1, vec_get(v1, 0) - vec_get(v2, 0), vec_get(v1, 1) - vec_get(v2, 1),vec_get(v1, 2) - vec_get(v2, 2));
        v1.d[0] -= v2.d[0];
        v1.d[1] -= v2.d[1];
        v1.d[2] -= v2.d[2];
    }
    FORCE_INLINE void vec_sub_c(int idx, Vec &v1, const Vec &v2) {
        // make_vec(v1, vec_get(v1, 0) - vec_get(v2, 0), vec_get(v1, 1) - vec_get(v2, 1),vec_get(v1, 2) - vec_get(v2, 2));
        if (idx < 3) v1.d[idx] -= v2.d[idx];
    }
    FORCE_INLINE void vec_sub(const Vec &v1, Flt f, Vec &out) {
        // make_vec(out, vec_get(v1, 0) - f, vec_get(v1, 1) - f,vec_get(v1, 2) - f);
        out.d[0] = v1.d[0] - f;
        out.d[1] = v1.d[1] - f;
        out.d[2] = v1.d[2] - f;
    }
    FORCE_INLINE void vec_product(const Vec &v1, const Vec &v2, Vec &out) {
        // make_vec(out, vec_get(v1, 0) * vec_get(v2, 0), vec_get(v1, 1) * vec_get(v2, 1),vec_get(v1, 2) * vec_get(v2, 2));
        out.d[0] = v1.d[0] * v2.d[0];
        out.d[1] = v1.d[1] * v2.d[1];
        out.d[2] = v1.d[2] * v2.d[2];
    }
    FORCE_INLINE void vec_product(const Vec &v1, Flt f, Vec &out) {
        // make_vec(out, vec_get(v1, 0) * f, vec_get(v1, 1) * f,vec_get(v1, 2) * f);
        out.d[0] = v1.d[0] * f;
        out.d[1] = v1.d[1] * f;
        out.d[2] = v1.d[2] * f;
    }
    FORCE_INLINE Flt vec_product_sum(const Vec &v1, const Vec &v2) {
        // return vec_get(v1, 0) * vec_get(v2, 0)+ vec_get(v1, 1) * vec_get(v2, 1)+vec_get(v1, 2) * vec_get(v2, 2);
        return v1.d[0] * v2.d[0]+ v1.d[1] * v2.d[1]+ v1.d[2] * v2.d[2];
    }
    FORCE_INLINE void cross_product(const Vec &a, const Vec &b, Vec &out) {
        // return make_vec(out, vec_get(a, 1) * vec_get(b, 2) - vec_get(a, 2) * vec_get(b, 1),
        //                 vec_get(a, 2) * vec_get(b, 0) - vec_get(a, 0) * vec_get(b, 2),
        //                 vec_get(a, 0) * vec_get(b, 1) - vec_get(a, 1) * vec_get(b, 0));
        out.d[0] = a.d[1] * b.d[2] - a.d[2] * b.d[1];
        out.d[1] = a.d[2] * b.d[0] - a.d[0] * b.d[2];
        out.d[2] = a.d[0] * b.d[1] - a.d[1] * b.d[0];
    }
    FORCE_INLINE void cross_product_c(int idx, const Vec &a, const Vec &b, Vec &out) {
        // return make_vec(out, vec_get(a, 1) * vec_get(b, 2) - vec_get(a, 2) * vec_get(b, 1),
        //                 vec_get(a, 2) * vec_get(b, 0) - vec_get(a, 0) * vec_get(b, 2),
        //                 vec_get(a, 0) * vec_get(b, 1) - vec_get(a, 1) * vec_get(b, 0));
        if (idx < 3) {
            int n1 = idx+1, n2 = idx+2;
            if (n1 > 3) n1-=3;
            if (n2 > 3) n2-=3;
            out.d[idx] = a.d[n1] * b.d[n2] - a.d[n2] * b.d[n1];
        }
    }

    // out += cross_product_c(idx, a, b)
    FORCE_INLINE void cross_product_add_c(int idx, const Flt *a, const Vec &b, Vec &out) {
        // return make_vec(out, vec_get(a, 1) * vec_get(b, 2) - vec_get(a, 2) * vec_get(b, 1),
        //                 vec_get(a, 2) * vec_get(b, 0) - vec_get(a, 0) * vec_get(b, 2),
        //                 vec_get(a, 0) * vec_get(b, 1) - vec_get(a, 1) * vec_get(b, 0));
        if (idx < 3) {
            int n1 = idx+1, n2 = idx+2;
            if (n1 > 3) n1-=3;
            if (n2 > 3) n2-=3;
            out.d[idx] += a[n1] * b.d[n2] - a[n2] * b.d[n1];
        }
    }
    // out += (left-right) x b
    FORCE_INLINE void sub_cross_product_add_c(int idx, const Vec &left, const Vec &right, const Vec &b, Vec &out) {
        // return make_vec(out, vec_get(a, 1) * vec_get(b, 2) - vec_get(a, 2) * vec_get(b, 1),
        //                 vec_get(a, 2) * vec_get(b, 0) - vec_get(a, 0) * vec_get(b, 2),
        //                 vec_get(a, 0) * vec_get(b, 1) - vec_get(a, 1) * vec_get(b, 0));
        if (idx < 3) {
            int n1 = idx+1, n2 = idx+2;
            if (n1 >= 3) n1-=3;
            if (n2 >= 3) n2-=3;
            out.d[idx] += (left.d[n1] - right.d[n1]) * b.d[n2] - (left.d[n2] - right.d[n2]) * b.d[n1];
        }
    }
    FORCE_INLINE void vecp_clear(Vecp &p) {
        make_vec(p.first, 0, 0, 0);
        make_vec(p.second, 0, 0, 0);
    }
    FORCE_INLINE void vecp_add(Vecp &p1, const Vecp &p2) {
        vec_add(p1.first, p2.first);
        vec_add(p1.second, p2.second);
    }
    FORCE_INLINE void qt_set(Qt &dst, const Qt &src) {
        dst = src;
    }
    FORCE_INLINE void qt_set(Qt &dst, Flt x, Flt y, Flt z, Flt w) {
        dst.d[0] = x, dst.d[1] = y, dst.d[2] = z, dst.d[3] = w;
    }
    FORCE_INLINE void qt_set(Qt &dst, int idx, Flt x) {
        dst.d[idx] = x;
    }
    FORCE_INLINE void qt_multiple(Qt &dst, Flt x) {
        dst.d[0] *= x;
        dst.d[1] *= x;
        dst.d[2] *= x;
        dst.d[3] *= x;
    }
    FORCE_INLINE void qt_multiple(Flt *dst, Flt x) {
        dst[0] *= x;
        dst[1] *= x;
        dst[2] *= x;
        dst[3] *= x;
    }
    FORCE_INLINE void qt_multiple(Qt &dst, const Qt &src) {
        Flt xr = src.d[0], yr = src.d[1], zr = src.d[2], wr = src.d[3];
        Flt x = dst.d[0], y = dst.d[1], z = dst.d[2], w = dst.d[3];
        qt_set(dst, x * xr - y * yr - z * zr - w * wr, x * yr + y * xr + z * wr - w * zr,
               x * zr - y * wr + z * xr + w * yr, x * wr + y * zr - z * yr + w * xr);
    }
    FORCE_INLINE void mat_multiple(const Flt *mat, const Vec &v, Vec &out) {
		// make_vec(out,
        //            mat[0]*v.x + mat[3]*v.y + mat[6]*v.z, 
		// 	       mat[1]*v.x + mat[4]*v.y + mat[7]*v.z,
		// 		   mat[2]*v.x + mat[5]*v.y + mat[8]*v.z);
        out.d[0] = mat[0] * v.d[0] + mat[3] * v.d[1] + mat[6] * v.d[2];
        out.d[1] = mat[1] * v.d[0] + mat[4] * v.d[1] + mat[7] * v.d[2];
        out.d[2] = mat[2] * v.d[0] + mat[5] * v.d[1] + mat[8] * v.d[2];
    }
    FORCE_INLINE void mat_multiple_c(int idx, const Flt *mat, const Vec &v, Vec &out) {
        if (idx < 3) {
            out.d[idx] = mat[idx] * v.d[0] + mat[idx+3] * v.d[1] + mat[idx+6] * v.d[2];
        }
    }

    FORCE_INLINE void mat_set(Flt *mat, int i, int j, Flt v) {
        mat[i + 3 *j] = v;
    }
    __device__ int qt_to_mat_map[] = {
        // (aa + bb - cc - dd)
        1, 0, 0, 1, 1, 1, 1, 2, 2, -1, 3, 3, -1, // 0
        // 2 * (ad + bc)
        2, 0, 3, 1, 1, 2, 1,  0, 0, 0, 0, 0, 0, // 1
        // 2 * (-ac + bd)
        2, 0, 2, -1, 1, 3, 1,  0, 0, 0, 0, 0, 0, // 2
        // 2 * (-ad + bc)
        2, 0, 3, -1, 1, 2, 1, 0, 0, 0, 0, 0, 0, // 3
        // (aa - bb + cc - dd)
        1, 0, 0, 1, 1, 1, -1, 2, 2, 1, 3, 3, -1, // 4
        // 2 * (ab + cd)
        2, 0, 1, 1, 2, 3, 1,  0, 0, 0, 0, 0, 0, // 5
        //2 * (ac + bd));
        2, 0, 2, 1, 1, 3, 1,  0, 0, 0, 0, 0, 0, // 6
        // (-ab + cd)
        2, 0, 1, -1, 2, 3, 1,  0, 0, 0, 0, 0, 0, // 7
        // (aa - bb - cc + dd)
        1, 0, 0, 1, 1, 1, -1, 2, 2, -1, 3, 3, 1, // 8
    };
    FORCE_INLINE void qt_to_mat_c(int idx, int blk, const Flt *qt, Flt *mat) {
        while(idx < 9) {
            auto m = qt_to_mat_map + 13 * idx;
            Flt mul = m[0];
            m++;
            Flt sum = 0;
            FOR(i, 4) {
                if (m[2] != 0) sum += qt[m[0]] * qt[m[1]] * ((Flt)m[2]);
                m += 3;
            }
            mat[idx] = mul * sum;
            idx += blk;
        }
    }
    FORCE_INLINE void qt_to_mat(const Flt *qt, Flt *mat) {
        const Flt a = qt[0];
        const Flt b = qt[1];
        const Flt c = qt[2];
        const Flt d = qt[3];

        const Flt aa = a * a;
        const Flt ab = a * b;
        const Flt ac = a * c;
        const Flt ad = a * d;
        const Flt bb = b * b;
        const Flt bc = b * c;
        const Flt bd = b * d;
        const Flt cc = c * c;
        const Flt cd = c * d;
        const Flt dd = d * d;

        mat_set(mat, 0, 0, (aa + bb - cc - dd)); // 0
        mat_set(mat, 0, 1, 2 * (-ad + bc)); // 3
        mat_set(mat, 0, 2, 2 * (ac + bd)); // 6
        mat_set(mat, 1, 0, 2 * (ad + bc)); // 1
        mat_set(mat, 1, 1, (aa - bb + cc - dd)); // 4
        mat_set(mat, 1, 2, 2 * (-ab + cd));
        mat_set(mat, 2, 0, 2 * (-ac + bd));
        mat_set(mat, 2, 1, 2 * (ab + cd));
        mat_set(mat, 2, 2, (aa - bb - cc + dd));
    }
    FORCE_INLINE void qt_to_mat(const Qt &qt, Flt *mat) {
        const Flt a = qt.d[0];
        const Flt b = qt.d[1];
        const Flt c = qt.d[2];
        const Flt d = qt.d[3];

        const Flt aa = a * a;
        const Flt ab = a * b;
        const Flt ac = a * c;
        const Flt ad = a * d;
        const Flt bb = b * b;
        const Flt bc = b * c;
        const Flt bd = b * d;
        const Flt cc = c * c;
        const Flt cd = c * d;
        const Flt dd = d * d;

        mat_set(mat, 0, 0, (aa + bb - cc - dd));
        mat_set(mat, 0, 1, 2 * (-ad + bc));
        mat_set(mat, 0, 2, 2 * (ac + bd));
        mat_set(mat, 1, 0, 2 * (ad + bc));
        mat_set(mat, 1, 1, (aa - bb + cc - dd));
        mat_set(mat, 1, 2, 2 * (-ab + cd));
        mat_set(mat, 2, 0, 2 * (-ac + bd));
        mat_set(mat, 2, 1, 2 * (ab + cd));
        mat_set(mat, 2, 2, (aa - bb - cc + dd));
    }
    template <typename T>
    FORCE_INLINE T arr3d_get(const struct Arr3d<T> *a, int i, int j, int k) {
        return a->data[i + a->dim[0] * (j + a->dim[1] * k)];
    }
    FORCE_INLINE Grid *cache_get_grid(Cache &c, int idx) {
        return &c.grids[idx];
    }

// #define FLT_EPSILON       1.19209e-07
#define DBL_EPSILON       2.22045e-16
#define EPSILON DBL_EPSILON

#define FLOAT_MAX           3.40282e+38
#define DOUBLE_MAX           1.79769e+308
#define FLT_MAX DOUBLE_MAX


#define FLT_NOT_MAX(f) ((f) < (FLT_MAX * 0.1))

FORCE_INLINE void curl(Flt& e, Flt& deriv, Flt v) {
	if(e > 0 && FLT_NOT_MAX(v)) { // FIXME authentic_v can be gotten rid of everywhere now
		Flt tmp = (v < DBL_EPSILON) ? 0 : (v / (v + e));
		e *= tmp;
		deriv *= tmp * tmp;
	}
}
FORCE_INLINE void curl(Flt& e, Vec& deriv, Flt v) {
	if(e > 0 && FLT_NOT_MAX(v)) { // FIXME authentic_v can be gotten rid of everywhere now
		Flt tmp = (v < DBL_EPSILON) ? 0 : (v / (v + e));
		e *= tmp;
        tmp *= tmp;
        vec_product(deriv, tmp, deriv);
	}
}

FORCE_INLINE void curl(Flt& e, Flt v) {
	if(e > 0 && FLT_NOT_MAX(v)) {
		Flt tmp = (v < DBL_EPSILON) ? 0 : (v / (v + e));
		e *= tmp;
	}
}
FORCE_INLINE void normalize_angle(Flt& x) { // subtract or add enough 2*pi's to make x be in [-pi, pi]
    while(true) {
        // if(threadIdx.z == 2) MCUDBG("normangle %f", x);
        if (x > 3 * PI) {                   // very large
            Flt n = (x - PI) * R2PI;    // how many 2*PI's do you want to subtract?
            auto cn = CEIL(n);
            // if(threadIdx.z == 2) MCUDBG("3pi ceil(%f) = %f", n, cn);
            x -= 2 * PI * cn;  // ceil can be very slow, but this should not be called often

        } else if (x < -3 * PI) {   // very small
            Flt n = (-x - PI) * R2PI;  // how many 2*PI's do you want to add?
            auto cn = CEIL(n);
            // if(threadIdx.z == 2) MCUDBG("-3pi ceil(%f) = %f", n, cn);
            x += 2 * PI * cn;  // ceil can be very slow, but this should not be called often
        } else {
            if (x > PI) {                // in (   PI, 3*PI]
                x -= 2 * PI;
                // if(threadIdx.z == 2) MCUDBG("pi %f", x);
            } else if (x < -PI) {        // in [-3*PI,  -PI)
                x += 2 * PI;
                // if(threadIdx.z == 2) MCUDBG("-pi %f", x);
            }
            // in [-pi, pi]
            break;
        }
    }
}
FORCE_INLINE Flt normalized_angle(Flt x) { // subtract or add enough 2*pi's to make x be in [-pi, pi]
    normalize_angle(x);
    return x;
}
FORCE_INLINE void angle_to_quaternion(const Flt *axis, Flt angle, Flt *out) { // axis is assumed to be a unit vector
	//assert(eq(tvmet::norm2(axis), 1));
	// assert(eq(axis.norm(), 1));
	normalize_angle(angle); // this is probably only necessary if angles can be very big
	Flt c = COS(angle/2);
	Flt s = SIN(angle/2);
	// if(threadIdx.z == 2) MCUDBG("normalized angle %f cos %f sin %f", angle, c, s);
    out[0] = c, out[1] = s * axis[0], out[2] = s * axis[1], out[3] = s * axis[2];
}
// require 1 flt tmp
FORCE_INLINE void angle_to_quaternion_x(const Flt *axis, Flt angle, Flt *out, Flt *tmp) { // axis is assumed to be a unit vector
	//assert(eq(tvmet::norm2(axis), 1));
	// assert(eq(axis.norm(), 1));
    int dx = threadIdx.x;
    if (dx == 0) {
        normalize_angle(angle); // this is probably only necessary if angles can be very big
        angle = angle * 0.5;
        out[0] = COS(angle);
        *tmp = SIN(angle);
    }
    WARPSYNC();
    if (dx > 0 && dx < 4) {
        out[dx] = *tmp * axis[dx-1];
    }
}
FORCE_INLINE void angle_to_quaternion(const Flt *rotation, Flt *out) {
	//fl angle = tvmet::norm2(rotation); 
	Flt angle = norm3d(rotation[0], rotation[1], rotation[2]); 
	// if(threadIdx.z == 2) MCUDBG("angle %f epsilon %f", angle, EPSILON);
	if(angle > EPSILON) {
		//vec axis; 
		//axis = rotation / angle;	
        Flt r = reciprocal(angle);
        Flt axis[3];
        axis[0] = rotation[0] * r, axis[1] = rotation[1] * r, axis[2] = rotation[2] * r;
		// if(threadIdx.z == 2) MCUDBG("axis %f %f %f", axis[0], axis[1], axis[2]);
        angle_to_quaternion(axis, angle, out);
    } else {
        out[0] = 1., out[1] = 0, out[2] = 0, out[3] = 0;
    }
}
// require tmp: 6 flts
FORCE_INLINE void angle_to_quaternion_x(const Flt *rotation, Flt *out, Flt *tmp) {
	//fl angle = tvmet::norm2(rotation); 
    int offset = 0;
    Flt *angle, *r, *axis, *atq;
    angle = tmp + offset, offset ++;
    r = tmp + offset, offset ++;
    axis = tmp + offset, offset += 3;
    atq = tmp + offset, offset++;

    int dx = threadIdx.x;
    if (dx == 0) {
        *angle = norm3d(rotation[0], rotation[1], rotation[2]); 
        *r = reciprocal(*angle);
    }
    WARPSYNC();
    if (dx < 3) {
        out[dx] = dx > 0 ? 0. : 1.;
    }
    WARPSYNC();
	if(*angle > EPSILON) {
		//vec axis; 
		//axis = rotation / angle;	
        if (dx < 3) {
            axis[dx] = rotation[dx] * *r;
        }
        angle_to_quaternion_x(axis, *angle, out, atq);
    }
}
FORCE_INLINE void qt_multiple(Flt *dst, const Flt *left, const Flt *right) {
    Flt xr = right[0], yr = right[1], zr = right[2], wr = right[3];
    Flt x = left[0], y = left[1], z = left[2], w = left[3];
    dst[0] = x * xr - y * yr - z * zr - w * wr, dst[1] = x * yr + y * xr + z * wr - w * zr,
    dst[2] = x * zr - y * wr + z * xr + w * yr, dst[3] = x * wr + y * zr - z * yr + w * xr;
}
// left *= right
// d: dx, or dy, or dz
// tmp: 4 Flts temp
// this must be run inside one single warp
__device__ int qtm_seq[] = {
    0, 3, 2,
    3, 0, 1,
    2, 1, 0
};
FORCE_INLINE void qt_multiple_c(int idx,Flt *left, const Flt *right) {
    if (idx < 4) {
        Flt d[4];
        int ridx[4] = {idx, 1, 2, 3};
        if (idx > 0) {
            int offset = 3 * (idx-1);
            ridx[1] = qtm_seq[offset], ridx[2] = qtm_seq[offset+1], ridx[3] = qtm_seq[offset+2]; 
        }
        d[0] = left[0], d[1] = left[1], d[2] = left[2], d[3] = left[3];

        if (idx == 0 || idx == 2) {
            d[1] = -d[1];
        }
        if (idx == 0 || idx == 3) {
            d[2] = -d[2];
        }
        if (idx < 2) {
            d[3] = -d[3];
        }
        // right:
        //        0  1  2  3
        //        ----------
        // idx 0: 0, 1, 2, 3
        // idx 1: 1, 0, 3, 2
        // idx 2: 2, 3, 0, 1
        // idx 3: 3, 2, 1, 0
        // because this routine runs inside one single warp, we do not worry the write before read problem
        // if (threadIdx.z == 0) printf("%d %d %d: d %f %f %f %f\n", threadIdx.x, threadIdx.y, threadIdx.z, d[0], d[1], d[2], d[3]);
        left[idx] = d[0] * right[ridx[0]] + d[1] * right[ridx[1]] + d[2] * right[ridx[2]] + d[3] * right[ridx[3]];

    }
}
FORCE_INLINE void qt_multiple(Flt *dst, const Flt *src) {
    Flt xr = src[0], yr = src[1], zr = src[2], wr = src[3];
    Flt x = dst[0], y = dst[1], z = dst[2], w = dst[3];
    dst[0] = x * xr - y * yr - z * zr - w * wr, dst[1] = x * yr + y * xr + z * wr - w * zr,
    dst[2] = x * zr - y * wr + z * xr + w * yr, dst[3] = x * wr + y * zr - z * yr + w * xr;
}
FORCE_INLINE void angle_to_quaternion(const Vec& axis, Flt angle, Qt &out) { // axis is assumed to be a unit vector
	//assert(eq(tvmet::norm2(axis), 1));
	normalize_angle(angle); // this is probably only necessary if angles can be very big
    angle = angle * 0.5;
	Flt c = COS(angle);
	Flt s = SIN(angle);
	return qt_set(out, c, s*vec_get(axis, 0), s*vec_get(axis, 1), s*vec_get(axis, 2));
}
FORCE_INLINE void angle_to_quaternion_c(int idx, const Vec& axis, Flt angle, Qt &out) { // axis is assumed to be a unit vector
	//assert(eq(tvmet::norm2(axis), 1));
	normalize_angle(angle); // this is probably only necessary if angles can be very big
    angle = angle * 0.5;
    if (idx == 0) {
        qt_set(out, 0, COS(angle));
    } else if (idx < 4) {
        qt_set(out, idx, SIN(angle) * vec_get(axis, idx-1));
    }
}
FORCE_INLINE void angle_to_quaternion_c(int idx, const Vec& axis, Flt angle, Flt *out) { // axis is assumed to be a unit vector
	//assert(eq(tvmet::norm2(axis), 1));
	normalize_angle(angle); // this is probably only necessary if angles can be very big
    angle = angle * 0.5;
    if (idx == 0) {
        out[0] = COS(angle);
    } else if (idx < 4) {
        out[idx] = SIN(angle) * vec_get(axis, idx-1);
    }
}
FORCE_INLINE void angle_to_quaternion_c(int idx, const Vec& axis, const Flt *cs, Flt *out) { // axis is assumed to be a unit vector
	//assert(eq(tvmet::norm2(axis), 1));
    if (idx == 0) {
        out[0] = cs[0];
    } else if (idx < 4) {
        out[idx] = cs[1] * vec_get(axis, idx-1);
    }
}
FORCE_INLINE Flt quaternion_norm_sqr(const Qt& q) { // equivalent to sqr(boost::math::abs(const qt&))
	return SQR(q.d[0]) + SQR(q.d[1]) + SQR(q.d[2]) + SQR(q.d[3]);
}
FORCE_INLINE void quaternion_normalize_approx(Qt& q, const Flt tolerance = 1e-6) {
	const Flt s = quaternion_norm_sqr(q);
	if(ABS(s - 1) < tolerance)
		; // most likely scenario
	else {
		const Flt a = RSQRT(s);
        qt_multiple(q, a);
	}
}
FORCE_INLINE Flt quaternion_norm_sqr(const Flt * q) { // equivalent to sqr(boost::math::abs(const qt&))
	return SQR(q[0]) + SQR(q[1]) + SQR(q[2]) + SQR(q[3]);
}
// todo
FORCE_INLINE void quaternion_normalize_approx(Flt * q, const Flt tolerance = 1e-6) {
	const Flt s = quaternion_norm_sqr(q);
	if(ABS(s - 1) < tolerance)
		; // most likely scenario
	else {
		const Flt a = RSQRT(s);
        qt_multiple(q, a);
	}
}
template<typename T>// T = Flt * or const Flt *
FORCE_INLINE T get_ligand_change(SrcModel *src, T g, int idx) {
    int offset = 0;
    for (int i = 0; i < idx; i++) {
        offset += src->ligands[i].nr_node - 1 + 6;// 3+3 for position and oritentation, rest will be torsions
    }
    return g + offset; 
}
#define get_ligand_change_torsion(g, idx) (g)[6+(idx)]

template<typename T> // T = Flt * or const Flt *
FORCE_INLINE T get_flex_change(SrcModel *src, T g, int idx) {
    int offset = src->nrfligands;
    for (int i = 0; i < idx; i++) {
        offset += src->flex[i].nr_node;
    }
    return g + offset; 
}
#define get_flex_change_torsion(g, idx) (g)[idx]

template<typename T>// T = Flt * or const Flt *
FORCE_INLINE T get_ligand_conf(SrcModel *src, T g, int idx) {
    int offset = 0;
    for (int i = 0; i < idx; i++) {
        offset += src->ligands[i].nr_node - 1 + 7;// 3+4 for position and oritentation, rest will be torsions
    }
    return g + offset; 
}
#define get_ligand_conf_torsion(g, idx) (g)[7+(idx)]


template<typename T>// T = Flt * or const Flt *
FORCE_INLINE T get_flex_conf(SrcModel *src, T g, int idx) {
    int offset = src->nrfligands;
    for (int i = 0; i < idx; i++) {
        offset += src->flex[i].nr_node;
    }
    return g + offset; 
}
#define get_flex_conf_torsion(g, idx)  (g)[idx]


// we'll calculate each torsion's cos/sin and save them inside an array for conf set
FORCE_INLINE int get_ligand_conf_angle_offset(SrcModel *src, int idx) {
    int offset = 0;
    for (int i = 0; i < idx; i++) {
        offset += src->ligands[i].nr_node - 1;
    }
    return offset * 2;  // for cos + sin
}
FORCE_INLINE int get_flex_conf_angle_offset(SrcModel *src, int idx) {
    int offset = 0;
    for (int i = 0; i < idx; i++) {
        offset += src->flex[i].nr_node - 1;// 3+4 for position and oritentation, rest will be torsions
    }
    return offset; 
}
};
#endif