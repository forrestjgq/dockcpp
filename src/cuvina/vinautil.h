
#ifndef VINA_UTIL_H
#define VINA_UTIL_H

#include "cuvina/vinadef.h"
#if USE_CUDA_VINA == 0
#include <cmath>
#endif

namespace dock {

#define FOR(i, n) for(int i = 0; i < n; i++)
#define FOR_RANGE(i, start, end) for(int i = start; i < end; i++)

#define SQR(x) ((x) * (x))
#if USE_CUDA_VINA
    #define FORCE_INLINE __forceinline__ __device__
    #define COULD_INLINE __forceinline__ __device__
    #define GLOBAL __global__ 
    #define CU_FOR(i, n) for (int i = threadIdx.x; i < n; i += blockDim.x)
    #define SYNC() __syncthreads()
    #define IS_MAIN_THREAD() (threadIdx.x == 0)
    #define IS_SUB_THREAD() (threadIdx.x == 1)
    #define IS_GRID(n) (blockIdx.x == n)
    #define CEIL(x) ceil(x)
    #define SIN(x) sin(x)
    #define COS(x) cos(x)
    #define QTNORM(qt) norm4d(qt.x, qt.y, qt.z, qt.w)
    #define QTRNORM(qt) rnorm4d(qt.x, qt.y, qt.z, qt.w)
    #define ABS(x) abs(x)
    #define SQRT(x) sqrt(x)
    #define RSQRT(x) rsqrt(x)
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
    FORCE_INLINE void qt_set(Qt &dst, const Qt &src) {
        dst = src;
    }
    FORCE_INLINE void qt_set(Qt &dst, Flt x, Flt y, Flt z, Flt w) {
        dst.x = x, dst.y = y, dst.z = z, dst.w = w;
    }
    FORCE_INLINE void qt_multiple(Qt &dst, Flt x) {
        dst.x *= x;
        dst.y *= x;
        dst.z *= x;
        dst.w *= x;
    }
    FORCE_INLINE void qt_multiple(Qt &dst, const Qt &src) {
        Flt xr = src.x, yr = src.y, zr = src.z, wr = src.w;
        Flt x = dst.x, y = dst.y, z = dst.z, w = dst.w;
        qt_set(dst, x * xr - y * yr - z * zr - w * wr, x * yr + y * xr + z * wr - w * zr,
               x * zr - y * wr + z * xr + w * yr, x * wr + y * zr - z * yr + w * xr);
    }
    FORCE_INLINE void mat_multiple(const Flt *mat, const Vec &v, Vec &out) {
		make_vec(out,
                   mat[0]*v.x + mat[3]*v.y + mat[6]*v.z, 
			       mat[1]*v.x + mat[4]*v.y + mat[7]*v.z,
				   mat[2]*v.x + mat[5]*v.y + mat[8]*v.z);
    }

    FORCE_INLINE void mat_set(Flt *mat, int i, int j, Flt v) {
        mat[i + 3 *j] = v;
    }
    FORCE_INLINE void qt_to_mat(const Qt &qt, Flt *mat) {
        const Flt a = qt.x;
        const Flt b = qt.y;
        const Flt c = qt.z;
        const Flt d = qt.w;

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
FORCE_INLINE void normalize_angle(Flt& x) { // subtract or add enough 2*pi's to make x be in [-pi, pi]
    while(true) {
        if (x > 3 * PI) {                   // very large
            Flt n = (x - PI) / (2 * PI);    // how many 2*PI's do you want to subtract?
            x -= 2 * PI * CEIL(n);  // ceil can be very slow, but this should not be called often
        } else if (x < -3 * PI) {   // very small
            Flt n = (-x - PI) / (2 * PI);  // how many 2*PI's do you want to add?
            x += 2 * PI * CEIL(n);  // ceil can be very slow, but this should not be called often
        } else {
            if (x > PI) {                // in (   PI, 3*PI]
                x -= 2 * PI;
            } else if (x < -PI) {        // in [-3*PI,  -PI)
                x += 2 * PI;
            }
            // in [-pi, pi]
            break;
        }
    }
}
FORCE_INLINE void angle_to_quaternion(const Vec& axis, Flt angle, Qt &out) { // axis is assumed to be a unit vector
	//assert(eq(tvmet::norm2(axis), 1));
	normalize_angle(angle); // this is probably only necessary if angles can be very big
	Flt c = COS(angle/2);
	Flt s = SIN(angle/2);
	return qt_set(out, c, s*vec_get(axis, 0), s*vec_get(axis, 1), s*vec_get(axis, 2));
}
FORCE_INLINE Flt quaternion_norm_sqr(const Qt& q) { // equivalent to sqr(boost::math::abs(const qt&))
	return SQR(q.x) + SQR(q.y) + SQR(q.z) + SQR(q.w);
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
};
#endif