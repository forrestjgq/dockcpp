#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
#include <algorithm>
#include "model_desc.cuh"

#define GRID_DEBUG 0
#define GRIDIDX 16

#define GRIDDBG(fmt, ...) do{ printf("%d [%d:%d] [%d:%d:%d]\t" fmt "\n",  __LINE__, lane_id(), warp_id(), threadIdx.x, threadIdx.y, threadIdx.z,  __VA_ARGS__);}while(0)
namespace dock {
// nc
FORCE_INLINE Flt grid_eval(const Grid *g, const Vec &location, Flt slope, Flt v, Vec *deriv) {
    Vec t, s;
    vec_sub(location, g->m_init, t);
    vec_product(t, g->m_factor, s);
	CUVDUMP("grid location", location);
	CUVDUMP("grid init", g->m_init);
	CUVDUMP("grid factor", g->m_factor);
	CUVDUMP("grid ele product", s);

    Vec miss;
    int region[3];
    Size a[3], a1[3];

    FOR (i, 3) {
        Flt si = vec_get(s, i);
        Flt minus = vec_get(g->m_dim_fl_minus_1, i);
        Flt m = 0;
        if (si < 0) {
            m = -si;
            region[i] = -1;
            a[i] = 0;
            si = 0;
        } else if(si >= minus) {
            m = si - minus;
            region[i] = 1;
            a[i] = g->m_data.dim[i] - 2;
            si = 1;
        } else {
            region[i] = 0;
            a[i] = (Size)si;
            si -= a[i];
        }
        vec_set(miss, i, m);
        vec_set(s, i, si);
    }
	const Flt penalty = slope * vec_product_sum(miss, g->m_factor_inv); // FIXME check that inv_factor is correctly initialized and serialized

	const Size x0 = a[0];
	const Size y0 = a[1];
	const Size z0 = a[2];

	const Size x1 = x0+1;
	const Size y1 = y0+1;
	const Size z1 = z0+1;

	const Flt f000 = arr3d_get<Flt>(&g->m_data,x0, y0, z0);
	const Flt f100 = arr3d_get<Flt>(&g->m_data,x1, y0, z0);
	const Flt f010 = arr3d_get<Flt>(&g->m_data,x0, y1, z0);
	const Flt f110 = arr3d_get<Flt>(&g->m_data,x1, y1, z0);
	const Flt f001 = arr3d_get<Flt>(&g->m_data,x0, y0, z1);
	const Flt f101 = arr3d_get<Flt>(&g->m_data,x1, y0, z1);
	const Flt f011 = arr3d_get<Flt>(&g->m_data,x0, y1, z1);
	const Flt f111 = arr3d_get<Flt>(&g->m_data,x1, y1, z1);

	const Flt x = vec_get(s,0);
	const Flt y = vec_get(s,1);
	const Flt z = vec_get(s,2);

	const Flt mx = 1-x;
	const Flt my = 1-y;
	const Flt mz = 1-z;

	Flt f = 
		f000 *  mx * my * mz  +
		f100 *   x * my * mz  +
		f010 *  mx *  y * mz  + 
		f110 *   x *  y * mz  +
		f001 *  mx * my *  z  +
		f101 *   x * my *  z  +
		f011 *  mx *  y *  z  +
		f111 *   x *  y *  z  ;

	if(deriv) { // valid pointer
		const Flt x_g = 
			f000 * (-1)* my * mz  +
			f100 *   1 * my * mz  +
			f010 * (-1)*  y * mz  + 
			f110 *   1 *  y * mz  +
			f001 * (-1)* my *  z  +
			f101 *   1 * my *  z  +
			f011 * (-1)*  y *  z  +
			f111 *   1 *  y *  z  ;


		const Flt y_g = 
			f000 *  mx *(-1)* mz  +
			f100 *   x *(-1)* mz  +
			f010 *  mx *  1 * mz  + 
			f110 *   x *  1 * mz  +
			f001 *  mx *(-1)*  z  +
			f101 *   x *(-1)*  z  +
			f011 *  mx *  1 *  z  +
			f111 *   x *  1 *  z  ;


		const Flt z_g =  
			f000 *  mx * my *(-1) +
			f100 *   x * my *(-1) +
			f010 *  mx *  y *(-1) + 
			f110 *   x *  y *(-1) +
			f001 *  mx * my *  1  +
			f101 *   x * my *  1  +
			f011 *  mx *  y *  1  +
			f111 *   x *  y *  1  ;

		Vec gradient;
        make_vec(gradient, x_g, y_g, z_g);

		curl(f, gradient, v);
        Flt gradient_everywhere[3];

		FOR(i, 3) {
			gradient_everywhere[i] = ((region[i] == 0) ? vec_get(gradient,i) : 0);
			Flt t = vec_get(g->m_factor,i) * gradient_everywhere[i] + slope * region[i];
            vec_set(*deriv, i, t);
		}

		return f + penalty;
	}
	else {
		curl(f, v);
		return f + penalty;
	}

}
// make sure grid_eval_x runs inside a single warp
// tmp require 3*5+8+4 = 27 Flt
FORCE_INLINE void grid_eval_x(int seq, const Grid *g, const Vec &location, Flt slope, Flt v, Vec *deriv, Flt *e, Flt *tmp) {
	const int dx = threadIdx.x;
	Flt *s, *miss, *f, *gs;
	int *region;
	Size *a, *b;

	int offset = 0;

    s    = tmp + offset, offset += 3;
    miss = tmp + offset, offset += 3;
    f    = tmp + offset, offset += 8;
    gs   = tmp + offset, offset += 4;

    region     = (int *)(tmp + offset), offset += 3;
	a          = (Size *)(tmp + offset), offset += 3;
	b          = (Size *)(tmp + offset), offset += 3;
	if (dx < 3) {

        s[dx]            = (location.d[dx] - g->m_init.d[dx]) * g->m_factor.d[dx];
        Flt &si          = s[dx];
        const Flt &minus = g->m_dim_fl_minus_1.d[dx];
#if GRID_DEBUG
		if (seq == GRIDIDX) GRIDDBG("s[%d] %f minus %f", dx, s[dx], minus);
#endif
        Flt m            = 0;
        if (si < 0) {
            m          = -si;
            region[dx] = -1;
            a[dx]      = 0;
            si         = 0;
        } else if (si >= minus) {
            m          = si - minus;
            region[dx] = 1;
            a[dx]      = g->m_data.dim[dx] - 2;
            si         = 1;
        } else {
            region[dx] = 0;
            a[dx]      = (Size)si;
            si -= a[dx];
        }
        miss[dx] = m * g->m_factor_inv.d[dx] * slope;
        // FIXME check that inv_factor is correctly initialized and serialized
    }

#if GRID_DEBUG
	if (seq == GRIDIDX && dx == 0)  {
		GRIDDBG("a %lu %lu %lu", a[0], a[1], a[2]);
	}
#endif
	if (dx < 4)  {
		// there are 8 f-s, f[0] - f[7], each dx in [0, 3] will calc 2 of them
		// const fl f000 = m_data(x0, y0, z0);
		// const fl f001 = m_data(x0, y0, z1);
		// const fl f010 = m_data(x0, y1, z0);
		// const fl f011 = m_data(x0, y1, z1);
		// const fl f100 = m_data(x1, y0, z0);
		// const fl f101 = m_data(x1, y0, z1);
		// const fl f110 = m_data(x1, y1, z0);
		// const fl f111 = m_data(x1, y1, z1);
		Size aa[3] = {a[0], a[1], a[2]};
		
        if (dx > 1) aa[0] += 1;
        if (dx & 1) aa[1] += 1;
#if GRID_DEBUG
		if (seq == GRIDIDX && dx == 0)  {
			GRIDDBG("changed a %lu %lu %lu", aa[0], aa[1], aa[2]);
		}
#endif

        // return a->data[i + a->dim[0] * (j + a->dim[1] * k)];
        int fidx    = dx << 1;
        f[fidx]     = arr3d_get<Flt>(&g->m_data, aa[0], aa[1], aa[2]);
        f[fidx + 1] = arr3d_get<Flt>(&g->m_data, aa[0], aa[1], aa[2] + 1);
#if GRID_DEBUG
		if (seq == GRIDIDX) {
			GRIDDBG("f[%d] = %f", fidx, f[fidx]);
			GRIDDBG("f[%d] = %f", fidx+1, f[fidx+1]);
		}
#endif

        Flt s1[3], s2[3];
        // as x, y, z
        s1[0] = s[0], s1[1] = s[1], s1[2] = s[2];
        // as mx, my, mz
        s2[0] = 1 - s[0], s2[1] = 1 - s[1], s2[2] = 1 - s[2];
        if (dx < 4 && dx > 0) {
            int idx = dx - 1;  // 0, 1, 2
			// the pattern of f and x/y/z _g calc is f = f[0..7] * x * y *z
			// for x_g/y_g/z_g the x/y/z part will be replaced with 1/-1, here we set to 1 first,
			// later we will set f[0..7] = -f[0..7]
			s2[idx] = -1;
			s1[idx] = 1;
        }
        gs[dx] = f[0] * s2[0] * s2[1] * s2[2] + 
		         f[1] * s2[0] * s2[1] * s1[2] + 
				 f[2] * s2[0] * s1[1] * s2[2] + 
				 f[3] * s2[0] * s1[1] * s1[2] + 
				 f[4] * s1[0] * s2[1] * s2[2] + 
				 f[5] * s1[0] * s2[1] * s1[2] + 
				 f[6] * s1[0] * s1[1] * s2[2] + 
				 f[7] * s1[0] * s1[1] * s1[2];

#if GRID_DEBUG
		if (seq == GRIDIDX && dx == 0)  {
			GRIDDBG("f %f %f %f %f %f %f %f %f", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
			GRIDDBG("s1 %f %f %f s2 %f %f %f", s1[0], s1[1], s1[2],  s2[0], s2[1], s2[2]);
			GRIDDBG("s %f %f %f gs %f %f %f %f", s[0], s[1], s[2], gs[0], gs[1], gs[2], gs[3]);
		}
#endif
        Vec gradient;
        make_vec(gradient, gs[1], gs[2], gs[3]);

        curl(gs[0], gradient, v);
        Flt gradient_everywhere[3];

        if (dx < 3) {
            gradient_everywhere[dx] = ((region[dx] == 0) ? gradient.d[dx] : 0);
            deriv->d[dx] = g->m_factor.d[dx] * gradient_everywhere[dx] + slope * region[dx];
        }
        if (dx == 0) {
            *e = gs[0] + miss[0] + miss[1] + miss[2];
        }
    }

}
// oute must has model movable_atoms size
// depends on model coords
__device__ void c_cache_eval_deriv_xy(const Cache *c, const ModelDesc *m, Flt *md, const Flt *vs) {
	auto src = m->src;
	CU_FOR2(i, src->movable_atoms) {
		Flt e = 0;
		Size t = src->xs_sizes[i];
		auto &force = *model_minus_forces(src, m, md, i);
		make_vec(force, 0, 0, 0);
		// NNCUDBG("cache %d t %lu ncoords %d md %p mforce offset %d", i, t, m->ncoords, md, m->minus_forces);
		if (t < c->ngrids) {
			auto *g = &c->grids[t];
			auto coord = model_coords(src, m, md, i);
			e = grid_eval(g, *coord, c->m_slope, vs[1], &force);
		}
		auto me = model_movable_e(src, m, md, i);
		*me = e;
		CUDBG("cache %d eval e %f", i, e);
		CUVDUMP("force deriv", force);
		
	}
}
#define CACHE_EVAL_MEM_SIZE(src) (((src)->movable_atoms) * 27)
// each grid eval requires 27 flt, tmp requires src->movable_atoms * 27 Flts
__device__ void c_cache_eval_deriv_xyz(const Cache *c, const ModelDesc *m, Flt *md, const Flt *vs, Flt *tmp) {
	auto src = m->src;

	// use yz for atoms visiting, x for grid evaluation
	for (int i = threadIdx.y + threadIdx.z * blockDim.y; i < src->movable_atoms; i += blockDim.y * blockDim.z) {
		Flt *e = model_movable_e(src, m, md, i);
		*e = 0;
		Size t = src->xs_sizes[i];
		auto force = model_minus_forces(src, m, md, i);
		if (threadIdx.x < 3) {
			force->d[threadIdx.x] = 0;
		}
		// NNCUDBG("cache %d t %lu ncoords %d md %p mforce offset %d", i, t, m->ncoords, md, m->minus_forces);
		if (t < c->ngrids) {
			auto *g = &c->grids[t];
			auto coord = model_coords(src, m, md, i);
#if GRID_DEBUG
			if (threadIdx.x == 0 && i == GRIDIDX) {
				GRIDDBG("cache %d coord %f %f %f", i, coord->d[0], coord->d[1], coord->d[2]);
			}
#endif
			grid_eval_x(i, g, *coord, c->m_slope, vs[1], force, e, tmp + 27 * i);
		}
#if GRID_DEBUG
		if (threadIdx.x == 0 && i == GRIDIDX) {
			GRIDDBG("cache %d eval e %f", i, *e);
			GRIDDBG("force derive: %f %f %f", force->d[0], force->d[1], force->d[2]);
		}
#endif
		
	}
}
};  // namespace dock
