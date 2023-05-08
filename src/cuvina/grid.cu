#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
#include <algorithm>


namespace dock {
// nc
COULD_INLINE Flt grid_eval(Grid *g, Vec &location, Flt slope, Flt v, Vec *deriv) {
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

// oute must has model movable_atoms size
// c1
GLOBAL void cache_eval_deriv(Cache *c, Model *m) {
	SrcModel *src = m->src;
	CU_FOR(i, src->movable_atoms) {
		Vec deriv;
		make_vec(deriv, 0, 0, 0);
		Flt e = 0;
		Size t = src->xs_sizes[i];
		CUDBG("cache %d t %lu", i, t);
		if (t != CU_INVALID_XS_SIZE) {
			Grid *g = &c->grids[t];
			e = grid_eval(g, m->coords[i], c->m_slope, m->vs[1], &deriv);
		}
		vec_set( m->minus_forces[i], deriv);
		m->movable_e[i] = e;
		CUDBG("cache %d eval e %f", i, e);
		CUVDUMP("force deriv", deriv);
		
	}
}
#if USE_CUDA_VINA
void cuda_cache_eval_deriv(Model *cpum, Cache *c, Model *m, cudaStream_t stream) {
    dim3 block(std::min(256, cpum->src->movable_atoms));
    dim3 grid(1);
    cache_eval_deriv<<<grid, block, 0, stream>>>(c, m);
}
#endif
};  // namespace dock
