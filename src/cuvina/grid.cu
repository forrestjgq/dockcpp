#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"


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
COULD_INLINE void cache_eval_deriv(Cache &c, Model &m) {
	SrcModel *src = m.src;
	ModelConf *conf = m.conf;
	CU_FOR(i, src->movable_atoms) {
		Vec deriv;
		make_vec(deriv, 0, 0, 0);
		Flt e = 0;
		Size t = src->xs_sizes[i];
		CUDBG("cache %d t %lu", i, t);
		if (t != CU_INVALID_XS_SIZE) {
			Grid *g = cache_get_grid(c, t);
			e = grid_eval(g, conf->coords[i], c.m_slope, src->movable_v, &deriv);
		}
		vec_set( m.minus_forces[i], deriv);
		m.movable_e[i] = e;
		CUDBG("cache %d eval e %f", i, e);
		CUVDUMP("force deriv", deriv);
		
	}
}

COULD_INLINE void prec_ele_eval_deriv(PrecalculateElement &pe, Flt r2, Flt *oute, Flt *outdor) {
        Flt r2_factored = pe.factor * r2;
        Size i1 = Size(r2_factored);
        Size i2 = i1 + 1; // r2 is expected < cutoff_sqr, and cutoff_sqr * factor + 1 < n, so no overflow
        Flt rem = r2_factored - i1;
        Flt * p1 = pe.smooth + (i1 << 1);
        Flt * p2 = pe.smooth + (i2 << 1);
        *oute = p1[0] + rem * (p2[0] - p1[0]);
        *outdor = p1[1] + rem * (p2[1] - p1[1]);
}
FORCE_INLINE Size triangular_matrix_index(Size n, Size i, Size j) {
	return i + j*(j+1)/2; 
}
FORCE_INLINE void prec_by_atom_eval_deriv(PrecalculateByAtom &pa, Size i, Size j, Flt r2, Flt *oute, Flt *outdor) {
	Size idx = triangular_matrix_index(pa.pe_dim, i, j);
	prec_ele_eval_deriv(pa.data[idx], r2, oute, outdor);
}

// outes size == npair and fill with e-s
// forces
COULD_INLINE void eval_interacting_pairs_deriv(PrecalculateByAtom &p, Flt v, InteractingPair *pairs, int npair, Vec *coords, Vec *out_forces, bool with_max_cutoff, Flt *outes) {
	Flt cutoff_sqr = p.cutoff_sqr;
	if (with_max_cutoff) {
		cutoff_sqr = p.max_cutoff_sqr;
	}

	CU_FOR(i, npair) {
		InteractingPair &ip = pairs[i];
		Vec r;
		vec_sub(coords[ip.b], coords[ip.a], r);
		Flt r2 = vec_sum(r);
		outes[i] = 0;
		Vec force;
		make_vec(force, 0, 0, 0);
		if (r2 < cutoff_sqr) {
			Flt e, dor;
			prec_by_atom_eval_deriv(p, ip.a, ip.b, r2, &e, &dor);
			vec_product(r, dor, force);
			curl(e, force, v);
			outes[i] = e;

			// vec_sub(forces[ip.a], force);
			// vec_add(forces[ip.b], force);
		}
		vec_set(out_forces[i], force);
	}
}
COULD_INLINE void eval_interacting_pair_deriv(int seq, PrecalculateByAtom &p, const InteractingPair &ip,
                                 Vec *coords, PairEvalResult &res) {
    Vec r;
    vec_sub(coords[ip.b], coords[ip.a], r);
    Flt r2   = vec_sqr_sum(r);
	CUDBG("eval seq %d , a %lu b %lu r2 %f cutoff %f v %f", seq, ip.a, ip.b, r2, ip.cutoff_sqr, ip.v);
	CUVDUMP("r", r);
    res.e = 0;
    Vec force;
    make_vec(force, 0, 0, 0);
    if (r2 < ip.cutoff_sqr) {
        Flt e, dor;
        prec_by_atom_eval_deriv(p, ip.a, ip.b, r2, &e, &dor);
		CUDBG("atom der %f %f", e, dor);
        vec_product(r, dor, force);
        curl(e, force, ip.v);
        res.e = e;
		CUDBG("e %f", e);
		CUVDUMP("force", force);
    }
    vec_set(res.force, force);
}
template <typename AtomFrame> // see atom_frame, takes: Size begin, end, Vec origin
COULD_INLINE void sum_force_and_torque(AtomFrame &frame, const Vec &origin, Vec *coords, Vec *forces, Vecp &out) {
	Vec product, sub;
	vecp_clear(out);
	CUDBG("frame begin %d end %d", frame.begin, frame.end);
	FOR_RANGE(i, frame.begin, frame.end) {
		vec_add(out.first, forces[i]);
		vec_sub(coords[i], origin, sub);
		cross_product(sub, forces[i], product);
		vec_add(out.second, product);
		CUDBG("    frame %d", i);
		CUVDUMP("    force", forces[i]);
		CUVDUMP("    coords", coords[i]);
		CUVDUMP("    origin", origin);
		CUVDUMP("    sub", sub);
		CUVDUMP("    product", product);
		CUVDUMP("    second", out.second);
	}
	CUVECPDUMP("sumft", out);
}

template <typename T>
COULD_INLINE void branches_derivative(T *b, int nb, const Vec& origin, Vec *coords, Vec *forces, Vecp& out, Flt *d) { // adds to out
	FOR(i, nb) {
		Vecp ft;
		derivative(b[i], coords, forces, d);
		Vec r, product;
		vec_sub(b[i].node.origin , origin, r);
		cross_product(r, ft.first, product);
		vec_add(ft.second, product);
		vecp_add(out, ft);
	}
}

FORCE_INLINE void set_derivative(const Vec &axis, const Vecp &force_torque, Flt &p) {
	p = vec_product_sum(force_torque.second, axis);
}

FORCE_INLINE void set_derivative(const Vecp& force_torque, RigidChange& c) {
	vec_set(c.position, force_torque.first);
	vec_set(c.orientation, force_torque.second);
}
// see tree::derivative
// template <typename T>
// COULD_INLINE void tree_derivative(T &t, Vec *coords, Vec *forces, Flt *&p) {
// 	Vec ft, product, sub;
// 	sum_force_and_torque(t.node, coords, forces, &ft);
// 	Flt &d = *p;
// 	++p;
// 	branch_derivative(t.children, t.node.orign, coords, forces, ft, p);
// 	set_derivative(t.node, ft, d);
// }
// see heterotree:: void derivative(const vecv& coords, const vecv& forces, ligand_change& c)  
// template <typename Node> // Node has (Vec axis)
// COULD_INLINE void ligand_htree_derivative(Node &t, Vec *coords, Vec *forces, LigandChange *lc) {
// 	Vecp ft;
// 	sum_force_and_torque(t.node, coords, forces, ft);
// 	Flt *p = lc.torsions;
// 	branches_derivative(t.children, t.node.origin, coords, forces, ft, p);
// 	set_derivative(t.node, ft, c.rigid);
// }
// see heterotree:: void derivative(const vecv& coords, const vecv& forces, residue_change& c)  
// template <typename Node> // Node has (Vec axis)
// COULD_INLINE void flex_htree_derivative(Node &t, Vec *coords, Vec *forces, ResidueChange *lc) {
// 	Vecp ft;
// 	sum_force_and_torque(t.node, coords, forces, ft);
// 	Flt *p = lc.torsions;
// 	Flt &d = *p;
// 	++p;
// 	branches_derivative(t.children, t.node.origin, coords, forces, ft, p);
// 	set_derivative(t.node, ft, d);
// }
// template<typename T, typename C>
// void vecm_derivative(T *t, Vec *coords, Vec *forces, C *c, int n) {
// 	FOR(i, nc) {
// 		htree_derivative(t[i], coords, forces, c[i]);
// 	}
// }

GLOBAL  void model_eval_deriv(Model &m, PrecalculateByAtom &p, Cache &c, Change &g) {
	SrcModel *src = m.src;
	ModelConf *conf = m.conf;

	// eval cache and set initial minus_forces
	cache_eval_deriv(c, m);

	CU_FOR(i, src->npairs) {
		eval_interacting_pair_deriv(i, p, src->pairs[i], conf->coords, m.pair_res[i]);
	}
	SYNC();
	// accumulate e as deriviative result
	if (IS_MAIN_THREAD()) {
		m.e = 0;
		FOR(i, src->movable_atoms) {
			m.e += m.movable_e[i];
		}
		FOR(i, src->npairs) {
			PairEvalResult &res = m.pair_res[i];
			m.e += res.e;
		}
	}
	// accumulate minus_forces
	if (IS_SUB_THREAD()) {
		FOR(i, src->npairs) {
			InteractingPair &pair = src->pairs[i];
			PairEvalResult &res = m.pair_res[i];
			vec_sub(m.minus_forces[pair.a], res.force);
			vec_add(m.minus_forces[pair.b], res.force);
		}
	}
	SYNC();

	// ligands deriviative
	CU_FOR(i, conf->nligand) {
		Ligand &ligand = conf->ligands[i];
		LigandVars &var = m.ligands[i];
		// first calculate all node force and torque, only for node itself, not include sub-nodes
		FOR(j, ligand.nr_node) {
			CUDBG("ligand %d", j);
			sum_force_and_torque<Segment>(ligand.tree[j], var.tree[j].origin, conf->coords, m.minus_forces, var.tree[j].ft);
			var.tree[j].dirty = 0; // once force accumulated, dirty will be set to non-0 and torque should be re-calculated
		}
		// climbing from the leaves to root and accumulate force and torque
		FOR(j, ligand.nr_node) {
			Segment &seg = ligand.tree[j];
			SegmentVars &segvar = var.tree[j];
			CUDBG("ligand %d", j);
            if (seg.parent >= 0) {
                Segment &parent        = ligand.tree[seg.parent];
                SegmentVars &parentVar = var.tree[seg.parent];
				CUDBG("    parent %d", seg.parent);
				CUVECPDUMP("    parent ft", parentVar.ft);
				CUVECPDUMP("    child ft", segvar.ft);
				CUVDUMP("    parent origin", parentVar.origin);
				CUVDUMP("    child origin", segvar.origin);
                vec_add(parentVar.ft.first, segvar.ft.first);
                var.tree[seg.parent].dirty++;

                // if (segvar.dirty > 0) {
                //     // this is not a leaf, calculate torque with new force
                //     Vec r, product;
                //     vec_sub(segvar.origin, parentVar.origin, r);
                //     cross_product(r, segvar.ft.first, product);
                //     vec_add(segvar.ft.second, product);
                // }
                    // this is not a leaf, calculate torque with new force
                    Vec r, product;
                    vec_sub(segvar.origin, parentVar.origin, r);
                    cross_product(r, segvar.ft.first, product);
					vec_add(product, segvar.ft.second);
                    vec_add(parentVar.ft.second, product);
				CUVECPDUMP("    childft added", parentVar.ft);
				// note that torsions has reversed order from segments
                set_derivative(segvar.axis, segvar.ft,
                               g.ligands[i].torsions[ligand.nr_node - j - 2]);
				CUDBG("set ligands %d ligand %d torsion[%d] %f", i, j, ligand.nr_node - j - 2, g.ligands[i].torsions[ligand.nr_node - j - 2]);
				CUVDUMP("    axis", segvar.axis);
				CUVECPDUMP("    ft", segvar.ft);
            } else {
				// root
				CUVECPDUMP("    root ft", segvar.ft);
				set_derivative(segvar.ft, g.ligands[i].rigid);
			}
        }
	}
	CU_FOR(i, conf->nflex) {
		Residue &flex = conf->flex[i];
		ResidueVars &var = m.flex[i];
		// first calculate all node force and torque, only for node itself, not include sub-nodes
		FOR(j, flex.nr_node) {
			CUDBG("flex %d", j);
			sum_force_and_torque<Segment>(flex.tree[j], var.tree[j].origin, conf->coords, m.minus_forces, var.tree[j].ft);
			var.tree[j].dirty = 0; // once force accumulated, dirty will be set to non-0 and torque should be re-calculated
		}
		// climbing from the leaves to root and accumulate force and torque
		FOR(j, flex.nr_node) {
			Segment &seg = flex.tree[j];
			SegmentVars &segvar = var.tree[j];
            if (seg.parent >= 0) {
                Segment &parent        = flex.tree[seg.parent];
                SegmentVars &parentVar = var.tree[seg.parent];
                vec_add(parentVar.ft.first, segvar.ft.first);
                var.tree[seg.parent].dirty++;

                // if (segvar.dirty > 0) {
                    // this is not a leaf, calculate torque with new force
                    Vec r, product;
                    vec_sub(segvar.origin, parentVar.origin, r);
                    cross_product(r, segvar.ft.first, product);
					vec_add(product, segvar.ft.second);
                    vec_add(parentVar.ft.second, product);
                // }
			}
			// note that torsions has reversed order from segments
			set_derivative(segvar.axis, segvar.ft,
							g.flex[i].torsions[flex.nr_node - j - 1]);
        }
	}
}

};  // namespace dock
