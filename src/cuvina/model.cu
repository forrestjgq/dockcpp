
#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"


namespace dock {

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
FORCE_INLINE void prec_by_atom_eval_deriv(PrecalculateByAtom *pa, Size i, Size j, Flt r2, Flt *oute, Flt *outdor) {
	Size idx = triangular_matrix_index(pa->pe_dim, i, j);
	prec_ele_eval_deriv(pa->data[idx], r2, oute, outdor);
}

// outes size == npair and fill with e-s
// forces
// COULD_INLINE void eval_interacting_pairs_deriv(PrecalculateByAtom &p, Flt v, InteractingPair *pairs, int npair, Vec *coords, Vec *out_forces, bool with_max_cutoff, Flt *outes) {
// 	Flt cutoff_sqr = p.cutoff_sqr;
// 	if (with_max_cutoff) {
// 		cutoff_sqr = p.max_cutoff_sqr;
// 	}

// 	CU_FOR(i, npair) {
// 		InteractingPair &ip = pairs[i];
// 		Vec r;
// 		vec_sub(coords[ip.b], coords[ip.a], r);
// 		Flt r2 = vec_sum(r);
// 		outes[i] = 0;
// 		Vec force;
// 		make_vec(force, 0, 0, 0);
// 		if (r2 < cutoff_sqr) {
// 			Flt e, dor;
// 			prec_by_atom_eval_deriv(p, ip.a, ip.b, r2, &e, &dor);
// 			vec_product(r, dor, force);
// 			curl(e, force, v);
// 			outes[i] = e;

// 			// vec_sub(forces[ip.a], force);
// 			// vec_add(forces[ip.b], force);
// 		}
// 		vec_set(out_forces[i], force);
// 	}
// }
COULD_INLINE void eval_interacting_pair_deriv(int seq, PrecalculateByAtom *p, const InteractingPair &ip,
                                 Vec *coords, PairEvalResult &res, Flt *vs) {
    Vec r;
    vec_sub(coords[ip.b], coords[ip.a], r);
    Flt r2   = vec_sqr_sum(r);
	CUDBG("eval seq %d , a %lu b %lu r2 %f cutoff %f v %d", seq, ip.a, ip.b, r2, ip.cutoff_sqr, ip.v);
	CUVDUMP("r", r);
    res.e = 0;
    Vec force;
    make_vec(force, 0, 0, 0);
    if (r2 < ip.cutoff_sqr) {
        Flt e, dor;
        prec_by_atom_eval_deriv(p, ip.a, ip.b, r2, &e, &dor);
		CUDBG("atom der %f %f", e, dor);
        vec_product(r, dor, force);
        curl(e, force, vs[ip.v]);
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

GLOBAL  void model_eval_deriv(Model *m, PrecalculateByAtom *p, BFGSCtx *ctx) {

	SrcModel *src = m->src;
    Change *g = &ctx->g;

	CU_FOR(i, src->npairs) {
		eval_interacting_pair_deriv(i, p, src->pairs[i], m->coords, m->pair_res[i], m->vs);
	}
	SYNC();
	// accumulate e as deriviative result
	if (IS_MAIN_THREAD()) {
		ctx->e = 0;
		FOR(i, src->movable_atoms) {
			ctx->e += m->movable_e[i];
		}
		FOR(i, src->npairs) {
			PairEvalResult &res = m->pair_res[i];
			ctx->e += res.e;
		}
	}
	// accumulate minus_forces
	if (IS_SUB_THREAD()) {
		FOR(i, src->npairs) {
			InteractingPair &pair = src->pairs[i];
			PairEvalResult &res = m->pair_res[i];
			vec_sub(m->minus_forces[pair.a], res.force);
			vec_add(m->minus_forces[pair.b], res.force);
		}
	}
	SYNC();

	// ligands deriviative
	CU_FOR(i, src->nligand) {
		Ligand &ligand = src->ligands[i];
		LigandVars &var = m->ligands[i];
		// first calculate all node force and torque, only for node itself, not include sub-nodes
		FOR(j, ligand.nr_node) {
			CUDBG("ligand %d", j);
			sum_force_and_torque<Segment>(ligand.tree[j], var.tree[j].origin, m->coords, m->minus_forces, var.tree[j].ft);
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
                               g->ligands[i].torsions[ligand.nr_node - j - 2]);
				CUDBG("set ligands %d ligand %d torsion[%d] %f", i, j, ligand.nr_node - j - 2, g->ligands[i].torsions[ligand.nr_node - j - 2]);
				CUVDUMP("    axis", segvar.axis);
				CUVECPDUMP("    ft", segvar.ft);
            } else {
				// root
				CUVECPDUMP("    root ft", segvar.ft);
				set_derivative(segvar.ft, g->ligands[i].rigid);
			}
        }
	}
	CU_FOR(i, src->nflex) {
		Residue &flex = src->flex[i];
		ResidueVars &var = m->flex[i];
		// first calculate all node force and torque, only for node itself, not include sub-nodes
		FOR(j, flex.nr_node) {
			CUDBG("flex %d", j);
			sum_force_and_torque<Segment>(flex.tree[j], var.tree[j].origin, m->coords, m->minus_forces, var.tree[j].ft);
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
							g->flex[i].torsions[flex.nr_node - j - 1]);
        }
	}
}

#if USE_CUDA_VINA
void cu_model_eval_deriv(Model *cpum, Model *m, PrecalculateByAtom *p, BFGSCtx *ctx, cudaStream_t stream) {
    dim3 block(std::min(256, std::max(cpum->src->nligand, cpum->src->nflex)));
    dim3 grid(1);
    model_eval_deriv<<<grid, block, 0, stream>>>(m, p, ctx);
}
#endif
};  // namespace dock
