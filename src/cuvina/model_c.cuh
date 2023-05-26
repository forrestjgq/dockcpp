
#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
#include "model_desc.cuh"

namespace dock {

#define MAIN_INDEX threadIdx.y

COULD_INLINE void prec_ele_eval_deriv(const PrecalculateElement &pe, Flt r2, Flt *oute, Flt *outdor) {
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
	return i + ((j*(j+1)) >> 1); 
}
FORCE_INLINE void prec_by_atom_eval_deriv(const PrecalculateByAtom *pa, Size i, Size j, Flt r2, Flt *oute, Flt *outdor) {
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
COULD_INLINE void eval_interacting_pair_deriv(int seq, const PrecalculateByAtom *p, const InteractingPair &ip,
                                 Vec *coords, PairEvalResult &res, const Flt *vs) {
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

// r should be rigid part in change floats
FORCE_INLINE void set_derivative(const Vecp& force_torque, Flt *r) {
    vec_copy_to(force_torque.first, r);
    vec_copy_to(force_torque.second, r+3);

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

// eval pairs der, stores in model pair_res, and model coords will be refreshed
FORCE_INLINE  void c_model_eval_deriv_pairs(ModelDesc *m, const PrecalculateByAtom *p, Flt *md, const Flt *vs) {
	SrcModel *src = m->src;
	CU_FOR2(i, src->npairs) {
        auto coords = model_coords(src, m, md);
        auto pair_res = model_pair_res(src, m, md, i);
		eval_interacting_pair_deriv(i, p, src->pairs[i], coords, *pair_res, vs);
        // printf("pair %d e %f\n force %f %f %f\n", i, pair_res->e, pair_res->force.x , pair_res->force.y, pair_res->force.z );
	}
}
template <size_t WARP_SIZE>
void __global__ warp_asum_kernel(
	const size_t n,
	const unsigned *src_d,
	unsigned *tmp_d)
{
	const size_t
		global_id = threadIdx.x + blockDim.x * blockIdx.x,
		lane_id = global_id % WARP_SIZE;
	unsigned
		val = global_id < n ? src_d[global_id] : 0;
	for (size_t offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
		val += __shfl_xor_sync(0xffffffff, val, offset, WARP_SIZE);
	if (lane_id == 0)
		tmp_d[global_id / WARP_SIZE] = val;
}
// collect e and setup forces
// make sure x*y = 32
COULD_INLINE void c_model_collect_deriv(ModelDesc *m, Flt *e, Flt *md) {
	SrcModel *src = m->src;

    // reduce e in movable atoms and pairs
    Flt val = 0;
    const int n = src->movable_atoms;
    CU_FOR2(i, src->movable_atoms) {
        val += (*model_movable_e(src, m, md, i));
    }
    CU_FOR2(i, src->npairs) {
        val += model_pair_res(src, m, md, i)->e;
    }
	for (size_t offset = 32 >> 1; offset > 0; offset >>= 1)
		val += __shfl_xor_sync(0xffffffff, val, offset, 32);
	if (IS_2DMAIN()) *e = val;
#if 0
    if (ZIS(0)) {
        for (int i = 0; i < src->movable_atoms + src->npairs; i++) {
            printf("gpu map add %d: %d\n", i, src->force_pair_map_add[i]);
        }
        #if 0
        for(int i = 0; i < src->movable_atoms; i++) {
            int d = src->idx_add[i];
            printf("index add %d: %d\n", i, d);
            if (d >= 0) {
                int n = src->force_pair_map_add[d];
                for (int j = 0, start = d+1; j < n; j++, start++) {
                    auto k = src->force_pair_map_add[start];
                    printf("\tpair %d\n", k);
                }

            }
        }
        for(int i = 0; i < src->movable_atoms; i++) {
            printf("index sub %d: %d\n", i, src->idx_sub[i]);
        }
        #endif
    }
#endif
#if 1
    CU_FOR2(i, src->movable_atoms) {
        int offset = src->idx_add[i];
        // printf("add minus forces %d offset %d\n", i, offset);
        if (offset >= 0) {
            int n = src->force_pair_map_add[offset];
            auto force = model_minus_forces(src, m, md, i);
            for (auto j = 0; j < n; j++) {
                auto pairidx = src->force_pair_map_add[offset + j + 1];
                auto res = model_pair_res(src, m, md, pairidx); 
                // printf("add %d to %d\n", pairidx, i);
                vec_add(*force, res->force);
            }
        }
    }
    CU_FOR2(i, src->movable_atoms) {
        int offset = src->idx_sub[i];
        if (offset >= 0) {
            int n = src->force_pair_map_sub[offset];
            auto force = model_minus_forces(src, m, md, i);
            for (auto j = 0; j < n; j++) {
                auto pairidx = src->force_pair_map_sub[offset + j + 1];
                auto res = model_pair_res(src, m, md, pairidx); 
                vec_sub(*force, res->force);
            }
        }
    }
#else
	// accumulate minus_forces
	if (IS_2DMAIN()) {
		FOR(i, src->npairs) {
			InteractingPair &pair = src->pairs[i];
            if (threadIdx.z == 0)printf("force %d pair %p a %d b %d\n", i, &pair, pair.a, pair.b);
			auto res = model_pair_res(src, m, md, i);
            auto paira = model_minus_forces(src, m, md, pair.a);
            auto pairb = model_minus_forces(src, m, md, pair.b);
			vec_sub(*paira, res->force);
			vec_add(*pairb, res->force);
		}
	}
#endif
}
FORCE_INLINE void c_model_collect_deriv_1(ModelDesc *m, Flt *e, Flt *md) {
	SrcModel *src = m->src;

	// accumulate e as deriviative result
	if (IS_2DMAIN()) {
		*e = 0;
		FOR(i, src->movable_atoms) {
            auto me = model_movable_e(src, m, md, i);
			*e += *me;
            CUDBG("me %d %f %p total %f", i, *me, me, *e);
		}
        CUDBG("ce0 %f %p", *e, e);
		FOR(i, src->npairs) {
			auto res = model_pair_res(src, m, md, i);
			*e += res->e;
            // printf("pe %d %f\n", i, res->e);
		}
        CUDBG("ce1 %f %p", *e, e);
	}
	// accumulate minus_forces
	if (IS_2DSUB()) {
		FOR(i, src->npairs) {
			InteractingPair &pair = src->pairs[i];
			auto res = model_pair_res(src, m, md, i);
            auto paira = model_minus_forces(src, m, md, pair.a);
            auto pairb = model_minus_forces(src, m, md, pair.b);
			vec_sub(*paira, res->force);
			vec_add(*pairb, res->force);
		}
	}
}
FORCE_INLINE void c_model_eval_deriv_ligand_sum_ft(ModelDesc *m, Flt *md) {
    SrcModel *src = m->src;
    auto coords = model_coords(src, m, md);
    CU_FORY(i, src->nligand) {
        Ligand &ligand  = src->ligands[i];
        // first calculate all node force and torque, only for node itself, not include sub-nodes
        CU_FOR(j, ligand.nr_node) {
            CUDBG("ligand %d", j);
            auto var = model_ligand(src, m, md, i, j);
            auto minus_forces = model_minus_forces(src, m, md);
            sum_force_and_torque<Segment>(ligand.tree[j], var->origin, coords, minus_forces, var->ft);
        }
    }

}

FORCE_INLINE void c_model_eval_deriv_ligand(ModelDesc *m, Flt *gs, Flt *md) {
    SrcModel *src = m->src;
    CU_FOR2(i, src->nligand) {
        Flt *g = get_ligand_change(src, gs, i);

        Ligand &ligand  = src->ligands[i];
        // climbing from the leaves to root and accumulate force and torque
        if (IS_2DMAIN()) {
            FOR(j, ligand.nr_node - 1) {
                Segment &seg        = ligand.tree[j];
                auto segvar = model_ligand(src, m, md, i, j);
                CUDBG("ligand %d", j);
                Segment &parent        = ligand.tree[seg.parent];
                auto parentVar = model_ligand(src, m, md, i, seg.parent);
                CUDBG("    parent %d", seg.parent);
                CUVECPDUMP("    parent ft", parentVar->ft);
                CUVECPDUMP("    child ft", segvar->ft);
                CUVDUMP("    parent origin", parentVar->origin);
                CUVDUMP("    child origin", segvar->origin);
                vec_add(parentVar->ft.first, segvar->ft.first);

                // this is not a leaf, calculate torque with new force
                Vec r, product;
                vec_sub(segvar->origin, parentVar->origin, r);
                cross_product(r, segvar->ft.first, product);
                vec_add(product, segvar->ft.second);
                vec_add(parentVar->ft.second, product);
                CUVECPDUMP("    childft added", parentVar->ft);
                // note that torsions has reversed order from segments
                set_derivative(segvar->axis, segvar->ft,
                               get_ligand_change_torsion(g, ligand.nr_node - j - 2));
                CUDBG("set ligands %d ligand %d torsion[%d] %f", i, j, ligand.nr_node - j - 2,
                      get_ligand_change_torsion(g, ligand.nr_node - j - 2));
                CUVDUMP("    axis", segvar->axis);
                CUVECPDUMP("    ft", segvar->ft);
            }
            auto svar = model_ligand(src, m, md, i, ligand.nr_node - 1);
            set_derivative(svar->ft, g);
        }
    }
}
FORCE_INLINE void c_model_eval_deriv_flex_sum_ft(ModelDesc *m, Flt *md) {
    SrcModel *src = m->src;
    CU_FORY(i, src->nflex) {
        Residue &flex    = src->flex[i];
        // first calculate all node force and torque, only for node itself, not include sub-nodes
        CU_FOR(j, flex.nr_node) {
            CUDBG("flex %d", j);
            auto coords = model_coords(src, m, md);
            auto minus_forces = model_minus_forces(src, m, md);
            auto seg = model_flex(src, m, md, i, j);
            sum_force_and_torque<Segment>(flex.tree[j], seg->origin, coords, minus_forces, seg->ft);
        }
    }
}
FORCE_INLINE void c_model_eval_deriv_flex(ModelDesc *m, Flt *gs, Flt *md) {
    SrcModel *src = m->src;
    CU_FOR2(i, src->nflex) {
        Flt *g           = get_flex_change(src, gs, i);
        Residue &flex    = src->flex[i];
        if (IS_2DMAIN()) {
            FOR(j, flex.nr_node) {
                Segment &seg        = flex.tree[j];
                auto segvar = model_flex(src, m, md, i, j);
                if (seg.parent >= 0) {
                    Segment &parent        = flex.tree[seg.parent];
                    auto parentVar = model_flex(src, m, md, i, seg.parent);
                    vec_add(parentVar->ft.first, segvar->ft.first);

                    // this is not a leaf, calculate torque with new force
                    Vec r, product;
                    vec_sub(segvar->origin, parentVar->origin, r);
                    cross_product(r, segvar->ft.first, product);
                    vec_add(product, segvar->ft.second);
                    vec_add(parentVar->ft.second, product);
                }
                // note that torsions has reversed order from segments
                set_derivative(segvar->axis, segvar->ft,
                               get_flex_change_torsion(g, flex.nr_node - j - 1));
            }
        }
    }
    // climbing from the leaves to root and accumulate force and torque
}

};  // namespace dock
