
#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
#include "grid_c.cuh"
#include "conf_c.cuh"
#include "model_c.cuh"
#include <algorithm>
#include <cmath>
namespace dock {

struct threadids {
    int tid, blksz;
};
#define INMAIN() (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
#define X0 (threadIdx.x == 0)
#define Z0 (threadIdx.z == 0)
#define TIDS tids
#define DECL_TIDS threadids tids

#define copy_change(dst, src) copy_array<Flt>(dst, src, ng, tids)
#define copy_conf(dst, src) copy_array<Flt>(dst, src, nc, tids)


// todo: do we need these? if we need, use threadIdx correctly
template <typename Ligand> // T = LigandChange, or LigandConf
FORCE_INLINE void copy_ligand(int ntorsion, Ligand * dst, const Ligand *src) {
    // 3 for rigid position, 3 for rigid orientation, ntorsion for torsion
    for (int x = threadIdx.x; x < 6+ntorsion; x += blockDim.x) {
        if (x < 3) {
            vec_set(dst->rigid.position, x, vec_get(src->rigid.position, x));
        } else if(x < 6) {
            vec_set(dst->rigid.orientation, x-3, vec_get(src->rigid.orientation, x-3));
        } else {
            dst->torsions[x-6] = src->torsions[x-6];
        }
    }
}
template <typename Residue> // T = ResidueChange, or ResidueConf
FORCE_INLINE void copy_flex(int ntorsion, Residue * dst, const Residue *src) {
    for (int x = threadIdx.x; x < ntorsion; x += blockDim.x) {
        dst->torsions[x] = src->torsions[x];
    }
}
template<typename T, typename Ligand, typename Residue>
FORCE_INLINE void copy_change_conf(SrcModel *sm, T * dst, const T *src) {
    for (int y = threadIdx.y; y < sm->nligand + sm->nflex; y += blockDim.y) {
        // y to choose ligand or flex
        if (y < sm->nligand) {
            int ntorsion = sm->ligands[y].nr_node - 1;
            copy_ligand<Ligand>(ntorsion, dst->ligands+y, src->ligands+y);
        } else {
            int idx = y-sm->nligand;
            int ntorsion = sm->flex[idx].nr_node;
            copy_flex<Residue>(ntorsion, dst->flex+idx, src->flex+idx);
        }
    }
}

#define THREADID (blockDim.x *blockDim.y * threadIdx.z +blockDim.x * threadIdx.y + threadIdx.x)
#define BLOCKSZ (blockDim.x * blockDim.y * blockDim.z)
#define FOR_TID(i, sz) for(int i = tids.tid; i < sz; i += tids.blksz)

FORCE_INLINE void convert_conf(SrcModel *sm, Conf * dst, const Flt *src) {
    if (threadIdx.z == 0) {
        CU_FORY(idx, sm->nligand) {
            LigandConf *ligand = &dst->ligands[idx];
            auto p = get_ligand_conf(sm, src, idx);
            CU_FOR (i, sm->ligands[idx].nr_node-1) {
                if(i == 0) {
                    make_vec(ligand->rigid.position, p[0], p[1], p[2]);
                    qt_set(ligand->rigid.orientation, p[3], p[4], p[5], p[6]);
                }
                ligand->torsions[i] = p[i + 7];
            }
        }

    } else if(threadIdx.z == 1) {
        CU_FORY(idx, sm->nflex) {
            ResidueConf *flex = &dst->flex[idx];
            auto p = get_flex_conf(sm, src, idx);
            CU_FOR (i, sm->flex[idx].nr_node-1) {
                flex->torsions[i] = p[i];
            }
        }
    }
}
FORCE_INLINE void read_conf(SrcModel *sm, const Conf * dst, Flt *src) {
    if (threadIdx.z == 0) {
        CU_FORY(idx, sm->nligand) {
            LigandConf *ligand = &dst->ligands[idx];
            auto p = get_ligand_conf(sm, src, idx);
            CU_FOR (i, sm->ligands[idx].nr_node-1) {
                if(i == 0) {
                    p[0] = ligand->rigid.position.x;
                    p[1] = ligand->rigid.position.y;
                    p[2] = ligand->rigid.position.z;
                    p[3] = ligand->rigid.orientation.x;
                    p[4] = ligand->rigid.orientation.y;
                    p[5] = ligand->rigid.orientation.w;
                    p[6] = ligand->rigid.orientation.z;
                }
                p[i+7] = ligand->torsions[i];
            }
        }

    } else if(threadIdx.z == 1) {
        CU_FORY(idx, sm->nflex) {
            ResidueConf *flex = &dst->flex[idx];
            auto p = get_flex_conf(sm, src, idx);
            CU_FOR (i, sm->flex[idx].nr_node-1) {
                 p[i] = flex->torsions[i];
            }
        }
    }
}

// dst = src
template<typename T>
FORCE_INLINE void copy_array(T * dst, const T *src, int sz, const threadids &tids) {
    FOR_TID(idx, sz) {
        dst[idx] = src[idx];
    }
}

// dst -= src
template<typename T>
FORCE_INLINE void sub_array(T * dst, const T *src, int sz,  const threadids &tids) {
    FOR_TID(idx, sz) {
        dst[idx] -= src[idx];
    }
}

// dst = left - right
template<typename T>
FORCE_INLINE void sub_array(T * dst, const T *left, const T *right, int sz, const threadids &tids) {
    FOR_TID(idx, sz) {
        dst[idx] = left[idx] - right[idx];
    }
}
FORCE_INLINE void init_triangle_mat(Flt *v, int ng) {
    // x for index in line
    // y index of line
    if (Z0) {
        for (int y = threadIdx.y; y < ng; y += blockDim.y) {
            int w = y + 1;  // w elements in line y
            int offset = ((y + 1) * (y + 2)) >> 1; // where this line starts
            for (int x = threadIdx.x; x < w; x += blockDim.x) {
                v[offset + x] = x == y ? 1.0 : 0.;
            }
        }
    }
}
FORCE_INLINE int triangular_matrix_index(int ng, int i, int j) {
	return i + j*(j+1)/2; 
}
FORCE_INLINE int triangular_matrix_index_permissive(int ng, int i, int j) {
	return (i <= j) ? triangular_matrix_index(ng, i, j)
		            : triangular_matrix_index(ng, j, i);
}
FORCE_INLINE Flt get_change_value(const SrcModel *sm, const Change *c, int idx) {
    for (int i = 0; i < sm->nligand; i++) {
        LigandChange *lc = c->ligands + i;
        if (idx < 3) return vec_get(lc->rigid.position, idx);
        idx -= 3;
        if (idx < 3) return vec_get(lc->rigid.orientation, idx);
        idx -= 3;
        if (idx < sm->ligands[i].nr_node-1) return lc->torsions[idx];
        idx -= sm->ligands[i].nr_node-1;
    }
    for (int i = 0; i < sm->nflex; i++) {
        ResidueChange *rc = c->flex+i;
        if (idx < sm->flex[i].nr_node) return rc->torsions[idx];
        idx -= sm->flex[i].nr_node;
    }
    printf("get_change_value fails\n");
    return 0;

}
FORCE_INLINE void set_change_value(const SrcModel *sm, const Change *c, int idx, Flt v) {
    for (int i = 0; i < sm->nligand; i++) {
        LigandChange *lc = c->ligands + i;
        if (idx < 3) {
            vec_set(lc->rigid.position, idx, v);
            return;
        }
        idx -= 3;
        if (idx < 3) {
            vec_set(lc->rigid.orientation, idx, v);
            return;
        }
        idx -= 3;
        if (idx < sm->ligands[i].nr_node-1) {
            lc->torsions[idx] = v;
            return;
        }
        idx -= sm->ligands[i].nr_node-1;
    }
    for (int i = 0; i < sm->nflex; i++) {
        ResidueChange *rc = c->flex+i;
        if (idx < sm->flex[i].nr_node){
             rc->torsions[idx] = v;
             return;
        }
        idx -= sm->flex[i].nr_node;
    }
    printf("set_change_value fails\n");

}
// FORCE_INLINE void minus_mat_vec_product(const BFGSCtx *ctx, const SrcModel *sm, const Flt* h, const	Change * in, Change* out, Flt *tmp) {
// 	int ng = ctx->dimh;
//     // use blockDim.x * blockDim.y for sums
//     Flt *sums = tmp; // [rows: blockDim.y, cols: blockDim.x]
// 	for (int i = threadIdx.y; i < ng; i += blockDim.y) {
//         Flt *sum = sums + i * blockDim.x;
// 		for (int j = threadIdx.x; j < ng; j += blockDim.x) {
// 			sum[threadIdx.x] += h[triangular_matrix_index_permissive(ng, i, j)] * get_change_value(sm, in, j);
// 		}
// 	}
//     SYNC();
//     if (threadIdx.x == 0) {
//         for (int i = threadIdx.y; i < ng; i += blockDim.y) {
//             Flt *sum = sums + i * blockDim.x;
//             for (int j =1; j < blockDim.x; j++) {
//                 sum[0] += sum[j];
//             }
//             set_change_value(sm, out, i, -sum[0]);
//         }
//     }

// }
// tmp: size requirement blockDim.x * blockDim.y
// todo:
FORCE_INLINE void minus_mat_vec_product(const Flt* h, const	Flt * in, Flt* out, Flt *tmp, int ng, const threadids &tids) {
    // use blockDim.x * blockDim.y for sums
    Flt *sums = tmp; // [rows: blockDim.y, cols: blockDim.x]
    FOR_TID(idx, blockDim.x * blockDim.y) {
        sums[idx] = 0;
    }
    if (Z0) {
        for (int i = threadIdx.y; i < ng; i += blockDim.y) {
            Flt *sum = sums + i * blockDim.x;
            for (int j = threadIdx.x; j < ng; j += blockDim.x) {
                sum[threadIdx.x] += h[triangular_matrix_index_permissive(ng, i, j)] * in[j];
            }
        }
    }
    SYNC();
    if (X0 && Z0) {
        for (int i = threadIdx.y; i < ng; i += blockDim.y) {
            Flt *sum = sums + i * blockDim.x;
            for (int j =1; j < blockDim.x; j++) {
                sum[0] += sum[j];
            }
            out[i] = -sum[0];
        }
    }

}
// calc sqrt(sum(v[x]^2)) and set to v[0], v will be corrupted
FORCE_INLINE void sqrt_product_array(Flt *v, int n, const threadids &tids) {
    FOR_TID(idx, n) {
        Flt item = v[idx];
        v[idx] = item * item;
    }
    SYNC();
    if (INMAIN()) {
        Flt sum = 0;
        for (int i = 0; i < n; i++) {
            sum += v[i];
        }
        v[0] = sqrt(sum);
    }
}
// calc sqrt(sum(v[x]^2)) and set to tmp[0], tmp should has same siz of v, which is n
FORCE_INLINE void scalar_product_array(const Flt *v, Flt *tmp, int n, const threadids &tids) {
    FOR_TID(idx, n) {
        Flt item = v[idx];
        tmp[idx] = item * item;
    }
    SYNC();
    if (INMAIN()) {
        Flt sum = 0;
        for (int i = 0; i < n; i++) {
            sum += tmp[i];
        }
        tmp[0] = sqrt(sum);
    }
}
// calc sqrt(sum(v[x]^2)) and set to tmp[0], tmp should has same siz of v
// tmp should have size of n
FORCE_INLINE void scalar_product_array(const Flt *v1, const Flt *v2,  Flt *tmp, int n, const threadids &tids) {
    FOR_TID(idx, n) {
        tmp[idx] = v1[idx] * v2[idx];
    }
    SYNC();
    if (INMAIN()) {
        Flt sum = 0;
        for (int i = 0; i < n; i++) {
            sum += tmp[i];
        }
        tmp[0] = sqrt(sum);
    }
}
FORCE_INLINE void set_diagonal(Flt *h, Flt v, int ng, const threadids &tids) {
    // x for index in line
    // y index of line
    FOR_TID(i, ng) {
        int idx = (i * (i+1)) << 1 - 1;
        h[idx] = v;
    }
}
#if 0
FORCE_INLINE void set_diagonal(Flt *h, Flt v, int ng) {
    // x for index in line
    // y index of line
    if (Z0) {
    for (int y = threadIdx.y; y < ng; y += blockDim.y) {
        int w = y + 1;  // w elements in line y
        int offset = ((y + 1) * (y + 2)) >> 1; // where this line starts
        for (int x = threadIdx.x; x < w; x += blockDim.x) {
            if (x == y) {
                h[offset + x] = v;
            }
        }
    }

    }
}
#endif

// todo


// cf: input conf, read only
// cg: output change, write only
// e: output loss
FORCE_INLINE void model_eval_deriv(Model *m, const PrecalculateByAtom *p, const Cache *c, const Flt *cf, Flt *cg, Flt *e) {
    // sed model::set, update conf
    model_set_conf_ligand_c(m, cf);
    model_set_conf_flex_c(m, cf);
    SYNC();

    c_cache_eval_deriv(c, m);
    SYNC();
    // depends on c_cache_eval_deriv
    c_model_eval_deriv_pairs(m, p);
    SYNC();

    c_model_collect_deriv(m, e);
    SYNC();

    c_model_eval_deriv_ligand_sum_ft(m);
    c_model_eval_deriv_flex_sum_ft(m);
    SYNC();

    c_model_eval_deriv_ligand(m, cg);
    c_model_eval_deriv_flex(m, cg);

}

// require max(blockDim.x*blockDim.y, ng)
FORCE_INLINE void bfgs_update(Flt *h, const Flt *p, const Flt *y, Flt *hy, Flt *tmp, Flt alpha, int ng, const threadids &tids) {
    scalar_product_array(y, p, tmp, ng, tids);
    SYNC();
    Flt yp = tmp[0];
    Flt beta = alpha * yp;
    if (beta >= EPSILON) {
        minus_mat_vec_product(h, y, hy, tmp, ng, tids);
        SYNC();
        scalar_product_array(y, hy, tmp, ng, tids);
        SYNC();
        Flt yhy = -tmp[0];
        Flt r = reciprocal(beta);
        if (Z0) {
            for (int i = threadIdx.y; i < ng; i += blockDim.y) {
                // for (j = i; j < ng; j++)
                // turn to :
                // for (k = threadIdx.x; k < ng - i; k += blockDim.x) {
                //     j = k + i
                // }
                // // this will enable the threadIdx.x working when threadIdx.x < i
                #if 0
                for (int j = threadIdx.x; j < ng; j += blockDim.x) {
                    if (j >= i) {
                        int idx = triangular_matrix_index(ng, i, j);
                        h[idx] += alpha * r * hy[i] * p[j] + hy[j] * p[i]
                                  + alpha * alpha * (r * r * yhy + r) * p[i] * p[j];
                    }
                }
                #else
                for (int k = threadIdx.x; k < ng - i; k += blockDim.x) {
                    int j = k + i; // note that j >= i && j < ng 
                    int idx = triangular_matrix_index(ng, i, j);
                    h[idx] += alpha * r * hy[i] * p[j] + hy[j] * p[i]
                                + alpha * alpha * (r * r * yhy + r) * p[i] * p[j];
                }
                #endif
            }
        }

    }
}

FORCE_INLINE void quaternion_increment(Flt *out, const Flt * q, const Flt * rotation) {
	// assert(quaternion_is_normalized(q));
    Flt qt[4];
    angle_to_quaternion(rotation, qt);
    qt_multiple(out, qt, q);
	quaternion_normalize_approx(out); // normalization added in 1.1.2
	//quaternion_normalize(q); // normalization added in 1.1.2
}
// use threadIdx.x only
// g ligand change, c: ligand conf, out: output ligand conf
FORCE_INLINE void ligand_conf_increament(const Flt *g, const Flt *c, Flt *out, int ntorsion, Flt alpha) {
    // rigid conf increament
    out[0] = c[0] + alpha * g[0], out[1] = c[1] + alpha * g[1], out[2] = c[2]+ alpha * g[2]; // set position
    Flt rot[3];
    rot[0] = alpha * g[3], rot[1] = alpha * g[4], rot[2] = alpha * g[5];
    quaternion_increment(out+3, c+3, rot); // set orientation

    // torsions increament
    const Flt *gt = g + 6;
    const Flt *ct = c + 7;
    Flt *outct = out + 7;
    for (int i = 0; i < ntorsion; i++) {
        outct[i] = normalized_angle(ct[i] + normalized_angle(gt[i] * alpha));
    }
}
FORCE_INLINE void flex_conf_increament(const Flt *g, const Flt *c, Flt *out, int ntorsion, Flt alpha) {
    // torsions increament
    const Flt *gt = g;
    const Flt *ct = c;
    Flt *outct = out;
    for (int i = 0; i < ntorsion; i++) {
        outct[i] = normalized_angle(ct[i] + normalized_angle(gt[i] * alpha));
    }
}
FORCE_INLINE void conf_increament(SrcModel *sm, const Flt *g, const Flt *c, Flt *out, Flt alpha) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bsz = blockDim.x * blockDim.y;
    if (tid < 32) {
        for (int dx = tid; dx < sm->nligand; dx += bsz) {
            int ntorsion  = sm->ligands[dx].nr_node - 1;
            const Flt *lc = get_ligand_conf(sm, c, dx);
            Flt *outc     = get_ligand_conf(sm, out, dx);
            const Flt *lg = get_ligand_change(sm, g, dx);
            ligand_conf_increament(lg, lc, outc, ntorsion, alpha);
            if (dx < sm->nligand) {
            } else {
                int idx       = dx - sm->nligand;  // index of flex
                int ntorsion  = sm->flex[idx].nr_node - 1;
                const Flt *lc = get_flex_conf(sm, c, idx);
                Flt *outc     = get_flex_conf(sm, out, idx);
                const Flt *lg = get_flex_change(sm, g, idx);
                flex_conf_increament(lg, lc, outc, ntorsion, alpha);
            }
        }
    } else if (tid < 64) {
        for (int dx = tid; dx < sm->nflex; dx += bsz) {
            int ntorsion  = sm->flex[dx].nr_node - 1;
            const Flt *lc = get_flex_conf(sm, c, dx);
            Flt *outc     = get_flex_conf(sm, out, dx);
            const Flt *lg = get_flex_change(sm, g, dx);
            flex_conf_increament(lg, lc, outc, ntorsion, alpha);
            
        }
    }
}
// since warp is 32, make it 16 is much better
#define MAX_TRIALS 8
__device__ Flt multipliers[MAX_TRIALS] = { 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125/*, 0.00390625, 0.001953125 */};

// ng, nc: size of Flt for a change and conf
// tmp: require MAX_TRIALS * (nc+ng+1) Flts
// outc: write only, output conf
// outg: write only, output change
__device__ void line_search(Model *m, PrecalculateByAtom *pa, Cache *ch, Flt f0, Flt &f1,
                const Flt *c, const Flt *g,Flt *p, Flt *tmp, Flt *outc, Flt *outg, int ng, int nc, int &evalcnt, Flt &out_alpha, const threadids &tids) {
    const Flt c0 = 0.0001;
    scalar_product_array(p, g, tmp, ng, TIDS); // tmp require n
    SYNC();
    Flt pg = tmp[0];

    Flt *es = tmp;
    tmp += MAX_TRIALS;
    for (int dz = threadIdx.z; dz < MAX_TRIALS; dz += blockDim.z) {
        Flt alpha = multipliers[dz];
        Flt *tc = tmp + (nc+ng) * dz; // c_new
        Flt *tg = tc + nc; // g_new
        Flt *e = &es[dz];

        conf_increament(m->src, p, c, tc, alpha);
        SYNC();
        model_eval_deriv(m, pa, ch, tc, tg, e);
        SYNC();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            CUDBG("line search %d der e %f", dz, *e);
        }
    }

    int best      = -1;
    if (INMAIN()) {
        // find min(f1) for (f1 - f0 < threshold)
        for (int dy = 0; dy < MAX_TRIALS; dy++) {
            Flt alpha = multipliers[dy];
            Flt threshold = c0 * alpha * pg;
            if (es[dy] - f0 < threshold) {
                if (best == -1 || es[dy] < f1) {
                    best = dy, f1 = es[dy];
                }
            }
        }

        if (best == -1) {
            best = MAX_TRIALS - 1;
            f1   = es[best];
        }
        evalcnt += best+1;
        out_alpha = multipliers[best];
    }
    SYNC();
    Flt *tc = tmp + (nc+ng) * best; // c_new
    Flt *tg = tc + nc; // g_new
    copy_change(outg, tg);
    copy_conf(outc, tc);

}

#define BFGSIDX threadIdx.z
__device__ void bfgs(Model *m, PrecalculateByAtom *pa, Cache *ch, BFGSCtx *ctx, int max_steps, Flt average_required_improvement, Size over, Flt *c, Flt *f0, Flt *mem) {
    int offset = 0; // mem alloc offset
    DECL_TIDS;
    tids.tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x *blockDim.y;
    tids.blksz = blockDim.x * blockDim.y * blockDim.z;
    SrcModel *srcm = m->src;
    int ng = srcm->nrflts_change;
    int nc = srcm->nrflts_conf;

    // mem requirement:
    //    4 + ((ng * (ng+1)) >> 1) + 6 *ng + 3 *nc + (max_steps + 1) + max(max(blockDim.x * blockDim.y, ng), MAX_TRIALS * (nc+ng+1)) 
    Flt *h, *g, *g_new, *c_new, *g_orig, *c_orig, *p, *y, *fs, *hy, *tmp;
    Flt * f_orig, *f1, *alpha;
    // f0 = mem + offset, offset += 1;
    f1 = mem + offset, offset += 1;
    f_orig = mem + offset, offset += 1;
    alpha = mem + offset, offset += 1;
    h = mem + offset, offset += ((ng * (ng+1)) >> 1);
    g = mem + offset, offset += ng;
    // c = mem + offset, offset += nc;
    g_new = mem + offset, offset += ng;
    c_new = mem + offset, offset += nc;
    g_orig = mem + offset, offset += ng;
    c_orig = mem + offset, offset += nc;
    p = mem + offset, offset += ng;
    y = mem + offset, offset += ng;
    hy = mem + offset, offset += ng;
    fs = mem + offset, offset += max_steps + 1;
    tmp = mem + offset; offset += max(max(blockDim.x * blockDim.y, ng), MAX_TRIALS * (nc+ng+1)) ;

    init_triangle_mat(h, ng);
    // in cpu version, here is a declare and copy, but there is no need because it will be used in linesearch for
    // output only.
    // copy_change(g_new, g);
    // copy_conf(c_new, c);
    // SYNC();

    if (BFGSIDX == 0) {
        // initial evaluation, get f0, initial conf and change
        model_eval_deriv(m, pa, ch, c, g, f0);
    }
    SYNC();

    copy_change(g_orig, g);
    copy_conf(c_orig, c);
    copy_change(p, g);
    if (INMAIN()) {
        *f_orig = *f0;
        fs[0] = *f0;
    }

    SYNC();
    for(int step = 0; step < max_steps; step++) {
        minus_mat_vec_product(h, g, p, mem, ng, TIDS);
        SYNC();
        // requires MAX_TRIALS * (nc+ng+1)
        line_search(m, pa, ch, *f0, *f1, c, g, p, tmp, c_new, g_new, ng, nc, ctx->eval_cnt, *alpha, TIDS);
        SYNC();
        // y = g_new - g
        sub_array<Flt>(y, g_new, g, ng, TIDS);
        if (INMAIN()) {
            fs[step+1] = *f1; 
            *f0 = *f1;
        }
        copy_conf(c, c_new);
        sqrt_product_array(g, ng , TIDS);
        SYNC();
        if (!(g[0] >= 1e-5)) {
            // done
            break;
        }
        
        if (step == 0) {
            // use g as tmp, note that we move the g = g_new after this
            scalar_product_array(y, g, ng, TIDS);
            SYNC();
            Flt yy = g[0];
            if (abs(yy) > EPSILON) {
                scalar_product_array(y, p, g, ng , TIDS);
                SYNC();
                Flt beta = *alpha * g[0] / yy;
                set_diagonal(h, beta, ng, TIDS);
                SYNC();
            }
        }

        copy_change(g, g_new);
        // require max(blockDim.x*blockDim.y, ng)
        bfgs_update(h, p, y, hy, tmp, *alpha, ng, TIDS);
        SYNC();
    }

    if (!(*f0 <= *f_orig)) {
        *f0 = *f_orig;
        copy_conf(c, c_orig);
        copy_change(g, g_orig);
    }
}
// used for single mc call
GLOBAL void bfgs_kernel(Model *m, PrecalculateByAtom *pa, Cache *ch, BFGSCtx *ctx, int max_steps, Flt average_required_improvement, Size over) {
    extern __shared__ Flt sm[];
    int nc = m->src->nrflts_conf;
    int offset = 0;
    Flt *c, *f0, *mem;
    c = sm + offset, offset += nc; // bfgs output conf
    f0 = sm + offset,  offset += 1;
    mem = sm + offset; // bfgs temp memory
    read_conf(m->src, &ctx->c, c);
    SYNC();

    bfgs(m, pa, ch, ctx, max_steps, average_required_improvement, over, c, f0, mem);
    SYNC();

    convert_conf(m->src, &ctx->c, c);
    if (INMAIN()) {
        ctx->e = *f0;
    }
}
void run_bfgs(Model *cpum, Model *m, PrecalculateByAtom *pa, Cache *ch, BFGSCtx *ctx, int max_steps, Flt average_required_improvement, Size over , cudaStream_t stream) {
    auto srcm = cpum->src;
    int ng = srcm->nrflts_change;
    int nc = srcm->nrflts_conf;

    // sm requirement:
    dim3 block(4, 8, MAX_TRIALS);
    dim3 grid(1);
    int sm =  4 + ((ng * (ng+1)) >> 1) + 6 *ng + 3 *nc + (max_steps + 1) + std::max(std::max(int(block.x * block.y), ng), MAX_TRIALS * (nc+ng+1));
    sm *= sizeof(Flt);
    printf("ng %d nc %d sm size %d\n", ng, nc, sm);
    bfgs_kernel<<<grid, block, sm, stream>>>(m, pa, ch, ctx, max_steps, average_required_improvement, over);
}
};