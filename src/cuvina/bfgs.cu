#include <assert.h>
#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
#include "grid_c.cuh"
#include "conf_c.cuh"
#include "model_c.cuh"
#include <algorithm>
#include <cmath>
#include "model_desc.cuh"
#include "bfgsdef.h"
namespace dock {
struct threadids {
    int tid, blksz;
};
#define BFGSDEBUG 0
#if BFGSDEBUG
#define REQUIRED() (threadIdx.z == 0 && IS_2DMAIN())
#define MUSTED() (threadIdx.z == 0 && IS_2DMAIN())
#else
#define REQUIRED() false
#define MUSTED() false
#endif

#define INMAIN() (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
#define X0 (threadIdx.x == 0)
#define Z0 (threadIdx.z == 0)
#define TIDS tids
#define DECL_TIDS threadids tids

#define copy_change(dst, src) copy_array<Flt>(dst, src, ng, tids)
#define copy_conf(dst, src) copy_array<Flt>(dst, src, nc, tids)
#define copy_change2d(dst, src) copy_array2d<Flt>(dst, src, ng)
#define copy_conf2d(dst, src) copy_array2d<Flt>(dst, src, nc)

#if 0
#define MCPRT(fmt, ...) do{ if(XY0()) printf("%d %d:%d:%d@%d\t" fmt "\n", \
 __LINE__,  threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.z,  __VA_ARGS__);}while(0)
#define MARK() do { if (XY0()) printf("%d %d:%d:%d@%d:%d:%d mark\n", __LINE__, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, lane_id(), warp_id()); } while(0)
#define MARKSTEP(step) do { if (XY0()) printf("%d %d:%d:%d@%d:%d:%d mark step %d\n", __LINE__, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, lane_id(), warp_id(), step); } while(0)
#define MARKXY() do { if (XY0()) printf("%d %d:%d:%d@%d:%d:%d mark %d %d\n", __LINE__, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, lane_id(), warp_id(), sx, sy); } while(0)
#define MARKXYSTEP() do { if (XY0()) printf("%d %d:%d:%d@%d:%d:%d mark %d %d step %d\n", __LINE__, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, lane_id(), warp_id(), sx, sy, step); } while(0)
#else
#define MARK() 
#define MARKSTEP(step) 
#define MARKXY() 
#define MARKXYSTEP() 
#define MCPRT(fmt, ...)
#endif

__device__ void printc(SrcModel *sm, const Flt *c) {
    for (int i = 0; i < sm->nligand; i++) {
        printf("(%f %f %f) (%f %f %f %f)[", c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
        for (int j = 0; j < sm->ligands[i].nr_node-1; j++) {
            if (j > 0) {
                printf(" ");
            }
            printf("%f", c[7+j]);
        }
        printf("]\n");
        c += 7 + sm->ligands[i].nr_node-1;
    }
    for (int i = 0; i < sm->nflex; i++) {
        printf("[");
        for (int j = 0; j < sm->flex[i].nr_node; j++) {
            if (j > 0) {
                printf(" ");
            }
            printf("%f", c[j]);
        }
        printf("]\n");
        c += sm->flex[i].nr_node;
    }
}
__device__ void printg(SrcModel *sm, const Flt *c) {
    for (int i = 0; i < sm->nligand; i++) {
        printf("(%f %f %f) (%f %f %f)[", c[0], c[1], c[2], c[3], c[4], c[5]);
        for (int j = 0; j < sm->ligands[i].nr_node-1; j++) {
            if (j > 0) {
                printf(" ");
            }
            printf("%f", c[6+j]);
        }
        printf("]\n");
        c += 6 + sm->ligands[i].nr_node-1;
    }
    for (int i = 0; i < sm->nflex; i++) {
        printf("[");
        for (int j = 0; j < sm->flex[i].nr_node; j++) {
            if (j > 0) {
                printf(" ");
            }
            printf("%f", c[j]); 
        }
        printf("]\n");
        c += sm->flex[i].nr_node;
    }
}
__device__ void printh(int ng, const Flt *h) {
    int idx = 0;
    for (auto row = 0; row < ng; row++) {
        printf("[%d]:", row);
        for (auto col = 0; col <= row; col++) {
            printf(" %f", h[idx++]);
        }
        printf("\n");
    }
}

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

#define FOR_TID(i, sz) for(int i = tids.tid; i < sz; i += tids.blksz)

FORCE_INLINE void convert_conf(SrcModel *sm, Conf * dst, const Flt *src) {
    if (Z0) {
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
                    p[0] = ligand->rigid.position.d[0];
                    p[1] = ligand->rigid.position.d[1];
                    p[2] = ligand->rigid.position.d[2];
                    p[3] = ligand->rigid.orientation.d[0];
                    p[4] = ligand->rigid.orientation.d[1];
                    p[5] = ligand->rigid.orientation.d[2];
                    p[6] = ligand->rigid.orientation.d[3];
                    // printf("conf %f %f %f %f %f %f %f\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
                }
                p[i+7] = ligand->torsions[i];
                // printf("ct %d: %f\n", i, ligand->torsions[i]);
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
template<typename T>
FORCE_INLINE void copy_array2d(T * dst, const T *src, int sz) {
    CU_FOR2(idx, sz) {
        dst[idx] = src[idx];
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
FORCE_INLINE void init_triangle_mat_xy(Flt *v, int ng) {
    // x for index in line
    // y index of line
    for (int y = threadIdx.y; y < ng; y += blockDim.y) {
        int w = y + 1;  // w elements in line y
        int offset = (y * (y + 1)) >> 1; // where this line starts
        for (int x = threadIdx.x; x < w; x += blockDim.x) {
            v[offset + x] = x == y ? 1.0 : 0.;
        }
    }
}
FORCE_INLINE void init_triangle_mat_xyz(Flt *v, int ng) {
    // x for index in line
    // y index of line
    int blk = blockDim.y *blockDim.z;
    for (int y = threadIdx.y + threadIdx.z * blockDim.y; y < ng; y += blk) {
        int w = y + 1;  // w elements in line y
        int offset = (y * (y + 1)) >> 1; // where this line starts
        for (int x = threadIdx.x; x < w; x += blockDim.x) {
            v[offset + x] = x == y ? 1.0 : 0.;
        }
    }
}
FORCE_INLINE int triangular_matrix_index(int ng, int i, int j) {
	return i + ((j*(j+1)) >> 1); 
}
FORCE_INLINE int triangular_matrix_index_permissive(int ng, int i, int j) {
	return (i < j) ? triangular_matrix_index(ng, i, j)
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
// tmp: size requirement ng * ng
// out: ng flts
FORCE_INLINE void minus_mat_vec_product(const Flt* h, const	Flt * in, Flt* out, Flt *tmp, int ng) {
    // use blockDim.x * blockDim.y for sums
    Flt *sums = tmp; // [rows: blockDim.y, cols: blockDim.x]
    if (Z0) {
        for (int i = threadIdx.y; i < ng; i += blockDim.y) {
            // row i
            Flt *sum = sums + i * ng;
            for (int j = threadIdx.x; j < ng; j += blockDim.x) {
                // row i col j
                CUDBG("mmvps %d %d: h[%d] %f * %f", i, j,triangular_matrix_index_permissive(ng, i, j) , h[triangular_matrix_index_permissive(ng, i, j)], in[j]);
                sum[j] = h[triangular_matrix_index_permissive(ng, i, j)] * in[j];
            }
        }
        // sum up each line
        CU_FOR2(row, ng) {
            auto sum = sums + row * ng;
            for (int i = 1; i < ng; i++) {
                sum[0] += sum[i];
            }
            CUDBG("minus_mat_vec_product sum %d: %f", row, sum[0]);
            out[row] = -sum[0];
        }
    }
}
// out: 1D, ng
COULD_INLINE void minus_mat_vec_product3(const Flt *h, const Flt *in, Flt *out, Flt *tmp, int ng,
                                        const threadids &tids) {
    // use blockDim.x * blockDim.y for sums
    Flt *sums = tmp;  // [rows: blockDim.y, cols: blockDim.x]
    CU_FORZ(i, ng) {
        // row i
        Flt *sum = sums + i * ng;
        CU_FOR2(j, ng) {
            // row i col j
            CUDBG("mmvps %d %d: h[%d] %f * %f", i, j, triangular_matrix_index_permissive(ng, i, j),
                  h[triangular_matrix_index_permissive(ng, i, j)], in[j]);
            sum[j] = h[triangular_matrix_index_permissive(ng, i, j)] * in[j];
        }
        // because this block of code runs inside one single warp, so there is no need to sync
        if (IS_2DMAIN()) {
            out[i] = 0;
            FOR(j, ng) {
                out[i] += sum[j];
            }
            out[i] = -out[i];
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
FORCE_INLINE void sqrt_product_array(const Flt *v, int n, Flt *tmp, const threadids &tids) {
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
// calc sum(v[x]^2) and set to out, tmp should has same siz of v, which is n
FORCE_INLINE void scalar_product_array(const Flt *v, Flt *tmp, Flt *out,int n, const threadids &tids) {
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
        *out = sum;
    }
}
// calc sum(v1[x] * v2[x]) for x in [0,n-1] and set to out, tmp should has same siz of v1 and v1, which is n
// out requires n, tmp requires n
FORCE_INLINE void scalar_product_array1(const Flt *v1, const Flt *v2, Flt *out,  Flt *tmp, int n, const threadids &tids) {
    FOR_TID(idx, n) {
        tmp[idx] = v1[idx] * v2[idx];
    }
    SYNC();
    if (INMAIN()) {
        Flt sum = 0;
        for (int i = 0; i < n; i++) {
            sum += tmp[i];
            // printf("scalar %d %f * %f = %f, sum %f\n", i, v1[i], v2[i], tmp[i], sum);
        }
        *out = sum;
    }
}
FORCE_INLINE void scalar_product_array(const Flt *v1, const Flt *v2, Flt *out,  Flt *tmp, int n, const threadids &tids) {
    FOR_TID(idx, n) {
        tmp[idx] = v1[idx] * v2[idx];
    }
    SYNC();
    if (INMAIN()) {
        Flt sum = 0;
        for (int i = 0; i < n; i++) {
            sum += tmp[i];
        }
        *out = sum;
    }
}
FORCE_INLINE void set_diagonal(Flt *h, Flt v, int ng, const threadids &tids) {
    // x for index in line
    // y index of line
    FOR_TID(i, ng) {
        int idx = triangular_matrix_index(ng, i, i);
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

// cf: input conf, read only
// cg: output change, write only
// e: output loss
// m: model description, used for parsing md
// md: model variable data, will be updated
// this runs only in one single warp
FORCE_INLINE void model_eval_deriv(ModelDesc *m, const PrecalculateByAtom *p, const Cache *c, const Flt *cf, Flt *cg, Flt *e, Flt *md, const Flt *vs) {
    // 970
    // sed model::set, update conf
    model_set_conf_ligand_xy_old(m, cf, md);
    model_set_conf_flex_xy(m, cf, md);
    WARPSYNC();

    // 2800
    c_cache_eval_deriv_xy(c, m, md, vs);
    WARPSYNC();

    // 3200
    // depends on c_cache_eval_deriv
    c_model_eval_deriv_pairs(m, p, md, vs);
    WARPSYNC();

    //3980
    c_model_collect_deriv(m, e, md);
    WARPSYNC();

    // 5000
    c_model_eval_deriv_ligand_sum_ft(m, md);
    c_model_eval_deriv_flex_sum_ft(m, md);
    WARPSYNC();

    // 5400
    c_model_eval_deriv_ligand(m, cg, md);
    c_model_eval_deriv_flex(m, cg, md);
}
// evaluate deriviate and get loss value
// already synchronized
FORCE_INLINE void model_eval_deriv_e_xy(ModelDesc *m, const PrecalculateByAtom *p, const Cache *c,
                                         const Flt *cf,  Flt *e, Flt *md, const Flt *vs, Flt * tmp) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blk = blockDim.x * blockDim.y;
    // 2300
    // sed model::set, update conf
    // model_set_conf_ligand_xy_old(m, cf, md); // 4518
    // model_set_conf_ligand_xy(m, cf, md, tmp); // 4194
    model_set_conf_ligand_coords_xy(m, cf, md, tmp);// 3857
    model_set_conf_flex_xy(m, cf, md);
    XYSYNC();

    // 300 ~ 1000
    c_cache_eval_deriv_xy(c, m, md, vs);
    XYSYNC();

    // 800
    // depends on c_cache_eval_deriv
    c_model_eval_deriv_pairs_c(tid, blk, m, p, md, vs);
    XYSYNC();

    //100
    c_model_collect_deriv_e_xy(m, e, md);
    XYSYNC();
}


FORCE_INLINE void model_eval_deriv_update_2warp(ModelDesc *m, Flt *cg, Flt *md) {

    //3980
    c_model_update_forces_xy(m, md);
    XYSYNC();

    // 5000
    c_model_eval_deriv_ligand_sum_ft(m, md);
    c_model_eval_deriv_flex_sum_ft(m, md);
    XYSYNC();

    // 5400
    c_model_eval_deriv_ligand(m, cg, md);
    c_model_eval_deriv_flex(m, cg, md);
}
// for an accepted conf, update deriviate and output change to cg
FORCE_INLINE void model_update_xyz(ModelDesc *m, Flt *cg, Flt *md) {

    //3980
    c_model_update_forces_xyz(m, md);
    SYNC();

    // 5000
    c_model_eval_deriv_ligand_sum_ft_xyz(m, md);
    c_model_eval_deriv_flex_sum_ft_xyz(m, md);
    SYNC();

    // 5400
    c_model_eval_deriv_ligand_xyz(m, cg, md);
    c_model_eval_deriv_flex_xyz(m, cg, md);
    SYNC();
}
#define MODEL_EVAL_DERIV_MEM_SIZE(src) (CACHE_EVAL_MEM_SIZE(src) + WARPSZ + (src)->nrflts_conf * 2)
FORCE_INLINE void model_eval_deriv_e_xyz(ModelDesc *m, const PrecalculateByAtom *p, const Cache *c,
                                         const Flt *cf,  Flt *e, Flt *md, const Flt *vs, Flt *tmp) {
    int tid = THREADID;
    int blk = BLOCKSZ;
    // sed model::set, update conf
    model_set_conf_ligand_xyz(m, cf, md, tmp);
    // if (threadIdx.z == 0) model_set_conf_ligand_xy_old(m, cf, md);
    if (threadIdx.z == 1) model_set_conf_flex_xy(m, cf, md);
    SYNC();

    // 2800
    c_cache_eval_deriv_xyz(c, m, md, vs, tmp);
    DUMP_SEGVARS(m, md);
    SYNC();

    // 3200
    // depends on c_cache_eval_deriv
    c_model_eval_deriv_pairs_c(tid, blk, m, p, md, vs);
    DUMP_SEGVARS(m, md);
    SYNC();

    //3980
    c_model_collect_deriv_e_xyz(m, e, md, tmp);
    DUMP_SEGVARS(m, md);
    SYNC();
}

#define MODEL_EVAL_MEM_SIZE(src) MODEL_EVAL_DERIV_MEM_SIZE(src)
FORCE_INLINE void model_eval_xyz(ModelDesc *m, const PrecalculateByAtom *p, const Cache *c,
                                 const Flt *cf, Flt *cg, Flt *e, Flt *md, const Flt *vs, Flt *tmp) {
    model_eval_deriv_e_xyz(m, p, c, cf, e, md, vs, tmp);
    model_update_xyz(m, cg, md);
}
#define BFGS_UPD_SZ(ng) (ng * ng + ng + 4)
// require ng * ng + ng+ 4
FORCE_INLINE void bfgs_update3(Flt *h, const Flt *p, const Flt *y, Flt *tmp, Flt alpha, int ng, const threadids &tids) {
    Flt *yp, *beta, *r, *yhy, *hy;
    int offset = 0;
    yp = tmp + offset, offset++;
    beta = tmp + offset, offset++;
    r = tmp + offset, offset++;
    yhy = tmp + offset, offset++;
    hy = tmp + offset, offset += ng;
    tmp += offset;
    scalar_product_array(y, p, yp, tmp, ng, tids);
    if (INMAIN()) {
        *beta = alpha * (*yp);
        *r = reciprocal(*beta);
        CUDBG("yp %f r %f", *yp, *r);
    }
    SYNC();
    if (*beta >= EPSILON) {
        minus_mat_vec_product3(h, y, hy, tmp, ng, tids); // require ng * ng
        SYNC();
        scalar_product_array(y, hy, yhy, tmp, ng, tids); // require ng
        if(INMAIN()) {
            Flt tyhy = *yhy, tr = *r;
            tyhy = -tyhy;
            CUDBG("yhy %f", tyhy, tr);
            // alpha * alpha * (r*r * yhy  + r) * p(i) * p(j)
            *yhy = alpha * alpha * (tr * tr * tyhy + tr);
            // alpha * r |* (minus_hy(i) * p(j)|
            *r = tr * alpha;
        }
        SYNC();
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
                    CUDBG("bfgs before update [%d,%d]:h: %f", i, j, h[idx]);
                    h[idx] += (*r) * (hy[i] * p[j] + hy[j] * p[i]) + (*yhy) * p[i] * p[j];

                    CUDBG("bfgs update [%d,%d]: p %f %f hy %f %f h: %f", i, j, p[i], p[j], hy[i], hy[j], h[idx]);
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
#if PERFDEBUG
    if (threadIdx.z == 0) {
        MCUDBG("qin %f %f %f %f", q[0], q[1], q[2], q[3]);
        MCUDBG("qt %f %f %f %f", qt[0], qt[1], qt[2], qt[3]);
    }
#endif
    qt_multiple(out, qt, q);
#if PERFDEBUG
    if (threadIdx.z == 0) {
        MCUDBG("qt out %f %f %f %f", out[0], out[1], out[2], out[3]);
    }
#endif
	quaternion_normalize_approx(out); // normalization added in 1.1.2
	// quaternion_normalize(q); // normalization added in 1.1.2
#if PERFDEBUG
    if (threadIdx.z == 0) {
        MCUDBG("qt final %f %f %f %f", out[0], out[1], out[2], out[3]);
    }
#endif
}
FORCE_INLINE void quaternion_increment_y(Flt *out, const Flt * q, const Flt * rotation) {
	// assert(quaternion_is_normalized(q));
    if (threadIdx.y == 0) {
        angle_to_quaternion(rotation, out);
    }

#if PERFDEBUG
    if (INMAIN()) {
        MCUDBG("qin %f %f %f %f", q[0], q[1], q[2], q[3]);
        MCUDBG("qt %f %f %f %f", out[0], out[1], out[2], out[3]);
    }
#endif
    qt_multiple_c(threadIdx.y, out, q);
#if PERFDEBUG
    if (threadIdx.z == 0) {
        MCUDBG("qt out %f %f %f %f", out[0], out[1], out[2], out[3]);
    }
#endif
    if (threadIdx.y == 0) {
        quaternion_normalize_approx(out); // normalization added in 1.1.2
    }
#if PERFDEBUG
    if (INMAIN()) {
        MCUDBG("qt final %f %f %f %f", out[0], out[1], out[2], out[3]);
    }
#endif
}
// require block size of idx >= 4
FORCE_INLINE void quaternion_increment_c(int idx, Flt *out, const Flt * q, const Flt * rotation) {
	// assert(quaternion_is_normalized(q));
    if (idx == 0) {
        angle_to_quaternion(rotation, out);
    }

#if PERFDEBUG
    if (INMAIN()) {
        MCUDBG("qin %f %f %f %f", q[0], q[1], q[2], q[3]);
        MCUDBG("qt %f %f %f %f", out[0], out[1], out[2], out[3]);
    }
#endif
    qt_multiple_c(idx, out, q);
#if PERFDEBUG
    if (threadIdx.z == 0) {
        MCUDBG("qt out %f %f %f %f", out[0], out[1], out[2], out[3]);
    }
#endif
    if (idx == 0) {
        quaternion_normalize_approx(out); // normalization added in 1.1.2
    }
#if PERFDEBUG
    if (INMAIN()) {
        MCUDBG("qt final %f %f %f %f", out[0], out[1], out[2], out[3]);
    }
#endif
}
// use threadIdx.x only
// g ligand change, c: ligand conf, out: output ligand conf
FORCE_INLINE void ligand_conf_increament(const Flt *g, const Flt *c, Flt *out, int ntorsion, Flt alpha) {
    // rigid conf increament
    out[0] = c[0] + alpha * g[0], out[1] = c[1] + alpha * g[1], out[2] = c[2]+ alpha * g[2]; // set position
    // if(threadIdx.z == 2) MCUDBG("ligand ci torsion %d alpha %f", ntorsion, alpha);
    // if(threadIdx.z == 2) MCUDBG("position g %f %f %f, c %f %f %f", g[0], g[1], g[2], out[0], out[1], out[2]);
    Flt rot[3];
    rot[0] = alpha * g[3], rot[1] = alpha * g[4], rot[2] = alpha * g[5];
    // if(threadIdx.z == 2) MCUDBG("rotation g: %f %f %f, r %f %f %f", g[3], g[4], g[5], rot[0], rot[1], rot[2]);
    // if(threadIdx.z == 2) MCUDBG("conf or: %f %f %f %f", c[3], c[4], c[5], c[6]);
    quaternion_increment(out+3, c+3, rot); // set orientation
    // if(threadIdx.z == 2) MCUDBG("orientation c: %f %f %f %f", out[3], out[4], out[5], out[6]);

    // torsions increament
    const Flt *gt = g + 6;
    const Flt *ct = c + 7;
    Flt *outct = out + 7;
    for (int i = 0; i < ntorsion; i++) {
        outct[i] = normalized_angle(ct[i] + normalized_angle(gt[i] * alpha));
    }
}
FORCE_INLINE void ligand_conf_increament_c(int idx, int blk, const Flt *g, const Flt *c, Flt *out, int ntorsion, Flt alpha) {
    Flt *rot = out; // use out for temp rotation storage
    // rigid conf increament
    if (idx < 3) {
        rot[idx] = alpha * g[idx + 3];
    }
    quaternion_increment_c(idx, out+3, c+3, rot); // set orientation

    if (idx < 3) {
        out[idx] = c[idx] + alpha * g[idx];
    }

    // torsions increament
    const Flt *gt = g + 6;
    const Flt *ct = c + 7;
    Flt *outct = out + 7;
    for(int i = idx; i < ntorsion; i+= blk) {
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
FORCE_INLINE void flex_conf_increament_c(int idx, int blk, const Flt *g, const Flt *c, Flt *out, int ntorsion, Flt alpha) {
    // torsions increament
    const Flt *gt = g;
    const Flt *ct = c;
    Flt *outct = out;
    for(int i = idx; i < ntorsion; i+=blk) {
        outct[i] = normalized_angle(ct[i] + normalized_angle(gt[i] * alpha));
    }
}
FORCE_INLINE void conf_increament(SrcModel *sm, const Flt *g, const Flt *c, Flt *out, Flt alpha) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bsz = blockDim.x * blockDim.y;
    for (int dx = tid; dx < sm->nligand; dx += bsz) {
        int ntorsion  = sm->ligands[dx].nr_node - 1;
        const Flt *lc = get_ligand_conf(sm, c, dx);
        Flt *outc     = get_ligand_conf(sm, out, dx);
        const Flt *lg = get_ligand_change(sm, g, dx);
        ligand_conf_increament(lg, lc, outc, ntorsion, alpha);
    }
    for (int dx = tid; dx < sm->nflex; dx += bsz) {
        int ntorsion  = sm->flex[dx].nr_node - 1;
        const Flt *lc = get_flex_conf(sm, c, dx);
        Flt *outc     = get_flex_conf(sm, out, dx);
        const Flt *lg = get_flex_change(sm, g, dx);
        flex_conf_increament(lg, lc, outc, ntorsion, alpha);
        
    }
}
FORCE_INLINE void conf_increament_xy(SrcModel *sm, const Flt *g, const Flt *c, Flt *out, Flt alpha) {
    CU_FORY (i, sm->nligand) {
        int ntorsion  = sm->ligands[i].nr_node - 1;
        const Flt *lc = get_ligand_conf(sm, c, i);
        Flt *outc     = get_ligand_conf(sm, out, i);
        const Flt *lg = get_ligand_change(sm, g, i);
        ligand_conf_increament_c(threadIdx.x, blockDim.x, lg, lc, outc, ntorsion, alpha);
    }
    CU_FORY (i , sm->nflex) {
        int ntorsion  = sm->flex[i].nr_node - 1;
        const Flt *lc = get_flex_conf(sm, c, i);
        Flt *outc     = get_flex_conf(sm, out, i);
        const Flt *lg = get_flex_change(sm, g, i);
        flex_conf_increament_c(threadIdx.x, blockDim.x, lg, lc, outc, ntorsion, alpha);
        
    }
}
// since warp is 32, make it 16 is much better
extern int bfgs_max_trials() {
    return MAX_TRIALS;
}
__device__ Flt multipliers[]
    =  //{ 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125,
       //0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125};
    {
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.0078125,
        0.00390625,
        0.001953125,
        0.0009765625,
        0.00048828125,
        0.000244140625,
        0.0001220703125,
        6.103515625e-05,
        3.0517578125e-05,
        1.52587890625e-05,
        7.62939453125e-06,
        3.814697265625e-06,
        1.9073486328125e-06
    };

#define LINESRCH_SIZE(nc, ng) (MAX_TRIALS * ((3 * (nc))+ 2) + (ng) + 2)
// #define LINESRCH_SIZE(nc, ng) (MAX_TRIALS * ((nc)+2) + (ng) + 2)

// ng, nc: size of Flt for a change and conf
// tmp: require MAX_TRIALS * (nc+2) + ng + 2 Flts
// outc: write only, output conf
// outg: write only, output change
// tmp layout:
// | pg(1) | es(MAX_TRIALS) | tc0(nc) tg0(ng) | tc1(nc) tg1(ng) | ... | tc-MAX_TRIALS(nc) tg-MAX_TRIALS(ng)|
// output: outc, outg, out_alpha,
// evalcnt will be added to steps count used

__device__ void line_search(ModelDesc *m, PrecalculateByAtom *pa, Cache *ch, Flt f0, Flt &f1,
                            const Flt *c, const Flt *g, const Flt *p, Flt *tmp, Flt *outc,
                            Flt *outg, int ng, int nc, int &evalcnt, Flt &out_alpha,
                            const threadids &tids, Flt *md, const Flt *vs, int step) {
    const Flt c0 = 0.0001;

    Flt *pg = tmp;
    tmp++;

    Flt *active_md = md + m->active *m->szflt;

    scalar_product_array1(p, g, pg, tmp, ng, TIDS); // tmp require ng
    SYNC();

#if BFGSDEBUG
    if(REQUIRED()) {
        printf("step %d pg: %f\n", step, *pg);
        printf("p:\n");
        printg(m->src, p);
    }
    SYNC();
#endif

    Flt c0pg = c0 * (*pg);
    Flt *es = tmp;
    tmp += MAX_TRIALS;
    int *flags = (int *)tmp;
    tmp += MAX_TRIALS;

    bool exit = false;
    for (int dz = threadIdx.z; dz < MAX_TRIALS; dz += blockDim.z) {
        flags[dz] = 0;
    }
    int lsblk = 3 * nc;
    for (int dz = threadIdx.z; dz < MAX_TRIALS && !exit; dz += blockDim.z) {
#if 1
        Flt *tc = tmp + lsblk * dz; // c_new
        Flt *cs = tc + nc;

#else
        Flt *tc = tmp + nc * dz; // c_new
        Flt *cs = nullptr;
#endif
        Flt &e = es[dz];
        Flt *mdz = md + dz * m->szflt;
        Flt alpha = multipliers[dz];

        if (dz != m->active) {
            copy_array2d<Flt>(mdz, active_md, m->szflt);
        }
        conf_increament_xy(m->src, p, c, tc, alpha);
        XYSYNC();
#if 0
        if(REQUIRED()) {
            printf("before conf:\n");
            printc(m->src, c);
            printf("after conf:\n");
            printc(m->src, tc);
        }
#endif
        model_eval_deriv_e_xy(m, pa, ch, tc,  &e, mdz, vs, cs);
#if BFGSDEBUG
        SYNC();
        if (ZIS(0)) {
            for(int zidx = 0; zidx < blockDim.z; zidx++) {
                int trialIdx = zidx + dz;
                printf("line search step %d trial %d der e %f\n",step, trialIdx, es[trialIdx]);
            }
        }
        // if(REQUIRED()) printf("\n\n\n");
        SYNC();
#endif
        // update my flag
        if (XY0() && e - f0 < c0pg * alpha) {
            flags[dz] = 1;
        }
        SYNC(); // to get all threads result

        // check all threads for result, abort main for loop in this thread if there is any succeeds
        FOR(k, MAX_TRIALS) {
            if (flags[k] > 0) {
                // printf("gpu breaks %d %d %d @ %d\n", threadIdx.x, threadIdx.y, threadIdx.z, k);
                exit = true;
                break;
            }
        }

    }
    SYNC();
    // change active model data
    if (INMAIN()) {
        m->active = MAX_TRIALS - 1;
        bool found = false;
        // todo should we find the first passed case? or should we find best case?
        FOR(k, MAX_TRIALS) {
            if (flags[k] > 0) {
                m->active = k;
                found = true;
                break;
            }
        }
        f1 = es[m->active];
        out_alpha = multipliers[m->active];
        evalcnt += m->active+1;
        // note that in cpu bfgs, if no satisfied f1 is found,
        // alpha still will be multiplied by 0.5 and return,
        // so we should also do that
        if (!found) out_alpha *= 0.5;
    }
    SYNC();

    copy_conf(outc, tmp + lsblk * m->active);
    model_update_xyz(m, outg, md + m->active * m->szflt);
}
__device__ int seg_begins[] = {
    17, 11, 9, 33, 32, 26, 22,0
};
__device__ void bfgs_print(ModelDesc *m) {
    
    auto data = m->data + m->active * m->szflt;
    auto coords = model_coords(m->src, m, data);
	printf("coords:\n");
	for (auto i = 0u; i < m->ncoords; i++) {
		printf("\t%u: %f %f %f\n", i, coords[i].d[0], coords[i].d[1], coords[i].d[2]);
	}
    auto src = m->src;
    for (auto i = 0; i < src->nligand; i++) {
        auto &lg = src->ligands[i];
        printf("==== ligand %d ====\n", i);
        for (auto k = 0; k < lg.nr_node; k++) {
            auto begin = seg_begins[k];
            int j = 0;
            while(lg.tree[j].begin != begin) j++;

            auto &seg = lg.tree[j];
            auto &segvar = *model_ligand(src, m, data, i, j);
            if (seg.parent < 0) {
                printf("\trigid body begin %d end %d\n", seg.begin, seg.end);
            } else {
                printf("\tsegment begin %d end %d\n", seg.begin, seg.end);
            }
            printf("\t\taxis: %f %f %f\n", segvar.axis.d[0], segvar.axis.d[1], segvar.axis.d[2]);
            if (seg.parent >= 0) {
                printf("\t\trelative axis: %f %f %f\n", seg.relative_axis.d[0], seg.relative_axis.d[1],
                    seg.relative_axis.d[2]);
                printf("\t\trelative origin: %f %f %f\n", seg.relative_origin.d[0],
                    seg.relative_origin.d[1], seg.relative_origin.d[2]);
            }
            printf("\t\torigin: %f %f %f\n", segvar.origin.d[0], segvar.origin.d[1], segvar.origin.d[2]);
            printf("\t\torq: %f %f %f %f\n", segvar.orq.d[0],segvar.orq.d[1],segvar.orq.d[2],segvar.orq.d[3]);
            ;
            printf("\t\torm: %f %f %f %f %f %f %f %f %f\n", segvar.orm[0],
                   segvar.orm[1], segvar.orm[2], segvar.orm[3],
                   segvar.orm[4], segvar.orm[5], segvar.orm[6],
                   segvar.orm[7], segvar.orm[8]);
        }
    }
}
#define BFGSIDX threadIdx.z
// mem requirement
#define SHARED_TMPSZ(src, nc, ng) std::max(std::max(BFGS_UPD_SZ(ng), LINESRCH_SIZE(nc, ng)), MODEL_EVAL_MEM_SIZE(src))
#define SMSIZE(src, ng, nc, max_steps) (32 + ((ng * (ng+1)) >> 1) + 6 *ng + 2 *nc + (max_steps + 1) + SHARED_TMPSZ(src, nc, ng))
// Model data will be updated to m->data, conf will be write to c
__device__ void bfgs(ModelDesc *m, PrecalculateByAtom *pa, Cache *ch, int max_steps,
                     Flt average_required_improvement, Size over, Flt *c, Flt *f0, int &evalcnt,
                     Flt *vs, Flt *mem, int sx=0, int sy=0) {
    MCPRT("bfgs in %d %d", sx, sy);
    int offset = 0;  // mem alloc offset
    DECL_TIDS;
    tids.tid       = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    tids.blksz     = blockDim.x * blockDim.y * blockDim.z;
    SrcModel *srcm = m->src;
    int ng         = srcm->nrflts_change;
    int nc         = srcm->nrflts_conf;

    Flt *md = m->data;
    Flt *h, *g, *g_new, *c_new, *g_orig, *c_orig, *p, *y, *fs, *tmp;
    Flt *f_orig, *f1, *alpha, *yy, *yp, *beta;
    Flt *gg;

    f1     = mem + offset, offset += 1;
    f_orig = mem + offset, offset += 1;
    alpha  = mem + offset, offset += 1;
    beta   = mem + offset, offset += 1;
    yy     = mem + offset, offset += 1;
    yp     = mem + offset, offset += 1;

    h      = mem + offset, offset += ((ng * (ng + 1)) >> 1);
    g      = mem + offset, offset += ng;
    gg      = mem + offset, offset += ng;
    g_new  = mem + offset, offset += ng;
    c_new  = mem + offset, offset += nc;
    g_orig = mem + offset, offset += ng;
    c_orig = mem + offset, offset += nc;
    p      = mem + offset, offset += ng;
    y      = mem + offset, offset += ng;
    fs     = mem + offset, offset += max_steps + 1;
    tmp    = mem + offset;
    offset += max(max(BFGS_UPD_SZ(ng), LINESRCH_SIZE(nc, ng)), MODEL_EVAL_MEM_SIZE(srcm));

    // enable this line if sm is used for model data
    // copy input data into shared memory
    // copy_array<Flt>(md, m->data, m->szflt, TIDS);
    if (threadIdx.z == blockDim.z - 1) {
        init_triangle_mat_xy(h, ng);
    }
    // in cpu version, here is a declare and copy, but there is no need because it will be used in
    // linesearch for output only. copy_change(g_new, g); copy_conf(c_new, c); SYNC();

    // initial evaluation, get f0, initial conf and change
    model_eval_xyz(m, pa, ch, c, g, f0, md + m->active * m->szflt, vs, tmp);
    if (INMAIN()) {
        evalcnt++;
        // *flag = 1;
    }
    SYNC();

#if BFGSDEBUG
    if (INMAIN()) {
        printf("gpu f0 %f\n", *f0);
        printf("c:\n");
        printc(srcm, c);
        printf("g:\n");
        printg(srcm, g);
    }
    SYNC();
#endif

    copy_change(g_orig, g);
    copy_conf(c_orig, c);
    copy_change(p, g);

    if (INMAIN()) {
        *f_orig = *f0;
        fs[0]   = *f0;
        // printf("f0 %f\n", *f0);
    }

    // require max(ng * ng, MAX_TRIALS * (nc+ng+1) + 2, ng * ng + 4)
    for (int step = 0; step < max_steps; step++) {
        SYNC();
        minus_mat_vec_product3(h, g, p, tmp, ng, tids);  // require ng * ng
        SYNC();
#if BFGSDEBUG
        if (INMAIN()) {
            printf("GPU step %d\n", step);
            printf("g:\n");
            printg(srcm, g);
            printf("p:\n");
            printg(srcm, p);
            printf("c:\n");
            printc(srcm, c);
        }
        SYNC();
#endif
        line_search(m, pa, ch, *f0, *f1, c, g, p, tmp, c_new, g_new, ng, nc, evalcnt, *alpha, TIDS,
                    md, vs, step);
        SYNC();
        // y = g_new - g
        sub_array<Flt>(y, g_new, g, ng, TIDS);
        if (INMAIN()) {
            fs[step + 1] = *f1;
            *f0          = *f1;
        }
#if BFGSDEBUG
        if (INMAIN()) {
            printf("alpha %f f1 %f\n", *alpha, *f1);
            printf("c_new:\n");
            printc(srcm, c_new);
            printf("g:\n");
            printg(srcm, g);
            printf("g_new\n");
            printg(srcm, g_new);
            printf("\n\n\n\n");
            // auto cc = model_coords(srcm, m, m->data + m->szflt * m->active);
            // printf("step %d alpha %f f1 %f coord %f %f %f\n", step, *alpha, *f1, cc->d[0], cc->d[1], cc->d[2]);
            printf("step %d alpha %f f1 %f\n", step, *alpha, *f1);
            bfgs_print(m);
        }
#endif
        SYNC();
        copy_conf(c, c_new);
        sqrt_product_array(g, ng, gg, TIDS);
        SYNC();
#if BFGSDEBUG
        if (INMAIN()) {
            printf("sqrt g %f\n", g[0]);
            printf("y:\n");
            printg(srcm, y);
        }
        SYNC();
#endif
        if (!(gg[0] >= 1e-5)) {
            // done
            // printf("%d %d:%d:%d@%d:%d:%d %d:%d >>>>> BREAKS <<<<<\n", __LINE__, threadIdx.x,
            //        threadIdx.y, threadIdx.z, blockIdx.x, lane_id(), warp_id(), sx, sy);
            break;
        }
        if (step == 0) {
            scalar_product_array(y, tmp, yy, ng, TIDS);  // require ng
            SYNC();
#if BFGSDEBUG
            if (INMAIN()) {
                printf("yy %f\n", *yy);
            }
#endif
            if (abs(*yy) > EPSILON) {
                scalar_product_array(y, p, yp, tmp, ng, TIDS);  // require ng
                if (INMAIN()) {
                    *beta = (*alpha) * (*yp) / (*yy);
#if BFGSDEBUG
                    printf("set diag %f\n", *beta);
#endif                    
                }
                SYNC();
#if 0//BFGSDEBUG
                if (INMAIN()) {
                    printf("H before diag:\n");
                    printh(ng, h);
                }
                SYNC();
#endif                    
                set_diagonal(h, *beta, ng, TIDS);
                SYNC();
            }
        }

        copy_change(g, g_new);
#if 0//BFGSDEBUG
        SYNC();
        if (INMAIN()) {
            printf("H before update:\n");
            printh(ng, h);
        }
        SYNC();
#endif                    
        bfgs_update3(h, p, y, tmp, *alpha, ng, TIDS);
#if 0//BFGSDEBUG
        SYNC();
        if (INMAIN()) {
            printf("H after update:\n");
            printh(ng, h);
        }
#endif                    
    }
    SYNC();

    if (!(*f0 <= *f_orig)) {
        copy_conf(c, c_orig);
        // todo, no output change?
        // copy_change(g, g_orig);
    }
    if (INMAIN()) {
        if (!(*f0 <= *f_orig)) {
            *f0 = *f_orig;
        }
    }
    MCPRT("bfgs out %d %d", sx, sy);
}
// flags at least has atom size, acc has 3 Flts for each atom
// tmp require src->natoms * 4
// rout = 1/out
FORCE_INLINE void gyration_radius(/*const*/ ModelDesc *m, Flt *md, int idx, Flt *out, Flt *rout, Flt *tmp) {

    Flt *acc = tmp;
    tmp += m->src->natoms * 3;
    int *flags = (int *)tmp;
    tmp += m->src->natoms;

    auto &lig = m->src->ligands[idx];
    auto &root = lig.tree[lig.nr_node-1];
    CU_FOR2(i, m->src->natoms) {
        flags[i] = 0;
    }
    WARPSYNC();
    int dx = threadIdx.x;
    auto *seg = model_ligand(m->src, m, md, idx, lig.nr_node-1); // root is last seg
    for (int dy = threadIdx.y + root.begin; dy < root.end; dy += blockDim.y){
        if(m->src->atoms[dy].el != 0 /*EL_TYPE_H*/ && dx < 3) {
            auto *coord = model_coords(m->src, m, md, dy);
            Flt sub = vec_get(*coord, dx) - vec_get(seg->origin, dx);
            acc[dy * 3 + dx] = sub * sub;
            if (dx == 0) {
                flags[dy] = 1;
            }
        }
    }
    WARPSYNC();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int cnt = 0;
        Flt total = 0;
        for(int i = 0; i < m->src->natoms; i++) {
            if (flags[i] == 1) {
                auto *p = acc + i * 3;
                total += p[0] + p[1] + p[2];
                cnt++;
            }
        }
        *out = 0;
        if (cnt > 0) {
            *out = sqrt(total *reciprocal(cnt));
            if (*out > EPSILON) {
                *rout = reciprocal(*out);
            }
        }
    }
}
// conf will be updated
// tmp require SrcModel.natoms * 4 + 2
__device__ void mutate_conf_xy(/*const*/ ModelDesc *m, const MCCtx *ctx, const MCStepInput *in, Flt *c, Flt *md, Flt *tmp) {
    if (ctx->num_mutable_entities > 0 && threadIdx.z == 0) {
        auto g = in->groups;
        if (g[0] == 0) {
            // update ligand position
            if (threadIdx.x == 0) {
                Flt *f = get_ligand_conf(m->src, c, g[1]);
                for (int dy = threadIdx.y; dy < 3; dy += blockDim.y) {
                    f[dy] += vec_get(in->rsphere, dy) * ctx->amplitude;
                }
            }
        } else if (g[0] == 1) {
            // update ligand orientation
            Flt *gy = tmp;
            tmp++;
            Flt *rgy = tmp;
            tmp++;
            gyration_radius(m, md, g[1], gy, rgy, tmp);  // require natoms * 4
            WARPSYNC();
            if (*gy > EPSILON && threadIdx.x == 0 && threadIdx.y == 0) {
                Flt rot[3];
                *rgy *= ctx->amplitude;
                rot[0] = (*rgy) * in->rsphere.d[0], rot[1] = (*rgy) * in->rsphere.d[1], rot[2] = (*rgy) * in->rsphere.d[2];
                Flt *f = get_ligand_conf(m->src, c, g[1]);
                Flt q[4];
                q[0] = f[0], q[1] = f[1], q[2] = f[2], q[3] = f[3];
                quaternion_increment(f, q, rot);
            }

        } else if (g[0] == 2) {
            // update ligand torsion
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                Flt *f                           = get_ligand_conf(m->src, c, g[1]);
                get_ligand_conf_torsion(f, g[2]) = in->rpi;
            }
        } else {
            // update flex torsion
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                Flt *f                         = get_flex_conf(m->src, c, g[1]);
                get_flex_conf_torsion(f, g[2]) = in->rpi;
            }
        }
    }
}

__device__ bool metropolis_accept(Flt old_f, Flt new_f, Flt rtemperature, Flt threshold) {
    if (new_f < old_f) {
        return true;
    } else {
        Flt acc = exp(old_f - new_f) * rtemperature;
        return threshold < acc;
    }
}

#if CUDEBUG
__device__ void dump_md(ModelDesc *m) {
    auto data = m->data + m->active * m->szflt;
    printf("szflt %d ncoords %d active %d data %p end of data %p\n", m->szflt, m->ncoords, m->active, data, data+m->szflt);
    auto src = m->src;
    printf("Coords:\n");
    for (auto i = 0; i < m->ncoords; i++) {
        auto c = model_coords(src, m, data, i);
        printf("\t%d: %f %f %f\n", i, c->d[0], c->d[1], c->d[2]);
    }
    for (auto i = 0; i < src->nligand; i++) {
    for (auto j = 0; j < src->ligands[i].nr_node; j++) {
        printf("ligand %d node %d\n", i, j);
        auto seg = model_ligand(src, m, data, i, j);
        printf("\taxis: %f %f %f\n", seg->axis.d[0], seg->axis.d[1], seg->axis.d[2]);
        printf("\torigin: %f %f %f\n", seg->origin.d[0], seg->origin.d[1], seg->origin.d[2]);
        printf("\torq: %f %f %f %f\n", seg->orq.d[0], seg->orq.d[1], seg->orq.d[2], seg->orq.d[3]);
        auto a = seg->orm;
        printf("\torm: %f %f %f %f %f %f %f %f %f\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    }
    }
}
#endif
#define MCSIZE(src, natoms, ng, nc, local_steps) \
    (3 + 2 * nc + std::max((int)(4 * (natoms) + 2), SMSIZE(src, ng, nc, local_steps)))
template <bool forceadd>
__device__ void mc_xyz(ModelDesc *m, PrecalculateByAtom *pa, Cache *ch, MCCtx *ctx, MCInputs *ins,
                       MCOutputs *outs, Flt *e_and_c, Flt *mem, int *evalcnt, int steps) {
    int offset = 0;
    DECL_TIDS;
    tids.tid   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    tids.blksz = blockDim.x * blockDim.y * blockDim.z;

    SrcModel *srcm = m->src;
    int ng         = srcm->nrflts_change;
    int nc         = srcm->nrflts_conf;
    Flt *md        = m->data + m->szflt * m->active;
    Flt *src_conf  = e_and_c + 1;
    Flt *g_best_e  = e_and_c;

    Flt *ctemp, *candidate, *tmp, *beste;
    // ctemp and candidate includes Flt e + nc Flt conf
    ctemp     = mem + offset, offset += nc + 1;
    candidate = mem + offset, offset += nc + 1;
    beste     = mem + offset, offset++;

    tmp = mem + offset,
    offset += max((int)(4 * srcm->natoms + 2), SMSIZE(srcm, ng, nc, ctx->local_steps));

#define CANDIDATE   (candidate + 1)
#define TEMP        (ctemp + 1)
#define FCANDIDATE  (*candidate)
#define FTEMP       (*ctemp)
#define PFCANDIDATE candidate
#define PFTEMP      ctemp

#define SWAP() \
    do { \
        Flt *p    = ctemp; \
        ctemp     = candidate; \
        candidate = p; \
    } while (0)

    if (INMAIN()) {
        FTEMP  = 0;
        *beste = *g_best_e;
    }
    copy_conf(TEMP, src_conf);
    SYNC();

    int commited = 0;
    for (int dz = 0; dz < steps; dz++) {
        // if(INMAIN()) {
            // printf("%d:%d:%d@%d mc step %d/%d before sync\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, dz, steps);
            MCPRT("mc step %d/%d before sync", dz, steps);
        // }
        SYNC();
            // printf("%d:%d:%d@%d mc step %d/%d after sync\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, dz, steps);
            MCPRT("mc step %d/%d after sync", dz, steps);
        if (ctx->max_evalcnt > 0 && *evalcnt > ctx->max_evalcnt) {
            break;
        }
        auto in = ins->in + dz;
        copy_conf(CANDIDATE, TEMP);
        SYNC();

        mutate_conf_xy(m, ctx, in, CANDIDATE, md, tmp);  // update c, requires 4 * srcm->natoms + 2
        SYNC();
#if CUDEBUG
        if (INMAIN()) {
            printf("candidate c:\n");
            printc(srcm, CANDIDATE);
        }
#endif

        bfgs(m, pa, ch, ctx->local_steps, ctx->average_required_improvement, ctx->over, CANDIDATE,
            PFCANDIDATE, *evalcnt, ctx->vs, tmp, dz, 0);  // requires SMSIZE()

        SYNC();
        #if 0
        if (INMAIN()) {
            printf("candidate e : %f\n", *(PFCANDIDATE));
        }
        #endif
#if 1
        bool satisfied = dz == 0 || metropolis_accept(FCANDIDATE, FTEMP, ctx->rtemperature, in->raccept);
        MCPRT("satisfied %d candidate %f %p temp %f %p", satisfied, FCANDIDATE, candidate, FTEMP, ctemp);
        SYNC();
        if (satisfied) {
            // discard temp, use candidate
            SWAP();  // candidate <-> temp, in next for loop new conf will be used as conf
            SYNC();
            // check if this conf should be commited, if so, fine tune with new vs
            // the bfgs should be performed on TEMP because we swapped before
            MCPRT("accepted %d beste %f ftemp %f", dz, *beste, FTEMP);
            if (forceadd || FTEMP < *beste) {
                MCPRT("fine bfgs beste %f ftemp %f", *beste, FTEMP);
                bfgs(m, pa, ch, ctx->local_steps, ctx->average_required_improvement, ctx->over,
                     TEMP, PFTEMP, *evalcnt, ctx->vs + 3, tmp, dz, 1);  // requires SMSIZE()
                SYNC();
                // now add temp as new output
                auto &out = outs->out[outs->n];
                // copy e and conf
                copy_array<Flt>(out.e_and_c, ctemp, nc + 1, TIDS);
                FOR_TID(i, m->ncoords) {
                    vec_set(*(out.coords + i), *model_coords(srcm, m, m->data + m->szflt * m->active, i));
                }
                commited++;
                if (INMAIN()) {
                    if (FTEMP < *beste) {
                        *beste = FTEMP;
                    }
                }
                MCPRT("fine bfgs done, commited %d", commited);
            }
        }
        MCPRT("satisfied out %d", dz);
#endif
    }
    SYNC();
    copy_conf(src_conf, TEMP);
    if (INMAIN()) {
        *g_best_e = *beste;
        outs->n   = commited;
    }
#undef CANDIDATE
#undef SWAP
#undef TEMP
}
// used for single mc call
GLOBAL void bfgs_kernel(ModelDesc *m, PrecalculateByAtom *pa, Cache *ch, BFGSCtx *ctx, int max_steps, Flt average_required_improvement, Size over) {
    extern __shared__ Flt sm[];
    int nc = m->src->nrflts_conf;
    int offset = 0;

        #if 0
    if (ZIS(0)) {
        auto src = m->src;
        for (int i = 0; i < src->movable_atoms + src->npairs; i++) {
            printf("gpu map add %d: %d\n", i, src->force_pair_map_add[i]);
        }
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
    }
        #endif


    Flt *c, *f0, *mem;
    c = sm + offset, offset += nc; // bfgs output conf
    f0 = sm + offset,  offset += 1;
    mem = sm + offset; // bfgs temp memory
    read_conf(m->src, &ctx->c, c);
    SYNC();

    bfgs(m, pa, ch, max_steps, average_required_improvement, over, c, f0, ctx->eval_cnt, ctx->vs, mem);
    SYNC();

    auto data = m->data + m->szflt * m->active;
    auto cds = model_coords(m->src, m, data);
    CU_FOR3(i, m->ncoords) {
        vec_set(ctx->coords[i], cds[i]);
    }
    convert_conf(m->src, &ctx->c, c);
    if (INMAIN()) {
        ctx->e = *f0;
    }
    // this covers model.set(out.c) in the end of quasi_newton()
    // Flt *active = m->data + m->szflt * m->active;
    // model_set_conf_ligand_xy(m, c, active);
    // model_set_conf_flex_xy(m, c, active);
}
#define MDDBG(fmt, ...) do{ printf("\t" fmt "\n",  __LINE__, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,  __VA_ARGS__);}while(0)
#define MDPV(hdr, pv, ...) MDDBG(hdr ": %f %f %f", __VA_ARGS__, (pv)->d[0], (pv)->d[1], (pv)->d[2])
#define MDV(hdr, v, ...) MDDBG(hdr ": %f %f %f", __VA_ARGS__, (v).d[0], (v).d[1], (v).d[2])
GLOBAL void mc_kernel(ModelDesc *m, PrecalculateByAtom *pa, Cache *ch, MCCtx *ctx, MCInputs *ins, MCOutputs *outs, int steps) {
    extern __shared__ Flt smmc[];
    Flt *mem = smmc;
    SrcModel *srcm = m->src;
    int nc = srcm->nrflts_conf;
    int mcidx = blockIdx.x;
    auto c = ctx->best_e_and_c + (nc+1) * mcidx;
#if CUDEBUG
    if (blockIdx.x == 0 && INMAIN()) {
        printf("cu runs curr eval cnt %d max %d\n", *(ctx->curr_evalcnt + mcidx), ctx->max_evalcnt);
    }
#endif
    mc_xyz<false>(m+mcidx, pa, ch, ctx, ins+mcidx, outs+mcidx, c, mem, ctx->curr_evalcnt+mcidx, steps);

}
void run_bfgs(ModelDesc *cpum, ModelDesc *m, PrecalculateByAtom *pa, Cache *ch, BFGSCtx *ctx, int max_steps, Flt average_required_improvement, Size over , cudaStream_t stream) {
    auto srcm = cpum->src;
    dim3 block(DIMX, DIMY, DIMZ);
    dim3 grid(1);

    int ng = srcm->nrflts_change;
    int nc = srcm->nrflts_conf;
    // sm requirement:
    int sm = (nc + 1 + SMSIZE(cpum->src, ng, nc, max_steps)) * sizeof(Flt);

#if CUDEBUG
    printf("gpu ng %d nc %d sm size %d\n", ng, nc, sm);
#endif
    bfgs_kernel<<<grid, block, sm, stream>>>(m, pa, ch, ctx, max_steps, average_required_improvement, over);
}
void run_mc(SrcModel *srcm, ModelDesc *m, PrecalculateByAtom *pa, Cache *ch, MCCtx *ctx,
            MCInputs *ins, MCOutputs *outs, int nmc, int steps, int local_steps, cudaStream_t stream) {
    dim3 block(DIMX, DIMY, DIMZ);
    dim3 grid(nmc);

    int ng = srcm->nrflts_change;
    int nc = srcm->nrflts_conf;
    // sm requirement:
    int sm = MCSIZE(srcm, srcm->natoms, ng, nc, local_steps) * sizeof(Flt) *2;

#if CUDEBUG
    printf("nmc %d ng %d nc %d sm size %d\n", nmc, ng, nc, sm);
#endif
    mc_kernel<<<grid, block, sm, stream>>>(m, pa, ch, ctx, ins, outs, steps);
}
GLOBAL void bfgs_debug(ModelDesc *m) {
    bfgs_print(m);
}
void dump_bfgs(ModelDesc *m, BFGSCtx *ctx, cudaStream_t stream) {
    dim3 block(1);
    dim3 grid(1);
    bfgs_debug<<<grid, block, 0, stream>>>(m);
}
};