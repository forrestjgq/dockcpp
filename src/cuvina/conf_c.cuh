
#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
#include <algorithm>
#include "model_desc.cuh"

namespace dock {
FORCE_INLINE void frame_set_orientation(SegmentVars *segvar, const Flt *q) {
    CUDBG("set orientation: %f %f %f %f", q[0], q[1], q[2], q[3]);
    qt_set(segvar->orq, q[0], q[1], q[2], q[3]);
    qt_to_mat(q, segvar->orm);
}
FORCE_INLINE void frame_set_orientation_c(int idx, int blk, SegmentVars *segvar, const Flt *q) {
    CUDBG("set orientation: %f %f %f %f", q[0], q[1], q[2], q[3]);
    qt_set(segvar->orq, idx, q[idx]);
    qt_to_mat_c(idx, blk, q, segvar->orm);
}
FORCE_INLINE void frame_set_orientation(SegmentVars *segvar, const Qt &q) {
    CUDBG("set orientation: %f %f %f %f", q.x, q.y, q.z, q.w);
    qt_set(segvar->orq, q);
    qt_to_mat(q, segvar->orm);
}
FORCE_INLINE void frame_local_to_lab(SegmentVars *segvar, const Vec &local_coords, Vec &out) {
    // CUVDUMP("    orm", segvar->orm);
    CUVDUMP("    local coords(atoms)", local_coords);
    CUVDUMP("    origin", segvar->origin);
    auto m = segvar->orm;
    CUDBG("    orm: %f %f %f %f %f %f %f %f %f", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    mat_multiple(segvar->orm, local_coords, out);
    CUVDUMP("    before coord", out);
    vec_add(out, segvar->origin);
    CUVDUMP("    coord", out);
}
FORCE_INLINE void frame_local_to_lab_c(int idx, SegmentVars *segvar, const Vec &local_coords, Vec &out) {
    auto m = segvar->orm;
    mat_multiple_c(idx, segvar->orm, local_coords, out);
    vec_add_c(idx, out, segvar->origin);
}
FORCE_INLINE void frame_local_to_lab_direction(SegmentVars *segvar, const Vec &local_direction, Vec &out) {
    mat_multiple(segvar->orm, local_direction, out);
}
FORCE_INLINE void frame_local_to_lab_direction_c(int idx, SegmentVars *segvar, const Vec &local_direction, Vec &out) {
    mat_multiple_c(idx, segvar->orm, local_direction, out);
}
FORCE_INLINE void atom_frame_set_coords(Atom *atoms, Segment *seg, SegmentVars *segvar, Vec *coords) {
    CUDBG("begin %d end %d", seg->begin, seg->end);
    FOR_RANGE(i, seg->begin, seg->end) {
        CUDBG("set coords %d", i);
        frame_local_to_lab(segvar, atoms[i].coords, coords[i]);
        CUVDUMP("update coords", coords[i]);
    }
}
FORCE_INLINE void atom_frame_set_coords_c(int idx, int blk, Atom *atoms, Segment *seg, SegmentVars *segvar, Vec *coords) {
    for(int i = idx + seg->begin; i < seg->end; i+= blk) {
        frame_local_to_lab(segvar, atoms[i].coords, coords[i]);
    }
}
FORCE_INLINE void rigid_body_set_conf(Segment *seg, SegmentVars *segvar, Atom *atoms, Vec *coords,
                         const Flt *rigid) {
    CUDBG("conf position %f %f %f", rigid[0], rigid[1], rigid[2]);
    CUDBG("conf orientation %f %f %f %f", rigid[3], rigid[4], rigid[5], rigid[6]);
    make_vec(segvar->origin, rigid[0], rigid[1], rigid[2]);
    frame_set_orientation(segvar, rigid+3);
    atom_frame_set_coords(atoms, seg, segvar, coords);
}
FORCE_INLINE void rigid_body_set_conf_c(int idx, int blk, Segment *seg, SegmentVars *segvar, Atom *atoms, Vec *coords,
                         const Flt *rigid) {
    // origin = rigid[0:3]
    if (idx < 3) segvar->origin.d[idx] = rigid[idx];
    frame_set_orientation_c(idx, blk, segvar, rigid+3);
    atom_frame_set_coords_c(idx, blk, atoms, seg, segvar, coords);
}

FORCE_INLINE void segment_set_conf(SegmentVars *parent, SegmentVars *segvar, Segment *seg, Atom *atoms,
                      Vec *coords, Flt torsion) {
    frame_local_to_lab(parent, seg->relative_origin, segvar->origin);
    CUVDUMP("    local axis", seg->relative_axis);
    frame_local_to_lab_direction(parent, seg->relative_axis, segvar->axis);
    CUVDUMP("    my origin", segvar->origin);
    CUVDUMP("    axis", segvar->axis);
    CUDBG("torsion %f", torsion);
    CUDBG("parent orientation %f %f %f %f", parent->orq.x , parent->orq.y , parent->orq.z , parent->orq.w );
    Qt tmp;
    angle_to_quaternion(segvar->axis, torsion, tmp);
    qt_multiple(tmp, parent->orq);
    CUDBG("tmp %f %f %f %f", tmp.x, tmp.y, tmp.z, tmp.w);
    quaternion_normalize_approx(tmp);  // normalization added in 1.1.2
    CUDBG("approx tmp %f %f %f %f", tmp.x, tmp.y, tmp.z, tmp.w);
    // quaternion_normalize(tmp); // normalization added in 1.1.2
    frame_set_orientation(segvar, tmp);
    atom_frame_set_coords(atoms, seg, segvar, coords);
}
FORCE_INLINE void segment_set_conf_c(int idx, int blk, SegmentVars *parent, SegmentVars *segvar, Segment *seg, Atom *atoms,
                      Vec *coords, Flt torsion) {
    frame_local_to_lab_c(idx, parent, seg->relative_origin, segvar->origin);
    frame_local_to_lab_direction_c(idx, parent, seg->relative_axis, segvar->axis);
    auto tmp = segvar->tmp; // use orm for temperary variable
    angle_to_quaternion_c(idx, segvar->axis, torsion, tmp);
    qt_multiple_c(idx, tmp, (Flt *)&(parent->orq));
    if (IS_2DMAIN()) {
        quaternion_normalize_approx(tmp);  // normalization added in 1.1.2
    }
    // quaternion_normalize(tmp); // normalization added in 1.1.2
    frame_set_orientation_c(idx, blk, segvar, tmp);
    atom_frame_set_coords_c(idx, blk, atoms, seg, segvar, coords);
}
FORCE_INLINE void first_segment_set_conf(Segment *seg, SegmentVars *segvar, Atom *atoms, Vec *coords,
                            Flt torsion) {
    Qt tmp;
    angle_to_quaternion(segvar->axis, torsion, tmp);
    frame_set_orientation(segvar, tmp);
    atom_frame_set_coords(atoms, seg, segvar, coords);
}
FORCE_INLINE void first_segment_set_conf_c(int idx, int blk, Segment *seg, SegmentVars *segvar, Atom *atoms, Vec *coords,
                            Flt torsion) {
    auto tmp = &segvar->tmp[0];
    if (idx == 0) angle_to_quaternion(segvar->axis.d, torsion, tmp);
    frame_set_orientation_c(idx, blk, segvar, tmp);
    atom_frame_set_coords_c(idx, blk, atoms, seg, segvar, coords);
}

__device__ void dump_ligands(ModelDesc *m, Flt *md) {
    SrcModel *src = m->src;
    FOR (i, src->nligand) {
        printf("ligand %d\n", i);
        Ligand &ligand  = src->ligands[i];
        FOR(k, ligand.nr_node) {
            int j               = ligand.nr_node - k - 1;  // from root to leaf
            printf("\tnode %d/%d:\n", j, ligand.nr_node);
            Segment &seg        = ligand.tree[j];
            auto segvar = model_ligand(src, m, md, i, j);
            printf("\t\tcoord begin %d end %d\n", seg.begin, seg.end);
            printf("\t\taxis %f %f %f\n", segvar->axis.d[0], segvar->axis.d[1], segvar->axis.d[2]);
            printf("\t\torign %f %f %f\n", segvar->origin.d[0], segvar->origin.d[1], segvar->origin.d[2]);
            printf("\t\torq %f %f %f %f\n", segvar->orq.x, segvar->orq.y, segvar->orq.z, segvar->orq.w);
            printf("\t\torm %f %f %f \n\t\t%f %f %f \n\t\t%f %f %f\n", segvar->orm[0], segvar->orm[1], segvar->orm[2], segvar->orm[3], segvar->orm[4], segvar->orm[5], segvar->orm[6], segvar->orm[7], segvar->orm[8]);
        }
    }
}
__device__ void dump_coords(ModelDesc *m, Flt *md) {
    SrcModel *src = m->src;
    auto coords = model_coords(src, m, md);
    printf("coords:\n");
    FOR(i, m->ncoords) {
        auto c = coords + i;
        printf("\t%d/%d: %f %f %f\n",i, m->ncoords, c->d[0], c->d[1], c->d[2]);
    }
}
__device__ void dump_segvars(ModelDesc *m, Flt *md) {
    if (ZIS(0)) {
        dump_ligands(m, md);
        dump_coords(m, md);
    }
}
// single
FORCE_INLINE void model_set_conf_ligand_1(ModelDesc *m, const Flt *c, Flt *md) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        CUDBG("nligand %d", src->nligand);
    }
    CU_FOR2 (i, src->nligand) {
        Ligand &ligand  = src->ligands[i];
        auto p          = get_ligand_conf(src, c, i);
        FOR(k, ligand.nr_node) {
            int j               = ligand.nr_node - k - 1;  // from root to leaf
            Segment &seg        = ligand.tree[j];
            auto segvar = model_ligand(src, m, md, i, j);
            auto coords = model_coords(src, m, md);
            CUDBG("ligand %d parent %d k %d", j, seg.parent, k);
            if (seg.parent >= 0) {
                Segment &parent        = ligand.tree[seg.parent];
                auto parentVar = model_ligand(src, m, md, i, seg.parent);
                segment_set_conf(parentVar, segvar, &seg, atoms, coords, get_ligand_conf_torsion(p, k - 1));
            } else {
                // root
                rigid_body_set_conf(&seg, segvar, atoms, coords, p);
            }
        }
    }
    // dump_segvars(m, md);
}
FORCE_INLINE void model_set_conf_ligand_c(ModelDesc *m, const Flt *c, Flt *md) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        CUDBG("nligand %d", src->nligand);
    }
    CU_FOR (i, src->nligand) {
        Ligand &ligand  = src->ligands[i];
        auto p          = get_ligand_conf(src, c, i);
        FOR(k, ligand.nr_node) {
            int j               = ligand.nr_node - k - 1;  // from root to leaf
            Segment &seg        = ligand.tree[j];
            auto segvar = model_ligand(src, m, md, i, j);
            auto coords = model_coords(src, m, md);
            CUDBG("ligand %d parent %d k %d", j, seg.parent, k);
            if (seg.parent >= 0) {
                Segment &parent        = ligand.tree[seg.parent];
                auto parentVar = model_ligand(src, m, md, i, seg.parent);
                segment_set_conf_c(threadIdx.y, blockDim.y, parentVar, segvar, &seg, atoms, coords, get_ligand_conf_torsion(p, k - 1));
            } else {
                // root
                rigid_body_set_conf_c(threadIdx.y, blockDim.y, &seg, segvar, atoms, coords, p);
            }
        }
    }


    // dump_segvars(m, md);
}
FORCE_INLINE void model_set_conf_flex_1(ModelDesc *m, const Flt *c, Flt *md) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        CUDBG("nflex: %d", src->nflex);
    }
    CU_FOR2 (i, src->nflex) {
        Residue &flex    = src->flex[i];
        auto *p          = get_flex_conf(src, c, i);
        // climbing from the leaves to root and accumulate force and torque
        FOR(k, flex.nr_node) {
            int j               = src->nflex - k - 1;
            Segment &seg        = flex.tree[j];
            auto segvar = model_flex(src, m, md, i, j);
            auto coords = model_coords(src, m, md);
            if (seg.parent >= 0) {
                Segment &parent        = flex.tree[seg.parent];
                auto parentVar = model_flex(src, m, md, i, seg.parent);
                segment_set_conf(parentVar, segvar, &seg, atoms, coords, get_flex_conf_torsion(p, k));
            } else {
                first_segment_set_conf(&seg, segvar, atoms, coords, get_flex_conf_torsion(p, k));
            }
        }
    }
}
// c: conf floats
// todo:
FORCE_INLINE void model_set_conf_flex_c(ModelDesc *m, const Flt *c, Flt *md) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        CUDBG("nflex: %d", src->nflex);
    }
    CU_FOR (i, src->nflex) {
        Residue &flex    = src->flex[i];
        auto *p          = get_flex_conf(src, c, i);
        // climbing from the leaves to root and accumulate force and torque
        FOR(k, flex.nr_node) {
            int j               = src->nflex - k - 1;
            Segment &seg        = flex.tree[j];
            auto segvar = model_flex(src, m, md, i, j);
            auto coords = model_coords(src, m, md);
            if (seg.parent >= 0) {
                Segment &parent        = flex.tree[seg.parent];
                auto parentVar = model_flex(src, m, md, i, seg.parent);
                segment_set_conf_c(threadIdx.y, blockDim.y, parentVar, segvar, &seg, atoms, coords, get_flex_conf_torsion(p, k));
            } else {
                first_segment_set_conf_c(threadIdx.y, blockDim.y, &seg, segvar, atoms, coords, get_flex_conf_torsion(p, k));
            }
        }
    }
}

};  // namespace dock