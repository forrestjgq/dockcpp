
#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
#include <algorithm>

namespace dock {
FORCE_INLINE void frame_set_orientation(SegmentVars *segvar, const Flt *q) {
    qt_set(segvar->orq, q[0], q[1], q[2], q[3]);
    qt_to_mat(q, segvar->orm);
}
FORCE_INLINE void frame_set_orientation(SegmentVars *segvar, const Qt &q) {
    qt_set(segvar->orq, q);
    qt_to_mat(q, segvar->orm);
}
FORCE_INLINE void frame_local_to_lab(SegmentVars *segvar, const Vec &local_coords, Vec &out) {
    // CUVDUMP("    orm", segvar->orm);
    CUVDUMP("    local coords(atoms)", local_coords);
    mat_multiple(segvar->orm, local_coords, out);
    CUVDUMP("    before coord", out);
    vec_add(out, segvar->origin);
    CUVDUMP("    coord", out);
}
FORCE_INLINE void frame_local_to_lab_direction(SegmentVars *segvar, const Vec &local_direction, Vec &out) {
    mat_multiple(segvar->orm, local_direction, out);
}
FORCE_INLINE void atom_frame_set_coords(Atom *atoms, Segment *seg, SegmentVars *segvar, Vec *coords) {
    CUDBG("begin %d end %d", seg->begin, seg->end);
    FOR_RANGE(i, seg->begin, seg->end) {
        CUDBG("set coords %d", i);
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

FORCE_INLINE void segment_set_conf(SegmentVars *parent, SegmentVars *segvar, Segment *seg, Atom *atoms,
                      Vec *coords, Flt torsion) {
    frame_local_to_lab(parent, seg->relative_origin, segvar->origin);
    frame_local_to_lab_direction(parent, seg->relative_axis, segvar->axis);
    Qt tmp;
    angle_to_quaternion(segvar->axis, torsion, tmp);
    qt_multiple(tmp, parent->orq);
    quaternion_normalize_approx(tmp);  // normalization added in 1.1.2
    // quaternion_normalize(tmp); // normalization added in 1.1.2
    frame_set_orientation(segvar, tmp);
    atom_frame_set_coords(atoms, seg, segvar, coords);
}
FORCE_INLINE void first_segment_set_conf(Segment *seg, SegmentVars *segvar, Atom *atoms, Vec *coords,
                            Flt torsion) {
    Qt tmp;
    angle_to_quaternion(segvar->axis, torsion, tmp);
    frame_set_orientation(segvar, tmp);
    atom_frame_set_coords(atoms, seg, segvar, coords);
}

FORCE_INLINE void model_set_conf_ligand_c(Model *m, const Flt *c) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    CU_FOR2 (i, src->nligand) {
        Ligand &ligand  = src->ligands[i];
        LigandVars &var = m->ligands[i];
        auto p          = get_ligand_conf(src, c, i);
        FOR(k, ligand.nr_node) {
            int j               = ligand.nr_node - k - 1;  // from root to leaf
            Segment &seg        = ligand.tree[j];
            SegmentVars &segvar = var.tree[j];
            CUDBG("ligand %d", j);
            if (seg.parent >= 0) {
                Segment &parent        = ligand.tree[seg.parent];
                SegmentVars &parentVar = var.tree[seg.parent];
                segment_set_conf(&parentVar, &segvar, &seg, atoms, m->coords,
                                 get_ligand_conf_torsion(p, k - 1));
            } else {
                // root
                rigid_body_set_conf(&seg, &segvar, atoms, m->coords, c);
            }
        }
    }


}
// c: conf floats
FORCE_INLINE void model_set_conf_flex_c(Model *m, const Flt *c) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    CU_FOR2 (i, src->nflex) {
        Residue &flex    = src->flex[i];
        ResidueVars &var = m->flex[i];
        auto *p          = get_flex_conf(src, c, i);
        // climbing from the leaves to root and accumulate force and torque
        FOR(k, flex.nr_node) {
            int j               = src->nflex - k - 1;
            Segment &seg        = flex.tree[j];
            SegmentVars &segvar = var.tree[j];
            if (seg.parent >= 0) {
                Segment &parent        = flex.tree[seg.parent];
                SegmentVars &parentVar = var.tree[seg.parent];
                segment_set_conf(&parentVar, &segvar, &seg, atoms, m->coords,
                                 get_flex_conf_torsion(p, k));
            } else {
                first_segment_set_conf(&seg, &segvar, atoms, m->coords,
                                       get_flex_conf_torsion(p, k));
            }
        }
    }
}

};  // namespace dock