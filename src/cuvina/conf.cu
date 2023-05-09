
#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
#include <algorithm>

namespace dock {
FORCE_INLINE void frame_set_orientation(SegmentVars *segvar, const Qt &q) {
    qt_set(segvar->orq, q);
    qt_to_mat(q, segvar->orm);
}
FORCE_INLINE void frame_local_to_lab(SegmentVars *segvar, const Vec &local_coords, Vec &out) {
    mat_multiple(segvar->orm, local_coords, out);
    vec_add(out, segvar->origin);
}
FORCE_INLINE void frame_local_to_lab_direction(SegmentVars *segvar, const Vec &local_direction, Vec &out) {
    mat_multiple(segvar->orm, local_direction, out);
}
FORCE_INLINE void atom_frame_set_coords(Atom *atoms, Segment *seg, SegmentVars *segvar, Vec *coords) {
    FOR_RANGE(i, seg->begin, seg->end) {
        frame_local_to_lab(segvar, atoms[i].coords, coords[i]);
    }
}
FORCE_INLINE void rigid_body_set_conf(Segment *seg, SegmentVars *segvar, Atom *atoms, Vec *coords,
                         const RigidConf &c) {
    vec_set(segvar->origin, c.position);
    frame_set_orientation(segvar, c.orientation);
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

GLOBAL void model_set_conf_ligand(Model *m, Conf *c) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    int i         = blockIdx.x;

    Ligand &ligand  = src->ligands[i];
    LigandVars &var = m->ligands[i];
    Flt *p          = c->ligands[i].torsions;
    FOR(k, ligand.nr_node) {
        int j               = ligand.nr_node - k - 1;  // from root to leaf
        Segment &seg        = ligand.tree[j];
        SegmentVars &segvar = var.tree[j];
        CUDBG("ligand %d", j);
        if (seg.parent >= 0) {
            Segment &parent        = ligand.tree[seg.parent];
            SegmentVars &parentVar = var.tree[seg.parent];
            segment_set_conf(&parentVar, &segvar, &seg, atoms, m->coords, p[k - 1]);
        } else {
            // root
            rigid_body_set_conf(&seg, &segvar, atoms, m->coords, c->ligands[i].rigid);
        }
    }
}
GLOBAL void model_set_conf_flex(Model *m, Conf *c) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    int i         = blockIdx.x;

    Residue &flex    = src->flex[i];
    ResidueVars &var = m->flex[i];
    Flt *p           = c->flex[i].torsions;
    // climbing from the leaves to root and accumulate force and torque
    FOR(k, flex.nr_node) {
        int j               = src->nflex - k - 1;
        Segment &seg        = flex.tree[j];
        SegmentVars &segvar = var.tree[j];
        if (seg.parent >= 0) {
            Segment &parent        = flex.tree[seg.parent];
            SegmentVars &parentVar = var.tree[seg.parent];
            segment_set_conf(&parentVar, &segvar, &seg, atoms, m->coords, p[k]);
        } else {
            first_segment_set_conf(&seg, &segvar, atoms, m->coords, p[k]);
        }
    }
}

#if USE_CUDA_VINA
void cuda_model_set_conf(Model &cpum, Model &m, Conf &c, cudaStream_t stream) {
    int nligand = cpum.src->nligand;
    int nflex = cpum.src->nflex;
    if (nligand > 0) {
        int maxnode = 0;
        for (auto i = 0; i < nligand; i++) {
            if (cpum.src->ligands[i].nr_node > maxnode) {
                maxnode = cpum.src->ligands[i].nr_node;
            }
        }
        model_set_conf_ligand<<<dim3(nligand), dim3(1), 0, stream>>>(&m, &c);
    }
    if (nflex > 0) {
        int maxnode = 0;
        for (auto i = 0; i < nflex; i++) {
            if (cpum.src->flex[i].nr_node > maxnode) {
                maxnode = cpum.src->flex[i].nr_node;
            }
        }
        model_set_conf_flex<<<dim3(nflex), dim3(1), 0, stream>>>(&m, &c);
    }
}
#endif
};  // namespace dock