
#include "vinautil.h"
#include "stdio.h"
#include <cstring>
#include "culog.h"
#include <algorithm>
#include "model_desc.cuh"
#include "bfgsdef.h"

namespace dock {
FORCE_INLINE void frame_set_orientation(SegmentVars *segvar, const Flt *q) {
    CUDBG("set orientation: %f %f %f %f", q[0], q[1], q[2], q[3]);
    qt_set(segvar->orq, q[0], q[1], q[2], q[3]);
    qt_to_mat(q, segvar->orm);
}
FORCE_INLINE void frame_set_orientation_c(int idx, int blk, SegmentVars *segvar, const Flt *q) {
    if (idx == 0) CUDBG("set orientation: %f %f %f %f", q[0], q[1], q[2], q[3]);
    if (idx < 4) qt_set(segvar->orq, idx, q[idx]);
    qt_to_mat_c(idx, blk, q, segvar->orm);
}
FORCE_INLINE void frame_set_orientation(SegmentVars *segvar, const Qt &q) {
    CUDBG("set orientation: %f %f %f %f", q.d[0], q.d[1], q.d[2], q.d[3]);
    qt_set(segvar->orq, q);
    qt_to_mat(q, segvar->orm);
}
FORCE_INLINE void frame_local_to_lab(SegmentVars *segvar, const Vec &local_coords, Vec &out) {
#if CUDEBUG
    // CUVDUMP("    orm", segvar->orm);
    CUVDUMP("    local coords(atoms)", local_coords);
    CUVDUMP("    origin", segvar->origin);
    auto m = segvar->orm;
    CUDBG("    orm: %f %f %f %f %f %f %f %f %f", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
#endif
    mat_multiple(segvar->orm, local_coords, out);
    CUVDUMP("    before coord", out);
    vec_add(out, segvar->origin);
    CUVDUMP("    coord", out);
}
FORCE_INLINE void frame_local_to_lab_c(int idx, const SegmentVars *segvar, const Vec &local_coords, Vec &out) {
    mat_multiple_c(idx, segvar->orm, local_coords, out);
    vec_add_c(idx, out, segvar->origin);
}
FORCE_INLINE void frame_local_to_lab_direction(const SegmentVars *segvar, const Vec &local_direction, Vec &out) {
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
// use conf to update origin/orm/orq, and use origin/orm + atoms coords to update coords
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
    CUDBG("parent orientation %f %f %f %f", parent->orq.d[0] , parent->orq.d[1] , parent->orq.d[2] , parent->orq.d[3] );
    Qt tmp;
    angle_to_quaternion(segvar->axis, torsion, tmp);
    qt_multiple(tmp, parent->orq);
    CUDBG("tmp %f %f %f %f", tmp.d[0], tmp.d[1], tmp.d[2], tmp.d[3]);
    quaternion_normalize_approx(tmp);  // normalization added in 1.1.2
    CUDBG("approx tmp %f %f %f %f", tmp.d[0], tmp.d[1], tmp.d[2], tmp.d[3]);
    // quaternion_normalize(tmp); // normalization added in 1.1.2
    frame_set_orientation(segvar, tmp);
    atom_frame_set_coords(atoms, seg, segvar, coords);
}
FORCE_INLINE void segment_set_conf_c(int idx, int blk, SegmentVars *parent, SegmentVars *segvar, Segment *seg, Atom *atoms,
                      Vec *coords, const Flt torsion) {
    // update segvar origin and axis
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
FORCE_INLINE void segment_set_conf_c(int idx, int blk, SegmentVars *parent, SegmentVars *segvar, Segment *seg, Atom *atoms,
                      Vec *coords, const Flt* cs) {
    // update segvar origin and axis
    frame_local_to_lab_c(idx, parent, seg->relative_origin, segvar->origin);
    frame_local_to_lab_direction_c(idx, parent, seg->relative_axis, segvar->axis);
    auto tmp = segvar->tmp; // use orm for temperary variable
    angle_to_quaternion_c(idx, segvar->axis, cs, tmp);
    qt_multiple_c(idx, tmp, (Flt *)&(parent->orq));
    if (IS_2DMAIN()) {
        quaternion_normalize_approx(tmp);  // normalization added in 1.1.2
    }
    // quaternion_normalize(tmp); // normalization added in 1.1.2
    frame_set_orientation_c(idx, blk, segvar, tmp);
    atom_frame_set_coords_c(idx, blk, atoms, seg, segvar, coords);
}
FORCE_INLINE void segment_set_conf_c(int idx, int blk, SegmentVars *parent, SegmentVars *segvar, Segment *seg, const Flt* cs) {
    // update segvar origin and axis
    frame_local_to_lab_c(idx, parent, seg->relative_origin, segvar->origin);
    frame_local_to_lab_direction_c(idx, parent, seg->relative_axis, segvar->axis);
    auto tmp = segvar->tmp; // use orm for temperary variable
    angle_to_quaternion_c(idx, segvar->axis, cs, tmp);
    qt_multiple_c(idx, tmp, (Flt *)&(parent->orq));
    if (IS_2DMAIN()) {
        quaternion_normalize_approx(tmp);  // normalization added in 1.1.2
    }
    // quaternion_normalize(tmp); // normalization added in 1.1.2
    frame_set_orientation_c(idx, blk, segvar, tmp);
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
            printf("\tparent %d layer %d\n", seg.parent, seg.layer);
            printf("\t\tcoord begin %d end %d\n", seg.begin, seg.end);
            printf("\t\taxis %f %f %f\n", segvar->axis.d[0], segvar->axis.d[1], segvar->axis.d[2]);
            printf("\t\torign %f %f %f\n", segvar->origin.d[0], segvar->origin.d[1], segvar->origin.d[2]);
            printf("\t\torq %f %f %f %f\n", segvar->orq.d[0], segvar->orq.d[1], segvar->orq.d[2], segvar->orq.d[3]);
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
__device__ void dump_segvars(int line, ModelDesc *m, Flt *md) {
    SYNC();
    if (ZIS(0)) {
        printf("dump segvars at line %d\n", line);
        dump_ligands(m, md);
        dump_coords(m, md);
    }
    SYNC();
}
#define DUMP_SEGVARS(m, md) dump_segvars(__LINE__, m, md)
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
    DUMP_SEGVARS(m, md);
}

#define LIGAND_LAYER_SET 1


// require tmp: src->nrflts_change * 2
FORCE_INLINE void model_set_conf_ligand_coords_xy(ModelDesc *m, const Flt *c, Flt *md, Flt *tmp) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    auto coords = model_coords(src, m, md);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        CUDBG("nligand %d", src->nligand);
    }
    const int tid = threadIdx.x;
    const int blk = blockDim.x;
    
    // set root conf and prepare cos/sin angle for all ligands
    CU_FORY(i, src->nall) {
        Ligand &ligand = src->ligands[i];
        auto p         = get_ligand_conf(src, c, i);
        int offset = get_ligand_conf_angle_offset(src, i);
        Flt *cs = tmp + offset; // save torsion angles
        CU_FOR(j, ligand.nr_node-1+ligand.bflex) {
            auto angle = get_ligand_conf_torsion(p, j);
            normalize_angle(angle);
            angle *= 0.5;
            cs[j << 1] = COS(angle);
            cs[(j << 1)+1] = SIN(angle);
        }
    }
    XYSYNC();

    // calc root
    CU_FORY(i, src->nall) {
        Ligand &ligand = src->ligands[i];
        auto p         = get_ligand_conf(src, c, i);
        auto segvar    = model_ligand(src, m, md, i, ligand.nr_node - 1);
        Segment &seg   = src->segs[ligand.seg_offset + ligand.nr_node - 1];
        const Flt * orientation;
        // setup root
        if (ligand.bflex) {
            int offset = get_ligand_conf_angle_offset(src, i);
            Flt *cs = tmp + offset; // save torsion angles
            if (tid == 0) {
                segvar->tmp[0] = cs[0];
            } else if (tid < 4) {
                segvar->tmp[tid] = cs[1] * segvar->axis.d[tid-1];
            }
            orientation = segvar->tmp;
        } else {
            if (tid < 3) segvar->origin.d[tid] = p[tid];
            orientation = p + 3;
        }
        frame_set_orientation_c(tid, blk, segvar, orientation);
        atom_frame_set_coords_c(tid, blk, atoms, &seg, segvar, coords);
    }

    // set root conf and prepare cos/sin angle for all ligands
    FOR(i, src->nligand) {
        Ligand &ligand  = src->ligands[i];
        auto tree = src->segs+ligand.seg_offset;
        auto p          = get_ligand_conf(src, c, i);
        int offset = get_ligand_conf_angle_offset(src, i);
        Flt *cs = tmp + offset; // save torsion angles
        for(int layeridx = 1; layeridx <ligand.nr_layers; layeridx++) { // except root(layer 0)
            XYSYNC();
            // this layer locates in layer map [mapidx, mapidx + layernodes - 1]
            int mapidx = ligand.layers[layeridx*2];
            int layernodes = ligand.layers[layeridx*2+1];
            CU_FORY(idx, layernodes) {
                int segidx = ligand.layer_map[mapidx + idx]; // segment index
                auto &seg = tree[segidx];
                auto segvar = model_ligand(src, m, md, i, segidx);
                auto &parent = tree[seg.parent];
                auto parentVar = model_ligand(src, m, md, i, seg.parent);
                // for ligand:
                //     ntorision = nr_node - 1, so segidx + torsionidx = nr_node - 2
                // for flex:
                //     ntorsion = nr_node, so segidx + torsionidx = nr_node - 1
                // do not forget cs occupies 2 flts
                int csidx = 2* (ligand.nr_node - 2 - segidx + ligand.bflex);
                // if (tid == 0) printf("layer %d mapidx %d layernodes %d idx %d segidx %d parent %d csid %d\n",
                //        layeridx, mapidx, layernodes, idx, segidx, seg.parent, csidx);
                segment_set_conf_c(tid, blk, parentVar, segvar, &seg,  cs + csidx);
            }
        }
    }
    XYSYNC();
    FOR(i, src->nligand) {
        Ligand &ligand  = src->ligands[i];
        auto tree = src->segs + ligand.seg_offset;
        CU_FOR2(j, int(src->natoms)) {
            auto segidx = ligand.atom_map[j];
            // printf("%d %d: atom %d seg %d\n", threadIdx.x, threadIdx.y, j, segidx);
            if (segidx >= 0) {
                auto &seg = tree[segidx];
                auto segvar = model_ligand(src, m, md, i, segidx);
                frame_local_to_lab(segvar, atoms[j].coords, coords[j]);
            }
        }
    }

    DUMP_SEGVARS(m, md);
}
FORCE_INLINE void model_set_conf_ligand_xyz(ModelDesc *m, const Flt *c, Flt *md, Flt *tmp) {
    SrcModel *src = m->src;
    Atom *atoms   = src->atoms;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        CUDBG("nligand %d", src->nligand);
    }
    CU_FORZ (i,  src->nligand) {
        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
        const int blk = blockDim.x * blockDim.y;
        Ligand &ligand  = src->ligands[i];
        auto p          = get_ligand_conf(src, c, i);
        int offset = get_ligand_conf_angle_offset(src, i);
        Flt *cs = tmp + offset; // save torsion angles

        CU_FOR2(j, ligand.nr_node-1+ligand.bflex) {
            auto angle = get_ligand_conf_torsion(p, j);
            normalize_angle(angle);
            angle *= 0.5;
            cs[j << 1] = COS(angle);
            cs[(j << 1)+1] = SIN(angle); // todo: sqrt(1-cos^2)?
        }
    }
    SYNC();

    // calc root
    CU_FORYZ(i, src->nall) {
        const int tid = threadIdx.x;
        const int blk = blockDim.x;
        Ligand &ligand = src->ligands[i];
        auto p         = get_ligand_conf(src, c, i);
        auto segvar    = model_ligand(src, m, md, i, ligand.nr_node - 1);
        Segment &seg   = src->segs[ligand.seg_offset + ligand.nr_node - 1];
        const Flt * orientation;
        // setup root
        if (ligand.bflex) {
            int offset = get_ligand_conf_angle_offset(src, i);
            Flt *cs = tmp + offset; // save torsion angles
            if (tid == 0) {
                segvar->tmp[0] = cs[0];
            } else if (tid < 4) {
                segvar->tmp[tid] = cs[1] * segvar->axis.d[tid-1];
            }
            orientation = segvar->tmp;
        } else {
            if (tid < 3) segvar->origin.d[tid] = p[tid];
            orientation = p + 3;
        }
        frame_set_orientation_c(tid, blk, segvar, orientation);
        atom_frame_set_coords_c(tid, blk, atoms, &seg, segvar, coords);
    }
    SYNC();
    // use Y for ligands, X for ligand to make sure:
    // all set_conf for one segment is executed inside same warp to avoid sync
    // the nz/valid/max_ligand_layers are all used to make sure sync() execute on all threads
    int nz = blockDim.z;
    while(src->nall > nz) {
        nz += blockDim.z;
    }
    CU_FORZ(zidx, nz) {
        const int tid = threadIdx.x;
        const int blk = blockDim.x;
        int i = zidx;
        bool valid = zidx < src->nall;
        if (!valid) {
            i = 0;
        }
        Ligand &ligand  = src->ligands[i];
        auto tree = src->segs + ligand.seg_offset;
        auto p          = get_ligand_conf(src, c, i);
        int offset = get_ligand_conf_angle_offset(src, i);
        Flt *cs = tmp + offset; // save torsion angles

        for(int layeridx = 1; layeridx < src->max_ligand_layers; layeridx++) { // except root(layer 0)
            SYNC();
            if (valid && layeridx < ligand.nr_layers) {
                // this layer locates in layer map [mapidx, mapidx + layernodes - 1]
                int mapidx     = ligand.layers[layeridx * 2];
                int layernodes = ligand.layers[layeridx * 2 + 1];
                CU_FORY(idx, layernodes) {
                    int segidx     = ligand.layer_map[mapidx + idx];  // segment index
                    auto &seg      = tree[segidx];
                    auto segvar    = model_ligand(src, m, md, i, segidx);
                    auto &parent   = tree[seg.parent];
                    auto parentVar = model_ligand(src, m, md, i, seg.parent);
                    // ntorision = nr_node - 1, so segidx + torsionidx = nr_node - 2
                    // do not forget cs occupies 2 flts
                    int csidx = 2 * (ligand.nr_node - 2 - segidx + ligand.bflex);
                    // if (tid == 0) printf("layer %d mapidx %d layernodes %d idx %d segidx %d
                    // parent %d csid %d\n",
                    //        layeridx, mapidx, layernodes, idx, segidx, seg.parent, csidx);
                    segment_set_conf_c(tid, blk, parentVar, segvar, &seg, cs + csidx);
                }
            }
        }
    }
    SYNC();
    // update coords
    auto coords = model_coords(src, m, md);

    CU_FORZ(i, src->nligand) {
        Ligand &ligand  = src->ligands[i];
        auto tree = src->segs + ligand.seg_offset;
        CU_FOR2(j, int(src->natoms)) {
            auto segidx = ligand.atom_map[j];
            // printf("%d %d: atom %d seg %d\n", threadIdx.x, threadIdx.y, j, segidx);
            if (segidx >= 0) {
                auto &seg = tree[segidx];
                auto segvar = model_ligand(src, m, md, i, segidx);
                frame_local_to_lab(segvar, atoms[j].coords, coords[j]);
            }
        }
    }


    DUMP_SEGVARS(m, md);
}

};  // namespace dock