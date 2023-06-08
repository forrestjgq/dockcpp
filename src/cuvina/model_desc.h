#ifdef ATTR
namespace dock {
    ATTR Flt *model_data(ModelDesc *d) {
        return d->data;
    }
    ATTR Vec *model_coords(const SrcModel *sm, const ModelDesc *d, Flt *m, int idx = 0) {
        Vec * coords = (Vec *)(m + d->coords);
        return coords + idx;
    }
    ATTR SegmentVars *model_ligand(const SrcModel *sm, const ModelDesc *d, Flt *m, int idx_ligand = 0, int idx_segvar = 0) {
        // m + d->ligands[idx_ligand] is the start point of ligands
        // it takes several trees, each contains some SegmentVars, the number of SegmentVars is defined in
        // SrcModel.ligands[idx_ligand].nr_node
        SegmentVars * ligands = (SegmentVars *)(m + d->ligands[idx_ligand]);
        return ligands + idx_segvar;
    }
    // ATTR SegmentVars *model_flex(const SrcModel *sm, const ModelDesc *d, Flt *m, int idx_flex = 0, int idx_segvar = 0) {
    //     // m + d->flex[idx_flex] is the start point of flex
    //     // it takes several trees, each contains some SegmentVars, the number of SegmentVars is defined in
    //     // SrcModel.flex[idx_flex].nr_node
    //     SegmentVars * flex = (SegmentVars *)(m + d->flex[idx_flex]);
    //     return flex + idx_segvar;
    // }
    ATTR Vec *model_minus_forces(const SrcModel *sm, const ModelDesc *d, Flt *m, int idx = 0) {
        Vec * f = (Vec *)(m + d->minus_forces);
        return f + idx;
    }
    ATTR Flt *model_movable_e(const SrcModel *sm, const ModelDesc *d, Flt *m, int idx = 0) {
        return m + d->movable_e + idx;
    }
    ATTR PairEvalResult *model_pair_res(const SrcModel *sm, const ModelDesc *d, Flt *m, int idx = 0) {
        PairEvalResult * res = (PairEvalResult *)(m + d->pair_res);
        return res + idx;
    }
};
#endif