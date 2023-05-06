
#include "vina/cache.h"
#include "vina/model.h"
#include "vina/precalculate.h"
#include "vinadef.h"
#include "culog.h"
#include "cuvina.h"

namespace dock {
class Memory {
public:
    // f_alloc_ is used to allocate a memory, it returns the memory address at least has sz bytes.
    // for device mapped memory, the pptr will be filled with device memory address(while returning
    // host address), for other memories, pptr will be filled with returning address.
    // pptr will NOT be filled unless its not NULL
    using f_alloc_ = std::function<void *(size_t sz)>;
    // free memory allocated by f_alloc_
    using f_dealloc = std::function<void(void *)>;

    explicit Memory(f_alloc_ alloc, f_dealloc deaclloc, size_t blksz, size_t align = sizeof(double))
        : alloc_(std::move(alloc)), dealloc_(deaclloc), blksize_(blksz), align_(align) {
        blksize_ = (blksize_ + align - 1) / align * align;
        ptr_ = (uint8_t *)alloc_(blksize_);
    }
    ~Memory() {
        if (ptr_) {
            dealloc_((void *)ptr_);
            ptr_ = nullptr;
        }
    }

    bool empty() {
        return ptr_ == nullptr;
    }
    void reset() {
        offset_ = 0;
    }
    void *crop(size_t sz, size_t align) {
        sz = (sz + align - 1) / align * align;
        if (left() >= sz) {
            auto p = ptr_ + offset_;
            offset_ += sz;
            return p;
        }
        std::cerr << "fail to alloc " << sz << " bytes, left " << left() << std::endl;
        return nullptr;
    }
    size_t left() {
        return blksize_ - offset_;
    }

private:
    f_alloc_ alloc_;
    f_dealloc dealloc_;
    size_t blksize_;
    size_t align_;
    uint8_t *ptr_ = nullptr;
    size_t offset_   = 0;
};

using Memsp = std::shared_ptr<Memory>;

Memsp makeCpuMemory(Size size) {
    auto mem = std::make_shared<Memory>([](size_t sz) {
        return malloc(sz);
    }, [](void *p) {
        free(p);
    }, size, sizeof(Flt));
    if (mem->empty()) {
        return nullptr;
    }
    return mem;
}
Memsp makeCudaMemory(Size size) {
    auto mem = std::make_shared<Memory>([](size_t sz) {
        void *p = nullptr;
        cudaMalloc(&p, sz);
        return p;
    }, [](void *p) {
        if (p) {
            cudaFree(p);
        }
    }, size, sizeof(Flt));
    if (mem->empty()) {
        return nullptr;
    }
    return mem;
}

#if USE_CUDA
#define makeMemory makeCudaMemory
#else
#define makeMemory makeCpuMemory
#endif

std::shared_ptr<CuObj> makeCuObject(Memsp mem, void *obj) {
    auto co = std::make_shared<CuObj>();
    co->ctrl = mem;
    co->obj = obj;
    return co;
}
static inline void make_vec(Vec &v, Flt a, Flt b, Flt c) {
    v.x = a, v.y = b, v.z = c;
}
static inline void vec_set(Vec &v, const vec &src) {
    v.x = src.data[0], v.y = src.data[1], v.z = src.data[2];
}
static inline void vec_set(vec &v, const Vec &src) {
    v.data[0] = src.x, v.data[1] = src.y, v.data[2] = src.z; 
}
void copy_vec(Vec &dst, const vec &src) {
    make_vec(dst, src.data[0], src.data[1], src.data[2]);
}
void copy_vecs(Vec *dst, const vecv &srcs) {
    for (auto &v : srcs) {
        copy_vec(*dst, v);
        dst++;
    }
}
void copy_pair(InteractingPair *dst, interacting_pair *src,Flt cutoff_sqr, Flt v) {
    dst->a = src->a, dst->b = src->b, dst->type_pair_index = src->type_pair_index;
    dst->cutoff_sqr = cutoff_sqr;
    dst->v = v;
}
void copy_pairs(InteractingPair * &dst, interacting_pairs &src,Flt cutoff_sqr, Flt v) {
    for (auto &p : src)  {
        copy_pair(dst, &p, cutoff_sqr, v);
        dst++;
    }
}
template<typename T>
int tree_nodes_size(struct tree<T> &tree) {
    tree.nr_nodes = 0;
    for (auto &t : tree.children) {
        tree.nr_nodes += tree_nodes_size(t);
    }
    return tree.nr_nodes + 1;
}
void seg_tree_nodes_copy(int parent, int idx, Segment *segs, struct tree<segment> &tree) {
    auto myidx = idx + tree.nr_nodes;
    // fill child in reverse order first into segs
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        seg_tree_nodes_copy(myidx, idx, segs, *it);
        idx += it->nr_nodes + 1;
    }
    auto &seg = segs[myidx];
    seg.begin = tree.node.begin;
    seg.end = tree.node.end;
    seg.parent = parent;
    tree.seq = myidx;
    CUDBG("COPY HSubTree, my idx %d parent %d child %d begin %d end %d", myidx, parent, tree.nr_nodes, seg.begin, seg.end);
}
void segvar_tree_nodes_copy(int parent, int idx, SegmentVars *segs, const struct tree<segment> &tree) {
    auto myidx = idx + tree.nr_nodes;
    // fill child in reverse order first into segs
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        segvar_tree_nodes_copy(myidx, idx, segs, *it);
        idx += it->nr_nodes + 1;
    }
    auto &seg = segs[myidx];
    vec_set(seg.axis, tree.node.axis);
    vec_set(seg.origin, tree.node.get_origin());
    DBG("copy child node %d", myidx);
    CUVDUMP("    axis", seg.axis);
    CUVDUMP("    origin", seg.origin);
}
template<typename T>
int htree_nodes_size(struct heterotree<T> &tree) {
    tree.nr_nodes = 0;
    for (auto &branch : tree.children) {
        tree.nr_nodes += tree_nodes_size<segment>(branch);
    }
    return tree.nr_nodes + 1;
}
template<typename T>
void htree_nodes_copy(Segment *segs, struct heterotree<T> &tree) {
    int idx = 0;
    int myidx = tree.nr_nodes; // where this node is stored
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        seg_tree_nodes_copy(myidx, idx, segs, *it);
        idx += it->nr_nodes + 1;
    }
    auto &seg = segs[myidx];
    seg.begin = tree.node.begin;
    seg.end = tree.node.end;
    seg.parent = -1;
    CUDBG("COPY HTree, my idx %d child %d begin %d end %d", myidx, tree.nr_nodes, seg.begin, seg.end);
}
template<typename T>
void htree_var_nodes_copy(SegmentVars *segs, const struct heterotree<T> &tree) {
    int idx = 0;
    int myidx = tree.nr_nodes; // where this node is stored
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        segvar_tree_nodes_copy(myidx, idx, segs, *it);
        idx += it->nr_nodes + 1;
    }
    auto &seg = segs[myidx];
    vec_set(seg.axis, tree.node.axis);
    vec_set(seg.origin, tree.node.get_origin());
    DBG("copy root node %d", myidx);
    CUVDUMP("    axis", seg.axis);
    CUVDUMP("    origin", seg.origin);
}
        // std::cout << __FILE__ << ":" << __LINE__ << " " << "alloc " #type " size " << sizeof(type) << std::endl;
    #define ALLOC(dst, type) do {\
        dst = (type *)mem->crop(sizeof(type), sizeof(double));\
        if (dst == nullptr) goto cleanup;\
    } while(0)

        // std::cout << __FILE__ << ":" << __LINE__ << " " << "alloc arr " #type " size " << sizeof(type) << " cout " << n << " total " << sizeof(type) * n << std::endl;
    #define ALLOC_ARR(dst, type, n) do {\
        dst = (type *)mem->crop(sizeof(type) * n, sizeof(double));\
        if (dst == nullptr) goto cleanup;\
    } while(0)
SrcModel *make_src_model(Memory *mem, model *m, const vec &v, const precalculate_byatom &p) {

    SrcModel *sm;
    InteractingPair *pair;
    Flt sqr, max_sqr;
    ALLOC(sm, SrcModel);
    sm->movable_atoms = m->num_movable_atoms();
    sm->movable_v = v[1];
    sm->xs_nat = num_atom_types(atom_type::XS);
    ALLOC_ARR(sm->xs_sizes, Size, sm->movable_atoms);

    // determine grids index, see cache::eval
    for(int i = 0; i < sm->movable_atoms; i++) {
        auto &xsz = sm->xs_sizes[i];
        // if (m->is_atom_in_ligand(i)) {
        //     xsz = CU_INVALID_XS_SIZE;
        // } else {
            auto t = m->atoms[i].get(atom_type::XS);
            if (t >= sm->xs_nat) {
                xsz = CU_INVALID_XS_SIZE;
            } else {
                switch (t) {
                case XS_TYPE_G0:
                case XS_TYPE_G1:
                case XS_TYPE_G2:
                case XS_TYPE_G3:
                    xsz = CU_INVALID_XS_SIZE;
                    break;
                case XS_TYPE_C_H_CG0:
                case XS_TYPE_C_H_CG1:
                case XS_TYPE_C_H_CG2:
                case XS_TYPE_C_H_CG3:
                    xsz = XS_TYPE_C_H;
                    break;
                case XS_TYPE_C_P_CG0:
                case XS_TYPE_C_P_CG1:
                case XS_TYPE_C_P_CG2:
                case XS_TYPE_C_P_CG3:
                    xsz = XS_TYPE_C_P;
                    break;
                default:
                    xsz = t;
                    break;
                }
            }
        // }
    }

    // copy atom coords
    sm->ncoords = m->coords.size();
    ALLOC_ARR(sm->coords, Vec, sm->ncoords);
    copy_vecs(sm->coords, m->coords);

    sm->npairs = int(m->inter_pairs.size() + m->glue_pairs.size() + m->other_pairs.size());
    for (auto &ligand : m->ligands) {
        sm->npairs += int(ligand.pairs.size());
    }
    ALLOC_ARR(sm->pairs, InteractingPair, sm->npairs);
    pair = sm->pairs;
    sqr = p.cutoff_sqr();
    max_sqr = p.max_cutoff_sqr();
    for (auto &ligand : m->ligands) {
        copy_pairs(pair, ligand.pairs, sqr, v[0]);
    }
    copy_pairs(pair, m->inter_pairs, sqr, v[2]);
    copy_pairs(pair, m->other_pairs, sqr, v[2]);
    copy_pairs(pair, m->glue_pairs, max_sqr, v[2]);

    // prepare ligands and flex, in the reverse order to store tree
    sm->nligand = m->ligands.size();
    sm->nflex = m->flex.size();
    ALLOC_ARR(sm->ligands, Ligand, sm->nligand);
    ALLOC_ARR(sm->flex, Residue, sm->nflex);
    for (int i = 0; i < sm->nligand; i++) {
        auto &ligand = sm->ligands[i];
        ligand.nr_node = htree_nodes_size(m->ligands[i]);
        ALLOC_ARR(ligand.tree, Segment, ligand.nr_node);
        htree_nodes_copy(ligand.tree, m->ligands[i]);
    }
    for (int i = 0; i < sm->nflex; i++) {
        auto &flex = sm->flex[i];
        flex.nr_node = htree_nodes_size(m->flex[i]);
        ALLOC_ARR(flex.tree, Segment, flex.nr_node);
        htree_nodes_copy(flex.tree, m->flex[i]);
    }
    return sm;
cleanup:
    return nullptr;
}

Model *make_model(Memory *mem, SrcModel *sm, model *m) {
    Model *md;
    ALLOC(md, Model);
    md->src = sm;
    ALLOC_ARR(md->ligands, LigandVars, sm->nligand);
    ALLOC_ARR(md->flex, ResidueVars, sm->nflex);
    for (int i = 0; i < sm->nligand; i++) {
        auto &ligand = sm->ligands[i];
        auto &ligandvar = md->ligands[i];
        ALLOC_ARR(ligandvar.tree, SegmentVars, ligand.nr_node);
        htree_var_nodes_copy(ligandvar.tree, m->ligands[i]);
    }
    for (int i = 0; i < sm->nflex; i++) {
        auto &flex = sm->flex[i];
        auto &flexvar = md->flex[i];
        ALLOC_ARR(flexvar.tree, SegmentVars, flex.nr_node);
        htree_var_nodes_copy(flexvar.tree, m->flex[i]);
    }
    ALLOC_ARR(md->minus_forces, Vec, sm->movable_atoms);
    ALLOC_ARR(md->movable_e, Flt, sm->movable_atoms);
    ALLOC_ARR(md->pair_res, PairEvalResult, sm->npairs);

    return md;
cleanup:
    return nullptr;
}
Cache *make_cache(Memory *mem, const cache &c) {
    Cache *ch;
    ALLOC(ch, Cache);
    ch->m_slope = c.m_slope;
    ch->ngrids = int(c.m_grids.size());
    ALLOC_ARR(ch->grids, Grid, ch->ngrids);
    for (int i = 0; i < ch->ngrids; i++) {
        auto &dst = ch->grids[i];
        auto &src = c.m_grids[i];
        vec_set(dst.m_dim_fl_minus_1 , src.m_dim_fl_minus_1);
        vec_set(dst.m_factor, src.m_factor);
        vec_set(dst.m_init, src.m_init);
        vec_set(dst.m_factor_inv, src.m_factor_inv);
        dst.m_data.dim[0] = src.m_data.dim0();
        dst.m_data.dim[1] = src.m_data.dim1();
        dst.m_data.dim[2] = src.m_data.dim2();
        DBG("grid %d dim %d %d %d data size %lu", i, dst.m_data.dim[0], dst.m_data.dim[1], dst.m_data.dim[2], src.m_data.m_data.size());
        ALLOC_ARR(dst.m_data.data, Flt, src.m_data.m_data.size());
        memcpy(dst.m_data.data, src.m_data.m_data.data(), src.m_data.m_data.size() * sizeof(Flt));
    }
    return ch;
cleanup:
    return nullptr;
}
Change *make_change(Memory *mem, change &g) {
    Change *c;
    ALLOC(c, Change);
    ALLOC_ARR(c->ligands, LigandChange, g.ligands.size());
    ALLOC_ARR(c->flex, ResidueChange, g.flex.size());
    for (auto i = 0u; i < g.ligands.size(); i++) {
        auto &dst  = c->ligands[i];
        auto &src = g.ligands[i];
        DBG("Alloc ligand %u torsion size %lu", i, src.torsions.size());
        ALLOC_ARR(dst.torsions, Flt, src.torsions.size());
    }
    for (auto i = 0u; i < g.flex.size(); i++) {
        auto &dst  = c->flex[i];
        auto &src = g.flex[i];
        DBG("Alloc flex %u torsion size %lu", i, src.torsions.size());
        ALLOC_ARR(dst.torsions, Flt, src.torsions.size());
    }
    return c;
cleanup:
    return nullptr;
}

std::shared_ptr<Memory> create_prec_byatom_memory(const precalculate_byatom &p) {
    auto pesz = p.m_data.m_data.size();
    Size sz = sizeof(PrecalculateByAtom) + sizeof(PrecalculateElement) * pesz;
    for (auto i = 0u; i < pesz; i++) {
        auto &src = p.m_data.m_data[i];
        sz += sizeof(Flt) * src.fast.size();
        sz += sizeof(Flt) * src.smooth.size() * 2;
    }
    sz = (sz + 4096 -1) / 4096*4096;

    return makeMemory(sz);
}
PrecalculateByAtom *make_prec_atom(Memory *mem, const precalculate_byatom &p) {
    PrecalculateByAtom *pa;
    ALLOC(pa, PrecalculateByAtom);
    pa->cutoff_sqr = p.m_cutoff_sqr;
    pa->max_cutoff_sqr = p.m_max_cutoff_sqr;
    pa->pe_dim = p.m_data.dim();
    pa->pe_sz = p.m_data.m_data.size();
    // std::cout << "prec by atom dim " << pa->pe_dim << " size " << pa->pe_sz<<std::endl;
    ALLOC_ARR(pa->data, PrecalculateElement, pa->pe_sz);
    for (auto i = 0u; i < pa->pe_sz; i++) {
        auto &dst = pa->data[i];
        auto &src = p.m_data.m_data[i];
        dst.factor = src.factor;
        // std::cout << "pe " << i << " fast size " << src.fast.size() << " smooth size " << src.smooth.size() << std::endl;
        ALLOC_ARR(dst.fast, Flt, src.fast.size());
        ALLOC_ARR(dst.smooth, Flt, src.smooth.size() * 2);
        memcpy(dst.fast, src.fast.data(), src.fast.size() * sizeof(Flt));
        memcpy(dst.smooth, src.smooth.data(), src.fast.size() * sizeof(Flt));
        for (auto j = 0u; j < src.smooth.size(); j++) {
            dst.smooth[j * 2] = src.smooth[j].first;
            dst.smooth[j * 2+1] = src.smooth[j].second;
        }
    }

    return pa;
cleanup:
    return nullptr;
}
bool makePrecalcByAtom(precalculate_byatom &p) {
    auto mem = create_prec_byatom_memory(p);
    if (mem) {
        auto pa = make_prec_atom(mem.get(), p);
        if (pa) {
            p.m_gpu = makeCuObject(mem, pa);
            return true;
        }
    }
    return false;
}
void output_ligand_change(ligand_change &dst, LigandChange &src) {
    vec_set(dst.rigid.position, src.rigid.position);
    vec_set(dst.rigid.orientation, src.rigid.orientation);
    DBG("torsion size %lu", dst.torsions.size());
    CUDBG("src torsion %f %f %f", src.torsions[0], src.torsions[1], src.torsions[2]);
    memcpy(dst.torsions.data(), src.torsions, dst.torsions.size() * sizeof(dst.torsions[0]));
    CUDBG("dst torsion %f %f %f", dst.torsions[0], dst.torsions[1], dst.torsions[2]);
}
void output_flex_change(residue_change &dst, ResidueChange &src) {
    memcpy(dst.torsions.data(), src.torsions, dst.torsions.size() * sizeof(dst.torsions[0]));
}
extern void model_eval_deriv(Model &m, PrecalculateByAtom &p, Cache &c, Change &g);
fl run_model_eval_deriv(model *m, const precalculate_byatom &p, const igrid &ig, const vec &v,
                        change &g) {
    auto &c = (const cache &)ig;
    Memory mem([](size_t sz) {
        return malloc(sz);
    }, [](void *p) {
        free(p);
    }, size_t(2048) * size_t(1024 * 1024), sizeof(Flt));

    // const
    auto sm = make_src_model(&mem, m, v, p);
    auto ch = make_cache(&mem, c);
    auto pa = make_prec_atom(&mem, p);

    auto md = make_model(&mem, sm, m);
    auto chg = make_change(&mem, g);

    model_eval_deriv(*md, *pa, *ch, *chg);

    // output changes
    for (auto i = 0u; i < g.ligands.size(); i ++) {
        output_ligand_change(g.ligands[0], chg->ligands[0]);
    }
    for (auto i = 0u; i < g.flex.size(); i ++) {
        output_flex_change(g.flex[0], chg->flex[0]);
    }


    return md->e;

}
void comp_model(model *m1, model *m2) {
}

static fl eps = 0.0000001;
void comp_change(fl c1, fl c2) {
    auto diff = c1 - c2;
    if (diff < 0) diff = -diff;
    assert (diff < eps);
}
void comp_change(flv &c1, flv &c2) {
    assert(c1.size() == c2.size());
    for (auto i = 0u; i < c1.size(); i++)
        comp_change(c1[i], c2[i]);
}
void comp_change(vec &c1, vec &c2) {
    for (int i = 0; i < 3; i++)
        comp_change(c1.data[i], c2.data[i]);
}
void comp_change(ligand_change &c1, ligand_change &c2) {
    comp_change(c1.rigid.orientation, c2.rigid.orientation);
    comp_change(c1.rigid.position, c2.rigid.position);
    comp_change(c1.torsions, c2.torsions);
}
void comp_change(residue_change &c1, residue_change &c2) {
    comp_change(c1.torsions, c2.torsions);
}
void comp_change(change &c1, change &c2) {
    assert(c1.ligands.size() == c2.ligands.size());
    assert(c1.flex.size() == c2.flex.size());
    for(auto i = 0u; i < c1.ligands.size(); i++) {
        CUDBG("compare ligand %u", i);
        auto &src = c1.ligands[i];
        auto &dst = c2.ligands[i];
        comp_change(src, dst);
    }
    for(auto i = 0u; i < c1.flex.size(); i++) {
        CUDBG("compare flex %u", i);
        auto &src = c1.flex[i];
        auto &dst = c2.flex[i];
        comp_change(src, dst);
    }

    CUDBG("Compare %lu ligands %lu flex Done, congratulations!", c1.ligands.size(), c1.flex.size());
}

};  // namespace dock