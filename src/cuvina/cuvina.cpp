
#include "vina/cache.h"
#include "vina/model.h"
#include "vina/precalculate.h"
#include "vinadef.h"
#include "culog.h"
#include "cuvina.h"
#include "vinasrv.h"
#include <unordered_map>

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
        m_.clear();
    }
    void crop(void **pptr, size_t sz, size_t align) {
        sz = (sz + align - 1) / align * align;
        if (left() >= sz) {
            auto p = ptr_ + offset_;
            *pptr = p;
            if (offset_ > 0) {
                m_[(size_t)pptr - (size_t)ptr_] = offset_;
            }
            offset_ += sz;
        } else {
            std::cerr << "fail to alloc " << sz << " bytes, left " << left() << std::endl;
            *pptr = nullptr;
        }
    }
    void dump(uint8_t *base) {
         for (auto &p : m_) {
            *(uint8_t **)(ptr_ + p.first) = base + p.second;
         }
    }
    void restore() {
        dump(ptr_);
    }
    size_t left() {
        return blksize_ - offset_;
    }
    size_t size() {
        return blksize_;
    }
    uint8_t *ptr() {
        return ptr_;
    }

private:
    f_alloc_ alloc_;
    f_dealloc dealloc_;
    size_t blksize_;
    size_t align_;
    uint8_t *ptr_ = nullptr;
    size_t offset_   = 0;
    // records ptr at *((uint8_t *)obj + key) == ((uint8_t *)obj + value)
    std::unordered_map<size_t, size_t> m_;
};

using Memsp = std::shared_ptr<Memory>;

Memsp makeCpuMemory(Size size) {
    auto mem = std::make_shared<Memory>([](size_t sz) {
        CUDBG("make cpu mem size %lu", sz);
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
        CUDBG("make cuda mem size %lu", sz);
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

#define makeMemory makeCpuMemory

template <typename T>
inline T *extract_object(std::shared_ptr<void> stub) {
    auto obj = std::static_pointer_cast<CuObject>(stub);
    return (T *)(obj->obj);
}
template <typename T>
inline T *extract_cuda_object(std::shared_ptr<void> stub) {
    auto obj = std::static_pointer_cast<CuObject>(stub);
    return (T *)(obj->cuobj);
}
inline std::shared_ptr<Memory> extract_memory(std::shared_ptr<void> stub) {
    auto obj = std::static_pointer_cast<CuObject>(stub);
    return std::static_pointer_cast<Memory>(obj->ctrl);
}
inline std::shared_ptr<Memory> extract_cuda_memory(std::shared_ptr<void> stub) {
    auto obj = std::static_pointer_cast<CuObject>(stub);
    return std::static_pointer_cast<Memory>(obj->cuctrl);
}
inline std::shared_ptr<CuObject> makeCuObject(Memsp mem, void *obj) {
    auto co = std::make_shared<CuObject>();
    co->ctrl = mem;
    co->obj = obj;
#if USE_CUDA_VINA
    submit_vina_server([=](cudaStream_t) {
        auto cudamem = makeCudaMemory(mem->size());
        if(cudamem) {
            auto cudabase = cudamem->ptr();
            mem->dump(cudabase);
            cudaMemcpy(cudabase, mem->ptr(), mem->size(), cudaMemcpyHostToDevice);
            co->cuctrl = cudamem;
            co->cuobj = cudabase;
        }
    });
    if (co->cuctrl) {
        mem->restore();
    } else {
        return nullptr;
    }
#endif
    return co;
}
inline void updateCuObject(std::shared_ptr<void> stub, void *obj) {
    auto cuobj = std::static_pointer_cast<CuObject>(stub);
    cuobj->obj = obj;
#if USE_CUDA_VINA
    if (cuobj->cuctrl) {
        auto mem = std::static_pointer_cast<Memory>(cuobj->ctrl);
        submit_vina_server([=](cudaStream_t) {
            auto cudamem = std::static_pointer_cast<Memory>(cuobj->cuctrl);
            auto cudabase = cudamem->ptr();
            mem->dump(cudabase);
            cudaMemcpy(cudabase, mem->ptr(), mem->size(), cudaMemcpyHostToDevice);
        });
        mem->restore();
    }
#endif
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
void copy_pair(InteractingPair *dst, interacting_pair *src,Flt cutoff_sqr, int v) {
    dst->a = src->a, dst->b = src->b, dst->type_pair_index = src->type_pair_index;
    dst->cutoff_sqr = cutoff_sqr;
    dst->v = v;
}
void copy_pairs(InteractingPair * &dst, interacting_pairs &src,Flt cutoff_sqr, int v) {
    for (auto &p : src)  {
        copy_pair(dst, &p, cutoff_sqr, v);
        dst++;
    }
}
void copy_qt(Qt &dst, const qt &src) {
    dst.x = src.R_component_1();
    dst.y = src.R_component_2();
    dst.z = src.R_component_3();
    dst.w = src.R_component_4();
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
    auto &axis = tree.node.relative_axis;
    auto &origin = tree.node.relative_origin;
    make_vec(seg.relative_axis, axis.data[0], axis.data[1], axis.data[2]);
    make_vec(seg.relative_origin, origin.data[0], origin.data[1], origin.data[2]);
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
        mem->crop((void **)&(dst), sizeof(type), sizeof(double));\
        if (dst == nullptr) goto cleanup;\
    } while(0)

        // std::cout << __FILE__ << ":" << __LINE__ << " " << "alloc arr " #type " size " << sizeof(type) << " cout " << n << " total " << sizeof(type) * n << std::endl;
    #define ALLOC_ARR(dst, type, n) do {\
        mem->crop((void **)&(dst), sizeof(type) * n, sizeof(double));\
        if (dst == nullptr) goto cleanup;\
    } while(0)

#define ALIGNMENT sizeof(double)
#define ALIGN(x, a) (((x) + (a) - 1) / (a) * (a))
#define SIZEOF(t) ALIGN(sizeof(t), ALIGNMENT)
#define SIZEOFARR(t, sz) ALIGN(sizeof(t)*sz, ALIGNMENT)
std::shared_ptr<Memory> create_src_model_memory(model *m) {

    auto npairs = int(m->inter_pairs.size() + m->glue_pairs.size() + m->other_pairs.size());
    for (auto &ligand : m->ligands) {
        npairs += int(ligand.pairs.size());
    }
    Size sz = SIZEOF(SrcModel) + SIZEOFARR(Size, m->num_movable_atoms()) +
    SIZEOFARR(Atom, m->atoms.size()) +
    SIZEOFARR(InteractingPair, npairs) +
    SIZEOFARR(Ligand, m->ligands.size()) +
    SIZEOFARR(Residue, m->flex.size());
    for (auto i = 0u; i < m->ligands.size(); i++) {
        auto nr_node = htree_nodes_size(m->ligands[i]);
        sz += SIZEOFARR(Segment, nr_node);
    }
    for (auto i = 0u; i < m->flex.size(); i++) {
        auto nr_node = htree_nodes_size(m->flex[i]);
        sz += SIZEOFARR(Segment, nr_node);
    }
    sz = (sz + 4096 -1) / 4096*4096;
    return makeMemory(sz);
}
SrcModel *make_src_model(Memory *mem, model *m, const precalculate_byatom &p) {

    SrcModel *sm;
    InteractingPair *pair;
    Flt sqr, max_sqr;
    ALLOC(sm, SrcModel);
    sm->movable_atoms = m->num_movable_atoms();
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

    sm->natoms = m->atoms.size();
    ALLOC_ARR(sm->atoms, Atom, sm->natoms);
    for (auto i = 0u; i < sm->natoms; i++) {
        auto &c = m->atoms[i].coords;
        make_vec(sm->atoms[i].coords, c.data[0], c.data[1], c.data[2]);
    }

    sm->npairs = int(m->inter_pairs.size() + m->glue_pairs.size() + m->other_pairs.size());
    for (auto &ligand : m->ligands) {
        sm->npairs += int(ligand.pairs.size());
    }
    ALLOC_ARR(sm->pairs, InteractingPair, sm->npairs);
    pair = sm->pairs;
    sqr = p.cutoff_sqr();
    max_sqr = p.max_cutoff_sqr();
    for (auto &ligand : m->ligands) {
        copy_pairs(pair, ligand.pairs, sqr, 0); // v0
    }
    copy_pairs(pair, m->inter_pairs, sqr, 2); // v2
    copy_pairs(pair, m->other_pairs, sqr, 2);
    copy_pairs(pair, m->glue_pairs, max_sqr, 2);

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
bool makeSrcModel(model *m, precalculate_byatom &p) {
    auto mem = create_src_model_memory(m);
    if (mem) {
        auto sm = make_src_model(mem.get(), m, p);
        if (sm) {
            m->m_gpu = makeCuObject(mem, sm);
            return true;
        }
    }
    return false;
}

std::shared_ptr<Memory> create_model_memory(model *m, SrcModel *sm) {
    Size sz = SIZEOF(Model) + SIZEOFARR(LigandVars, sm->nligand) + SIZEOFARR(ResidueVars, sm->nflex);
    for (int i = 0; i < sm->nligand; i++) {
        sz += SIZEOFARR(SegmentVars, sm->ligands[i].nr_node);
    }
    for (int i = 0; i < sm->nflex; i++) {
        sz += SIZEOFARR(SegmentVars, sm->flex[i].nr_node);
    }
    sz += SIZEOFARR(Vec, sm->movable_atoms);
    sz += SIZEOFARR(Flt, sm->movable_atoms);
    sz += SIZEOFARR(PairEvalResult, sm->npairs);
    sz += SIZEOFARR(Vec, m->coords.size());

    sz = (sz + 4096 -1) / 4096*4096;
    return makeMemory(sz);
}
Model *make_model(Memory *mem, SrcModel *sm, model *m, const vec &v) {
    Model *md;
    ALLOC(md, Model);
    md->src = sm;
    md->vs[0] = v.data[0], md->vs[1] = v.data[1], md->vs[2] = v.data[2];
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

    md->ncoords = m->coords.size();
    ALLOC_ARR(md->coords, Vec, md->ncoords);
    copy_vecs(md->coords, m->coords);
    return md;
cleanup:
    return nullptr;
}
std::shared_ptr<void> makeModel(model *m, const vec &v) {
    auto sm = extract_object<SrcModel>(m->m_gpu);
    auto mem = create_model_memory(m, sm);
    if (mem) {
        auto md = make_model(mem.get(), sm, m, v);
        if (md) {
            auto gpusm = extract_cuda_object<SrcModel>(m->m_gpu);
            md->src = gpusm;
            auto ret = makeCuObject(mem, md);
            md->src = sm;
            return ret;
        }
    }
    return nullptr;
}
std::shared_ptr<Memory> create_cache_memory(const cache &c) {
    Size sz = SIZEOF(Cache) + SIZEOFARR(Grid, c.m_grids.size());
    for (auto i = 0u; i <c.m_grids.size(); i++) {
        sz += SIZEOFARR(Flt, c.m_grids[i].m_data.m_data.size());
    }
    sz = (sz + 4096 -1) / 4096*4096;
    return makeMemory(sz);
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
bool makeCache(cache &c) {
    auto mem = create_cache_memory(c);
    if (mem) {
        auto ch = make_cache(mem.get(), c);
        if (ch) {
            c.m_gpu = makeCuObject(mem, ch);
            return true;
        }
    }
    return false;
}
Size eval_change_size(const change &g) {
    Size sz = SIZEOFARR(LigandChange, g.ligands.size()) + SIZEOFARR(LigandChange, g.ligands.size());
    for (auto i = 0u; i < g.ligands.size(); i++) {
        auto &src = g.ligands[i];
        sz += SIZEOFARR(Flt, src.torsions.size());
    }
    for (auto i = 0u; i < g.flex.size(); i++) {
        auto &src  = g.flex[i];
        sz += SIZEOFARR(Flt, src.torsions.size());
    }
    return sz;
}
Size eval_conf_size(const conf &c) {
    Size sz = 0;
    sz += SIZEOFARR(LigandConf, c.ligands.size());
    sz += SIZEOFARR(ResidueConf, c.flex.size());
    for (auto i = 0u; i < c.ligands.size(); i++) {
        auto &src = c.ligands[i];
        sz += SIZEOFARR(Flt, src.torsions.size());
    }
    for (auto i = 0u; i < c.flex.size(); i++) {
        auto &src = c.flex[i];
        sz += SIZEOFARR(Flt, src.torsions.size());
    }
    return sz;
}
std::shared_ptr<Memory> create_bfgs_memory(const change &g, const conf &c) {
    Size sz = SIZEOF(BFGSCtx) + eval_change_size(g) + eval_conf_size(c);
    sz = (sz + 4096 -1) / 4096*4096;
    return makeMemory(sz);
}
bool make_change(Memory *mem, Change *c, const change &g) {
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
    return true;
cleanup:
    return false;
}
bool make_conf(Memory *mem, Conf *ret, const conf &c) {
    ALLOC_ARR(ret->ligands, LigandConf, c.ligands.size());
    ALLOC_ARR(ret->flex, ResidueConf, c.flex.size());
    for (auto i = 0u; i < c.ligands.size(); i++) {
        auto &dst  = ret->ligands[i];
        auto &src = c.ligands[i];
        DBG("Alloc ligand %u torsion size %lu", i, src.torsions.size());
        ALLOC_ARR(dst.torsions, Flt, src.torsions.size());
        memcpy(dst.torsions, src.torsions.data(), src.torsions.size() * sizeof(src.torsions[0]));
        make_vec(dst.rigid.position, src.rigid.position.data[0], src.rigid.position.data[1], src.rigid.position.data[2]);
        copy_qt(dst.rigid.orientation, src.rigid.orientation);

    }
    for (auto i = 0u; i < c.flex.size(); i++) {
        auto &dst  = ret->flex[i];
        auto &src = c.flex[i];
        DBG("Alloc flex %u torsion size %lu", i, src.torsions.size());
        ALLOC_ARR(dst.torsions, Flt, src.torsions.size());
        memcpy(dst.torsions, src.torsions.data(), src.torsions.size() * sizeof(src.torsions[0]));
    }
    return true;
cleanup:
    return false;
}
BFGSCtx *make_bfgs(Memory *mem, const change &g, const conf &c) {
    BFGSCtx *ctx;
    ALLOC(ctx, BFGSCtx);
    if (!make_change(mem, &ctx->g, g)) goto cleanup;
    if (!make_conf(mem, &ctx->c, c)) goto cleanup;
    return ctx;
cleanup:
    return nullptr;
}

bool makeBFGSCtx(std::shared_ptr<void> &obj, const change &g, const conf &c) {
    std::shared_ptr<Memory> mem;
    if (!obj) {
        mem = create_bfgs_memory(g, c);
    } else {
        mem = extract_memory(obj);
    }
    if (!mem) {
        return false;
    }
    mem->reset();
    auto ctx = make_bfgs(mem.get(), g, c);
    if(!ctx) {
        return false;
    }
    if (!obj) {
        obj = makeCuObject(mem, ctx);
    } else {
        updateCuObject(obj, ctx);
    }
    return true;
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
    CUDBG("%d  make precalc by atom", 0);
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
#if USE_CUDA_VINA
void cuda_model_set_conf(Model &cpum, Model &m, Conf &c, cudaStream_t stream);
void cuda_cache_eval_deriv(Model *cpum, Cache *c, Model *m, cudaStream_t stream);
void cu_model_eval_deriv(Model *cpum, Model *m, PrecalculateByAtom *p, BFGSCtx *ctx, cudaStream_t stream);
fl run_model_eval_deriv(const precalculate_byatom &p, const igrid &ig, 
                        change &g, std::shared_ptr<void> mobj, std::shared_ptr<void> ctxobj) {
    submit_vina_server([&](cudaStream_t stream) {
        auto &che = (const cache &)ig;
        // const
        auto ch = extract_cuda_object<Cache>(che.m_gpu);
        auto pa = extract_cuda_object<PrecalculateByAtom>(p.m_gpu);
        auto md = extract_cuda_object<Model>(mobj);
        auto cpum = extract_object<Model>(mobj);
        auto ctx = extract_cuda_object<BFGSCtx>(ctxobj);

        cuda_model_set_conf(*cpum, *md, ctx->c, stream);
        cuda_cache_eval_deriv(cpum, ch, md, stream);
        cu_model_eval_deriv(cpum, md, pa, ctx, stream);

        auto cpumem = extract_memory(ctxobj);
        auto cudamem = extract_cuda_memory(ctxobj);
        cudaMemcpyAsync(cpumem->ptr(), cudamem->ptr(), cpumem->size(), cudaMemcpyDeviceToHost, stream);
        auto err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "vina eval fail, err: " << cudaGetErrorString(err) << std::endl;
        }
    }) ;

    auto cpumem = extract_memory(ctxobj);
    cpumem->restore();
    auto ctx = extract_object<BFGSCtx>(ctxobj);
    auto chg = &ctx->g;
    // output changes
    for (auto i = 0u; i < g.ligands.size(); i ++) {
        output_ligand_change(g.ligands[0], chg->ligands[0]);
    }
    for (auto i = 0u; i < g.flex.size(); i ++) {
        output_flex_change(g.flex[0], chg->flex[0]);
    }

    return ctx->e;

}
#else
extern void model_eval_deriv(Model &m, PrecalculateByAtom &p, Cache &c, Change &g);
extern void model_set_conf(Model &m, Conf &c);
fl run_model_eval_deriv(const precalculate_byatom &p, const igrid &ig, 
                        change &g, std::shared_ptr<void> mobj, std::shared_ptr<void> ctxobj) {
    auto &che = (const cache &)ig;

    // const
    auto ch = extract_object<Cache>(che.m_gpu);
    auto pa = extract_object<PrecalculateByAtom>(p.m_gpu);
    auto md = extract_object<Model>(mobj);
    auto ctx = extract_object<BFGSCtx>(ctxobj);
    auto chg = &ctx->g;
    auto cf = &ctx->c;

    model_set_conf(*md, *cf);
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
#endif

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