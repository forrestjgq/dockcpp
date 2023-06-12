
#include "vina/cache.h"
#include "vina/model.h"
#include "vina/precalculate.h"
#include "vinadef.h"
#include "vina/log.h"
#include "cuvina.h"
#include <unordered_map>
#include "cuda_context.h"
// #include "vinasrv.h"

#define ATTR inline
#include "model_desc.h"
namespace dock {
bool submit_vina_server(StreamCallback callback);

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

            // printf("alloc sz %lu: %p end %p\n", sz, p, ptr_+offset_);
        } else {
            std::cerr << "fail to alloc " << sz << " bytes, left " << left() << std::endl;
            *pptr = nullptr;
            assert(false);
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
        DBG("make cpu mem size %lu", sz);
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
        DBG("make cuda mem size %lu", sz);
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
    // v.x = a, v.y = b, v.z = c;
    v.d[0] = a, v.d[1] = b, v.d[2] = c;
}
static inline void vec_set(Vec &v, const vec &src) {
    // v.x = src.data[0], v.y = src.data[1], v.z = src.data[2];
    v.d[0] = src.data[0], v.d[1] = src.data[1], v.d[2] = src.data[2];
}
static inline void vec_set(vec &v, const Vec &src) {
    v.data[0] = src.d[0], v.data[1] = src.d[1], v.data[2] = src.d[2]; 
}
void copy_vec(Vec &dst, const vec &src) {
    make_vec(dst, src.data[0], src.data[1], src.data[2]);
}
void copy_vec(vec &dst, const Vec &src) {
    dst.data[0] = src.d[0], dst.data[1] = src.d[1], dst.data[2] = src.d[2];
}
void copy_vecs(Vec *dst, const vecv &srcs) {
    for (auto &v : srcs) {
        VDUMP("copy vec", v);
        copy_vec(*dst, v);
        dst++;
    }
}
void copy_vecs(vecv &dst, const Vec *srcs) {
    for (auto &v : dst) {
        copy_vec(v, *srcs);
        srcs++;
    }
}
static inline void qt_set(qt &dst, const Qt &src) {
    dst = qt(src.d[0], src.d[1], src.d[2], src.d[3]);
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
void eval_pairs(int &idx, std::map<int, std::vector<int>> &adds,std::map<int, std::vector<int>> &subs, interacting_pairs &src) {
    for (auto &p : src)  {
        subs[p.a].push_back(idx);
        adds[p.b].push_back(idx);
        idx++;
    }
}
void copy_qt(Qt &dst, const qt &src) {
    dst.d[0] = src.R_component_1();
    dst.d[1] = src.R_component_2();
    dst.d[2] = src.R_component_3();
    dst.d[3] = src.R_component_4();
}
// ok
template<typename T>
int tree_nodes_size(struct tree<T> &tree) {
    tree.nr_nodes = 0;
    for (auto &t : tree.children) {
        tree.nr_nodes += tree_nodes_size(t);
    }
    return tree.nr_nodes + 1;
}

int seg_tree_nodes_prep(struct tree<segment> &tree, int layer, std::map<int, std::vector<struct tree<segment> *>> &m) {
    int ret_layer = layer;
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        auto ret = seg_tree_nodes_prep(*it, layer+1, m);
        ret_layer = std::max(ret_layer, ret);
    }
    tree.layer = layer;
    auto it = m.find(layer);
    if (it == m.end()) {
        m[layer] = {&tree};
    } else {
        m[layer].push_back(&tree);
    }
    return ret_layer;
}
void seg_tree_nodes_set_parent(struct tree<segment> &tree, int parent) {
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        seg_tree_nodes_set_parent(*it, tree.idx);
    }
    tree.parentIdx = parent;
}
template<typename T>
int htree_nodes_prep(Segment *segs, struct heterotree<T> &tree, int *layermap) {
    std::map<int, std::vector<struct tree<segment> *>> m;
    int layerIdx = 0; // layer index for each node, 0 ~ layers
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        auto ret  = seg_tree_nodes_prep(*it, 1, m);
        layerIdx = std::max(ret, layerIdx);
    }
    int layers = layerIdx + 1; // how many layers

    // setup index for each segment
    int idx = 0;
    for (auto i = layerIdx; i > 0; i --) {
        for (auto node : m[i]) {
            node->idx = idx;
            idx++;
        }
    }
    // now idx is the root index
    // setup parent index for each segment
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        seg_tree_nodes_set_parent(*it, idx);
    }

    int mapidx = 0; // for one layer, mapidx indicates the pos of layer starting in segments
    for (auto i = layerIdx; i > 0; i --) {
        layermap[i * 2] = mapidx;
        layermap[i * 2+1] = int(m[i].size());
        mapidx += int(m[i].size());
        
        for (auto t : m[i]) {
            auto &seg    = segs[t->idx];
            auto &node   = t->node;
            seg.begin    = node.begin;
            seg.end      = node.end;
            seg.parent   = t->parentIdx;
            seg.layer    = t->layer;
            vec_set(seg.relative_axis, node.relative_axis);
            vec_set(seg.relative_origin, node.relative_origin);
            // auto &axis   = node.relative_axis;
            // auto &origin = node.relative_origin;
            // make_vec(seg.relative_axis, axis.data[0], axis.data[1], axis.data[2]);
            // make_vec(seg.relative_origin, origin.data[0], origin.data[1], origin.data[2]);
        }
    }
    layermap[0] = idx;
    layermap[1] = 1;

    auto &seg = segs[idx];
    seg.begin  = tree.node.begin;
    seg.end    = tree.node.end;
    seg.parent = -1;
    seg.layer  = 0;

    for (auto i = 0; i < idx; i++) {
        auto &seg = segs[i];
        printf("SEG %d/%d: parent %d begin %d end %d layer %d\n", i, idx, seg.parent, seg.begin, seg.end, seg.layer);
    }
    return layers;
}
// layer: which layer segment in tree, start from 0
int seg_tree_nodes_copy(int parent, int idx, Segment *segs, struct tree<segment> &tree, int layer) {
    int layers = 0;
    auto myidx = idx + tree.nr_nodes;
    // fill child in reverse order first into segs
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        auto sublayers = seg_tree_nodes_copy(myidx, idx, segs, *it, layer+1);
        idx += it->nr_nodes + 1;
        layers = std::max(sublayers, layers);
    }
    auto &seg = segs[myidx];
    seg.begin = tree.node.begin;
    seg.end = tree.node.end;
    seg.parent = parent;
    seg.layer = layer;
    auto &axis = tree.node.relative_axis;
    auto &origin = tree.node.relative_origin;
    make_vec(seg.relative_axis, axis.data[0], axis.data[1], axis.data[2]);
    make_vec(seg.relative_origin, origin.data[0], origin.data[1], origin.data[2]);
    tree.idx = myidx;
    printf("COPY HSubTree, my idx %d parent %d child %d begin %d end %d\n", myidx, parent, tree.nr_nodes, seg.begin, seg.end);
    return layers + 1;
}
void segvar_tree_nodes_restore(int parent, int idx, const SegmentVars *segs, struct tree<segment> &tree) {
    auto myidx = idx + tree.nr_nodes;
    // fill child in reverse order first into segs
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        segvar_tree_nodes_restore(myidx, idx, segs, *it);
        idx += it->nr_nodes + 1;
    }
    auto &seg = segs[myidx];
    vec_set(tree.node.axis, seg.axis);
    vec_set(tree.node.origin, seg.origin);
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
    DBG("    axis %f %f %f", seg.axis.d[0], seg.axis.d[1], seg.axis.d[2]);
    DBG("    origin %f %f %f", seg.origin.d[0], seg.origin.d[1], seg.origin.d[2]);
}
// ok
template<typename T>
int htree_nodes_size(struct heterotree<T> &tree) {
    tree.nr_nodes = 0;
    for (auto &branch : tree.children) {
        tree.nr_nodes += tree_nodes_size<segment>(branch);
    }
    return tree.nr_nodes + 1;
}
template<typename T>
int htree_nodes_copy(Segment *segs, struct heterotree<T> &tree) {
    int idx = 0;
    int myidx = tree.nr_nodes; // where this node is stored
    int layers = 0;
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        auto sublayers = seg_tree_nodes_copy(myidx, idx, segs, *it, 1);
        idx += it->nr_nodes + 1;
        layers = std::max(layers, sublayers);
    }
    auto &seg = segs[myidx];
    seg.begin = tree.node.begin;
    seg.end = tree.node.end;
    seg.parent = -1;
    seg.layer = 0;
    printf("COPY HTree, my idx %d child %d begin %d end %d\n", myidx, tree.nr_nodes, seg.begin, seg.end);
    return layers+1;
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
    DBG("    axis %f %f %f", seg.axis.d[0], seg.axis.d[1], seg.axis.d[2]);
    DBG("    origin %f %f %f", seg.origin.d[0], seg.origin.d[1], seg.origin.d[2]);
}
void segvar_tree_nodes_prep(SegmentVars *segs, const struct tree<segment> &tree) {
    // fill child in reverse order first into segs
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        segvar_tree_nodes_prep(segs, *it);
    }
    auto &seg = segs[tree.idx];
    vec_set(seg.axis, tree.node.axis);
    vec_set(seg.origin, tree.node.get_origin());
}
template<typename T>
void htree_var_nodes_prep(SegmentVars *segs, struct heterotree<T> &tree) {
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        segvar_tree_nodes_prep(segs, *it);
    }
    auto &seg = segs[tree.nr_nodes];
    vec_set(seg.axis, tree.node.axis);
    vec_set(seg.origin, tree.node.get_origin());
}
template<typename T>
void htree_var_nodes_restore(const SegmentVars *segs, struct heterotree<T> &tree) {
    int idx = 0;
    int myidx = tree.nr_nodes; // where this node is stored
    for (auto it = tree.children.rbegin(); it != tree.children.rend(); ++it) {
        segvar_tree_nodes_restore(myidx, idx, segs, *it);
        idx += it->nr_nodes + 1;
    }
    auto &seg = segs[myidx];
    vec_set(tree.node.axis, seg.axis);
    vec_set(tree.node.origin, seg.origin);
}
        // std::cout << __FILE__ << ":" << __LINE__ << " " << "alloc " #type " size " << sizeof(type) << std::endl;
    #define ALLOC(dst, type) do {\
        mem->crop((void **)&(dst), sizeof(type), sizeof(double));\
        if (dst == nullptr) goto cleanup;\
    } while(0)

        // std::cout << __FILE__ << ":" << __LINE__ << " " << "alloc arr " #type " size " << sizeof(type) << " cout " << n << " total " << sizeof(type) * n << std::endl;
    #define ALLOC_ARR(dst, type, n) do {\
        mem->crop((void **)&(dst), sizeof(type) * (n), sizeof(double));\
        if (dst == nullptr) goto cleanup;\
    } while(0)

#define ALIGNMENT sizeof(double)
#define ALIGN(x, a) (((x) + (a) - 1) / (a) * (a))
#define SIZEOF(t) ALIGN(sizeof(t), ALIGNMENT)
#define SIZEOFARR(t, sz) ALIGN(sizeof(t)*(sz), ALIGNMENT)
// ok
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


    // allocated memory space for idx_sub/add, force_pair_map_sub/add
    // note that idx_sub/add has fixed space requirement as movable_atoms ints
    sz += 2 * SIZEOFARR(int, m->num_movable_atoms());
    // and because the pair number is fixed to npairs, and each of them takes one
    // int in force_pair_map_sub/add, plus the prefix count(each minus-force has one count),
    // the total space will be:
    sz += 2 * SIZEOFARR(int, m->num_movable_atoms() + npairs);


    // for force_pair_map_sub and force_pair_map_add
    sz += SIZEOFARR(int, m->num_movable_atoms() * (npairs + 1));

    for (auto i = 0u; i < m->ligands.size(); i++) {
        auto nr_node = htree_nodes_size(m->ligands[i]);
        sz += SIZEOFARR(Segment, nr_node);
        sz += SIZEOFARR(int, nr_node * 2); // for Ligand.layers
        sz += SIZEOFARR(int, nr_node); // for Ligand.layer_map
        sz += SIZEOFARR(int, m->atoms.size()); // for Ligand.atom_map
    }
    for (auto i = 0u; i < m->flex.size(); i++) {
        auto nr_node = htree_nodes_size(m->flex[i]);
        sz += SIZEOFARR(Segment, nr_node);
        sz += SIZEOFARR(int, nr_node * 2); // for FLex.layers
        sz += SIZEOFARR(int, nr_node); // for Flex.layer_map
        sz += SIZEOFARR(int, m->atoms.size()); // for Flex.atom_map
    }
    sz = (sz + 4096 -1) / 4096*4096;
    return makeMemory(sz);
}
void checksm(SrcModel *sm, int mark, const char *f, int line) {

    for (auto i = 0; i < sm->movable_atoms + sm->npairs; i++) {
        auto k = sm->force_pair_map_add[i];
        if (k < 0) {
            printf("check false at %s %d mark %d\n", f, line, mark);
            fflush(stdout);
            assert(false);
        }
    }
}
#define CHECKSM(i) checksm(sm , i, __FILE__, __LINE__)
SrcModel *make_src_model(Memory *mem, model *m, const precalculate_byatom &p) {

    SrcModel *sm;
    InteractingPair *pair;
    Flt sqr, max_sqr;
    int offset_sub = 0, offset_add = 0;
    std::map<int, std::vector<int>> adds, subs;

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
        sm->atoms[i].el = m->atoms[i].el;
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

    // prepare force-pair mapping
    // key: index of min-force, v: indices of pairs
    for (int i = 0; i < sm->movable_atoms; i++) {
        adds[i] = {};
        subs[i] = {};
    }
    for (int i = 0; i < sm->npairs; i++) {
        auto &ip = sm->pairs[i];
        subs[ip.a].push_back(i);
        adds[ip.b].push_back(i);
    }
    ALLOC_ARR(sm->idx_sub, int, sm->movable_atoms);
    ALLOC_ARR(sm->idx_add, int, sm->movable_atoms);
    ALLOC_ARR(sm->force_pair_map_add, int, sm->movable_atoms + sm->npairs);
    // printf("ints %d, add %p -> %p\n", sm->movable_atoms + sm->npairs, sm->force_pair_map_add, sm->force_pair_map_add + sm->movable_atoms + sm->npairs);
    ALLOC_ARR(sm->force_pair_map_sub, int, sm->movable_atoms + sm->npairs);
    
    for (int i = 0; i < sm->movable_atoms; i++) {
        auto &add = adds[i];
        if (add.empty()) {
            sm->idx_add[i] = -1;
            // printf("minus-force %d, -1\n", i);
        } else {
            sm->idx_add[i] = offset_add;
            // printf("minus-force %d: offset %d size %d\n", i, offset_add, (int)(add.size()));
            auto *p = sm->force_pair_map_add + offset_add;
            *p++ = (int)(add.size());
            offset_add += (int)(add.size())+1;
            for (auto k : add) {
                *p++ = k;
                // printf("\tpair %d\n", k);
            }
        }
        auto &sub = subs[i];
        if (sub.empty()) {
            sm->idx_sub[i] = -1;
        } else {
            sm->idx_sub[i] = offset_sub;
            auto *p = sm->force_pair_map_sub + offset_sub;
            *p++ = (int)(sub.size());
            offset_sub += (int)(sub.size())+1;
            for (auto k : sub) {
                *p++ = k;
            }
        }
    }

    // prepare ligands and flex, in the reverse order to store tree
    sm->nligand = m->ligands.size();
    sm->nflex = m->flex.size();
    ALLOC_ARR(sm->ligands, Ligand, sm->nligand);
    ALLOC_ARR(sm->flex, Residue, sm->nflex);
    sm->nrfligands = 0;
    sm->nrfflex = 0;
    sm->max_ligand_layers = 0;
    sm->max_flex_layers = 0;
    for (int i = 0; i < sm->nligand; i++) {
        auto &ligand = sm->ligands[i];
        ligand.nr_node = htree_nodes_size(m->ligands[i]);
        // see change::num_floats, torsion size in ligand will be nr_node -1(no torsion for rigid-body)
        sm->nrfligands += 6 + ligand.nr_node - 1;
        ALLOC_ARR(ligand.tree, Segment, ligand.nr_node);
        ALLOC_ARR(ligand.layers, int, ligand.nr_node * 2);
        ALLOC_ARR(ligand.layer_map, int, ligand.nr_node);
        ALLOC_ARR(ligand.atom_map, int, sm->natoms);
#if TREE_LAYER
        ligand.nr_layers = htree_nodes_prep(ligand.tree, m->ligands[i], ligand.layers);
#else
        ligand.nr_layers = htree_nodes_copy(ligand.tree, m->ligands[i]);
#endif
        if (ligand.nr_layers > sm->max_ligand_layers) {
            sm->max_ligand_layers = ligand.nr_layers;
        }

        for (auto j = 0, layersidx = 0, mapidx = 0; j < ligand.nr_layers; j++, layersidx+=2) {
            ligand.layers[layersidx] = mapidx;
            int nr = 0;
            for (auto k = 0; k < ligand.nr_node; k++) {
                if (ligand.tree[k].layer == j) {
                    ligand.layer_map[mapidx] = k;
                    printf("layermap %d = %d\n", mapidx, k);
                    nr++;
                    mapidx++;
                }
            }
            ligand.layers[layersidx+1] = nr;
            printf("layer %d layersidx %d mapidx %d nr %d\n", j, layersidx, ligand.layers[layersidx], nr);
        }

        for (auto j = 0u; j < sm->natoms; j ++) {
            ligand.atom_map[j] = -1;
        }
        for (auto j = 0; j < ligand.nr_node; j++) {
            auto &seg = ligand.tree[j];
            for (auto k = seg.begin; k < seg.end; k++) {
                ligand.atom_map[k] = j;
            }
        }
        // CHECKSM(i);
    }
    for (int i = 0; i < sm->nflex; i++) {
        auto &flex = sm->flex[i];
        sm->nrfflex += flex.nr_node;
        flex.nr_node = htree_nodes_size(m->flex[i]);
        ALLOC_ARR(flex.tree, Segment, flex.nr_node);
        ALLOC_ARR(flex.layers, int, flex.nr_node * 2);
        ALLOC_ARR(flex.layer_map, int, flex.nr_node);
        ALLOC_ARR(flex.atom_map, int, sm->natoms);
#if TREE_LAYER
        flex.nr_layers = htree_nodes_prep(flex.tree, m->flex[i], flex.layers);
#else
        flex.nr_layers = htree_nodes_copy(flex.tree, m->flex[i]);
#endif
        if (flex.nr_layers > sm->max_flex_layers) {
            sm->max_flex_layers = flex.nr_layers;
        }
        for (auto j = 0, layersidx = 0, mapidx = 0; j < flex.nr_layers; j++, layersidx+=2) {
            flex.layers[layersidx] = mapidx;
            int nr = 0;
            for (auto k = 0; k < flex.nr_node; k++) {
                if (flex.tree[k].layer == j) {
                    flex.layer_map[mapidx++] = k;
                    nr++;
                }
            }
            flex.layers[layersidx+1] = nr;
        }
        for (auto j = 0u; j < sm->natoms; j ++) {
            flex.atom_map[j] = -1;
        }
        for (auto j = 0; j < flex.nr_node; j++) {
            auto &seg = flex.tree[j];
            for (auto k = seg.begin; k < seg.end; k++) {
                flex.atom_map[k] = j;
            }
        }
    }
    sm->nrflts_change = sm->nrfligands + sm->nrfflex;
    // conf has qt orientation and change has vec orientation, so each ligand in conf
    // has an extra floats
    sm->nrflts_conf = sm->nrflts_change + sm->nligand;
    // for (auto i = 0; i < sm->movable_atoms + sm->npairs; i++) {
    //         printf("cpu map add %d: %d\n", i, sm->force_pair_map_add[i]);
    // }
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

extern int bfgs_max_trials();
std::shared_ptr<Memory> create_model_desc_memory(model *m, SrcModel *sm, int nmc) {
    Size sz = SIZEOF(ModelDesc) + SIZEOFARR(int, sm->nligand)+ SIZEOFARR(int, sm->nflex);
    int offset = 0;
    /*md->coords = offset,*/ offset += sizeof(Vec) * m->coords.size();

    for (int i = 0; i < sm->nligand; i++) {
        auto &ligand = sm->ligands[i];
        /*md->ligands[i] = offset,*/ offset += sizeof(SegmentVars) * ligand.nr_node;
    }
    for (int i = 0; i < sm->nflex; i++) {
        auto &flex = sm->flex[i];
        /*md->flex[i] = offset,*/ offset += sizeof(SegmentVars) * flex.nr_node;
    }

    /*md->minus_forces = offset,*/ offset += sizeof(Vec) * sm->movable_atoms;
    /*md->movable_e = offset,*/ offset += sizeof(Flt) * sm->movable_atoms;
    /*md->pair_res = offset,*/ offset += sm->npairs * sizeof(PairEvalResult);

    sz += Size(offset * bfgs_max_trials());
    sz *= nmc;
    sz = (sz + 4096 -1) / 4096*4096;
    return makeMemory(sz);
}
template <typename T>
inline int set_desc_offset(int& offset, int n) {
    assert(sizeof(T) % sizeof(Flt) == 0);
    int cur = offset;
    offset += sizeof(T) /sizeof(Flt) * n;
    return cur;
}
bool make_model_desc(Memory *mem, SrcModel *sm, model *m, ModelDesc *md) {
    md->src = sm;
    md->ncoords = m->coords.size();
    int offset = 0;

    md->coords = set_desc_offset<Vec>(offset, md->ncoords);
    // md->coords = offset, offset += sizeof(Vec) * md->ncoords;
    ALLOC_ARR(md->ligands, int, sm->nligand);
    ALLOC_ARR(md->flex, int, sm->nflex);

    for (int i = 0; i < sm->nligand; i++) {
        auto &ligand = sm->ligands[i];
        // md->ligands[i] = offset, offset += sizeof(SegmentVars) * ligand.nr_node;
        md->ligands[i] = set_desc_offset<SegmentVars>(offset, ligand.nr_node);
    }
    for (int i = 0; i < sm->nflex; i++) {
        auto &flex = sm->flex[i];
        // md->flex[i] = offset, offset += sizeof(SegmentVars) * flex.nr_node;
        md->flex[i] = set_desc_offset<SegmentVars>(offset, flex.nr_node);
    }

    // md->minus_forces = offset, offset += sizeof(Vec) * sm->movable_atoms;
    md->minus_forces = set_desc_offset<Vec>(offset, sm->movable_atoms);
    // md->movable_e = offset, offset += sizeof(Flt) * sm->movable_atoms;
    md->movable_e = set_desc_offset<Flt>(offset, sm->movable_atoms);
    // md->pair_res = offset, offset += sm->npairs * sizeof(PairEvalResult);
    md->pair_res = set_desc_offset<PairEvalResult>(offset, sm->npairs);

    md->szflt = offset;
    md->active = 0;
    ALLOC_ARR(md->data, Flt, md->szflt);

    for (int i = 0; i < sm->nligand; i++) {
        auto ligandvar = model_ligand(sm, md, md->data, i, 0);
#if TREE_LAYER
        htree_var_nodes_prep(ligandvar, m->ligands[i]);
#else
        htree_var_nodes_copy(ligandvar, m->ligands[i]);
#endif
    }
    for (int i = 0; i < sm->nflex; i++) {
        auto flexvar = model_flex(sm, md, md->data, i, 0);
#if TREE_LAYER
        htree_var_nodes_prep(flexvar, m->flex[i]);
#else
        htree_var_nodes_copy(flexvar, m->flex[i]);
#endif
    }

    copy_vecs(model_coords(sm, md, md->data), m->coords);


    return true;
cleanup:
    return false;
}
ModelDesc *make_model_desc(Memory *mem, SrcModel *sm, model *m, int nmc) {
    ModelDesc *md;

    ALLOC_ARR(md, ModelDesc, nmc);
    for (auto i = 0; i < nmc; i++) {
        if (!make_model_desc(mem, sm, m, md+i)) {
            goto cleanup;
        }
    }
    return md;
cleanup:
    return nullptr;
}
void restore_model_desc(ModelDesc *md, SrcModel *sm, model *m) {

    for (int i = 0; i < sm->nligand; i++) {
        auto ligandvar = model_ligand(sm, md, md->data, i, 0);
        htree_var_nodes_restore(ligandvar, m->ligands[i]);
    }
    for (int i = 0; i < sm->nflex; i++) {
        auto flexvar = model_flex(sm, md, md->data, i, 0);
        htree_var_nodes_restore(flexvar, m->flex[i]);
    }

    copy_vecs(m->coords, model_coords(sm, md, md->data));
}
std::shared_ptr<void> makeModelDesc(model *m, int nmc) {
    auto sm = extract_object<SrcModel>(m->m_gpu);
    auto mem = create_model_desc_memory(m, sm, nmc);
    if (mem) {
        auto gpusm = extract_cuda_object<SrcModel>(m->m_gpu);
        auto md = make_model_desc(mem.get(), sm, m, nmc);
        if (md) {
            for (auto i = 0; i < nmc; i++) {
                md[i].src = gpusm;
            }
            auto ret = makeCuObject(mem, md);
            for (auto i = 0; i < nmc; i++) {
                md[i].src = sm;
            }
            return ret;
        }
    }
    return nullptr;
}
bool makeModelDesc(std::shared_ptr<void> &obj, model *m, int nmc) {
    if (obj) {
        auto mem = extract_memory(obj);
        if (mem) {
            mem->reset();
            auto sm = extract_object<SrcModel>(m->m_gpu);
            auto gpusm = extract_cuda_object<SrcModel>(m->m_gpu);
            auto md = make_model_desc(mem.get(), sm, m, nmc);
            if (md) {
                for (auto i = 0; i < nmc; i++) {
                    md[i].src = gpusm;
                }
                updateCuObject(obj, md);
                for (auto i = 0; i < nmc; i++) {
                    md[i].src = sm;
                }
                return true;
            }
        }
    }  else {
        obj = makeModelDesc(m, nmc);
        return bool(obj);
    }
    return false;
}
// ok
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

#if TREE_LAYER
        htree_var_nodes_prep(ligandvar.tree, m->ligands[i]);
#else
        htree_var_nodes_copy(ligandvar.tree, m->ligands[i]);
#endif
    }
    for (int i = 0; i < sm->nflex; i++) {
        auto &flex = sm->flex[i];
        auto &flexvar = md->flex[i];
        ALLOC_ARR(flexvar.tree, SegmentVars, flex.nr_node);
#if TREE_LAYER
        htree_var_nodes_prep(flexvar.tree, m->flex[i]);
#else
        htree_var_nodes_copy(flexvar.tree, m->flex[i]);
#endif
    }
    ALLOC_ARR(md->minus_forces, Vec, sm->movable_atoms);
    ALLOC_ARR(md->movable_e, Flt, sm->movable_atoms);
    ALLOC_ARR(md->pair_res, PairEvalResult, sm->npairs);

    md->ncoords = m->coords.size();
    ALLOC_ARR(md->coords, Vec, md->ncoords);
    PRINT("Copy model vec");
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
bool makeModel(std::shared_ptr<void> &obj, model *m, const vec &v) {
    if (obj) {
        auto mem = extract_memory(obj);
        if (mem) {
            mem->reset();
            auto sm = extract_object<SrcModel>(m->m_gpu);
            auto ctx = make_model(mem.get(), sm, m, v);
            if (ctx) {
                updateCuObject(obj, ctx);
                return true;
            }
        }
    }  else {
        obj = makeModel(m, v);
        return bool(obj);
    }
    return false;
}
// ok
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
// good
BFGSCtx *make_bfgs(Memory *mem, const model &m, const change &g, const conf &c, const vec &v, int evalcnt) {
    BFGSCtx *ctx;
    ALLOC(ctx, BFGSCtx);
    if (!make_change(mem, &ctx->g, g)) goto cleanup;
    if (!make_conf(mem, &ctx->c, c)) goto cleanup;
    ALLOC_ARR(ctx->coords, Vec, m.coords.size());
    ctx->vs[0] = v.data[0], ctx->vs[1] = v.data[1], ctx->vs[2] = v.data[2];
    ctx->eval_cnt = evalcnt;
    return ctx;
cleanup:
    return nullptr;
}

// good
static bool makeNewBFGSCtx(std::shared_ptr<void> &obj, const model &m, const change &g, const conf &c, const vec &v, int evalcnt) {
    auto mem = create_bfgs_memory(g, c);
    if (mem) {
        auto ctx = make_bfgs(mem.get(), m, g, c, v, evalcnt);
        if (ctx) {
            obj = makeCuObject(mem, ctx);
            return true;
        }
    }
    return true;
}
bool makeBFGSCtx(std::shared_ptr<void> &obj, const model &m, const change &g, const conf &c, const vec& v, int evalcnt) {
    if (obj) {
        auto mem = extract_memory(obj);
        if (mem) {
            mem->reset();
            auto ctx = make_bfgs(mem.get(), m, g, c, v, evalcnt);
            if (ctx) {
                updateCuObject(obj, ctx);
                return true;
            }
        }
    }  else {
        return makeNewBFGSCtx(obj, m, g, c, v, evalcnt);
    }
    return false;
}

std::shared_ptr<Memory> create_mcctx_memory(SrcModel *sm, int nmc) {
    Size sz = SIZEOF(MCCtx) + SIZEOFARR(int, nmc) + SIZEOFARR(Flt, nmc*(sm->nrflts_conf+1));
    sz = (sz + 4096 -1) / 4096*4096;
    return makeMemory(sz);
}
// init: used to create random conf for i-th MC instance
MCCtx * make_mcctx(Memory *mem, SrcModel *sm, int nmc, MCCtx &src, std::function<void (int i, Flt *c)> &init) {
    MCCtx *ctx;
    const Flt max_fl = (std::numeric_limits<Flt>::max)();

    ALLOC(ctx, MCCtx);
    *ctx = src;
    ALLOC_ARR(ctx->curr_evalcnt, int, nmc);
    memset(ctx->curr_evalcnt, 0, sizeof(int) * nmc);
    ALLOC_ARR(ctx->best_e_and_c, Flt, (sm->nrflts_conf + 1) * nmc);
    for (int i = 0; i < nmc; i++) {
        auto ec = ctx->best_e_and_c + i * (sm->nrflts_conf+1);
        *ec = max_fl;
        init(i, ec+1);
    }
    return ctx;
cleanup:
    return nullptr;
}
std::shared_ptr<void> makeMCCtx(const model &m, MCCtx &src, int nmc, std::function<void (int i, Flt *c)> &init) {
    auto sm = extract_object<SrcModel>(m.m_gpu);
    auto mem = create_mcctx_memory(sm, nmc);
    if (mem) {
        auto ctx = make_mcctx(mem.get(), sm, nmc, src, init);
        if (ctx) {
            return makeCuObject(mem, ctx);
        }
    }
    return nullptr;
}
std::shared_ptr<Memory> create_mcin_memory(int nmc) {
    Size sz = SIZEOFARR(MCInputs, nmc);
    sz = (sz + 4096 -1) / 4096*4096;
    return makeMemory(sz);
}
MCInputs * make_inputs(Memory *mem, const conf &c, int nmc, int num_mutable_entities, int mc_steps, std::vector<std::shared_ptr<rng>> & generators) {
    MCInputs *ret;
    assert (mc_steps <= MC_MAX_STEP_BATCH);
    ALLOC_ARR(ret, MCInputs, nmc);
    for (auto idx = 0; idx < nmc; idx++) {
        auto &generator = *(generators[idx]);
        auto ins = ret + idx;
        for (int i = 0; i < mc_steps; i++) {
            ins->mc_steps   = mc_steps;
            auto &in     = ins->in[i];
            auto v       = random_inside_sphere(generator);
            in.rsphere.d[0] = v.data[0], in.rsphere.d[1] = v.data[1], in.rsphere.d[2] = v.data[2];
            in.rpi       = random_fl(-pi, pi, generator);
            in.groups[0] = -1;

            int which_int = random_int(0, num_mutable_entities, generator);
            sz which      = sz(which_int);
            VINA_FOR_IN(i, c.ligands) {
                if (which == 0) {
                    in.groups[0] = 0, in.groups[1] = i;
                    break;
                }
                --which;
                if (which == 0) {
                    in.groups[0] = 1, in.groups[1] = i;
                    break;
                }
                --which;
                if (which < c.ligands[i].torsions.size()) {
                    in.groups[0] = 2, in.groups[1] = i, in.groups[2] = which;
                    break;
                }
                which -= c.ligands[i].torsions.size();
            }
            VINA_FOR_IN(i, c.flex) {
                if (which < c.flex[i].torsions.size()) {
                    in.groups[0] = 3, in.groups[1] = i, in.groups[2] = which;
                    break;
                }
                which -= c.flex[i].torsions.size();
            }
        }
    }
    return ret;
cleanup:
    return nullptr;
}
std::shared_ptr<void> makeMCInputs(const model &m, int nmc,int num_mutable_entities, int mc_steps, rngs &generators, conf &c) {
    auto mem = create_mcin_memory(nmc);
    if (mem) {
        auto ctx = make_inputs(mem.get(), c, nmc, num_mutable_entities, mc_steps, generators);
        if (ctx) {
            return makeCuObject(mem, ctx);
        }
    }
    return nullptr;
}
bool makeMCInputs(std::shared_ptr<void> &obj, const model &m, int nmc,int num_mutable_entities, int mc_steps, rngs &generators, conf &c) {
    if (obj) {
        auto mem = extract_memory(obj);
        if (mem) {
            mem->reset();
            auto ins = make_inputs(mem.get(), c, nmc, num_mutable_entities, mc_steps, generators);
            if (ins) {
                updateCuObject(obj, ins);
                return true;
            }
        }
        return false;
    } else {
        obj = makeMCInputs(m, nmc, num_mutable_entities, mc_steps, generators, c);
        return bool(obj);
    }
}


std::shared_ptr<Memory> create_mcout_memory(ModelDesc *md, SrcModel *sm, int steps, int nmc) {
    Size sz = SIZEOFARR(MCOutputs, nmc);
    for (auto i = 0; i < nmc; i++) {
        Size szsteps = 0;
        szsteps += SIZEOFARR(Flt, sm->nrflts_conf+1);
        szsteps += SIZEOFARR(Vec, md->ncoords);
        sz += szsteps * steps;
    }
    sz = (sz + 4096 -1) / 4096*4096;
    return makeMemory(sz);
}
MCOutputs * make_outputs(Memory *mem, SrcModel *sm, ModelDesc *md, int steps, int nmc) {
    MCOutputs *ret;
    ALLOC_ARR(ret, MCOutputs, nmc);
    for (int idx = 0; idx< nmc; idx++) {
        auto outs = ret + idx;
        for(int i = 0; i < steps; i++) {
            auto &out = outs->out[i];
            ALLOC_ARR(out.e_and_c, Flt, sm->nrflts_conf+1);
            ALLOC_ARR(out.coords, Vec, md->ncoords);
        }

    }
    return ret;
cleanup:
    return nullptr;
}
std::shared_ptr<void> makeMCOutputs(const model &m, SrcModel *sm, ModelDesc *md, int nmc, int steps) {
    auto mem = create_mcout_memory(md, sm, steps, nmc);
    if (mem) {
        auto ctx = make_outputs(mem.get(), sm, md, steps, nmc);
        if (ctx) {
            return makeCuObject(mem, ctx);
        }
    }
    return nullptr;
}

bool makeMC(std::shared_ptr<void> &pmd, std::shared_ptr<void> &pctx, std::shared_ptr<void> &pin,
            std::shared_ptr<void> &pout, model &m, int num_mutable_entities, int steps, int nmc,
            rngs &generators, conf &c, sz over, fl average_required_improvement, int local_steps,
            int max_evalcnt, const vec &v1, const vec &v2, fl amplitude, fl temperature, 
            std::function<void (int i, fl *c)> init) {

    ModelDesc * md;
    auto sm = extract_object<SrcModel>(m.m_gpu);
    if (!pmd) {
        // create once only
        if(!makeModelDesc(pmd, &m, nmc))  {
            goto cleanup;
        }
    }

    md = extract_object<ModelDesc>(pmd);
    if (!pctx) {
        MCCtx src;
        src.over = over;
        src.average_required_improvement = average_required_improvement;
        src.num_mutable_entities = num_mutable_entities;
        src.max_evalcnt = max_evalcnt;
        src.local_steps = local_steps;
        src.vs[0] = v1.data[0], src.vs[1] = v1.data[1], src.vs[2] = v1.data[2];
        src.vs[3] = v2.data[0], src.vs[4] = v2.data[1], src.vs[5] = v2.data[2];
        src.amplitude = amplitude;
        src.rtemperature = 1.0/temperature;
        pctx = makeMCCtx(m, src, nmc, init);
        if (!pctx) {
            goto cleanup;
        }
    }

    if (!pout) {
        pout = makeMCOutputs(m, sm, md, nmc, steps);
        if (!pout) {
            goto cleanup;
        }
    }
    if(!makeMCInputs(pin, m, nmc, num_mutable_entities, steps, generators, c)) {
        goto cleanup;
    }

    return true;
cleanup:
    return false;
}
void run_mc(SrcModel *srcm, ModelDesc *m, PrecalculateByAtom *pa, Cache *ch, MCCtx *ctx,
            MCInputs *ins, MCOutputs *outs, int nmc, int steps, int local_steps, cudaStream_t stream);

std::vector<std::vector<output_type>> run_cuda_mc(std::shared_ptr<void> &pmd, std::shared_ptr<void> &pctx,
                 std::shared_ptr<void> &pin, std::shared_ptr<void> &pout, model *m,
                 const precalculate_byatom &p, const igrid &ig, int nmc, int steps, int local_steps, conf_size &s) {
    submit_vina_server([&](cudaStream_t stream) {
        auto &che = (const cache &)ig;
        // const
        auto ch = extract_cuda_object<Cache>(che.m_gpu);
        auto pa = extract_cuda_object<PrecalculateByAtom>(p.m_gpu);
        auto md = extract_cuda_object<ModelDesc>(pmd);
        auto cpum = extract_object<ModelDesc>(pmd);
        auto ins = extract_cuda_object<MCInputs>(pin);
        auto outs = extract_cuda_object<MCOutputs>(pout);
        auto ctx = extract_cuda_object<MCCtx>(pctx);

        run_mc(cpum->src, md, pa, ch, ctx, ins, outs, nmc, steps, local_steps, stream);

        // copy output from cuda to host
        auto cpumem = extract_memory(pout);
        auto cudamem = extract_cuda_memory(pout);
        cudaMemcpyAsync(cpumem->ptr(), cudamem->ptr(), cpumem->size(), cudaMemcpyDeviceToHost, stream);
        auto err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "vina mc eval fail, err: " << cudaGetErrorString(err) << std::endl;
        }
    }) ;

    auto cpumem = extract_memory(pout);
    cpumem->restore();

    auto outs = extract_object<MCOutputs>(pout);
    std::vector<std::vector<output_type>> ret;
    ret.resize(nmc);
    for (int idx = 0; idx < nmc; idx++) {
        auto &v = ret[idx];
        auto &mouts = outs[idx];
        for (int i = 0; i < mouts.n; i++) {
            auto &out = outs->out[i];
            conf c(s, out.e_and_c+1);
            output_type ot(c, *out.e_and_c);
            VINA_FOR(j, m->num_movable_atoms()) {
                auto &pc = out.coords[j];
                if (m->atoms[j].el != EL_TYPE_H) {
                    ot.coords.emplace_back(pc.d[0], pc.d[1], pc.d[2]);
                }
            }
            v.push_back(ot);
        }
    }

    // todo
    // auto cpumd = extract_object<ModelDesc>(mobj);
    // restore_model_desc(cpumd, cpumd->src, m);

    // todo: update model
    return ret;
}
#if 0
bool makeBFGSCtx(std::shared_ptr<void> &obj, const change &g, const conf &c) {
#if 1
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
#endif
    return true;
}
#endif
#if 1
    // bad
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
    DBG("%d  make precalc by atom", 0);
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
// bad
void output_ligand_change(ligand_change &dst, LigandChange &src) {
    vec_set(dst.rigid.position, src.rigid.position);
    vec_set(dst.rigid.orientation, src.rigid.orientation);
    DBG("torsion size %lu", dst.torsions.size());
    DBG("src torsion %f %f %f", src.torsions[0], src.torsions[1], src.torsions[2]);
    memcpy(dst.torsions.data(), src.torsions, dst.torsions.size() * sizeof(dst.torsions[0]));
    DBG("dst torsion %f %f %f", dst.torsions[0], dst.torsions[1], dst.torsions[2]);
}
void output_flex_change(residue_change &dst, ResidueChange &src) {
    memcpy(dst.torsions.data(), src.torsions, dst.torsions.size() * sizeof(dst.torsions[0]));
}
void output_ligand_conf(ligand_conf &dst, LigandConf &src) {
    vec_set(dst.rigid.position, src.rigid.position);
    qt_set(dst.rigid.orientation, src.rigid.orientation);
    DBG("conf torsion size %lu", dst.torsions.size());
    DBG("conf src torsion %f %f %f", src.torsions[0], src.torsions[1], src.torsions[2]);
    memcpy(dst.torsions.data(), src.torsions, dst.torsions.size() * sizeof(dst.torsions[0]));
    DBG("conf dst torsion %f %f %f", dst.torsions[0], dst.torsions[1], dst.torsions[2]);
}
void output_flex_conf(residue_conf &dst, ResidueConf &src) {
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
        output_ligand_change(g.ligands[i], chg->ligands[i]);
    }
    for (auto i = 0u; i < g.flex.size(); i ++) {
        output_flex_change(g.flex[i], chg->flex[i]);
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

extern void run_bfgs(ModelDesc *cpum, ModelDesc *m, PrecalculateByAtom *pa, Cache *ch, BFGSCtx *ctx,
                     int max_steps, Flt average_required_improvement, Size over,
                     cudaStream_t stream);
void dump_bfgs(ModelDesc *m, BFGSCtx *ctx, cudaStream_t stream);
#define QNDEBUG 0
fl run_cuda_bfgs(model *m, const precalculate_byatom &p, const igrid &ig, change &g, conf &c, vecv &coords,
                 const unsigned max_steps, const fl average_required_improvement, const sz over,
                 int &evalcount, std::shared_ptr<void> mobj, std::shared_ptr<void> ctxobj) {
    submit_vina_server([&](cudaStream_t stream) {
        auto &che = (const cache &)ig;
        // const
        auto ch = extract_cuda_object<Cache>(che.m_gpu);
        auto pa = extract_cuda_object<PrecalculateByAtom>(p.m_gpu);
        auto md = extract_cuda_object<ModelDesc>(mobj);
        auto cpum = extract_object<ModelDesc>(mobj);
        auto ctx = extract_cuda_object<BFGSCtx>(ctxobj);
#if QNDEBUG
        PRINT("RUN BFGS CUDA");
        printf(">>>> before bfgs\n");
        dump_bfgs(md, ctx, stream);
        cudaStreamSynchronize(stream);
#endif
        run_bfgs(cpum, md, pa, ch, ctx, max_steps, average_required_improvement, over, stream);
#if QNDEBUG
        cudaStreamSynchronize(stream);
        printf(">>>> after bfgs\n");
        dump_bfgs(md, ctx, stream);
        cudaStreamSynchronize(stream);
#endif

        // copy output from cuda to host
        auto cpumem = extract_memory(ctxobj);
        auto cudamem = extract_cuda_memory(ctxobj);
        cudaMemcpyAsync(cpumem->ptr(), cudamem->ptr(), cpumem->size(), cudaMemcpyDeviceToHost, stream);
        cpumem = extract_memory(mobj);
        cudamem = extract_cuda_memory(mobj);
        cudaMemcpyAsync(cpumem->ptr(), cudamem->ptr(), cpumem->size(), cudaMemcpyDeviceToHost, stream);
        auto err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "vina eval fail, err: " << cudaGetErrorString(err) << std::endl;
        }
    }) ;

    auto cpumem = extract_memory(ctxobj);
    auto cpumdmem = extract_memory(mobj);
    cpumem->restore();
    cpumdmem->restore();

    auto ctx = extract_object<BFGSCtx>(ctxobj);
    auto cf = &ctx->c;
    // output conf
    for (auto i = 0u; i < c.ligands.size(); i ++) {
        output_ligand_conf(c.ligands[i], cf->ligands[i]);
    }
    for (auto i = 0u; i < c.flex.size(); i ++) {
        output_flex_conf(c.flex[i], cf->flex[i]);
    }
    for (auto i = 0u; i < m->coords.size();i++) {
        auto &src = ctx->coords[i];
        coords.emplace_back(src.d[0], src.d[1], src.d[2]);
    }
    evalcount = ctx->eval_cnt;

    // todo
    // auto cpumd = extract_object<ModelDesc>(mobj);
    // restore_model_desc(cpumd, cpumd->src, m);

    // todo: update model
    return ctx->e;
}

# define MDBG(fmt, ...) printf("%s:%d " fmt "\n", __FILE__, __LINE__, __VA_ARGS__)
# define MPT(fmt)       printf("%s:%d " fmt "\n", __FILE__, __LINE__)
static fl eps = 0.01;
bool comp_flt(fl c1, fl c2) {
    auto diff = c1 - c2;
    if (diff < 0) diff = -diff;
    return (diff < eps);
}
void comp_change(flv &c1, flv &c2) {
    assert(c1.size() == c2.size());
    for (auto i = 0u; i < c1.size(); i++)
        if(!comp_flt(c1[i], c2[i])) {
            MDBG("flv comp fail @ %u, %f != %f\n", i, c1[i], c2[i]);
            fflush(stdout);
            assert(false);
        }
}
void comp_change(vec &c1, vec &c2) {
    for (int i = 0; i < 3; i++)
        if(!comp_flt(c1.data[i], c2.data[i])){
            MDBG("vec comp fail @ %d, %f != %f\n", i, c1.data[i], c2.data[i]);
            fflush(stdout);
            assert(false);
        }
}
void comp_change(ligand_change &c1, ligand_change &c2) {
    MPT("compare rigid orientation");
    comp_change(c1.rigid.orientation, c2.rigid.orientation);
    MPT("compare rigid position");
    comp_change(c1.rigid.position, c2.rigid.position);
    MPT("compare rigid torsion");
    comp_change(c1.torsions, c2.torsions);
}
void comp_change(residue_change &c1, residue_change &c2) {
    MPT("compare flex torsion");
    comp_change(c1.torsions, c2.torsions);
}
void comp_change(change &c1, change &c2) {
    assert(c1.ligands.size() == c2.ligands.size());
    assert(c1.flex.size() == c2.flex.size());
    for(auto i = 0u; i < c1.ligands.size(); i++) {
        MDBG("compare ligand %u", i);
        auto &src = c1.ligands[i];
        auto &dst = c2.ligands[i];
        comp_change(src, dst);
    }
    for(auto i = 0u; i < c1.flex.size(); i++) {
        MDBG("compare flex %u", i);
        auto &src = c1.flex[i];
        auto &dst = c2.flex[i];
        comp_change(src, dst);
    }

    MDBG("Compare %lu ligands %lu flex Done, congratulations!", c1.ligands.size(), c1.flex.size());
}
#endif
class CuVinaSrv {
public:
    CuVinaSrv() = default;
    virtual ~CuVinaSrv() = default;
#if 1
    bool run(StreamCallback &callback) {
        int id = 0;
        std::uint64_t cnt = 9999999999;
        {
            std::unique_lock<std::mutex> lock(mt_);
            for (auto it = running_.begin(); it != running_.end(); ++it) {
                if (it->second < cnt) {
                    cnt = it->second;
                    id = it->first;
                }
            }
        }
        if (id == 0) {
            return false;
        }
         auto ctx = insts_[id];
         auto req = std::make_shared<StreamRequest>(callback);
         {
            std::unique_lock<std::mutex> lock(mt_);
            running_[id]++;
         }
         ctx->commit(req);
         {
            std::unique_lock<std::mutex> lock(mt_);
            running_[id]--;
         }
         return true;

    }
    bool init(int device, int nrinst) {
        if (!insts_.empty()) {
            std::cerr << "vina server already created" << std::endl;
            return false;
        }
        device_ = device;
        nrinst_ = nrinst;
        for(auto i = 0; i < nrinst_; i++) {
            auto ctx = std::make_shared<CudaContext>(device_);
            if (ctx->create()) {
                seq_++;
                insts_[seq_] = ctx;
                running_[seq_] = 0;
            } else {
                return false;
            }
        }
        return true;
    }
#endif
private:
    std::map<int, std::shared_ptr<CudaContext>> insts_;
    std::map<int, std::uint64_t> running_;
    std::uint64_t seq_ = 0;
    std::mutex mt_;
    int device_;
    int nrinst_;
};

static CuVinaSrv instance_;
bool create_vina_server(int device, int nrinst) {
    return instance_.init(device, nrinst);
}
void destroy_vina_server() {
}
bool submit_vina_server(StreamCallback callback) {
    return instance_.run(callback);
}
};  // namespace dock