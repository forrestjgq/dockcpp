#ifndef VINA_DEF_H
#define VINA_DEF_H

#include "cuda_runtime_api.h"
#include <stdint.h>

#define USE_CUDA_VINA 1
namespace dock {
    
    struct Qt {
        double x, y, z, w;
    };

    typedef double Flt;
    typedef double3 Vec;
    // typedef double4 Qt; // cuda require 16 alignment, do not use it
    typedef uint64_t Size;

    typedef struct {
        Vec first;
        Vec second;
    } Vecp;

    template <typename T>
    struct Arr3d {
        T *data;
        int dim[3];
    };


    typedef struct {
        Vec m_init;
        Vec m_factor;
        Vec m_factor_inv;
        Vec m_dim_fl_minus_1;
        Arr3d<Flt> m_data;
    } Grid ;

    typedef struct  {
        Size type_pair_index;
        Size a;
        Size b;
        // value from monte_carlo::operator()::authentic_v
        // for pairs from ligands, use v[0]
        // for pairs from model(inter, other, glue), use v[2]
        // see model::eval_deriv
        // here we use an index for incoming vs in eval
        int v; 
        Flt cutoff_sqr;  // for glue pairs, use max cutoff sqr, otherwise use cutoff sqr
    }InteractingPair;
    typedef struct {
        Flt *smooth; // vector of (e, dor)
        Flt *fast;
        Flt factor;
    } PrecalculateElement;
    typedef struct {
        Flt cutoff_sqr;
        Flt max_cutoff_sqr;
        PrecalculateElement *data;
        Size pe_dim; // dimension of data
        Size pe_sz;
    } PrecalculateByAtom;

    // constant part of ligands and flex
    typedef struct {
        int begin;
        int end;
        int parent; // -1 for root
        // for segment, not for rigid_body and first_segment
        Vec relative_axis;
        Vec relative_origin;
    } Segment;
    typedef struct {
        // inputs, changes in set_conf
        Vec axis; // for first_segment, it's const, for others updated in set_conf
        Vec origin;
        Qt orq; // see frame::orientation_q
        Flt orm[9]; // see frame::orientation_m

        // outputs
        Vecp ft; // force and torque

        // temp
        int dirty;
    } SegmentVars;

#define CU_XS_TYPE_SIZE 32
#define CU_INVALID_XS_SIZE 9999
    typedef struct {
        int nr_node; // how many nodes in tree
        Segment *tree;
    } Ligand;

    typedef struct {
        SegmentVars *tree;
    } LigandVars;

    typedef struct {
        int nr_node;
        Segment *tree;
    } Residue;
    typedef struct {
        SegmentVars *tree;
    } ResidueVars;

    typedef struct {
        Vec position;
        Vec orientation;
    } RigidChange;
    typedef struct {
        RigidChange rigid;
        // note that torsions has reversed order from segments
        Flt *torsions;
    } LigandChange;
    typedef struct {
        Flt * torsions;
    } ResidueChange;

    typedef struct {
        Vec position;
        Qt orientation;
    } RigidConf;
    typedef struct {
        RigidConf rigid;
        // note that torsions has reversed order from segments
        Flt *torsions;
    } LigandConf;
    typedef struct {
        Flt * torsions;
    } ResidueConf;

    typedef  struct {
        Vec coords;
    }Atom;

    // never changed model vars
    typedef struct {
        int movable_atoms;
        Size xs_nat; // num_atom_types(atom_type::XS)

        Size natoms;
        Atom *atoms;
        // array of movable atom size, value: atom_type::get(atom_type::XS)
        // if xs_sizes[i] == INVALID_XS_SIZEZ, minus_forces is 0, and cache eval_deriv should not be calc
        // see cache::eval_deriv
        Size *xs_sizes; 

        // int ninter, nother, nglue, nligand, nflex;
        // int nligandPairs; // how many pairs in all ligands
        // InteractingPair *inter_pairs;
        // InteractingPair *other_pairs;
        // InteractingPair *glue_pairs;

        int npairs; // include: ligands pairs, inter, other, glue pairs
        InteractingPair *pairs;

        int nligand, nflex;
        Ligand *ligands;
        Residue *flex;
    } SrcModel;

    typedef struct  {
        // to calc force, find corresponding pair, get a and b, 
        // forces[a] -= force, forces[b] += force
        Vec force;
        Flt e; // the total e will be accumulated e in all pair eval results
    } PairEvalResult;
    typedef struct {
        SrcModel *src;

        Flt vs[3];

        // changes at conf updating, and as input of der eval
        int ncoords;
        Vec * coords; // array of atom size, for atom coords, see model::coords

        // for der eval
        LigandVars *ligands;
        ResidueVars *flex;
        Vec *minus_forces; // size: movable atoms, see SrcModel.movable_atoms, used in eval_deriv, no init value
        Flt *movable_e; // result of movable atoms, size: src->movable_atoms, used in eval_deriv, no init value
        PairEvalResult *pair_res; // size: src->nparis, used in eval deriv, no init value
    } Model;
    typedef struct {
        Flt m_slope;
        int ngrids;
        Grid *grids; // cache.m_grids
    } Cache;
    typedef struct {
        LigandChange *ligands;
        ResidueChange *flex;
    } Change;
    typedef struct {
        LigandConf *ligands;
        ResidueConf *flex;
    } Conf;
    typedef struct  {
        Change g;
        Conf c;
        Flt e; // output loss
    } BFGSCtx;

    
const Flt PI = Flt(3.1415926535897931);



};


#endif