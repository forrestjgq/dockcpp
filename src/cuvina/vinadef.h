#ifndef VINA_DEF_H
#define VINA_DEF_H

#include "cuda_runtime_api.h"
#include <stdint.h>

namespace dock {
    typedef double Flt;
    struct Qt {
        double x, y, z, w;
    };
    struct Vec {
        Flt d[3];
    };

    // typedef double3 Vec;
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
        int layer; // 0 for root, 1 for child of root, ...
        int *children; //children id
        // for segment, not for rigid_body and first_segment
        Vec relative_axis;
        Vec relative_origin;
    } Segment;

    // This struct must NOT define any pointers and should be all Flts
    // Alignment: sizeof(Flt)
    typedef struct {
        // inputs, changes in set_conf
        Vec axis; // for first_segment, it's const, for others updated in set_conf
        Vec origin;
        Qt orq; // see frame::orientation_q
        Flt orm[9]; // see frame::orientation_m
        Flt tmp[32]; // used in set conf and sum ft

        // outputs
        Vecp ft; // force and torque
    } SegmentVars;

#define CU_XS_TYPE_SIZE 32
#define CU_INVALID_XS_SIZE 9999
    typedef struct {
        int nr_node; // how many nodes in tree
        int nr_layers; // indicates how many items in layers
        // alllocation size: 2 * nr_node
        // pair of <size, idx> size: how many nodes in this layer,
        //                     idx: start index of each layer in tree 
        int *layers; 
        Segment *tree;
    } Ligand;

    typedef struct {
        SegmentVars *tree; // array size: Ligand.nr_node
    } LigandVars;

    typedef struct {
        int nr_node;
        int nr_layers; // indicates how many items in layers
        int *layers; // index of each layer in relations, size nr_layers
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
        Size el;
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
        // idx_sub/add contains movable_atoms indices, each for one minus-forces
        // it indicates that offset of a group of ints in force_pair_map_xxx shows the pair and
        // minus-force relationship, this group of ints are defined as:
        // [n][pair-1][pair-2]...[pair-n]
        // assume the offset value is k, this means for minus-forces[k], it should add/sub
        // the force in pair result of pair-1 ~ pair-n, where n indicates how many related
        // pairs to minus-force[k]
        int *idx_sub, *idx_add;
        int *force_pair_map_sub;
        int *force_pair_map_add;

        int nligand, nflex; // how many ligand and flex in model
        int nrflts_change, nrflts_conf; // how many Flt in a change and conf
        // in a change, how may floats for ligands and flex
        // note that nrfligands + nligand will be Flt number of conf for ligand, and nrfflex are same for conf and change
        int nrfligands, nrfflex; 
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

        Flt vs[3]; // const value used to eval der

        // changes at conf updating, and as input of der eval
        int ncoords;
        Vec * coords; // array of atom size, for atom coords, see model::coords

        // for der eval
        LigandVars *ligands; // array size: SrcModel.nligand
        ResidueVars *flex; // array size: SrcModel.nflex
        Vec *minus_forces; // size: movable atoms, see SrcModel.movable_atoms, used in eval_deriv, no init value
        Flt *movable_e; // result of movable atoms, size: src->movable_atoms, used in eval_deriv, no init value
        PairEvalResult *pair_res; // size: src->nparis, used in eval deriv, no init value
    } Model;
    typedef struct {
        SrcModel *src;
        int szflt; // how many flts ModelVars takes
        int ncoords;
        int active = 0; // data contains some model description data, this indicates the active one

        // offset of each fields in Flt unit
        int coords; // will be updated after model der
        // offset of each ligands, array size: SrcModel.nligand
        int *ligands;  // will be updated after model der
        int *flex; // will be updated after model der
        int minus_forces; // used only inside model der
        int movable_e; // used only inside model der
        int pair_res; // used only inside model der

        Flt *data;
    } ModelDesc;


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
        Flt vs[3];
        Change g;
        Conf c; // input c
        Flt e; // output e
        Vec *coords; // output model coords , size: SrcModel:: ncoords
        // Flt *pair_e; // output loss
        // Flt *atom_e; // output loss
        int eval_cnt; // how many model eval called in this bfgs

        // tmp for bfgs

        // see bfgs.h(flmat) , todo
        // int dimh; // dim of h, its value will be SrcModel::nrflts
        // Flt *h; // size dimh, init value 0
        // Change g_new;
        // Conf c_new;

        // Change g_orig;
        // Conf c_orig;

        // Change p;
        // Change y;

        // int max_step;
        // Flt *fs; // size: max_step + 1
    } BFGSCtx;

    #define MC_MAX_STEP_BATCH 200
    typedef struct {
        Vec rsphere; // same size as which, random_inside_sphere(generator); 
        Flt rpi; // same size as which, random_fl(-pi, pi, generator)
        Flt raccept; // random probalility, see metropolis_accept

        // for each group there are 3 ints
        // int -1: no mutate conf, 0: 0-ligand pos, 1-ligand orientation, 2-ligand torsion, 3-flex torsion
        // int 1: ligand or flex index
        // int 2: torsion index
        int groups[3]; 
    } MCStepInput;
    typedef struct {
        int mc_steps;// how many mc steps at most should be executed
        MCStepInput in[MC_MAX_STEP_BATCH];
    } MCInputs;

    typedef struct {
        Flt *e_and_c; // conf
        Vec *coords; // size: SrcModel.ncoords, for all coords, not just heavy ones
    } MCStepOutput;

    // outputs for one MC instance
    typedef struct {
        int n; // num of out, it might be smaller then steps, or even 0
        MCStepOutput out[MC_MAX_STEP_BATCH]; // alloc size must be MCInputs.steps
    } MCOutputs;

    // MC const values
    // shared by all MC instances
    typedef struct {
        // newton parameters
        Size over;
        Flt average_required_improvement;

        int local_steps;// bfgs max steps
        int num_mutable_entities;
        int max_evalcnt; // max allowed model der evaluations
        int *curr_evalcnt; // array, each for one MC, records how many model der eval has been exeucted

        Flt vs[6]; // v[0-2] for initial newton, v[3-5] for promising one
        Flt amplitude;
        Flt rtemperature; // 1.0/temperature

        // group of e + conf for each MC
        // these values are maintained by cuda and never be updated by host
        // e: 1 Flt, initial with max flt
        // c: nc Flt, initial with random values
        Flt *best_e_and_c;

    } MCCtx;
    
const Flt PI = Flt(3.1415926535897931);
const Flt R2PI = Flt(0.15915494309189535); // 1/(2*pi)



};


#endif