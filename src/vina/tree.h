/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>, 
           The Olson Lab, 
           The Scripps Research Institute

*/

#ifndef VINA_TREE_H
#define VINA_TREE_H

#include "conf.h"
#include "atom.h"
#include "log.h"

struct frame {
	frame(const vec& origin_) : origin(origin_), orientation_q(qt_identity), orientation_m(quaternion_to_r3(qt_identity)) {}
	vec local_to_lab(const vec& local_coords) const {
		vec tmp;
		VDUMP("    origin", origin);
#if VINADEBUG
		auto m = orientation_m.data;
#endif
		DBG("    orm: %f %f %f %f %f %f %f %f %f", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
		tmp = origin + orientation_m*local_coords; 
		return tmp;
	}
	vec local_to_lab_direction(const vec& local_direction) const {
		vec tmp;
		tmp = orientation_m * local_direction;
		return tmp;
	}
	const qt& orientation() const { return orientation_q; }
	const vec& get_origin() const { return origin; }
public:
	vec origin;
	void set_orientation(const qt& q) { // does not normalize the orientation
#if VINADEBUG
			fl q1 = q.R_component_1();
			fl q2 = q.R_component_2();
			fl q3 = q.R_component_3();
			fl q4 = q.R_component_4();
			DBG("set orientation: %f %f %f %f", q1, q2, q3, q4);
#endif
		orientation_q = q;
		orientation_m = quaternion_to_r3(orientation_q);
	}
	qt  orientation_q;
	mat orientation_m;
};

struct atom_range {
    sz begin;
    sz end;
	atom_range(sz begin_, sz end_) : begin(begin_), end(end_) {}
	template<typename F>
	void transform(const F& f) {
		sz diff = end - begin;
		begin = f(begin);
		end   = begin + diff;
	}
};

struct atom_frame : public frame, public atom_range {
	atom_frame(const vec& origin_, sz begin_, sz end_) : frame(origin_), atom_range(begin_, end_) {}
	void set_coords(const atomv& atoms, vecv& coords) const {
		VINA_RANGE(i, begin, end) {
			DBG("set coords %lu", i);
			VDUMP("    local coords(atoms)", atoms[i].coords);
			VDUMP("    before coords", coords[i]);
			coords[i] = local_to_lab(atoms[i].coords);
			VDUMP("    coords", coords[i]);
		}
	}
	vecp sum_force_and_torque(const vecv& coords, const vecv& forces) const {
		vecp tmp;
		tmp.first.assign(0);
		tmp.second.assign(0);
		DBG("frame begin %lu end %lu", begin, end);
		VINA_RANGE(i, begin, end) {
			tmp.first  += forces[i]; 
			auto product = cross_product(coords[i] - origin, forces[i]);
			tmp.second += product;
			DBG("    frame %lu", i);
			VDUMP("    force", forces[i]);
			VDUMP("    coords", coords[i]);
			VDUMP("    origin", origin);
			VDUMP("    product", product);
			VDUMP("    second", tmp.second);
		}
		VECPDUMP("sumft", tmp);
		return tmp;
	}
};

struct rigid_body : public atom_frame {
	rigid_body(const vec& origin_, sz begin_, sz end_) : atom_frame(origin_, begin_, end_) {}
	void set_conf(const atomv& atoms, vecv& coords, const rigid_conf& c) {
		origin = c.position;
		set_orientation(c.orientation);
		set_coords(atoms, coords);
	}
	void count_torsions(sz& s) const {} // do nothing
	void set_derivative(const vecp& force_torque, rigid_change& c) const {
		c.position     = force_torque.first;
		c.orientation  = force_torque.second;
	}
	vec axis; // add this just for better code in cuvina
	void print() const {
		printf("\trigid body begin %lu end %lu\n", begin, end);
		printf("\t\taxis: %f %f %f\n", axis.data[0], axis.data[1], axis.data[2]);
		printf("\t\torigin: %f %f %f\n", origin.data[0], origin.data[1], origin.data[2]);
        printf("\t\torq: %f %f %f %f\n", orientation_q.R_component_1(),
               orientation_q.R_component_2(), orientation_q.R_component_3(),
               orientation_q.R_component_4());
        ;
        printf("\t\torm: %f %f %f %f %f %f %f %f %f\n", orientation_m.data[0],
               orientation_m.data[1], orientation_m.data[2], orientation_m.data[3],
               orientation_m.data[4], orientation_m.data[5], orientation_m.data[6],
               orientation_m.data[7], orientation_m.data[8]);
    }
};

struct axis_frame : public atom_frame {
	axis_frame(const vec& origin_, sz begin_, sz end_, const vec& axis_root) : atom_frame(origin_, begin_, end_) {
		vec diff; diff = origin - axis_root;
		fl nrm = diff.norm();
		VINA_CHECK(nrm >= epsilon_fl);
		axis = (1/nrm) * diff;
	}
	void set_derivative(const vecp& force_torque, fl& c) const {
		VDUMP("    set der torque", force_torque.second);
		VDUMP("    set der axis", axis);
		VDUMP("    set der origin", origin);
		c = force_torque.second * axis;
	}
public:
	vec axis;
};

struct segment : public axis_frame {
	segment(const vec& origin_, sz begin_, sz end_, const vec& axis_root, const frame& parent) : axis_frame(origin_, begin_, end_, axis_root) {
		VINA_CHECK(eq(parent.orientation(), qt_identity)); // the only initial parent orientation this c'tor supports
		relative_axis = axis;
		relative_origin = origin - parent.get_origin();
	}
	void set_conf(const frame& parent, const atomv& atoms, vecv& coords, flv::const_iterator& c) {
		const fl torsion = *c;
		++c;
		VDUMP("    local coords", relative_origin);
		VDUMP("    local axis", relative_axis);
		origin = parent.local_to_lab(relative_origin);
		axis = parent.local_to_lab_direction(relative_axis);
		VDUMP("    my origin", origin);
		VDUMP("    axis", axis);
		DBG("torsion %f", torsion);
#if VINADEBUG
		auto &t = parent.orientation();
			fl q1 = t.R_component_1();
			fl q2 = t.R_component_2();
			fl q3 = t.R_component_3();
			fl q4 = t.R_component_4();
		DBG("parent orientation %f %f %f %f", q1, q2, q3, q4);
#endif
		qt tmp = angle_to_quaternion(axis, torsion) * parent.orientation();
#if VINADEBUG
			q1 = tmp.R_component_1();
			q2 = tmp.R_component_2();
			q3 = tmp.R_component_3();
			q4 = tmp.R_component_4();
		DBG("tmp %f %f %f %f", q1, q2, q3, q4);
#endif

		quaternion_normalize_approx(tmp); // normalization added in 1.1.2
#if VINADEBUG
			q1 = tmp.R_component_1();
			q2 = tmp.R_component_2();
			q3 = tmp.R_component_3();
			q4 = tmp.R_component_4();
		DBG("approx tmp %f %f %f %f", q1, q2, q3, q4);
#endif
		//quaternion_normalize(tmp); // normalization added in 1.1.2
		set_orientation(tmp);
		set_coords(atoms, coords);
	}
	void count_torsions(sz& s) const {
		++s;
	}
	void print() const {
		printf("\tsegment begin %lu end %lu\n", begin, end);
		printf("\t\taxis: %f %f %f\n", axis.data[0], axis.data[1], axis.data[2]);
		printf("\t\trelative axis: %f %f %f\n", relative_axis.data[0], relative_axis.data[1], relative_axis.data[2]);
		printf("\t\trelative origin: %f %f %f\n", relative_origin.data[0], relative_origin.data[1], relative_origin.data[2]);
		printf("\t\torigin: %f %f %f\n", origin.data[0], origin.data[1], origin.data[2]);
        printf("\t\torq: %f %f %f %f\n", orientation_q.R_component_1(),
               orientation_q.R_component_2(), orientation_q.R_component_3(),
               orientation_q.R_component_4());
        ;
        printf("\t\torm: %f %f %f %f %f %f %f %f %f\n", orientation_m.data[0],
               orientation_m.data[1], orientation_m.data[2], orientation_m.data[3],
               orientation_m.data[4], orientation_m.data[5], orientation_m.data[6],
               orientation_m.data[7], orientation_m.data[8]);
	}
public:
	vec relative_axis;
	vec relative_origin;
};

struct first_segment : public axis_frame {
	first_segment(const segment& s) : axis_frame(s) {}
	first_segment(const vec& origin_, sz begin_, sz end_, const vec& axis_root) : axis_frame(origin_, begin_, end_, axis_root) {}
	void set_conf(const atomv& atoms, vecv& coords, fl torsion) {
		set_orientation(angle_to_quaternion(axis, torsion));
		set_coords(atoms, coords);
	}
	void count_torsions(sz& s) const {
		++s;
	}
};

template<typename T> // T == branch
void branches_set_conf(std::vector<T>& b, const frame& parent, const atomv& atoms, vecv& coords, flv::const_iterator& c) {
	VINA_FOR_IN(i, b)
		b[i].set_conf(parent, atoms, coords, c);
}

template<typename T> // T == branch
void branches_derivative(const std::vector<T>& b, const vec& origin, const vecv& coords, const vecv& forces, vecp& out, flv::iterator& d) { // adds to out
	VECPDUMP("startft", out);
	VINA_FOR_IN(i, b) {
		vecp force_torque = b[i].derivative(coords, forces, d);
		VECPDUMP("childft", force_torque);
		out.first  += force_torque.first;
		auto o = b[i].node.get_origin();
		vec r; r = o - origin;
		// todo
		VDUMP("my origin", origin);
		VDUMP("child origin", o);
		out.second += cross_product(r, force_torque.first) + force_torque.second;
		VECPDUMP("childft added", out);
	}
}

template<typename T> // T == segment
struct tree {
	T node;
	std::vector< tree<T> > children;
	int nr_nodes; // record how many T in children
	int idx;
	int parentIdx;
	int layer;
	tree(const T& node_) : node(node_) {}
	void set_conf(const frame& parent, const atomv& atoms, vecv& coords, flv::const_iterator& c) {
		node.set_conf(parent, atoms, coords, c);
		branches_set_conf(children, node, atoms, coords, c);
	}
	vecp derivative(const vecv& coords, const vecv& forces, flv::iterator& p) const {
		vecp force_torque = node.sum_force_and_torque(coords, forces);
		fl& d = *p; // reference
		++p;
		branches_derivative(children, node.get_origin(), coords, forces, force_torque, p);
		node.set_derivative(force_torque, d);
		DBG("TreeDer idx %d c %f", idx, d);
		VECPDUMP("    tree ft", force_torque);
		// VDUMP("    forces", forces);
		return force_torque;
	}
	void print() const {
		for (auto &c : children) {
			c.print();
		}
		node.print();
	}
};

typedef tree<segment> branch;
typedef std::vector<branch> branches;

template<typename Node> // Node == first_segment || rigid_body
struct heterotree {
	Node node;
	branches children;
	int nr_nodes = 0; // record how many T in children
	heterotree(const Node& node_) : node(node_) {}
	void set_conf(const atomv& atoms, vecv& coords, const ligand_conf& c) {
		node.set_conf(atoms, coords, c.rigid);
		flv::const_iterator p = c.torsions.begin();
		branches_set_conf(children, node, atoms, coords, p);
		assert(p == c.torsions.end());
	}
	void set_conf(const atomv& atoms, vecv& coords, const residue_conf& c) {
		flv::const_iterator p = c.torsions.begin();
		node.set_conf(atoms, coords, *p);
		++p;
		branches_set_conf(children, node, atoms, coords, p);
		assert(p == c.torsions.end());
	}
	void derivative(const vecv& coords, const vecv& forces, ligand_change& c) const {
		vecp force_torque = node.sum_force_and_torque(coords, forces);
		flv::iterator p = c.torsions.begin();
		branches_derivative(children, node.get_origin(), coords, forces, force_torque, p);
		node.set_derivative(force_torque, c.rigid);
		assert(p == c.torsions.end());
	}
	void derivative(const vecv& coords, const vecv& forces, residue_change& c) const {
		vecp force_torque = node.sum_force_and_torque(coords, forces);
		flv::iterator p = c.torsions.begin();
		fl& d = *p; // reference
		++p;
		branches_derivative(children, node.get_origin(), coords, forces, force_torque, p);
		node.set_derivative(force_torque, d);
		assert(p == c.torsions.end());
	}
	void print() const {
		for (auto &b : children) {
			b.print();
		}
		node.print();
	}
};

template<typename T> // T = main_branch, branch, flexible_body
void count_torsions(const T& t, sz& s) {
	t.node.count_torsions(s);
	VINA_FOR_IN(i, t.children)
		count_torsions(t.children[i], s);
}

typedef heterotree<rigid_body> flexible_body;
typedef heterotree<first_segment> main_branch;

template<typename T> // T == flexible_body || main_branch
struct vector_mutable : public std::vector<T> {
	template<typename C>
	void set_conf(const atomv& atoms, vecv& coords, const std::vector<C>& c) { // C == ligand_conf || residue_conf
		VINA_FOR_IN(i, (*this))
			(*this)[i].set_conf(atoms, coords, c[i]);
	}
	szv count_torsions() const {
		szv tmp(this->size(), 0);
		VINA_FOR_IN(i, (*this))
			::count_torsions((*this)[i], tmp[i]);
		return tmp;
	}
	template<typename C>
	void derivative(const vecv& coords, const vecv& forces, std::vector<C>& c) const { // C == ligand_change || residue_change
		VINA_FOR_IN(i, (*this))
			(*this)[i].derivative(coords, forces, c[i]);
	}
};

template<typename T, typename F> // tree or heterotree - like structure
void transform_ranges(T& t, const F& f) {
	t.node.transform(f);
	VINA_FOR_IN(i, t.children)
		transform_ranges(t.children[i], f);
}

#endif
