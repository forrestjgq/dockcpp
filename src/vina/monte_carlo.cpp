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

#include "monte_carlo.h"
#include "coords.h"
#include "mutate.h"
#include "quasi_newton.h"
#include "cuvina/cuvina.h"
namespace dock {
// void run_bfgs(Model *cpum, Model *m, PrecalculateByAtom *pa, Cache *ch, BFGSCtx *ctx, int max_steps, Flt average_required_improvement, Size over , cudaStream_t stream);
};

output_type monte_carlo::operator()(model& m, const precalculate_byatom& p, const igrid& ig, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const {
	output_container tmp;
	this->operator()(m, tmp, p, ig, corner1, corner2, increment_me, generator); // call the version that produces the whole container
	VINA_CHECK(!tmp.empty());
	return tmp.front();
}

bool metropolis_accept(fl old_f, fl new_f, fl temperature, rng& generator) {
	if(new_f < old_f) return true;
	const fl acceptance_probability = std::exp((old_f - new_f) / temperature);
	return random_fl(0, 1, generator) < acceptance_probability;
}

void monte_carlo::enable_gpu(bool enable) {
	use_gpu = enable;
}
// out is sorted
bool monte_carlo::gpu(model& m, output_container& out, const precalculate_byatom& p, const igrid& ig, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const {
	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl
	// if (use_gpu) {
	// 	m_gpu = dock::makeModel(m, authentic_v);
	// 	if (!m_gpu) {
	// 		use_gpu = false;
	// 	}
	// }
	// if (!use_gpu) {
	// 	return false;
	// }
    int evalcount = 0;
	conf_size s = m.get_size();
	change g(s);
	output_type tmp(s, 0);
	tmp.c.randomize(corner1, corner2, generator);
	fl best_e = max_fl;
	quasi_newton quasi_newton_par;
	quasi_newton_par.use_gpu = use_gpu;
    quasi_newton_par.max_steps = local_steps;
	int bfgscnt = 0;
	// printf("mc global steps %u max_evals %u\n", global_steps, max_evals);
	VINA_U_FOR(step, global_steps) {
		if(increment_me)
			++(*increment_me);
		if((max_evals > 0) & ((unsigned)evalcount > max_evals)) {
			// printf("evals %u times, step %u\n", evalcount, step);
			break;
		}
		output_type candidate = tmp;
		mutate_conf(candidate.c, m, mutation_amplitude, generator);
		bfgscnt++;
		// printf("mc calls bfgs %d\n", bfgscnt);
		quasi_newton_par(m, p, ig, candidate, g, hunt_cap, evalcount);
		if(step == 0 || metropolis_accept(tmp.e, candidate.e, temperature, generator)) {
			tmp = candidate;

			m.set(tmp.c); // FIXME? useless?

			// FIXME only for very promising ones
			if(tmp.e < best_e || out.size() < num_saved_mins) {
				bfgscnt++;
				// printf("    mc calls bfgs %d\n", bfgscnt);
				quasi_newton_par(m, p, ig, tmp, g, authentic_v, evalcount);
				m.set(tmp.c); // FIXME? useless?
				tmp.coords = m.get_heavy_atom_movable_coords();
				add_to_output_container(out, tmp, min_rmsd, num_saved_mins); // 20 - max size
				if(tmp.e < best_e)
					best_e = tmp.e;
			}
		}
	}
	VINA_CHECK(!out.empty());
	VINA_CHECK(out.front().e <= out.back().e); // make sure the sorting worked in the correct order
	return true;
}
void monte_carlo::cpu(model& m, output_container& out, const precalculate_byatom& p, const igrid& ig, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const {
    int evalcount = 0;
	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl
	conf_size s = m.get_size();
	change g(s);
	output_type tmp(s, 0);
	tmp.c.randomize(corner1, corner2, generator);
	fl best_e = max_fl;
	quasi_newton quasi_newton_par;
	quasi_newton_par.use_gpu = use_gpu;
    quasi_newton_par.max_steps = local_steps;
	int bfgscnt = 0;
	// printf("mc global steps %u max_evals %u\n", global_steps, max_evals);
	VINA_U_FOR(step, global_steps) {
		if(increment_me)
			++(*increment_me);
		if((max_evals > 0) & ((unsigned)evalcount > max_evals)) {
			// printf("evals %u times, step %u\n", evalcount, step);
			break;
		}
		output_type candidate = tmp;
		mutate_conf(candidate.c, m, mutation_amplitude, generator);
		bfgscnt++;
		// printf("mc calls bfgs %d\n", bfgscnt);
		quasi_newton_par(m, p, ig, candidate, g, hunt_cap, evalcount);
		if(step == 0 || metropolis_accept(tmp.e, candidate.e, temperature, generator)) {
			tmp = candidate;

			m.set(tmp.c); // FIXME? useless?

			// FIXME only for very promising ones
			if(tmp.e < best_e || out.size() < num_saved_mins) {
				bfgscnt++;
				// printf("    mc calls bfgs %d\n", bfgscnt);
				quasi_newton_par(m, p, ig, tmp, g, authentic_v, evalcount);
				m.set(tmp.c); // FIXME? useless?
				tmp.coords = m.get_heavy_atom_movable_coords();
				add_to_output_container(out, tmp, min_rmsd, num_saved_mins); // 20 - max size
				if(tmp.e < best_e)
					best_e = tmp.e;
			}
		}
	}
	VINA_CHECK(!out.empty());
	VINA_CHECK(out.front().e <= out.back().e); // make sure the sorting worked in the correct order
}
void monte_carlo::operator()(model& m, output_container& out, const precalculate_byatom& p, const igrid& ig, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const {
#if 0
	if (!gpu(m, out, p, ig, corner1, corner2, increment_me, generator)) {
        cpu(m, out, p, ig, corner1, corner2, increment_me, generator);
    }
#else
        cpu(m, out, p, ig, corner1, corner2, increment_me, generator);
#endif
}
