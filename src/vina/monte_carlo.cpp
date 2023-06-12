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
#include "utils.h"
namespace dock {
// void run_bfgs(Model *cpum, Model *m, PrecalculateByAtom *pa, Cache *ch, BFGSCtx *ctx, int max_steps, Flt average_required_improvement, Size over , cudaStream_t stream);
};

extern sz count_mutable_entities(const conf& c);
struct gpu_monte_carlo {
	const monte_carlo *host;
	bool enabled;
    std::shared_ptr<void> m_gpu_model_desc;
    std::shared_ptr<void> m_gpu_mc_ctx;
    std::shared_ptr<void> m_gpu_mc_inputs;
    std::shared_ptr<void> m_gpu_mc_outputs;

    std::vector<output_container> containers;
    std::vector<std::shared_ptr<rng>> generators;
    int gpu_steps = 200;
    conf_size cs;
    conf ctemplate;
    int nr_mc = 0;
    int nr_mutable_entities = 0;
    bool prepare_gpu(model& m, precalculate_byatom& p, const igrid& ig, const vec& corner1,
                     const vec& corner2, incrementable* increment_me, rng& generator, int nmc) {
        const cache* subclass = dynamic_cast<const struct cache*>(&ig);
        if (!subclass) {
            enabled = false;
            return false;
        }
        nr_mc = nmc;
        for (int i = 0; i < nmc; i++) {
            auto seed = random_int(0, 1000000, generator);
            generators.push_back(std::make_shared<rng>(static_cast<rng::result_type>(seed)));
        }

        cs                  = m.get_size();
        ctemplate           = conf(cs);
        nr_mutable_entities = count_mutable_entities(ctemplate);

        auto f = [&](int idx, fl* c) {
            output_type ot(cs, 0);
            ot.c.randomize(corner1, corner2, *generators[idx]);
			ot.c.dump_to(c);
        };

#define OVER                    10
#define AVERAGE_REQ_IMPROVEMENT 0
        vec authentic_v(1000, 1000, 1000);  // FIXME? this is here to avoid max_fl/max_fl
        enabled = dock::makeMC(m_gpu_model_desc, m_gpu_mc_ctx, m_gpu_mc_inputs, m_gpu_mc_outputs, m,
                               nr_mutable_entities, gpu_steps, nmc, generators, ctemplate, OVER,
                               AVERAGE_REQ_IMPROVEMENT, host->local_steps, host->max_evals, authentic_v,
                               host->hunt_cap, host->mutation_amplitude, 1.0 / host->temperature, f);

		if (enabled) {
			containers.resize(nr_mc);
		}
        return enabled;
    }
    bool operator()(model& m, output_container& out, precalculate_byatom& p, const igrid& ig,
                           const vec& corner1, const vec& corner2, incrementable* increment_me,
                           rng& generator, int nmc) {
		if (!prepare_gpu(m, p, ig, corner1, corner2, increment_me, generator, nmc)) {
			return false;
		}
        int step = 0;
		dock::Clock clk;
		clk.mark();
        while (step < int(host->global_steps)) {
            int batch = step == 0 ? host->num_saved_mins : gpu_steps;
			printf("run cuda mc batch %d bfgs steps %d\n", batch, host->local_steps);
            increment_me->increase(batch);
			step += batch;
            auto suc = dock::makeMCInputs(m_gpu_mc_inputs, m, nr_mc, nr_mutable_entities, batch,
                                          generators, ctemplate);
            if (!suc) {
                std::cerr << "make mc inputs failed, quit gpu processing" << std::endl;
                return false;
            }
            auto ret = dock::run_cuda_mc(m_gpu_model_desc, m_gpu_mc_ctx, m_gpu_mc_inputs,
                                         m_gpu_mc_outputs, &m, p, ig, nr_mc, batch,
                                         host->local_steps, cs);
			if (ret.empty()) {
				std::cerr << "cuda mc output is empty" << std::endl;
				return false;
			}
			if ((int)ret.size() != nr_mc) {
				std::cerr << "cuda mc output size " << ret.size() << " but expect " << nr_mc << std::endl;
				return false;
			}
			for (auto i = 0; i < nr_mc; i++) {
				auto &v = ret[i];
				for (auto &tmp : v) {
					add_to_output_container(containers[i], tmp, host->min_rmsd, host->num_saved_mins);
				}
			}
			auto us = clk.mark();
			printf("run step %d: %lu us\n", step, us);
        }
        for (auto i = 0; i < nr_mc; i++) {
            for (auto& tmp : containers) {
				if (!tmp.empty()) {
					add_to_output_container(out, tmp.front(), host->min_rmsd, host->num_saved_mins);
				}
            }
        }
        return !out.empty();
    }
};

output_type monte_carlo::operator()(model& m, precalculate_byatom& p, const igrid& ig,
                                    const vec& corner1, const vec& corner2,
                                    incrementable* increment_me, rng& generator, int gpu_nmc) const {
    output_container tmp;
	this->operator()(m, tmp, p, ig, corner1, corner2, increment_me, generator, gpu_nmc); // call the version that produces the whole container
	VINA_CHECK(!tmp.empty());
	return tmp.front();
}

bool metropolis_accept(fl old_f, fl new_f, fl temperature, rng& generator) {
	if(new_f < old_f) return true;
	const fl acceptance_probability = std::exp((old_f - new_f) / temperature);
	return random_fl(0, 1, generator) < acceptance_probability;
}

// out is sorted
void monte_carlo::cpu(model& m, output_container& out, precalculate_byatom& p,
                             const igrid& ig, const vec& corner1, const vec& corner2,
                             incrementable* increment_me, rng& generator) const {
    int evalcount = 0;
	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl
	conf_size s = m.get_size();
	change g(s);
	output_type tmp(s, 0);
	tmp.c.randomize(corner1, corner2, generator);

	// printf("random conf:\n");
	// tmp.c.print();

	fl best_e = max_fl;
	quasi_newton quasi_newton_par;
	quasi_newton_par.use_gpu = true;
    quasi_newton_par.max_steps = local_steps;
	int bfgscnt = 0;
	// printf("mc global steps %u max_evals %u\n", global_steps, max_evals);
	// sz last = 0;
	// dock::Clock clk;
	// clk.mark();
	VINA_U_FOR(step, global_steps) {
		// if (step - last == 200) {
		// 	last = step;
		// 	auto us = clk.mark();
		// 	printf("batch 200 cost %lu us\n", us);
		// }
		// if(step == 20) exit(0); // for test
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
		// printf("step %u e %f coord %f %f %f\n", step, candidate.e, m.coords[0].data[0], m.coords[0].data[1], m.coords[0].data[2]);
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
				printf("add step %u e %f coord %f %f %f\n", step, tmp.e, m.coords[0].data[0], m.coords[0].data[1], m.coords[0].data[2]);
				add_to_output_container(out, tmp, min_rmsd, num_saved_mins); // 20 - max size
				if(tmp.e < best_e)
					best_e = tmp.e;
			}
		}
	}
	VINA_CHECK(!out.empty());
	VINA_CHECK(out.front().e <= out.back().e); // make sure the sorting worked in the correct order

}
void monte_carlo::operator()(model& m, output_container& out, precalculate_byatom& p,
                             const igrid& ig, const vec& corner1, const vec& corner2,
                             incrementable* increment_me, rng& generator, int gpu_nmc) const {
	printf("mc num-min %lu steps %u local steps %u\n", num_saved_mins, global_steps, local_steps);
	bool run_cpu = true;
#if 0
	if (gpu_nmc > 0) {
		gpu_monte_carlo gmc;
		gmc.host = this;
		if (!gmc(m, out, p, ig, corner1, corner2, increment_me, generator, gpu_nmc)) {
			run_cpu = false;
		}
	}
#endif

	if (run_cpu) {
		this->cpu(m, out, p, ig, corner1, corner2, increment_me, generator); // call the version that produces the whole container
	}
}
