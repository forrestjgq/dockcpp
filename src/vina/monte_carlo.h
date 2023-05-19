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

#ifndef VINA_MONTE_CARLO_H
#define VINA_MONTE_CARLO_H

#include "incrementable.h"
#include "model.h"
#include "cuvina/cuvina.h"

struct monte_carlo {
    unsigned max_evals;
	unsigned global_steps;
	fl temperature;
	vec hunt_cap;
	fl min_rmsd;
	sz num_saved_mins;
	fl mutation_amplitude;
    bool use_gpu;
	unsigned local_steps;

    std::shared_ptr<void> m_gpu_model_desc;
    std::shared_ptr<void> m_gpu_mc_ctx;
    std::shared_ptr<void> m_gpu_mc_inputs;
    std::shared_ptr<void> m_gpu_mc_outputs;

    std::vector<output_container> containers;
    std::vector<std::shared_ptr<rng>> generators;
    int gpu_steps = 50;

    // T = 600K, R = 2cal/(K*mol) -> temperature = RT = 1.2;  global_steps = 50*lig_atoms = 2500
    monte_carlo() : max_evals(0), global_steps(2500), temperature(1.2), hunt_cap(10, 1.5, 10), min_rmsd(0.5), num_saved_mins(50), mutation_amplitude(2), use_gpu(false) {}

    void enable_gpu(bool enable);
    bool prepare_gpu(model& m, precalculate_byatom& p,const igrid& ig, const vec& corner1, const vec& corner2,
                     incrementable* increment_me, rng& generator, int nmc);
    output_type operator()(model& m, precalculate_byatom& p, const igrid& ig, const vec& corner1,
                           const vec& corner2, incrementable* increment_me, rng& generator) const;
	// out is sorted
	void operator()(model& m, output_container& out, precalculate_byatom& p, const igrid& ig,
                    const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const;
	void cpu(model& m, output_container& out, precalculate_byatom& p, const igrid& ig,
                    const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const;
	bool gpu(model& m, output_container& out, precalculate_byatom& p, const igrid& ig,
                    const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const;
};

#endif

