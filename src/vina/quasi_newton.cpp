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

#include "quasi_newton.h"
#include "bfgs.h"
#include "cache.h"
#include "cuvina/cuvina.h"


#define VINA_CUTEST 1
struct quasi_newton_aux {
    model* m;
    const precalculate_byatom* p;
    const igrid* ig;
    const vec v;
    bool use_gpu;
    std::shared_ptr<void> m_gpu; // for model
    std::shared_ptr<void> m_bfgs_ctx; // for bfgs g&c

    quasi_newton_aux(model* m_, const precalculate_byatom* p_, const igrid* ig_, const vec& v_, const bool use_gpu_) : m(m_), p(p_), ig(ig_), v(v_), use_gpu(use_gpu_) {
        if (use_gpu) {
            m_gpu = dock::makeModel(m, v);
            if (!m_gpu) {
                use_gpu = false;
            }
        }
    }
    fl cpu(const conf& c, change& g) {
        m->set(c);
        return m->eval_deriv(*p, *ig, v, g);
    }
    fl gpu(const conf& c, change& g) {
        if (use_gpu) {
            use_gpu = dock::makeBFGSCtx(m_bfgs_ctx, g, c);
        }
        // Before evaluating conf, we have to update model
        const cache *subclass = dynamic_cast<const struct cache *>(ig);
        if (subclass && use_gpu) {
            return dock::run_model_eval_deriv(*p, *ig,  g, m_gpu, m_bfgs_ctx);
        }
        return cpu(c, g);
    }
 #if VINA_CUTEST  == 0
    fl operator()(const conf& c, change& g) {
        // Before evaluating conf, we have to update model
        return cpu(c, g);
    }
 #elif VINA_CUTEST  == 1
    fl operator()(const conf& c, change& g) {
        return gpu(c, g);
    }
#else
    fl operator()(const conf& c, change& g) {
        const cache *subclass = dynamic_cast<const struct cache *>(ig);
        fl ret = 0;
        change g1 = g;

        // Before evaluating conf, we have to update model
        if (use_gpu && subclass != nullptr) {
            use_gpu = dock::makeBFGSCtx(m_bfgs_ctx, g, c);
        }

        // std::cout << "VINA EVAL" << std::endl;
        m->set(c);
        const fl tmp = m->eval_deriv(*p, *ig, v, g);

        if (use_gpu) {
            // std::cout  << std::endl << std::endl << std::endl << std::endl << "OWR EVAL" << std::endl;
            ret = dock::run_model_eval_deriv(*p, *ig,  g1, m_gpu, m_bfgs_ctx);
            dock::comp_change(g1, g);
            auto diff = std::abs(ret - tmp);
            if (diff > 1e-6)  {
                printf("cu ret %f cpu ret %f diff %f\n", ret, tmp, diff);
                // assert (false);
            }
        }
        return tmp;
    }
#endif
};

void quasi_newton::operator()(model& m, const precalculate_byatom& p, const igrid& ig, output_type& out, change& g, const vec& v, int& evalcount) const { // g must have correct size
    quasi_newton_aux aux(&m, &p, &ig, v, use_gpu);

    fl res = bfgs(aux, out.c, g, max_steps, average_required_improvement, 10, evalcount);

    // Update model a last time after optimization
    m.set(out.c);
    out.e = res;
}
