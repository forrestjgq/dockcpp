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
#include "utils.h"

#define QNDEBUG 0
#define CUBFGS 1
#define VINA_CUTEST 0 // 0: cpu, 1: gpu prefer 2: gpu & cpu & compare test
struct quasi_newton_aux {
    model* m;
    const precalculate_byatom* p;
    const igrid* ig;
    const vec v;
    bool use_gpu;
    std::shared_ptr<void> m_gpu; // for model
    std::shared_ptr<void> m_bfgs_ctx; // for bfgs g&c
    // int cnt = 0;

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
        auto f = m->eval_deriv(*p, *ig, v, g);
        return f;
    }
    void print() {
        m->print();
    }
    fl gpu(const conf& c, change& g) {
        // cnt++;
        // if (cnt > 1000) {
        //     exit(0);
        // }
        if (use_gpu) {
            // printf("gpu %d start\n", cnt);
            use_gpu = dock::makeBFGSCtx(m_bfgs_ctx, *m, g, c, v);
            // printf("gpu %d end\n", cnt);
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
        change g1 = g;
        printf("\n\n\nEVAL CPU\n");
        m->set(c);
        const fl tmp = m->eval_deriv(*p, *ig, v, g);

        printf("\n\n\nEVAL GPU\n");
        fl ret = gpu(c, g1);
        dock::comp_change(g1, g);
        auto diff = std::abs(ret - tmp);
        if (diff > 1e-6)  {
            printf("cu ret %f cpu ret %f diff %f\n", ret, tmp, diff);
            // assert (false);
        }
        return tmp;
    }
#endif
};

#ifdef BFGS_CPU
#if BFGS_CPU == 1
#define QNUSAGE 0 // 0-cpu, 1-gpu, 2-compare
#else
#define QNUSAGE 1 // 0-cpu, 1-gpu, 2-compare
#endif
#else
#define QNUSAGE 1 // 0-cpu, 1-gpu, 2-compare
#endif
void quasi_newton::operator()(model& m, const precalculate_byatom& p, const igrid& ig, output_type& out, change& g, const vec& v, int& evalcount) { // g must have correct size
#if QNUSAGE == 0
    cpu(m, p, ig, out, g, v, evalcount);
#elif QNUSAGE == 1
    gpu(m, p, ig, out, g, v, evalcount);
#else 
    int cnt = evalcount;
    model m1(m);
    change g1(g);
    output_type out1(out);

    dock::Clock clk;
    clk.mark();
    cpu(m, p, ig, out, g, v, evalcount);
    auto cdu = clk.mark();
    gpu(m1, p, ig, out1, g1, v, cnt);
    auto gdu = clk.mark();
    auto diff = int(cdu > gdu ? cdu - gdu : gdu - cdu);
    if (cdu < gdu) {
        diff = -diff;
    }
    printf("cpu %lu cnt %d gpu %lu cnt %d diff %d us\n\n", cdu, evalcount, gdu, cnt, diff);
    if (0) {
        printf("last eval cnt %d\n", evalcount);
        out.c.print();
    }
    // exit (0);
#endif
}

void quasi_newton::gpu(model& m, const precalculate_byatom& p, const igrid& ig, output_type& out, change& g, const vec& v, int &evalcount) { // g must have correct size
    fl res = 0;
    if (!use_gpu) {
        std::cerr << "newton init not use gpu" << std::endl;
    }
    if (use_gpu) {
        use_gpu = dock::makeModelDesc(m_gpu, &m);
        if (!use_gpu) {
            std::cerr << "make model not use gpu" << std::endl;
        }
        if (use_gpu) {
            use_gpu = dock::makeBFGSCtx(m_bfgs_ctx, m, g, out.c, v, evalcount);
            if (!use_gpu) {
                std::cerr << "make bfgs ctx not use gpu" << std::endl;
            }
        }

        if (use_gpu) {
            // std::cerr << "use gpu" << std::endl;
            // dock::Clock clk;
            // clk.mark();
            res = dock::run_cuda_bfgs(&m, p, ig, g, out.c, out.coords, max_steps, average_required_improvement, 10, evalcount, m_gpu, m_bfgs_ctx);
            // auto du = clk.mark();
            // printf("gpu bfgs use %lu us\n", du);
            // todo:
            // printf("gpu done, e: %f cnt %d\n", res, evalcount);
            // return;
        }
    }

    if (!use_gpu) {
        std::cerr << "use cpu" << std::endl;
        quasi_newton_aux aux(&m, &p, &ig, v, use_gpu);
        res = bfgs(aux, out.c, g, max_steps, average_required_improvement, 10, evalcount);
    }
    // printf("=========================\n");

#if 0
    printf("gpu done, e: %f cnt %d\n", res, evalcount);
    out.c.print();
#endif

    // Update model a last time after optimization
    m.set(out.c);
    out.e = res;
}
void quasi_newton::cpu(model& m, const precalculate_byatom& p, const igrid& ig, output_type& out, change& g, const vec& v, int& evalcount) { // g must have correct size
#if QNDEBUG
    printf(">>>> before bfgs\n");
    m.print();
#endif
    // std::cerr << "use cpu" << std::endl;
    quasi_newton_aux aux(&m, &p, &ig, v, use_gpu);
    auto res = bfgs(aux, out.c, g, max_steps, average_required_improvement, 10, evalcount);
#if QNDEBUG
    printf(">>>> after bfgs\n");
    m.print();
#endif
    // printf("=========================\n");

    // Update model a last time after optimization
#if 0
    printf("cpu done, e: %f cnt %d\n", res, evalcount);
    out.c.print();
#endif
    m.set(out.c);
    out.e = res;
    // exit (0);
}