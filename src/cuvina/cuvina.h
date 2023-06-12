#ifndef CUVINA_H
#define CUVINA_H
#include <memory>
#include "vina/precalculate.h"
#include "vina/cache.h"

namespace dock {
class CuObject {
public:
    std::shared_ptr<void> ctrl;
    void *obj;
    std::shared_ptr<void> cuctrl;
    void *cuobj;

};

extern void comp_change(change &c1, change &c2) ;
bool makePrecalcByAtom(precalculate_byatom &p);
bool makeSrcModel(model *m, precalculate_byatom &p);
bool makeCache(cache &c);
std::shared_ptr<void> makeModel(model *m, const vec &v);
bool makeModel(std::shared_ptr<void> &obj, model *m, const vec &v);
bool makeModelDesc(std::shared_ptr<void> &obj, model *m, int nmc=1);
bool makeBFGSCtx(std::shared_ptr<void> &obj, const model &m, const change &g, const conf &c, const vec &v, int evalcnt=0);

using rngs = std::vector<std::shared_ptr<rng>>;
bool makeMCInputs(std::shared_ptr<void> &obj, const model &m, int nmc, int num_mutable_entities,
                  int mc_steps, rngs &generators, conf &c);
bool makeMC(std::shared_ptr<void> &pmd, std::shared_ptr<void> &pctx, std::shared_ptr<void> &pin,
            std::shared_ptr<void> &pout, model &m, int num_mutable_entities, int steps, int nmc,
            rngs &generators, conf &c, sz Over, fl average_required_improvement, int local_steps,
            int max_evalcnt, const vec &v1, const vec &v2, fl amplitude, fl temperature, 
            std::function<void (int i, fl *c)> init);
fl run_model_eval_deriv(const precalculate_byatom &p, const igrid &ig, 
                        change &g, std::shared_ptr<void> mobj, std::shared_ptr<void> ctxobj);
fl run_cuda_bfgs(model *m, const precalculate_byatom &p, const igrid &ig, change &g, conf &c, vecv &coords,
                 const unsigned max_steps, const fl average_required_improvement, const sz over,
                 int &evalcount, std::shared_ptr<void> mobj, std::shared_ptr<void> ctxobj);
std::vector<std::vector<output_type>> run_cuda_mc(std::shared_ptr<void> &pmd, std::shared_ptr<void> &pctx,
                 std::shared_ptr<void> &pin, std::shared_ptr<void> &pout, model *m,
                 const precalculate_byatom &p, const igrid &ig, int nmc, int steps, int local_steps, conf_size &s);
bool create_vina_server(int device, int nrinst);
void destroy_vina_server();

};  // namespace dock
#endif