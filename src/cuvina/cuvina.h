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
bool makeModelDesc(std::shared_ptr<void> &obj, model *m, const vec &v);
bool makeBFGSCtx(std::shared_ptr<void> &obj, const change &g, const conf &c, int evalcnt=0);
fl run_model_eval_deriv(const precalculate_byatom &p, const igrid &ig, 
                        change &g, std::shared_ptr<void> mobj, std::shared_ptr<void> ctxobj);
fl run_cuda_bfgs(const precalculate_byatom &p, const igrid &ig, change &g, conf &c,
                 const unsigned max_steps, const fl average_required_improvement, const sz over,
                 int &evalcount, std::shared_ptr<void> mobj, std::shared_ptr<void> ctxobj);
bool create_vina_server(int device, int nrinst);
void destroy_vina_server();

};  // namespace dock
#endif