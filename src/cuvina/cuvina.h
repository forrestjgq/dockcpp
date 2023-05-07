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
};

extern void comp_change(change &c1, change &c2) ;
bool makePrecalcByAtom(precalculate_byatom &p);
bool makeSrcModel(model *m, precalculate_byatom &p);
bool makeCache(cache &c);
std::shared_ptr<void> makeModel(model *m, const vec &v);
bool makeBFGSCtx(std::shared_ptr<void> &obj, const change &g, const conf &c);
fl run_model_eval_deriv(const precalculate_byatom &p, const igrid &ig, const vec &v,
                        change &g, const conf &c, std::shared_ptr<void> mobj, std::shared_ptr<void> ctxobj);
};  // namespace dock
#endif