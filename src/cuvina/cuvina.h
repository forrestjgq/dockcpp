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

bool makePrecalcByAtom(precalculate_byatom &p);
bool makeSrcModel(model *m, precalculate_byatom &p);
bool makeCache(cache &c);
};  // namespace dock
#endif