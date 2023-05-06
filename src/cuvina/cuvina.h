#ifndef CUVINA_H
#define CUVINA_H
#include <memory>
#include "vina/precalculate.h"

namespace dock {
class CuObject {
public:
    std::shared_ptr<void> ctrl;
    void *obj;
};

bool makePrecalcByAtom(precalculate_byatom &p);
};  // namespace dock
#endif