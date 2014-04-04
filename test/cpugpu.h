#ifndef CPUGPU_H_
#define CPUGPU_H_

#include "base/global.h"

using namespace vecgeom;

void LocatePointsGpu(Precision *const x, Precision *const y, Precision *const z,
                     const unsigned size, const int depth, int *const output);

#endif // CPUGPU_H_