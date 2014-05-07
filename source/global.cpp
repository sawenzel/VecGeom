#include "base/global.h"

#ifdef VECGEOM_NVCC

namespace vecgeom_cuda {

const int Constants::kAlignmentBoundary =
    vecgeom::Constants::kAlignmentBoundary;
const Precision Constants::kDegToRad = vecgeom::Constants::kDegToRad;
const Precision Constants::kRadToDeg = vecgeom::Constants::kRadToDeg;
const Precision Constants::kInfinity = vecgeom::Constants::kInfinity;
const Precision Constants::kTiny = vecgeom::Constants::kTiny;
const Precision Constants::kTolerance = vecgeom::Constants::kTolerance;
const Precision Constants::kHalfTolerance = vecgeom::Constants::kHalfTolerance;

} // End namespace vecgeom_cuda

#endif