/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedron.h"

#include "volumes/PlacedPolyhedron.h"
#include "volumes/SpecializedPolyhedron.h"

#include <cmath>

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC
UnplacedPolyhedron::UnplacedPolyhedron(
    int sideCount,
    int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[])
    : fSideCount(sideCount), fOuter(zPlaneCount-1), fInner(zPlaneCount-1),
      fHasInnerRadii(false) {

  // Sanity check of input parameters
  {
    Assert(zPlaneCount > 1, "Need at least two z-planes to construct polyhedron"
           " segments.\n");
    Assert(fSideCount > 2, "Need at least three sides to construct polyhedron"
           " segments.\n");

    int innerRadii = 0;
    for (int i = 0; i < zPlaneCount; ++i) {
      if (rMin[i] > kTolerance) {
        fHasInnerRadii = true;
        Assert(innerRadii++ == i, "Isolated cavities in polyhedron not"
               " allowed.\n");
      }
    }
  }

}
#endif

} // End global namespace