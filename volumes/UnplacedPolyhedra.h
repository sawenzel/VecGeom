/// \file UnplacedPolyhedra.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYHEDRA_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYHEDRA_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/Polygon.h"
#include "volumes/UnplacedVolume.h"

namespace VECGEOM_NAMESPACE {

class UnplacedPolyhedra : public VUnplacedVolume, public AlignedBase {

private:

  int fSideCount;
  Precision fPhiStart;
  Precision fPhiDelta;
  bool fHasPhi;
  Polygon *fCorners;

public:

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedPolyhedra(
      Precision phiStart,
      Precision phiDelta,
      const int sideCount,
      const int cornerCount,
      const Precision zPlanes[],
      const Precision rInner[],
      const Precision rOuter[]);

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPOLYHEDRA_H_