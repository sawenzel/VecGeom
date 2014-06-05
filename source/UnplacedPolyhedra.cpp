/// \file UnplacedPolyhedra.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedra.h"

#include "volumes/Polygon.h"

#include <cmath>

namespace VECGEOM_NAMESPACE {

UnplacedPolyhedra::UnplacedPolyhedra(
    Precision phiStart, Precision phiDelta, const int sideCount,
    const int zPlaneCount, const Precision zPlane[], const Precision rInner[],
    const Precision rOuter[]) : fSideCount(sideCount) {

  assert(sideCount > 0 && "Polyhedra requires at least one side.\n");

  // Fix and set phi parameters

  if (phiStart < 0.) {
    phiStart += kTwoPi*(1 - static_cast<int>(phiStart / kTwoPi));
  }
  if (phiStart > kTwoPi) {
    phiStart -= kTwoPi*static_cast<int>(phiStart / kTwoPi);
  }
  fPhiStart = phiStart;

  if ((phiDelta <= 0.) || (phiDelta >= kTwoPi * (1. - kEpsilon))) {
    fPhiDelta = kTwoPi;
    fHasPhi = false;
  } else {
    fPhiDelta = phiDelta;
    fHasPhi = true;
  }
  Precision convertRad = 1. / cos(.5 * fPhiDelta / sideCount);

  // Check contiguity in segments

  for (int i = 0; i < zPlaneCount-1; ++i) {
    if (zPlane[i] == zPlane[i+1]) {
      assert(rInner[i] <= rOuter[i+1] && rInner[i+1] <= rOuter[i]);
    }
  }

  // Create corner polygon from segments are verify the outcome

  fCorners = new Polygon(rInner, rOuter, zPlane, zPlaneCount);
  fCorners->Scale(convertRad, 1.);

  assert(fCorners->GetXMin() >= 0.);

  if (fCorners->SurfaceArea() < -kTolerance) fCorners->ReverseOrder();
  assert(fCorners->SurfaceArea() >= -kTolerance);


}

} // End global namespace