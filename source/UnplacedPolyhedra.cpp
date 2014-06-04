/// \file UnplacedPolyhedra.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedra.h"

#include "volumes/Polygon.h"

namespace VECGEOM_NAMESPACE {

UnplacedPolyhedra::UnplacedPolyhedra(
    const Precision phiStart, Precision phiDelta, const int sideCount,
    const int zPlaneCount, const Precision zPlane[], const Precision rInner[],
    const Precision rOuter[]) {

  assert(sideCount > 0 && "Polyhedra requires at least one side.\n");

  if ((phiDelta <= 0.) || (phiDelta >= kTwoPi * (1. - kEpsilon))) {
    fPhiDelta = kTwoPi;
  } else {
    fPhiDelta = phiDelta;
  }
  Precision convertRad = 1. / cos(0.5 * fPhiDelta / sideCount);

  for (int i = 0; i < zPlaneCount-1; ++i) {
    if (zPlane[i] == zPlane[i+1]) {
      assert(rInner[i] <= rOuter[i+1] && rInner[i+1] <= rOuter[i]);
    }
  }

  Polygon rz = Polygon(rInner, rOuter, zPlane, zPlaneCount);
  rz.Scale(convertRad, 1.);

  assert(rz.GetXMin() >= 0.);

  if (rz.SurfaceArea() < -kTolerance) rz.ReverseOrder();
  assert(rz.SurfaceArea() >= -kTolerance);

}

} // End global namespace