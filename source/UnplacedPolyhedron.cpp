/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedron.h"

#include "volumes/Polygon.h"

#include <cmath>

namespace VECGEOM_NAMESPACE {

UnplacedPolyhedron::UnplacedPolyhedron(
    Precision phiStart, Precision phiDelta, int sideCount,
    const int zPlaneCount, const Precision zPlane[], const Precision rInner[],
    const Precision rOuter[]) {

  Assert(sideCount > 0, "Polyhedron requires at least one side.\n");

  // Fix and set phi parameters

  bool hasPhi;

  if (phiStart < 0.) {
    phiStart += kTwoPi*(1 - static_cast<int>(phiStart / kTwoPi));
  }
  if (phiStart > kTwoPi) {
    phiStart -= kTwoPi*static_cast<int>(phiStart / kTwoPi);
  }

  if ((phiDelta <= 0.) || (phiDelta >= kTwoPi * (1. - kEpsilon))) {
    phiDelta = kTwoPi;
    hasPhi = false;
  } else {
    phiDelta = phiDelta;
    hasPhi = true;
  }
  Precision convertRad = 1. / cos(.5 * phiDelta / sideCount);

  // Check contiguity in segments

  for (int i = 0; i < zPlaneCount-1; ++i) {
    if (zPlane[i] == zPlane[i+1]) {
      Assert(rInner[i] <= rOuter[i+1] && rInner[i+1] <= rOuter[i]);
    }
  }

  // Create corner polygon from segments are verify the outcome

  Polygon corners = Polygon(rInner, rOuter, zPlane, zPlaneCount);
  corners.Scale(convertRad, 1.);

  Assert(corners.GetXMin() >= 0.);

  if (corners.SurfaceArea() < -kTolerance) corners.ReverseOrder();
  Assert(corners.SurfaceArea() >= -kTolerance);

  // Create faces

  int faceCount = hasPhi ? corners.GetVertixCount()+2
                         : corners.GetVertixCount();
  fFaces.Allocate(faceCount);
}

UnplacedPolyhedron::~UnplacedPolyhedron() {
  for (Face** f = fFaces.begin(), **fEnd = fFaces.end(); f != fEnd; ++f) {
    delete *f;
  }
}

} // End global namespace