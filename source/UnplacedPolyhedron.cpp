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
    : fSideCount(sideCount), fHasInnerRadii(false), fSegments(zPlaneCount-1),
      fZPlanes(zPlaneCount) {

  typedef Vector3D<Precision> Vec_t;

  // Sanity check of input parameters
  Assert(zPlaneCount > 1, "Need at least two z-planes to construct polyhedron"
         " segments.\n");
  Assert(fSideCount > 2, "Need at least three sides to construct polyhedron"
         " segments.\n");

  copy(zPlanes, zPlanes+zPlaneCount, &fZPlanes[0]);
  // Initialize segments
  for (int i = 0; i < zPlaneCount-1; ++i) {
    fSegments[i].outer = Quadrilaterals<0>(fSideCount);
    if (rMin[i] > kTolerance || rMin[i+1] > kTolerance) {
      fHasInnerRadii = true;
      fSegments[i].hasInnerRadius = true;
      fSegments[i].inner = Quadrilaterals<0>(fSideCount);
    }
  }

  // Compute phi as the last cylindrical coordinate of the vertices
  Precision deltaPhi = kTwoPi / sideCount;
  Precision *vertixPhi = new Precision[sideCount];
  for (int i = 0; i < sideCount; ++i) vertixPhi[i] = i*deltaPhi;

  // Precompute all vertices to ensure that there are no numerical cracks in the
  // surface.
  Vec_t *outerVertices = new Vec_t[zPlaneCount*sideCount];
  Vec_t *innerVertices = new Vec_t[zPlaneCount*sideCount];
  for (int i = 0; i < zPlaneCount; ++i) {
    for (int j = 0; j < sideCount; ++j) {
      int index = i*sideCount + j;
      outerVertices[index] =
          Vec_t::FromCylindrical(rMax[i], vertixPhi[j], zPlanes[i]);
      innerVertices[index] =
          Vec_t::FromCylindrical(rMin[i], vertixPhi[j], zPlanes[i]);
    }
  }
  delete vertixPhi;

  // Build segments by drawing quadrilaterals between vertices
  for (int i = 0; i < zPlaneCount-1; ++i) {
    Vec_t corner0, corner1, corner2, corner3;
    // Sides of outer shell
    for (int j = 0; j < sideCount-1; ++j) {
      corner0 = outerVertices[i*sideCount + j];
      corner1 = outerVertices[i*sideCount + j+1];
      corner2 = outerVertices[(i+1)*sideCount + j+1];
      corner3 = outerVertices[(i+1)*sideCount + j];
      fSegments[i].outer.Set(j, corner0, corner1, corner2, corner3);
    }
    // Close the loop
    corner0 = corner1;
    corner1 = outerVertices[i*sideCount];
    corner2 = outerVertices[(i+1)*sideCount];
    corner3 = corner2;
    fSegments[i].outer.Set(sideCount-1, corner0, corner1, corner2, corner3);
    // Sides of inner shell (if needed)
    if (fSegments[i].hasInnerRadius) {
      for (int j = 0; j < sideCount; ++j) {
        corner0 = innerVertices[i*sideCount + j];
        corner1 = innerVertices[i*sideCount + j+1];
        corner2 = innerVertices[(i+1)*sideCount + j+1];
        corner3 = innerVertices[(i+1)*sideCount + j];
        fSegments[i].inner.Set(j, corner0, corner1, corner2, corner3);
      }
      // Close the loop
      corner0 = corner1;
      corner1 = innerVertices[i*sideCount];
      corner2 = innerVertices[(i+1)*sideCount];
      corner3 = corner2;
      fSegments[i].inner.Set(sideCount-1, corner0, corner1, corner2, corner3);
    }
  }
  delete outerVertices;
  delete innerVertices;
}
#endif

} // End global namespace