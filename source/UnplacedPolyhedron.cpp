/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedron.h"

#include "volumes/PlacedPolyhedron.h"
#include "volumes/SpecializedPolyhedron.h"

#include <cmath>

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC
UnplacedPolyhedron::UnplacedPolyhedron(
    const int sideCount,
    const int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[])
    : fSideCount(sideCount), fHasInnerRadii(false),
      fZBounds{zPlanes[0]-kTolerance, zPlanes[zPlaneCount-1]+kTolerance},
      fEndCaps(), fSegments(zPlaneCount-1), fZPlanes(zPlaneCount) {

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
    fSegments[i].zMax = zPlanes[i+1];
    Assert(zPlanes[i] <= zPlanes[i+1], "Polyhedron Z-planes must be "
           "monotonically increasing.\n");
    if (rMin[i] > kTolerance || rMin[i+1] > kTolerance) {
      fHasInnerRadii = true;
      fSegments[i].hasInnerRadius = true;
      fSegments[i].inner = Quadrilaterals<0>(fSideCount);
    } else {
      fSegments[i].hasInnerRadius = false;
    }
  }

  fEndCaps.Set(0, Vector3D<Precision>(0, 0, -1),
                  Vector3D<Precision>(0, 0, zPlanes[0]));
  fEndCaps.Set(1, Vector3D<Precision>(0, 0, 1),
                  Vector3D<Precision>(0, 0, zPlanes[zPlaneCount-1]));

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
  delete[] vertixPhi;

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

  delete[] outerVertices;
  delete[] innerVertices;
}
#endif

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedPolyhedron::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  UnplacedPolyhedron const *unplaced =
      static_cast<UnplacedPolyhedron const *>(volume->unplaced_volume());

  bool hasInner = unplaced->HasInnerRadii();
  int sideCount = unplaced->GetSideCount();

#ifndef VECGEOM_NVCC
  #define POLYHEDRON_CREATE_SPECIALIZATION(INNER, SIDES) \
  if (hasInner == INNER && sideCount == SIDES) { \
    if (placement) { \
      return new(placement) \
             SpecializedPolyhedron<INNER, SIDES>(volume, transformation); \
    } else { \
      return new SpecializedPolyhedron<INNER, SIDES>(volume, transformation); \
    } \
  }
#else
  #define POLYHEDRON_CREATE_SPECIALIZATION(INNER, SIDES) \
  if (hasInner == INNER && sideCount == SIDES) { \
    if (placement) { \
      return new(placement) \
             SpecializedPolyhedron<INNER, SIDES>(volume, transformation, id); \
    } else { \
      return new \
             SpecializedPolyhedron<INNER, SIDES>(volume, transformation, id); \
    } \
  }
#endif

  POLYHEDRON_CREATE_SPECIALIZATION(true, 3);
  POLYHEDRON_CREATE_SPECIALIZATION(true, 4);
  POLYHEDRON_CREATE_SPECIALIZATION(true, 5);
  POLYHEDRON_CREATE_SPECIALIZATION(true, 6);
  POLYHEDRON_CREATE_SPECIALIZATION(true, 7);
  POLYHEDRON_CREATE_SPECIALIZATION(true, 8);
  POLYHEDRON_CREATE_SPECIALIZATION(false, 3);
  POLYHEDRON_CREATE_SPECIALIZATION(false, 4);
  POLYHEDRON_CREATE_SPECIALIZATION(false, 5);
  POLYHEDRON_CREATE_SPECIALIZATION(false, 6);
  POLYHEDRON_CREATE_SPECIALIZATION(false, 7);
  POLYHEDRON_CREATE_SPECIALIZATION(false, 8);

#ifndef VECGEOM_NVCC
  if (placement) {
    return new(placement)
           SpecializedPolyhedron<true, 0>(volume, transformation);
  } else {
    return new SpecializedPolyhedron<true, 0>(volume, transformation);
  }
#else
  if (placement) {
    return new(placement)
           SpecializedPolyhedron<true, 0>(volume, transformation, id);
  } else {
    return new SpecializedPolyhedron<true, 0>(volume, transformation, id);
  }
#endif

  #undef POLYHEDRON_CREATE_SPECIALIZATION
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedPolyhedron::Print() const {
  printf("UnplacedPolyhedron {%i sides, %i segments, %s}",
         fSideCount, fSegments.size(),
         (fHasInnerRadii) ? "has inner radii" : "no inner radii");
}

void UnplacedPolyhedron::Print(std::ostream &os) const {
  os << "UnplacedPolyhedron {" << fSideCount << " sides, " << fSegments.size()
     << " segments, "
     << ((fHasInnerRadii) ? "has inner radii" : "no inner radii") << "}";
}

} // End global namespace