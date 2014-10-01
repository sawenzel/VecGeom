/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedron.h"

#include "volumes/PlacedPolyhedron.h"
#include "volumes/SpecializedPolyhedron.h"

#include <cmath>

namespace VECGEOM_NAMESPACE {

UnplacedPolyhedron::UnplacedPolyhedron(
    const int sideCount,
    const int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[])
    : fSideCount(sideCount), fHasInnerRadii(false),
      fZBounds{zPlanes[0]-kTolerance, zPlanes[zPlaneCount-1]+kTolerance},
      fEndCaps(), fEndCapsOuterRadii{0, 0}, fSegments(zPlaneCount-1),
      fZPlanes(zPlaneCount), fPhiSections(sideCount+1) {

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

  // Compute the cylindrical coordinate phi along which the corners are placed
  Precision deltaPhi = kTwoPi / sideCount;
  Precision *vertixPhi = new Precision[sideCount];
  for (int i = 0; i < sideCount; ++i) {
    vertixPhi[i] = i*deltaPhi;
    fPhiSections.set(i,
        Vec_t::FromCylindrical(1., vertixPhi[i], 0).Normalized().FixZeroes());
  }
  fPhiSections.set(sideCount, fPhiSections[0]); // Close the loop

  // Specified radii are to the sides, not to the corners. Change these values,
  // as corners and not sides are used to build the structure
  Precision cosHalfDeltaPhi = cos(0.5*deltaPhi);
  for (int i = 0; i < zPlaneCount; ++i) {
    rMin[i] /= cosHalfDeltaPhi;
    rMax[i] /= cosHalfDeltaPhi;
  }
  fEndCapsOuterRadii[0] = rMax[0];
  fEndCapsOuterRadii[1] = rMax[zPlaneCount-1];

  // Ease indexing into twodimensional vertix array
  auto GetVertix = [&sideCount] (int plane, int corner) {
    return plane*(sideCount+1) + corner;
  };

  // Precompute all vertices to ensure that there are no numerical cracks in the
  // surface.
  Vec_t *outerVertices = 0, *innerVertices = 0;
  outerVertices = new Vec_t[zPlaneCount*(sideCount+1)];
  if (fHasInnerRadii) innerVertices = new Vec_t[zPlaneCount*(sideCount+1)];
  for (int i = 0; i < zPlaneCount; ++i) {
    for (int j = 0; j < sideCount; ++j) {
      int index = GetVertix(i, j);
      outerVertices[index] =
          Vec_t::FromCylindrical(rMax[i], vertixPhi[j], zPlanes[i]).FixZeroes();
      if (fHasInnerRadii) {
        innerVertices[index] =
            Vec_t::FromCylindrical(
                rMin[i], vertixPhi[j], zPlanes[i]).FixZeroes();
      }
    }
    // Close the loop for easy indexing when building the sides
    outerVertices[GetVertix(i, sideCount)] = outerVertices[GetVertix(i, 0)];
    if (fHasInnerRadii) {
      innerVertices[GetVertix(i, sideCount)] = innerVertices[GetVertix(i, 0)];
    }
  }
  delete[] vertixPhi;

  // Build segments by drawing quadrilaterals between vertices
  for (int i = 0; i < zPlaneCount-1; ++i) {
    Vec_t corner0, corner1, corner2, corner3;
    // Sides of outer shell
    for (int j = 0; j < sideCount; ++j) {
      corner0 = outerVertices[GetVertix(i, j)];
      corner1 = outerVertices[GetVertix(i, j+1)];
      corner2 = outerVertices[GetVertix(i+1, j+1)];
      corner3 = outerVertices[GetVertix(i+1, j)];
      fSegments[i].outer.Set(j, corner0, corner1, corner2, corner3);
    }
    // Make sure the normals are pointing away from the Z-axis
    bool tiltingUp = rMax[i+1] - rMax[i] < 0;
    fSegments[i].outer.FixNormalSign(2, tiltingUp);
    // Sides of inner shell (if needed)
    if (fSegments[i].hasInnerRadius) {
      for (int j = 0; j < sideCount; ++j) {
        corner0 = innerVertices[GetVertix(i, j)];
        corner1 = innerVertices[GetVertix(i, j+1)];
        corner2 = innerVertices[GetVertix(i+1, j+1)];
        corner3 = innerVertices[GetVertix(i+1, j)];
        fSegments[i].inner.Set(j, corner0, corner1, corner2, corner3);
      }
      // Make sure the normals are pointing away from the Z-axis
      bool tiltingUp = rMin[i+1] - rMin[i] < 0;
      fSegments[i].inner.FixNormalSign(2, tiltingUp);
    }
  }

  delete[] outerVertices;
  if (fHasInnerRadii) delete[] innerVertices;
}

void UnplacedPolyhedron::ExtractZPlanes(
    Precision *z,
    Precision *rMin,
    Precision *rMax) const {

  const int zPlaneCount = fSegments.size()+1;

  // Find any corner that lies in the correct Z-plane
  auto GetCorner = [] (Quadrilaterals<0> const &q, Precision z) {
    Vector3D<Precision> corner;
    for (int j = 0; j < 4; ++j) {
      corner = q.GetCorners()[j][0];
      if (corner[2] == z) return corner;
    }
    Assert(0, "Unexpected corners during extraction of Z-planes in Polyhedron");
    return corner;
  };

  Vector3D<Precision> corner;
  for (int i = 0; i < zPlaneCount-1; ++i) {
    // Get Z-plane from the segment of which it is the beginning
    UnplacedPolyhedron::Segment const &segment = GetSegment(i);
    z[i] = fZPlanes[i];
    corner = GetCorner(segment.outer, z[i]);
    rMax[i] = sqrt(corner[0]*corner[0] + corner[1]*corner[1]);
    if (segment.hasInnerRadius) {
      corner = GetCorner(segment.inner, z[i]);
      rMin[i] = sqrt(corner[0]*corner[0] + corner[1]*corner[1]);
    } else {
      rMin[i] = 0;
    }
  }
  // Treat final Z-plane as the end-plane of the last segment
  UnplacedPolyhedron::Segment const &segment = GetSegment(GetSegmentCount()-1);
  z[zPlaneCount-1] = fZPlanes[GetSegmentCount()];
  corner = GetCorner(segment.outer, z[zPlaneCount-1]);
  rMax[zPlaneCount-1] =
      sqrt(corner[0]*corner[0] + corner[1]*corner[1]);
  if (segment.hasInnerRadius) {
    corner = GetCorner(segment.inner, z[zPlaneCount-1]);
    rMin[zPlaneCount-1] =
        sqrt(corner[0]*corner[0] + corner[1]*corner[1]);
  } else {
    rMin[zPlaneCount-1] = 0;
  }
  // Go from radii to corners to radii to sides
  Precision cosHalfDeltaPhi = cos(0.5*kTwoPi/fSideCount);
  for (int i = 0; i < zPlaneCount; ++i) {
    rMin[i] *= cosHalfDeltaPhi;
    rMax[i] *= cosHalfDeltaPhi;
  }

}

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
