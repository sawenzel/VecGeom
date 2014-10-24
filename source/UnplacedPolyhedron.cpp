/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedron.h"

#include "volumes/PlacedPolyhedron.h"
#include "volumes/SpecializedPolyhedron.h"

#include <cmath>
#include <memory>

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_STD_CXX11

UnplacedPolyhedron::UnplacedPolyhedron(
    const int sideCount,
    const int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[])
    : UnplacedPolyhedron(0, 360, sideCount, zPlaneCount, zPlanes, rMin, rMax) {}

UnplacedPolyhedron::UnplacedPolyhedron(
    Precision phiStart,
    Precision phiDelta,
    const int sideCount,
    const int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[])
    : fSideCount(sideCount), fHasInnerRadii(false),
      fHasPhiCutout(phiDelta < 360), fHasLargePhiCutout(phiDelta < 180),
      fEndCapsOuterRadii{0, 0}, fZSegments(zPlaneCount-1),
      fZPlanes(zPlaneCount), fRMin(zPlaneCount), fRMax(zPlaneCount),
      fPhiSections(sideCount+fHasPhiCutout), fBoundingTube(0, 1, 1, 0, 360) {

  typedef Vector3D<Precision> Vec_t;

  // Sanity check of input parameters
  Assert(zPlaneCount > 1, "Need at least two z-planes to construct polyhedron"
         " segments.\n");
  Assert(fSideCount > 2, "Need at least three sides to construct polyhedron"
         " segments.\n");

  copy(zPlanes, zPlanes+zPlaneCount, &fZPlanes[0]);
  copy(rMin, rMin+zPlaneCount, &fRMin[0]);
  copy(rMax, rMax+zPlaneCount, &fRMax[0]);
  // Initialize segments
  for (int i = 0; i < zPlaneCount-1; ++i) {
    Assert(zPlanes[i] <= zPlanes[i+1], "Polyhedron Z-planes must be "
           "monotonically increasing.\n");
    fZSegments[i].hasInnerRadius = rMin[i] > 0 || rMin[i+1] > 0;
    fZSegments[i].outer = Quadrilaterals(sideCount);
    if (fHasPhiCutout) {
      fZSegments[i].phi = Quadrilaterals(2);
    }
    if (fZSegments[i].hasInnerRadius) {
      fZSegments[i].inner = Quadrilaterals(sideCount);
      fHasInnerRadii = true;
    }
  }

  // Compute the cylindrical coordinate phi along which the corners are placed
  Assert(phiDelta > 0, "Invalid phi angle provided in polyhedron constructor. "
         "Value must be greater than zero.\n");
  phiStart = NormalizeAngle<kScalar>(kDegToRad*phiStart);
  phiDelta *= kDegToRad;
  if (phiDelta > kTwoPi) phiDelta = kTwoPi;
  Precision sidePhi = phiDelta / sideCount;
  std::unique_ptr<Precision[]> vertixPhi(new Precision[sideCount+1]);
  for (int i = 0, iMax = sideCount+fHasPhiCutout; i < iMax; ++i) {
    vertixPhi[i] = NormalizeAngle<kScalar>(phiStart + i*sidePhi);
    fPhiSections.set(i,
        Vec_t::FromCylindrical(1., vertixPhi[i], 0).Normalized().FixZeroes());
  }
  // If there is no phi cutout, last phi is equal to the first
  if (!fHasPhiCutout) vertixPhi[sideCount] = vertixPhi[0];

  // Specified radii are to the sides, not to the corners. Change these values,
  // as corners and not sides are used to build the structure
  Precision cosHalfDeltaPhi = cos(0.5*sidePhi);
  Precision innerRadius = kInfinity, outerRadius = -kInfinity;
  for (int i = 0; i < zPlaneCount; ++i) {
    // Use distance to side for minimizing inner radius of bounding tube
    if (rMin[i] < innerRadius) innerRadius = rMin[i];
    rMin[i] /= cosHalfDeltaPhi;
    rMax[i] /= cosHalfDeltaPhi;
    Assert(rMin[i] >= 0 && rMax[i] > 0, "Invalid radius provided to "
           "polyhedron constructor.");
    // Use distance to corner for minimizing outer radius of bounding tube
    if (rMax[i] > outerRadius) outerRadius = rMax[i];
  }
  // Out radius of endcaps are used for SafetyToIn
  fEndCapsOuterRadii[0] = rMax[0];
  fEndCapsOuterRadii[1] = rMax[zPlaneCount-1];
  // Create bounding tube with biggest outer radius and smallest inner radius
  Precision boundingTubeZ = zPlanes[zPlaneCount-1] - zPlanes[0] + 2.*kTolerance;
  Precision boundsPhiStart = !fHasPhiCutout ? 0 : phiStart;
  Precision boundsPhiDelta = !fHasPhiCutout ? 360 : phiDelta;
  fBoundingTube = UnplacedTube(innerRadius - kTolerance,
                               outerRadius + kTolerance, 0.5*boundingTubeZ,
                               boundsPhiStart, boundsPhiDelta);
  fBoundingTubeOffset = zPlanes[0] + 0.5*boundingTubeZ;

  // Ease indexing into twodimensional vertix array
  auto VertixIndex = [&sideCount] (int plane, int corner) {
    return plane*(sideCount+1) + corner;
  };

  // Precompute all vertices to ensure that there are no numerical cracks in the
  // surface.
  const int nVertices = zPlaneCount*(sideCount+1);
  std::unique_ptr<Vec_t[]> outerVertices(new Vec_t[nVertices]);
  std::unique_ptr<Vec_t[]> innerVertices(new Vec_t[nVertices]);
  for (int i = 0; i < zPlaneCount; ++i) {
    for (int j = 0, jMax = sideCount+fHasPhiCutout; j < jMax; ++j) {
      int index = VertixIndex(i, j);
      outerVertices[index] =
          Vec_t::FromCylindrical(rMax[i], vertixPhi[j], zPlanes[i]).FixZeroes();
      innerVertices[index] =
          Vec_t::FromCylindrical(rMin[i], vertixPhi[j], zPlanes[i]).FixZeroes();
    }
    // Non phi cutout case
    if (!fHasPhiCutout) {
      // Make last vertices identical to the first phi coordinate
      outerVertices[VertixIndex(i, sideCount)] =
          outerVertices[VertixIndex(i, 0)];
      innerVertices[VertixIndex(i, sideCount)] =
          innerVertices[VertixIndex(i, 0)];
    }
  }

  // Build segments by drawing quadrilaterals between vertices
  for (int iPlane = 0; iPlane < zPlaneCount-1; ++iPlane) {

    auto WrongNormal = [] (Vector3D<Precision> const &normal,
                           Vector3D<Precision> const &corner) {
      return normal[0]*corner[0] + normal[1]*corner[1] < 0;
    };

    // Draw the regular quadrilaterals along phi
    for (int iSide = 0; iSide < sideCount; ++iSide) {
      fZSegments[iPlane].outer.Set(
          iSide,
          outerVertices[VertixIndex(iPlane, iSide)],
          outerVertices[VertixIndex(iPlane, iSide+1)],
          outerVertices[VertixIndex(iPlane+1, iSide+1)],
          outerVertices[VertixIndex(iPlane+1, iSide)]);
      // Normal has to point away from Z-axis
      if (WrongNormal(fZSegments[iPlane].outer.GetNormal(iSide),
                      outerVertices[VertixIndex(iPlane, iSide)])) {
        fZSegments[iPlane].outer.FlipSign(iSide);
      }
      if (fZSegments[iPlane].hasInnerRadius) {
        fZSegments[iPlane].inner.Set(
            iSide,
            innerVertices[VertixIndex(iPlane, iSide)],
            innerVertices[VertixIndex(iPlane, iSide+1)],
            innerVertices[VertixIndex(iPlane+1, iSide+1)],
            innerVertices[VertixIndex(iPlane+1, iSide)]);
        // Normal has to point away from Z-axis
        if (WrongNormal(fZSegments[iPlane].inner.GetNormal(iSide),
                        innerVertices[VertixIndex(iPlane, iSide)])) {
          fZSegments[iPlane].inner.FlipSign(iSide);
        }
      }
    }

    if (fHasPhiCutout) {
      // If there's a phi cutout, draw two quadrilaterals connecting the four
      // corners (two inner, two outer) of the first and last phi coordinate,
      // respectively
      fZSegments[iPlane].phi.Set(
          0,
          innerVertices[VertixIndex(iPlane, 0)],
          innerVertices[VertixIndex(iPlane+1, 0)],
          outerVertices[VertixIndex(iPlane+1, 0)],
          outerVertices[VertixIndex(iPlane, 0)]);
      // Make sure normal points backward along phi
      if (fZSegments[iPlane].phi.GetNormal(0).Cross(fPhiSections[0])[2] < 0) {
        fZSegments[iPlane].phi.FlipSign(0);
      }
      fZSegments[iPlane].phi.Set(
          1,
          outerVertices[VertixIndex(iPlane, sideCount)],
          outerVertices[VertixIndex(iPlane+1, sideCount)],
          innerVertices[VertixIndex(iPlane+1, sideCount)],
          innerVertices[VertixIndex(iPlane, sideCount)]);
      // Make sure normal points forwards along phi
      if (fZSegments[iPlane].phi.GetNormal(1).Cross(
              fPhiSections[fSideCount])[2] > 0) {
        fZSegments[iPlane].phi.FlipSign(1);
      }
    }

  } // End loop over segments

}

#endif

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedPolyhedron::GetPhiStart() const {
  return kRadToDeg*NormalizeAngle<kScalar>(GetPhiSection(0).Phi());
}

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedPolyhedron::GetPhiEnd() const {
  return !HasPhiCutout() ? 360 :
         kRadToDeg*NormalizeAngle<kScalar>(GetPhiSection(GetSideCount()).Phi());
}

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedPolyhedron::GetPhiDelta() const {
  return GetPhiEnd() - GetPhiStart();
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

#ifndef VECGEOM_NVCC
  #define POLYHEDRON_CREATE_SPECIALIZATION(INNER) \
  if (hasInner == INNER) { \
    if (placement) { \
      return new(placement) \
             SpecializedPolyhedron<INNER>(volume, transformation); \
    } else { \
      return new SpecializedPolyhedron<INNER>(volume, transformation); \
    } \
  }
#else
  #define POLYHEDRON_CREATE_SPECIALIZATION(INNER) \
  if (hasInner == INNER) { \
    if (placement) { \
      return new(placement) \
             SpecializedPolyhedron<INNER>(volume, transformation, id); \
    } else { \
      return new \
             SpecializedPolyhedron<INNER>(volume, transformation, id); \
    } \
  }
#endif

  POLYHEDRON_CREATE_SPECIALIZATION(true);
  POLYHEDRON_CREATE_SPECIALIZATION(false);

#ifndef VECGEOM_NVCC
  if (placement) {
    return new(placement)
           SpecializedPolyhedron<true>(volume, transformation);
  } else {
    return new SpecializedPolyhedron<true>(volume, transformation);
  }
#else
  if (placement) {
    return new(placement)
           SpecializedPolyhedron<true>(volume, transformation, id);
  } else {
    return new SpecializedPolyhedron<true>(volume, transformation, id);
  }
#endif

  #undef POLYHEDRON_CREATE_SPECIALIZATION
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedPolyhedron::Print() const {
  printf("UnplacedPolyhedron {%i sides, %i segments, %s}",
         fSideCount, fZSegments.size(),
         (fHasInnerRadii) ? "has inner radii" : "no inner radii");
}

void UnplacedPolyhedron::Print(std::ostream &os) const {
  os << "UnplacedPolyhedron {" << fSideCount << " sides, " << fZSegments.size()
     << " segments, "
     << ((fHasInnerRadii) ? "has inner radii" : "no inner radii") << "}";
}

} // End global namespace
