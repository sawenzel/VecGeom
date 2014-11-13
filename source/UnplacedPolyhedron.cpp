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
      fZSegments(zPlaneCount-1), fZPlanes(zPlaneCount), fRMin(zPlaneCount),
      fRMax(zPlaneCount), fPhiSections(sideCount),
      fBoundingTube(0, 1, 1, 0, 360) {

  typedef Vector3D<Precision> Vec_t;

  // Sanity check of input parameters
  Assert(zPlaneCount > 1, "Need at least two z-planes to construct polyhedron"
         " segments.\n");
  Assert(fSideCount > 0, "Need at least one side to construct polyhedron"
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
  for (int i = 0, iMax = sideCount+1; i < iMax; ++i) {
    vertixPhi[i] = NormalizeAngle<kScalar>(phiStart + i*sidePhi);
    Vector3D<Precision> cornerVector =
        Vec_t::FromCylindrical(1., vertixPhi[i], 0).Normalized().FixZeroes();
    fPhiSections.set(
        i, Vector3D<Precision>(0, 0, 1).Cross(cornerVector).Normalized());
  }
  if (!fHasPhiCutout) {
    // If there is no phi cutout, last phi is equal to the first
    vertixPhi[sideCount] = vertixPhi[0];
  }

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
      // Make sure normal points forwards along phi
      if (fZSegments[iPlane].phi.GetNormal(0).Cross(fPhiSections[0])[2] > 0) {
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

#else // !VECGEOM_NVCC

__device__
UnplacedPolyhedron::UnplacedPolyhedron(
    int sideCount, bool hasInnerRadii, bool hasPhiCutout,
    bool hasLargePhiCutout, ZSegment *segmentData, Precision *zPlaneData,
    int zPlaneCount, Precision *phiSectionX, Precision *phiSectionY,
    Precision *phiSectionZ, UnplacedTube const &boundingTube,
    Precision boundingTubeOffset)
    : fSideCount(sideCount), fHasInnerRadii(hasInnerRadii),
      fHasPhiCutout(hasPhiCutout), fHasLargePhiCutout(hasLargePhiCutout),
      fZSegments(segmentData, zPlaneCount-1), fZPlanes(zPlaneData, zPlaneCount),
      fPhiSections(phiSectionX, phiSectionY, phiSectionZ, sideCount),
      fBoundingTube(boundingTube), fBoundingTubeOffset(boundingTubeOffset) {}


#endif // VECGEOM_NVCC

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedPolyhedron::GetPhiStart() const {
  return kRadToDeg*NormalizeAngle<kScalar>(
      GetPhiSection(0).Cross(Vector3D<Precision>(0, 0, 1)).Phi());
}

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedPolyhedron::GetPhiEnd() const {
  return !HasPhiCutout() ? 360 : kRadToDeg*NormalizeAngle<kScalar>(
      GetPhiSection(GetSideCount()).Cross(Vector3D<Precision>(0, 0, 1)).Phi());
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
  printf("UnplacedPolyhedron {%i sides, phi %f to %f, %i segments}",
         fSideCount, GetPhiStart(), GetPhiEnd(), fZSegments.size());
  printf("}");
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedPolyhedron::PrintSegments() const {
  printf("Printing %i polyhedron segments: ", fZSegments.size());
  for (int i = 0, iMax = fZSegments.size(); i < iMax; ++i) {
    printf("  Outer: ");
    fZSegments[i].outer.Print();
    printf("\n");
    if (fHasPhiCutout) {
      printf("  Phi: ");
      fZSegments[i].phi.Print();
      printf("\n");
    }
    if (fZSegments[i].hasInnerRadius) {
      printf("  Inner: ");
      fZSegments[i].inner.Print();
      printf("\n");
    }
  }
}

void UnplacedPolyhedron::Print(std::ostream &os) const {
  os << "UnplacedPolyhedron {" << fSideCount << " sides, " << fZSegments.size()
     << " segments, "
     << ((fHasInnerRadii) ? "has inner radii" : "no inner radii") << "}";
}

} // End global namespace

#ifdef VECGEOM_CUDA_INTERFACE
namespace vecgeom_cuda {
  class Quadrilaterals;
}
#endif

namespace vecgeom {

#ifdef VECGEOM_NVCC
class UnplacedTube;
class VUnplacedVolume;
#endif

// Has to return a void pointer as it's not possible to forward declare a nested
// class (vecgeom_cuda::UnplacedPolyhedron::ZSegment)
void* UnplacedPolyhedron_AllocateZSegments(int size, int &gpuMemorySize);

void UnplacedPolyhedron_ZSegment_GetAddresses(void *object, void *&outer,
                                              void *&inner, void *&phi,
                                              bool *&innerRadius);

void UnplacedPolyhedron_CopyToGpu(VUnplacedVolume *gpuPtr, int sideCount,
                                  bool hasInnerRadii, bool hasPhiCutout,
                                  bool hasLargePhiCutout, void *zSegments,
                                  Precision *zPlanes, int zPlaneCount,
                                  Precision *phiSectionsX,
                                  Precision *phiSectionsY,
                                  Precision *phiSectionsZ,
                                  VUnplacedVolume *boundingTube,
                                  Precision boundingTubeOffset);

#ifdef VECGEOM_CUDA_INTERFACE

VUnplacedVolume* UnplacedPolyhedron::CopyToGpu(
    VUnplacedVolume *const gpuPtr) const {
  int gpuMemorySize = 0;
  void *zSegmentsGpu =
      UnplacedPolyhedron_AllocateZSegments(fZSegments.size(), gpuMemorySize);
  for (int i = 0, iMax = fZSegments.size(); i < iMax; ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
    fZSegments[i].CopyToGpu(zSegmentsGpu + i*gpuMemorySize);
#pragma GCC diagnostic pop
  }
  // TODO: no one has responsibility for cleaning this up!! This has to be
  //       delegated rather urgently so memory at least isn't deliberately
  //       leaked...
  size_t planeBytes = fZPlanes.size()*sizeof(Precision);
  Precision *zPlanesGpu = AllocateOnGpu<Precision>(planeBytes);
  vecgeom::CopyToGpu(&fZPlanes[0], zPlanesGpu, planeBytes);
  size_t phiSectionBytes = fPhiSections.size()*sizeof(Precision);
  Precision *phiSectionsXGpu = AllocateOnGpu<Precision>(phiSectionBytes);
  Precision *phiSectionsYGpu = AllocateOnGpu<Precision>(phiSectionBytes);
  Precision *phiSectionsZGpu = AllocateOnGpu<Precision>(phiSectionBytes);
  vecgeom::CopyToGpu(fPhiSections.x(), phiSectionsXGpu, phiSectionBytes);
  vecgeom::CopyToGpu(fPhiSections.y(), phiSectionsYGpu, phiSectionBytes);
  vecgeom::CopyToGpu(fPhiSections.z(), phiSectionsZGpu, phiSectionBytes);
  VUnplacedVolume *boundingTubeGpu = fBoundingTube.CopyToGpu();
  UnplacedPolyhedron_CopyToGpu(gpuPtr, fSideCount, fHasInnerRadii,
                               fHasPhiCutout, fHasLargePhiCutout,
                               zSegmentsGpu, zPlanesGpu, fZPlanes.size(),
                               phiSectionsXGpu, phiSectionsYGpu,
                               phiSectionsZGpu, boundingTubeGpu,
                               fBoundingTubeOffset);
  vecgeom::CudaAssertError();
  return gpuPtr;
}

VUnplacedVolume* UnplacedPolyhedron::CopyToGpu() const {
  VUnplacedVolume *const gpuPtr = vecgeom::AllocateOnGpu<UnplacedPolyhedron>();
  return this->CopyToGpu(gpuPtr);
}

void UnplacedPolyhedron::ZSegment::CopyToGpu(void *gpuPtr) const {
  void *outerGpu, *innerGpu, *phiGpu;
  bool *innerRadiusGpu;
  UnplacedPolyhedron_ZSegment_GetAddresses(gpuPtr, outerGpu, innerGpu, phiGpu,
                                           innerRadiusGpu);
  outer.CopyToGpu(outerGpu);
  if (inner.size() > 0) inner.CopyToGpu(innerGpu);
  if (phi.size() > 0)   phi.CopyToGpu(phiGpu);
  vecgeom::CopyToGpu<bool>(&hasInnerRadius, innerRadiusGpu);
}

#endif

#ifdef VECGEOM_NVCC

void *UnplacedPolyhedron_AllocateZSegments(int size, int &gpuMemorySize) {
  gpuMemorySize = sizeof(vecgeom_cuda::UnplacedPolyhedron::ZSegment);
  return static_cast<void*>(
      AllocateOnGpu<vecgeom_cuda::UnplacedPolyhedron::ZSegment>(
          gpuMemorySize*size));
}

void UnplacedPolyhedron_ZSegment_GetAddresses(void *object, void *&outer,
                                              void *&inner, void *&phi,
                                              bool *&innerRadius) {
  vecgeom_cuda::UnplacedPolyhedron::ZSegment *segment =
      static_cast<vecgeom_cuda::UnplacedPolyhedron::ZSegment*>(object);
  outer = &segment->outer;
  inner = &segment->inner;
  phi   = &segment->phi;
  innerRadius = &segment->hasInnerRadius;
}

__global__
void UnplacedPolyhedron_ConstructOnGpu(
    VUnplacedVolume *gpuPtr, int sideCount, bool hasInnerRadii,
    bool hasPhiCutout, bool hasLargePhiCutout, void *zSegments,
    Precision *zPlanes, int zPlaneCount, Precision *phiSectionsX,
    Precision *phiSectionsY, Precision *phiSectionsZ,
    vecgeom_cuda::UnplacedTube *boundingTube, Precision boundingTubeOffset) {
  new (gpuPtr) vecgeom_cuda::UnplacedPolyhedron(
      sideCount, hasInnerRadii, hasPhiCutout, hasLargePhiCutout,
      static_cast<vecgeom_cuda::UnplacedPolyhedron::ZSegment*>(zSegments),
      zPlanes, zPlaneCount, phiSectionsX, phiSectionsY, phiSectionsZ,
      *boundingTube, boundingTubeOffset);
}

void UnplacedPolyhedron_CopyToGpu(VUnplacedVolume *gpuPtr, int sideCount,
                                  bool hasInnerRadii, bool hasPhiCutout,
                                  bool hasLargePhiCutout, void *zSegments,
                                  Precision *zPlanes, int zPlaneCount,
                                  Precision *phiSectionsX,
                                  Precision *phiSectionsY,
                                  Precision *phiSectionsZ,
                                  VUnplacedVolume *boundingTube,
                                  Precision boundingTubeOffset) {
  UnplacedPolyhedron_ConstructOnGpu<<<1, 1>>>(
      gpuPtr, sideCount, hasInnerRadii, hasPhiCutout, hasLargePhiCutout,
      zSegments, zPlanes, zPlaneCount, phiSectionsX, phiSectionsY, phiSectionsZ,
      reinterpret_cast<vecgeom_cuda::UnplacedTube*>(boundingTube),
      boundingTubeOffset);
}

#endif

} // End namespace vecgeom
