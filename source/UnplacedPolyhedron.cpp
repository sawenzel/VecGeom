/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/UnplacedPolyhedron.h"

#include "volumes/PlacedPolyhedron.h"
#include "volumes/SpecializedPolyhedron.h"

#include <cmath>
#include <memory>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

using namespace vecgeom::Polyhedron;


UnplacedPolyhedron::UnplacedPolyhedron(
    const int sideCount,
    const int zPlaneCount,
    Precision zPlanes[],
    Precision rMin[],
    Precision rMax[])
    : UnplacedPolyhedron(0, 360, sideCount, zPlaneCount, zPlanes, rMin, rMax) {}

VECGEOM_CUDA_HEADER_BOTH
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
      fPhiStart(phiStart),fPhiDelta(phiDelta),
      fZSegments(zPlaneCount-1), fZPlanes(zPlaneCount), fRMin(zPlaneCount),
      fRMax(zPlaneCount), fPhiSections(sideCount+1),
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
    new (&fZSegments[i].outer) Quadrilaterals(sideCount);
    if (fHasPhiCutout) {
      new (&fZSegments[i].phi) Quadrilaterals(2);
    }
    if (fZSegments[i].hasInnerRadius) {
      new (&fZSegments[i].inner) Quadrilaterals(sideCount);
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
  vecgeom::unique_ptr<Precision[]> vertixPhi(new Precision[sideCount+1]);
  for (int i = 0, iMax = sideCount+1; i < iMax; ++i) {
    vertixPhi[i] = NormalizeAngle<kScalar>(phiStart + i*sidePhi);
    Vector3D<Precision> cornerVector =
        Vec_t::FromCylindrical(1., vertixPhi[i], 0).Normalized().FixZeroes();
    fPhiSections.set(
        i, cornerVector.Normalized().Cross(Vector3D<Precision>(0, 0, -1)));
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
  vecgeom::unique_ptr<Vec_t[]> outerVertices(new Vec_t[nVertices]);
  vecgeom::unique_ptr<Vec_t[]> innerVertices(new Vec_t[nVertices]);
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
      // Make sure normal points backwards along phi
      if (fZSegments[iPlane].phi.GetNormal(0).Dot(fPhiSections[0]) > 0) {
        fZSegments[iPlane].phi.FlipSign(0);
      }
      fZSegments[iPlane].phi.Set(
          1,
          outerVertices[VertixIndex(iPlane, sideCount)],
          outerVertices[VertixIndex(iPlane+1, sideCount)],
          innerVertices[VertixIndex(iPlane+1, sideCount)],
          innerVertices[VertixIndex(iPlane, sideCount)]);
      // Make sure normal points forwards along phi
      if (fZSegments[iPlane].phi.GetNormal(1).Dot(fPhiSections[fSideCount]) < 0) {
        fZSegments[iPlane].phi.FlipSign(1);
      }
    }

  } // End loop over segments
} // end constructor

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedPolyhedron::GetPhiStart() const {
  return kRadToDeg*NormalizeAngle<kScalar>(
      fPhiSections[0].Cross(Vector3D<Precision>(0, 0, 1)).Phi());
}

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedPolyhedron::GetPhiEnd() const {
  return !HasPhiCutout() ? 360 : kRadToDeg*NormalizeAngle<kScalar>(
      fPhiSections[GetSideCount()].Cross(
          Vector3D<Precision>(0, 0, 1)).Phi());
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

  EInnerRadii innerRadii = unplaced->HasInnerRadii() ? EInnerRadii::kTrue
                                                     : EInnerRadii::kFalse;
  EPhiCutout phiCutout = unplaced->HasPhiCutout() ?
                      (unplaced->HasLargePhiCutout() ? EPhiCutout::kLarge
                                                     : EPhiCutout::kTrue)
                                                     : EPhiCutout::kFalse;

#ifndef VECGEOM_NVCC
  #define POLYHEDRON_CREATE_SPECIALIZATION(INNER, PHI) \
  if (innerRadii == INNER && phiCutout == PHI) { \
    if (placement) { \
      return new(placement) \
             SpecializedPolyhedron<INNER, PHI>(volume, transformation); \
    } else { \
      return new SpecializedPolyhedron<INNER, PHI>(volume, transformation); \
    } \
  }
#else
  #define POLYHEDRON_CREATE_SPECIALIZATION(INNER, PHI) \
  if (innerRadii == INNER && phiCutout == PHI) { \
    if (placement) { \
      return new(placement) \
             SpecializedPolyhedron<INNER, PHI>(volume, transformation, id); \
    } else { \
      return new \
             SpecializedPolyhedron<INNER, PHI>(volume, transformation, id); \
    } \
  }
#endif

#ifndef VECGEOM_NO_SPECIALIZATION

  POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kTrue, EPhiCutout::kFalse);
  POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kTrue, EPhiCutout::kTrue);
  POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kTrue, EPhiCutout::kLarge);
  POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kFalse, EPhiCutout::kFalse);
  POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kFalse, EPhiCutout::kTrue);
  POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kFalse, EPhiCutout::kLarge);

#endif

#ifndef VECGEOM_NVCC
  if (placement) {
    return new(placement)
           SpecializedPolyhedron<EInnerRadii::kGeneric, EPhiCutout::kGeneric>(
               volume, transformation);
  } else {
    return new SpecializedPolyhedron<EInnerRadii::kGeneric, EPhiCutout::kGeneric>(
        volume, transformation);
  }
#else
  if (placement) {
    return new(placement)
           SpecializedPolyhedron<EInnerRadii::kGeneric, EPhiCutout::kGeneric>(
               volume, transformation, id);
  } else {
    return new SpecializedPolyhedron<EInnerRadii::kGeneric, EPhiCutout::kGeneric>(
        volume, transformation, id);
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


#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedPolyhedron::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpuPtr) const {
    // idea: reconstruct defining arrays: copy them to GPU; then construct the UnplacedPolycon object from scratch
      // on the GPU


  Precision *z_gpu_ptr = AllocateOnGpu<Precision>( fZPlanes.size()*sizeof(Precision) );
  Precision *rmin_gpu_ptr = AllocateOnGpu<Precision>( fRMin.size()*sizeof(Precision) );
  Precision *rmax_gpu_ptr = AllocateOnGpu<Precision>( fRMax.size()*sizeof(Precision) );

  vecgeom::CopyToGpu(&fZPlanes[0], z_gpu_ptr, sizeof(Precision)*fZPlanes.size());
  vecgeom::CopyToGpu(&fRMin[0], rmin_gpu_ptr, sizeof(Precision)*fRMin.size());
  vecgeom::CopyToGpu(&fRMax[0], rmax_gpu_ptr, sizeof(Precision)*fRMax.size());

      DevicePtr<cuda::VUnplacedVolume> gpupolyhedra =  CopyToGpuImpl<UnplacedPolyhedron>(gpuPtr,
              fPhiStart, fPhiDelta, fSideCount, fZPlanes.size(), z_gpu_ptr, rmin_gpu_ptr, rmax_gpu_ptr);

  // remove temporary space from GPU : dangerous as the polyhedra keeps pointer to these
  FreeFromGpu(z_gpu_ptr);
  FreeFromGpu(rmin_gpu_ptr);
  FreeFromGpu(rmax_gpu_ptr);
  CudaAssertError();

 return gpupolyhedra;

 /* Philipps more elegant solution does not work yet: DevicePtr<double>::SizeOf undefined symbol
      DevicePtr<Precision> zPlanesGpu;
         zPlanesGpu.Allocate(fZPlanes.size());
         zPlanesGpu.ToDevice(&fZPlanes[0], fZPlanes.size());

         DevicePtr<Precision> rminGpu;
         rminGpu.Allocate(fZPlanes.size());
         rminGpu.ToDevice(&fRMin[0], fZPlanes.size());

         DevicePtr<Precision> rmaxGpu;
         rmaxGpu.Allocate(fZPlanes.size());
         rmaxGpu.ToDevice(&fRMax[0], fZPlanes.size());

         DevicePtr<cuda::VUnplacedVolume> result =
            CopyToGpuImpl<UnplacedPolyhedron>(gpuPtr,
                                              fPhiStart, fPhiDelta,
                                              fSideCount, fZPlanes.size(),
                                              zPlanesGpu, rminGpu, rmaxGpu
                                              );

        zPlanesGpu.Deallocate();
        rminGpu.Deallocate();
        rminGpu.Deallocate();

        CudaAssertError();
        return result;
*/

}

DevicePtr<cuda::VUnplacedVolume> UnplacedPolyhedron::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedPolyhedron>();
}

#endif


} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedPolyhedron>::SizeOf();
//template void DevicePtr<cuda::UnplacedPolyhedron>::Construct(Precision phiStart,
//                                 Precision phiDelta,
//                                 int sideCount,
//                                 int zPlaneCount,
//                                 DevicePtr<Precision> zPlanes,
//                                 DevicePtr<Precision> rMin,
//                                 DevicePtr<Precision> rMax) const;
//
//} // End cxx namespace
template void DevicePtr<cuda::UnplacedPolyhedron>::Construct(Precision phiStart,
                                 Precision phiDelta,
                                 int sideCount,
                                 int zPlaneCount,
                                 Precision zPlanes[],
                                 Precision rMin[],
                                 Precision rMax[]) const;

} // End cxx namespace


#endif
 
} // End namespace vecgeom
