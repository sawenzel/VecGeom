/// @file PlacedTrd.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/Trd.h"

#ifndef VECGEOM_NVCC

#ifdef VECGEOM_ROOT
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UTrd.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Trd.hh"
#endif

#endif // VECGEOM_NVCC

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedTrd::ConvertToUnspecialized() const {
  return new SimpleTrd(GetLabel().c_str(), logical_volume(), transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTrd::ConvertToRoot() const {
  if(dy1() == dy2())
    return new TGeoTrd1(GetLabel().c_str(), dx1(), dx2(), dy1(), dz());
  return new TGeoTrd2(GetLabel().c_str(), dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTrd::ConvertToUSolids() const {
  return new UTrd("", dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedTrd::ConvertToGeant4() const {
  return new G4Trd(GetLabel(), dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#endif // VECGEOM_NVCC

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedTrd_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedTrd::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedTrd_CopyToGpu(logical_volume, transformation, this->id(),
                                 gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedTrd::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedTrd>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedTrd_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleTrd(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedTrd_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedTrd_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                                id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} } // End global namespace

