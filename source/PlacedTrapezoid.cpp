/// \file PlacedTrapezoid.cpp

#include "volumes/PlacedTrapezoid.h"

#include "volumes/Trapezoid.h"

#include <cassert>

namespace VECGEOM_NAMESPACE {

VPlacedVolume const* PlacedTrapezoid::ConvertToUnspecialized() const {
  assert(0 && "NYI");
  return NULL;
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTrapezoid::ConvertToRoot() const {
  assert(0 && "NYI");
  return NULL;
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTrapezoid::ConvertToUSolids() const {
  assert(0 && "NYI");
  return NULL;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedTrapezoid::ConvertToGeant4() const {
  assert(0 && "NYI");
  return NULL;
}
#endif

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedTrapezoid_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedTrapezoid::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedTrapezoid_CopyToGpu(logical_volume, transformation, this->id(),
                             gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedTrapezoid::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedTrapezoid>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedTrapezoid_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleTrapezoid(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedTrapezoid_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedTrapezoid_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                            id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom