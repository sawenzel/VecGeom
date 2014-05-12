/// @file PlacedParaboloid.cpp

#include "volumes/PlacedParaboloid.h"

#include "volumes/Paraboloid.h"

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedParaboloid::ConvertToUnspecialized() const {
  assert(0 && "NYI");
  return NULL;
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedParaboloid::ConvertToRoot() const {
  assert(0 && "NYI");
  return NULL;
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedParaboloid::ConvertToUSolids() const {
  assert(0 && "NYI");
  return NULL;
}
#endif

#endif // VECGEOM_BENCHMARK

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedParaboloid_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedParaboloid::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedParaboloid_CopyToGpu(logical_volume, transformation, this->id(),
                             gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedParaboloid::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedParaboloid>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedParaboloid_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleParaboloid(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedParaboloid_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedParaboloid_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                            id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom