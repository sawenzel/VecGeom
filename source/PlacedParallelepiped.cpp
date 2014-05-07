/// @file PlacedParallelepiped.cpp
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedParallelepiped.h"

#include "volumes/Parallelepiped.h"

#if defined(VECGEOM_BENCHMARK) && defined(VECGEOM_ROOT)
#include "TGeoPara.h"
#endif

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedParallelepiped::ConvertToUnspecialized() const {
  return new SimpleParallelepiped(label().c_str(), logical_volume(),
                                  transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedParallelepiped::ConvertToRoot() const {
  return new TGeoPara(label().c_str(), GetX(), GetY(), GetZ(), GetAlpha(),
                      GetTheta(), GetPhi());
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedParallelepiped::ConvertToUSolids() const {
  assert(0 && "Parallelepiped unsupported for USolids.");
}
#endif

#endif // VECGEOM_BENCHMARK

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedParallelepiped_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedParallelepiped::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedParallelepiped_CopyToGpu(logical_volume, transformation, this->id(),
                                 gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedParallelepiped::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedParallelepiped>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedParallelepiped_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleParallelepiped(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedParallelepiped_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedParallelepiped_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                                id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom