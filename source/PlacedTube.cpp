/// @file PlacedTube.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/PlacedTube.h"
#include "volumes/Tube.h"
#include "volumes/SpecializedTube.h"

#if defined(VECGEOM_BENCHMARK) && defined(VECGEOM_ROOT)
#include "TGeoTube.h"
#endif

#if defined(VECGEOM_BENCHMARK) && defined(VECGEOM_USOLIDS)
#include "UTubs.hh"
#endif


namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedTube::ConvertToUnspecialized() const {
  return new SimpleTube(label().c_str(), logical_volume(), transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTube::ConvertToRoot() const {
  return new TGeoTubeSeg(label().c_str(), rmin(), rmax(), z(), sphi()*(180/M_PI), dphi()*(180/M_PI) );
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTube::ConvertToUSolids() const {
  return new UTubs("", rmin(), rmax(), z(), sphi(), dphi());
}
#endif

#endif // VECGEOM_BENCHMARK

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedTube_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedTube::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedTube_CopyToGpu(logical_volume, transformation, this->id(),
                                 gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedTube::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedTube>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedTube_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleTube(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedTube_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedTube_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                                id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom

