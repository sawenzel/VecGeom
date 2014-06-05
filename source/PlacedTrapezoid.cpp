/// @file PlacedTrapezoid.cpp
/// @author Guilherme Lima (lima at fnal dot gov)

#include "volumes/PlacedTrapezoid.h"
#include "volumes/Trapezoid.h"

#if defined(VECGEOM_BENCHMARK) && defined(VECGEOM_ROOT)
#include "TGeoArb8.h"
#endif
#ifdef VECGEOM_USOLIDS
#include "UTrap.hh"
#endif

namespace VECGEOM_NAMESPACE {

PlacedTrapezoid::~PlacedTrapezoid() {}

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedTrapezoid::ConvertToUnspecialized() const {
  return new SimpleTrapezoid(label().c_str(), logical_volume(), transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTrapezoid::ConvertToRoot() const {
  return new TGeoTrap( label().c_str(), GetDz(), GetTheta()*kRadToDeg, GetPhi()*kRadToDeg,
                       GetDy1(), GetDx1(), GetDx2(), GetTanAlpha1(),
                       GetDy2(), GetDx3(), GetDx4(), GetTanAlpha2() );
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTrapezoid::ConvertToUSolids() const {
  return new ::UTrap(label().c_str(), GetDz(), GetTheta(), GetPhi(),
                     GetDy1(), GetDx1(), GetDx2(), GetAlpha1(),
                     GetDy2(), GetDx3(), GetDx4(), GetAlpha2());
}
#endif

#endif // VECGEOM_BENCHMARK

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
