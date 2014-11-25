/// \file PlacedTorus.cpp

#include "volumes/PlacedTorus.h"
#include "volumes/Torus.h"
#include "volumes/SpecializedTorus.h"

#ifdef VECGEOM_ROOT
#include "TGeoTorus.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Torus.hh"
#endif

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedTorus::ConvertToUnspecialized() const {
  return new SimpleTorus(GetLabel().c_str(), logical_volume(), transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTorus::ConvertToRoot() const {
  return new TGeoTorus(GetLabel().c_str(), rtor(), rmin(), rmax(),
          sphi()*kRadToDeg, dphi()*kRadToDeg);
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTorus::ConvertToUSolids() const {
    return NULL;
    //  return new UTubs(GetLabel().c_str(), rmin(), rmax(), z(), sphi(), dphi());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedTorus::ConvertToGeant4() const {
  return new G4Torus(GetLabel().c_str(), rmin(), rmax(), rtor(), sphi(), dphi());
}
#endif

#endif // VECGEOM_BENCHMARK

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedTorus_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedTorus::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedTorus_CopyToGpu(logical_volume, transformation, this->id(),
                                 gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedTorus::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedTorus>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedTorus_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleTorus(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedTorus_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedTorus_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                                id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom

