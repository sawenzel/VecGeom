/// @file PlacedOrb.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/PlacedOrb.h"
#include "volumes/Orb.h"
#include "base/AOS3D.h"
#include "base/SOA3D.h"

#ifdef VECGEOM_USOLIDS
#include "UOrb.hh"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoSphere.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Orb.hh"
#endif

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedOrb::ConvertToUnspecialized() const {
  return new SimpleOrb(GetLabel().c_str(), logical_volume(),
                                  transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedOrb::ConvertToRoot() const {
  return new TGeoSphere(GetLabel().c_str(),0,GetRadius());
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedOrb::ConvertToUSolids() const {

return new UOrb(GetLabel().c_str(),GetRadius());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedOrb::ConvertToGeant4() const {
return new G4Orb(GetLabel().c_str(),GetRadius());
}
#endif

#endif // VECGEOM_NVCC

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedOrb_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedOrb::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  vecgeom::PlacedOrb_CopyToGpu(logical_volume, transformation, this->id(),
                                 gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedOrb::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedOrb>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedOrb_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleOrb(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedOrb_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedOrb_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                                id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom
