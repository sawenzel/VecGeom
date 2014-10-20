/// \file PlacedParaboloid.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/PlacedParaboloid.h"

#include "volumes/Paraboloid.h"

#ifndef VECGEOM_NVCC

#ifdef VECGEOM_ROOT
#include "TGeoParaboloid.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Paraboloid.hh"
#endif

#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#endif

#endif // VECGEOM_NVCC

#include <cassert>

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedParaboloid::ConvertToUnspecialized() const {
    return new SimpleParaboloid(GetLabel().c_str(), logical_volume(),transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedParaboloid::ConvertToRoot() const {
    return new TGeoParaboloid(GetLabel().c_str(), GetRlo(), GetRhi(), GetDz());
    
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedParaboloid::ConvertToUSolids() const {
    std::cerr << "**************************************************************\n";
    std::cerr << "WARNING: Paraboloid unsupported for USolids.; returning a box\n";
    std::cerr << "**************************************************************\n";
    return new UBox("",10,10,10);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedParaboloid::ConvertToGeant4() const {
    return new G4Paraboloid(GetLabel(), GetDz(), GetRlo(), GetRhi());
}
#endif

#endif // VECGEOM_NVCC

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
