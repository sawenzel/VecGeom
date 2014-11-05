/*
 * PlacedCone.cpp
 *
 *  Created on: Jun 13, 2014
 *      Author: swenzel
 */

#include "volumes/PlacedCone.h"
#include "volumes/Cone.h"
#include "volumes/SpecializedCone.h"

#if defined(VECGEOM_ROOT)
#include "TGeoCone.h"
#endif

#if defined(VECGEOM_USOLIDS)
#include "UCons.hh"
#endif

#if defined(VECGEOM_GEANT4)
#include "G4Cons.hh"
#endif


namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC
VPlacedVolume const* PlacedCone::ConvertToUnspecialized() const {
    return new SimpleCone(GetLabel().c_str(), logical_volume(), transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedCone::ConvertToRoot() const {
    if( GetDPhi() == 2.*M_PI )
    {
       return new TGeoCone("RootCone",GetDz(),GetRmin1(),GetRmax1(), GetRmin2(), GetRmax2());
    }
    else
    {
       return new TGeoConeSeg("RootCone", GetDz(),GetRmin1(),GetRmax1(), GetRmin2(), GetRmax2(),
               GetSPhi()*(180/M_PI), GetSPhi()+180*GetDPhi()/(M_PI) );
    }
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedCone::ConvertToUSolids() const {
  return new UCons("USolidCone", GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2(), GetDz(), GetSPhi(), GetDPhi());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const * PlacedCone::ConvertToGeant4() const {
  return new G4Cons("Geant4Cone", GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2(), GetDz(), GetSPhi(), GetDPhi());
}
#endif

#endif

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedCone_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedCone::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedCone_CopyToGpu(logical_volume, transformation, this->id(),
                                 gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedCone::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedCone>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedCone_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleCone(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedCone_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedCone_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                                id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom



