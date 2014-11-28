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

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

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

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VPlacedVolume> PlacedOrb::CopyToGpu(
   DevicePtr<cuda::LogicalVolume> const logical_volume,
   DevicePtr<cuda::Transformation3D> const transform,
   DevicePtr<cuda::VPlacedVolume> const in_gpu_ptr) const
{
   DevicePtr<cuda::PlacedOrb> gpu_ptr(in_gpu_ptr);
   gpu_ptr.Construct(logical_volume, transform, nullptr, this->id());
   CudaAssertError();
   return DevicePtr<cuda::VPlacedVolume>(gpu_ptr);
}

DevicePtr<cuda::VPlacedVolume> PlacedOrb::CopyToGpu(
      DevicePtr<cuda::LogicalVolume> const logical_volume,
      DevicePtr<cuda::Transformation3D> const transform) const
{
   DevicePtr<cuda::PlacedOrb> gpu_ptr;
   gpu_ptr.Allocate();
   return this->CopyToGpu(logical_volume,transform,DevicePtr<cuda::VPlacedVolume>(gpu_ptr));
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

template void DevicePtr<cuda::PlacedOrb>::SizeOf();
template void DevicePtr<cuda::PlacedOrb>::Construct(
   DevicePtr<cuda::LogicalVolume> const logical_volume,
   DevicePtr<cuda::Transformation3D> const transform,
   const int id);

#endif // VECGEOM_NVCC

} } // End global namespace
