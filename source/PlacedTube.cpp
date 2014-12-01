/// \file PlacedTube.cpp
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/PlacedTube.h"
#include "volumes/Tube.h"
#include "volumes/SpecializedTube.h"

#ifdef VECGEOM_ROOT
#include "TGeoTube.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UTubs.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Tubs.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedTube::ConvertToUnspecialized() const {
  return new SimpleTube(GetLabel().c_str(), logical_volume(), transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTube::ConvertToRoot() const {
  if(dphi() >= 2*M_PI)
     return new TGeoTube(GetLabel().c_str(), rmin(), rmax(), z());
  return new TGeoTubeSeg(GetLabel().c_str(), rmin(), rmax(), z(), sphi()*(180/M_PI), sphi()*(180/M_PI)+dphi()*(180/M_PI) );
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTube::ConvertToUSolids() const {
  return new UTubs(GetLabel().c_str(), rmin(), rmax(), z(), sphi(), dphi());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedTube::ConvertToGeant4() const {
  return new G4Tubs(GetLabel().c_str(), rmin(), rmax(), z(), sphi(), dphi());
}
#endif

#endif // VECGEOM_NVCC

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VPlacedVolume> PlacedTube::CopyToGpu(
   DevicePtr<cuda::LogicalVolume> const logical_volume,
   DevicePtr<cuda::Transformation3D> const transform,
   DevicePtr<cuda::VPlacedVolume> const gpu_ptr) const
{
   return CopyToGpuImpl<PlacedTube>(logical_volume, transform, gpu_ptr);
}

DevicePtr<cuda::VPlacedVolume> PlacedTube::CopyToGpu(
      DevicePtr<cuda::LogicalVolume> const logical_volume,
      DevicePtr<cuda::Transformation3D> const transform) const
{
   return CopyToGpuImpl<PlacedTube>(logical_volume, transform);
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::PlacedTube>::SizeOf();
template void DevicePtr<cuda::PlacedTube>::Construct(
   DevicePtr<cuda::LogicalVolume> const logical_volume,
   DevicePtr<cuda::Transformation3D> const transform,
   const int id) const;

} // End cxx namespace

#endif // VECGEOM_NVCC

} // End global namespace

