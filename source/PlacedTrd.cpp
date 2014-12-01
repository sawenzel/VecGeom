/// @file PlacedTrd.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/Trd.h"

#ifndef VECGEOM_NVCC

#ifdef VECGEOM_ROOT
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UTrd.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Trd.hh"
#endif

#endif // VECGEOM_NVCC

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedTrd::ConvertToUnspecialized() const {
  return new SimpleTrd(GetLabel().c_str(), logical_volume(), transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedTrd::ConvertToRoot() const {
  if(dy1() == dy2())
    return new TGeoTrd1(GetLabel().c_str(), dx1(), dx2(), dy1(), dz());
  return new TGeoTrd2(GetLabel().c_str(), dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedTrd::ConvertToUSolids() const {
  return new UTrd("", dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedTrd::ConvertToGeant4() const {
  return new G4Trd(GetLabel(), dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#endif // VECGEOM_NVCC

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VPlacedVolume> PlacedTrd::CopyToGpu(
   DevicePtr<cuda::LogicalVolume> const logical_volume,
   DevicePtr<cuda::Transformation3D> const transform,
   DevicePtr<cuda::VPlacedVolume> const gpu_ptr) const
{
   return CopyToGpuImpl<PlacedTrd>(logical_volume, transform, gpu_ptr);
}

DevicePtr<cuda::VPlacedVolume> PlacedTrd::CopyToGpu(
      DevicePtr<cuda::LogicalVolume> const logical_volume,
      DevicePtr<cuda::Transformation3D> const transform) const
{
   return CopyToGpuImpl<PlacedTrd>(logical_volume, transform);
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::PlacedTrd>::SizeOf();
template void DevicePtr<cuda::PlacedTrd>::Construct(
   DevicePtr<cuda::LogicalVolume> const logical_volume,
   DevicePtr<cuda::Transformation3D> const transform,
   const int id) const;

} // End cxx namespace

#endif // VECGEOM_NVCC

} // End global namespace

