/// \file PlacedBox.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedBox.h"

#include "base/AOS3D.h"
#include "base/SOA3D.h"
#include "volumes/SpecializedBox.h"
#ifdef VECGEOM_ROOT
#include "TGeoBBox.h"
#endif
#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Box.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void PlacedBox::PrintType() const {
  printf("PlacedBox");
}

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedBox::ConvertToUnspecialized() const {
  return new SimpleBox(GetLabel().c_str(), logical_volume_, transformation_);
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedBox::ConvertToRoot() const {
  return new TGeoBBox(GetLabel().c_str(), x(), y(), z());
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedBox::ConvertToUSolids() const {
  return new UBox(GetLabel(), x(), y(), z());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedBox::ConvertToGeant4() const {
  return new G4Box(GetLabel(), x(), y(), z());
}
#endif

#endif // VECGEOM_NVCC

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VPlacedVolume> PlacedBox::CopyToGpu(
   DevicePtr<cuda::LogicalVolume> const logical_volume,
   DevicePtr<cuda::Transformation3D> const transform,
   DevicePtr<cuda::VPlacedVolume> const gpu_ptr) const
{
   return CopyToGpuImpl<PlacedBox>(logical_volume, transform, gpu_ptr);
}

DevicePtr<cuda::VPlacedVolume> PlacedBox::CopyToGpu(
      DevicePtr<cuda::LogicalVolume> const logical_volume,
      DevicePtr<cuda::Transformation3D> const transform) const
{
   return CopyToGpuImpl<PlacedBox>(logical_volume, transform);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

template void DevicePtr<cuda::PlacedBox>::SizeOf();
template void DevicePtr<cuda::PlacedBox>::Construct(
   DevicePtr<cuda::LogicalVolume> const logical_volume,
   DevicePtr<cuda::Transformation3D> const transform,
   const int id);

#endif // VECGEOM_NVCC

} } // End namespace vecgeom
