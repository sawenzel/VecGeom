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

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void PlacedBox::PrintType() const {
  printf("PlacedBox");
}

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

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedBox_CopyToGpu(LogicalVolume const *const logical_volume,
                         Transformation3D const *const transformation,
                         const int id,
                         VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedBox::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  vecgeom::PlacedBox_CopyToGpu(logical_volume, transformation, this->id(),
                               gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedBox::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedBox>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedBox_ConstructOnGpu(LogicalVolume const *const logical_volume,
                              Transformation3D const *const transformation,
                              const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleBox(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    id
  );
}

void PlacedBox_CopyToGpu(LogicalVolume const *const logical_volume,
                         Transformation3D const *const transformation,
                         const int id,
                         VPlacedVolume *const gpu_ptr) {
  PlacedBox_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation, id,
                                     gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom
