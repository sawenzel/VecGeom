/// \file PlacedRootVolume.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "PlacedRootVolume.h"
#include "TGeoBBox.h"
#include "VecGeom/base/SOA3D.h"

namespace vecgeom {

PlacedRootVolume::PlacedRootVolume(char const *const label, TGeoShape const *const rootShape,
                                   LogicalVolume const *const logicalVolume,
                                   Transformation3D const *const transformation)
    : VPlacedVolume(label, logicalVolume, transformation)
{
}

PlacedRootVolume::PlacedRootVolume(TGeoShape const *const rootShape, LogicalVolume const *const logicalVolume,
                                   Transformation3D const *const transformation)
    : PlacedRootVolume(rootShape->GetName(), rootShape, logicalVolume, transformation)
{
}

void PlacedRootVolume::PrintType() const
{
  printf("PlacedRootVolume");
}

void PlacedRootVolume::PrintType(std::ostream &os) const
{
  os << "PlacedRootVolume";
}

Precision PlacedRootVolume::Capacity()
{
  return GetRootShape()->Capacity();
}

void PlacedRootVolume::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  TGeoBBox const *b = dynamic_cast<TGeoBBox const *>(GetRootShape());
  VECGEOM_ASSERT(b != nullptr);
  auto lx = b->GetDX();
  auto ly = b->GetDY();
  auto lz = b->GetDZ();
  auto o  = b->GetOrigin();
  aMin.Set(o[0] - lx, o[1] - ly, o[2] - lz);
  aMax.Set(o[0] + lx, o[1] + ly, o[2] + lz);
}

VPlacedVolume const *PlacedRootVolume::ConvertToUnspecialized() const
{
  VECGEOM_VALIDATE(0, << "Attempted to perform conversion on unsupported ROOT volume.");
  return NULL;
}

#ifdef VECGEOM_CUDA_INTERFACE
DevicePtr<cuda::VPlacedVolume> PlacedRootVolume::CopyToGpu(DevicePtr<cuda::LogicalVolume> const /* logical_volume */,
                                                           DevicePtr<cuda::Transformation3D> const /* transform */,
                                                           DevicePtr<cuda::VPlacedVolume> const /* in_gpu_ptr */) const
{
  VECGEOM_VALIDATE(0, << "Attempted to copy unsupported ROOT volume to GPU.");
  return DevicePtr<cuda::VPlacedVolume>(nullptr);
}
DevicePtr<cuda::VPlacedVolume> PlacedRootVolume::CopyToGpu(
    DevicePtr<cuda::LogicalVolume> const /*logical_volume*/,
    DevicePtr<cuda::Transformation3D> const /* transform */) const
{
  VECGEOM_VALIDATE(0, << "Attempted to copy unsupported ROOT volume to GPU.");
  return DevicePtr<cuda::VPlacedVolume>(nullptr);
}
#endif

} // End namespace vecgeom
