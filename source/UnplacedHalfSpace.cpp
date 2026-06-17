#include "VecGeom/volumes/UnplacedHalfSpace.h"

#include "VecGeom/base/Assert.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/SpecializedHalfSpace.h"

#ifdef VECGEOM_ROOT
#include "TGeoHalfSpace.h"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
UnplacedHalfSpace::UnplacedHalfSpace(Vector3D<Precision> const &point, Vector3D<Precision> const &normal) : fHalfSpace()
{
  fHalfSpace       = HalfSpaceStruct<Precision>(point, normal);
  fGlobalConvexity = true;
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
void UnplacedHalfSpace::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  aMin.Set(-kInfLength);
  aMax.Set(kInfLength);
}

Vector3D<Precision> UnplacedHalfSpace::SamplePointOnSurface() const { return fHalfSpace.fPoint; }

VECCORE_ATT_HOST_DEVICE
void UnplacedHalfSpace::Print() const
{
  printf("UnplacedHalfSpace {point=(%.2f, %.2f, %.2f), normal=(%.2f, %.2f, %.2f)}", GetPoint().x(), GetPoint().y(),
         GetPoint().z(), GetNormal().x(), GetNormal().y(), GetNormal().z());
}

void UnplacedHalfSpace::Print(std::ostream &os) const
{
  os << "UnplacedHalfSpace {point=" << GetPoint() << ", normal=" << GetNormal() << "}";
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedHalfSpace::CreateMesh3D(Transformation3D const &, size_t) const { return new SolidMesh(); }

#ifdef VECGEOM_ROOT
TGeoShape const *UnplacedHalfSpace::ConvertToRoot(char const *label) const
{
  double point[3]  = {GetPoint().x(), GetPoint().y(), GetPoint().z()};
  double normal[3] = {GetNormal().x(), GetNormal().y(), GetNormal().z()};
  return new TGeoHalfSpace(label, point, normal);
}
#endif

VPlacedVolume *UnplacedHalfSpace::Create(LogicalVolume const *const logical_volume,
                                         Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedHalfSpace(logical_volume, transformation);
    return placement;
  }
  return new SpecializedHalfSpace(logical_volume, transformation);
}

VPlacedVolume *UnplacedHalfSpace::SpecializedVolume(LogicalVolume const *const volume,
                                                    Transformation3D const *const transformation,
                                                    VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedHalfSpace>(volume, transformation, placement);
}
#else

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedHalfSpace::Create(LogicalVolume const *const logical_volume,
                                         Transformation3D const *const transformation, const int id, const int copy_no,
                                         const int child_id, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedHalfSpace(logical_volume, transformation, id, copy_no, child_id);
    return placement;
  }
  return new SpecializedHalfSpace(logical_volume, transformation, id, copy_no, child_id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedHalfSpace::SpecializedVolume(LogicalVolume const *const volume,
                                                    Transformation3D const *const transformation, const int id,
                                                    const int copy_no, const int child_id,
                                                    VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedHalfSpace>(volume, transformation, id, copy_no, child_id,
                                                                  placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedHalfSpace::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedHalfSpace>(in_gpu_ptr, GetPoint().x(), GetPoint().y(), GetPoint().z(), GetNormal().x(),
                                          GetNormal().y(), GetNormal().z());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedHalfSpace::CopyToGpu() const { return CopyToGpuImpl<UnplacedHalfSpace>(); }

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedHalfSpace>::SizeOf();
template void DevicePtr<cuda::UnplacedHalfSpace>::Construct(Precision px, Precision py, Precision pz, Precision nx,
                                                            Precision ny, Precision nz) const;

} // namespace cxx

#endif

} // namespace vecgeom
