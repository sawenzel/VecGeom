// LICENSING INFORMATION TBD

#include "VecGeom/volumes/UnplacedAssembly.h"
#include "VecGeom/volumes/PlacedAssembly.h"
#include "VecGeom/navigation/SimpleLevelLocator.h"
#include "VecGeom/management/ABBoxManager.h" // for Extent == bounding box calculation
#include "VecGeom/base/RNG.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
UnplacedAssembly::UnplacedAssembly() : fLogicalVolume(nullptr), fLowerCorner(-kInfLength), fUpperCorner(kInfLength)
{
  fIsAssembly = true;
}

UnplacedAssembly::~UnplacedAssembly() {}

void UnplacedAssembly::AddVolume(VPlacedVolume *const v) { fLogicalVolume->PlaceDaughter(v); }

VECCORE_ATT_HOST_DEVICE
void UnplacedAssembly::Print() const { printf("UnplacedAssembly "); }

void UnplacedAssembly::Print(std::ostream &os) const { os << "UnplacedAssembly "; }

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void UnplacedAssembly::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
#ifndef VECCORE_CUDA
  auto &abboxmgr = ABBoxManager<Precision>::Instance();

  // Returns the full 3D cartesian extent of the solid.
  // Loop nodes and get their extent
  aMin.Set(kInfLength);
  aMax.Set(-kInfLength);
  for (VPlacedVolume const *pv : fLogicalVolume->GetDaughters()) {
    Vector3D<Precision> lower, upper;
    abboxmgr.ComputeABBox(pv, &lower, &upper);
    aMin.Set(std::min(lower.x(), aMin.x()), std::min(lower.y(), aMin.y()), std::min(lower.z(), aMin.z()));
    aMax.Set(std::max(upper.x(), aMax.x()), std::max(upper.y(), aMax.y()), std::max(upper.z(), aMax.z()));
  }
#endif
}

Vector3D<Precision> UnplacedAssembly::SamplePointOnSurface() const
{
  // pick one of the constituents for now
  // should improve this to sample according to surface area
  const auto ndaughters = fLogicalVolume->GetDaughters().size();
  const size_t selected = RNG::Instance().uniform() * ndaughters;

  const auto selectedplaced = fLogicalVolume->GetDaughters()[selected];
  Vector3D<Precision> sp    = selectedplaced->GetUnplacedVolume()->SamplePointOnSurface();
  // this is in the reference frame of the selected daughter
  // we need to return it in the reference of this assembly

  return selectedplaced->GetTransformation()->InverseTransform(sp);
}

Precision UnplacedAssembly::Capacity() const
{
  Precision capacity = 0.;
  // loop over nodes and sum the capacity of all their unplaced volumes
  for (VPlacedVolume const *pv : fLogicalVolume->GetDaughters()) {
    capacity += const_cast<VPlacedVolume *>(pv)->Capacity();
  }
  return capacity;
}

Precision UnplacedAssembly::SurfaceArea() const
{
  Precision area = 0.;
  // loop over nodes and sum the area of all their unplaced volumes
  // (this might be incorrect in case 2 constituents are touching)
  for (VPlacedVolume const *pv : fLogicalVolume->GetDaughters()) {
    area += const_cast<VPlacedVolume *>(pv)->SurfaceArea();
  }
  return area;
}

#ifndef VECCORE_CUDA
VPlacedVolume *UnplacedAssembly::SpecializedVolume(LogicalVolume const *const volume,
                                                   Transformation3D const *const transformation,
                                                   VPlacedVolume *const placement) const
{
  if (placement) {
    return new (placement) PlacedAssembly("", volume, transformation);
  }
  return new PlacedAssembly("", volume, transformation);
}
#else
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedAssembly::SpecializedVolume(LogicalVolume const *const volume,
                                                   Transformation3D const *const transformation, const int id,
                                                   const int copy_no, const int child_id,
                                                   VPlacedVolume *const placement) const
{
  if (placement) {
    return new (placement) PlacedAssembly("", volume, transformation, id, copy_no, child_id);
  }
  return new PlacedAssembly("", volume, transformation, id, copy_no, child_id);
}
#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedAssembly::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedAssembly>(in_gpu_ptr);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedAssembly::CopyToGpu() const { return CopyToGpuImpl<UnplacedAssembly>(); }

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedAssembly>::SizeOf();
template void DevicePtr<cuda::UnplacedAssembly>::Construct() const;

} // namespace cxx

#endif

} // namespace vecgeom
