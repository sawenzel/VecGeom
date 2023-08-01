/*
 * UnplacedBooleanVolume.cpp
 *
 *  Created on: 07.11.2014
 *      Author: swenzel
 */

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"
#include "VecGeom/volumes/SpecializedBooleanVolume.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/utilities/GenerationUtilities.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"

#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/management/CudaManager.h"
#endif

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
template <>
Vector3D<Precision> UnplacedBooleanVolume<kUnion>::SamplePointOnSurface() const
{
  // implementation taken from G4
  int counter = 0;
  Vector3D<Precision> p;

  Precision arearatio(0.5);
  Precision leftarea, rightarea;

  // calculating surface area can be expensive
  // until there is a caching mechanism in place, we will cache these values here
  // in a static map
  // the caching mechanism will be put into place with the completion of the move to UnplacedVolume interfaces
  static std::map<size_t, Precision> idtoareamap;
  auto leftid = GetLeft()->GetLogicalVolume()->id();
  if (idtoareamap.find(leftid) != idtoareamap.end()) {
    leftarea = idtoareamap[leftid];
  } else { // insert
    leftarea = const_cast<VPlacedVolume *>(GetLeft())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, Precision>(leftid, leftarea));
  }

  auto rightid = GetRight()->GetLogicalVolume()->id();
  if (idtoareamap.find(rightid) != idtoareamap.end()) {
    rightarea = idtoareamap[rightid];
  } else { // insert
    rightarea = const_cast<VPlacedVolume *>(GetRight())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, Precision>(rightid, rightarea));
  }

  if (leftarea > 0. && rightarea > 0.) {
    arearatio = leftarea / (leftarea + rightarea);
  }
  do {
    counter++;
    if (counter > 1000) {
      std::cerr << "WARNING : COULD NOT GENERATE POINT ON SURFACE FOR BOOLEAN\n";
      return p;
    }

    auto *selected((RNG::Instance().uniform() < arearatio) ? GetLeft() : GetRight());
    auto transf = selected->GetTransformation();
    p           = transf->InverseTransform(selected->GetUnplacedVolume()->SamplePointOnSurface());
  } while (Inside(p) != vecgeom::kSurface);
  return p;
}

template <>
Vector3D<Precision> UnplacedBooleanVolume<kIntersection>::SamplePointOnSurface() const
{
  // implementation taken from G4
  int counter = 0;
  Vector3D<Precision> p;

  Precision arearatio(0.5);
  Precision leftarea, rightarea;

  // calculating surface area can be expensive
  // until there is a caching mechanism in place, we will cache these values here
  // in a static map
  // the caching mechanism will be put into place with the completion of the move to UnplacedVolume interfaces
  static std::map<size_t, Precision> idtoareamap;
  auto leftid = GetLeft()->GetLogicalVolume()->id();
  if (idtoareamap.find(leftid) != idtoareamap.end()) {
    leftarea = idtoareamap[leftid];
  } else { // insert
    leftarea = const_cast<VPlacedVolume *>(GetLeft())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, Precision>(leftid, leftarea));
  }

  auto rightid = GetRight()->GetLogicalVolume()->id();
  if (idtoareamap.find(rightid) != idtoareamap.end()) {
    rightarea = idtoareamap[rightid];
  } else { // insert
    rightarea = const_cast<VPlacedVolume *>(GetRight())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, Precision>(rightid, rightarea));
  }

  if (leftarea > 0. && rightarea > 0.) {
    arearatio = leftarea / (leftarea + rightarea);
  }
  do {
    counter++;
    if (counter > 1000) {
      std::cerr << "WARNING : COULD NOT GENERATE POINT ON SURFACE FOR BOOLEAN\n";
      return p;
    }

    auto *selected((RNG::Instance().uniform() < arearatio) ? GetLeft() : GetRight());
    auto transf = selected->GetTransformation();
    p           = transf->InverseTransform(selected->GetUnplacedVolume()->SamplePointOnSurface());
  } while (Inside(p) != vecgeom::kSurface);
  return p;
}

template <>
Vector3D<Precision> UnplacedBooleanVolume<kSubtraction>::SamplePointOnSurface() const
{
  // implementation taken from G4
  int counter = 0;
  Vector3D<Precision> p;

  Precision arearatio(0.5);
  Precision leftarea, rightarea;

  // calculating surface area can be expensive
  // until there is a caching mechanism in place, we will cache these values here
  // in a static map
  // the caching mechanism will be put into place with the completion of the move to UnplacedVolume interfaces
  static std::map<size_t, Precision> idtoareamap;
  auto leftid = GetLeft()->GetLogicalVolume()->id();
  if (idtoareamap.find(leftid) != idtoareamap.end()) {
    leftarea = idtoareamap[leftid];
  } else { // insert
    leftarea = const_cast<VPlacedVolume *>(GetLeft())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, Precision>(leftid, leftarea));
  }

  auto rightid = GetRight()->GetLogicalVolume()->id();
  if (idtoareamap.find(rightid) != idtoareamap.end()) {
    rightarea = idtoareamap[rightid];
  } else { // insert
    rightarea = const_cast<VPlacedVolume *>(GetRight())->SurfaceArea();
    idtoareamap.insert(std::pair<size_t, Precision>(rightid, rightarea));
  }

  if (leftarea > 0. && rightarea > 0.) {
    arearatio = leftarea / (leftarea + rightarea);
  }
  do {
    counter++;
    if (counter > 1000) {
      std::cerr << "WARNING : COULD NOT GENERATE POINT ON SURFACE FOR BOOLEAN\n";
      return p;
    }

    auto *selected((RNG::Instance().uniform() < arearatio) ? GetLeft() : GetRight());
    auto transf = selected->GetTransformation();
    p           = transf->InverseTransform(selected->GetUnplacedVolume()->SamplePointOnSurface());
  } while (Inside(p) != vecgeom::kSurface);
  return p;
}

VECCORE_ATT_HOST_DEVICE
BooleanStruct const *BooleanHelper::GetBooleanStruct(VUnplacedVolume const *unplaced)
{
  UnplacedBooleanVolume<kUnion> const *buni = dynamic_cast<UnplacedBooleanVolume<kUnion> const *>(unplaced);
  BooleanStruct const *bstruct              = (buni) ? &buni->GetStruct() : nullptr;

  if (!bstruct) {
    UnplacedBooleanVolume<kIntersection> const *bint =
        dynamic_cast<UnplacedBooleanVolume<kIntersection> const *>(unplaced);
    bstruct = (bint) ? &bint->GetStruct() : nullptr;
  }

  if (!bstruct) {
    UnplacedBooleanVolume<kSubtraction> const *bsub =
        dynamic_cast<UnplacedBooleanVolume<kSubtraction> const *>(unplaced);
    bstruct = (bsub) ? &bsub->GetStruct() : nullptr;
  }
  return bstruct;
}

VECCORE_ATT_HOST_DEVICE
size_t BooleanHelper::CountBooleanNodes(VUnplacedVolume const *unplaced, size_t &nunion, size_t &nintersection,
                                        size_t &nsubtraction)
{
  BooleanStruct const *bstruct = GetBooleanStruct(unplaced);
  if (!bstruct) return 0;

  nunion += bstruct->fOp == kUnion;
  nintersection += bstruct->fOp == kIntersection;
  nsubtraction += bstruct->fOp == kSubtraction;
  CountBooleanNodes(bstruct->fLeftVolume->GetUnplacedVolume(), nunion, nintersection, nsubtraction);
  CountBooleanNodes(bstruct->fRightVolume->GetUnplacedVolume(), nunion, nintersection, nsubtraction);
  return (nunion + nintersection + nsubtraction);
}

UnplacedMultiUnion *BooleanHelper::Flatten(VUnplacedVolume const *unplaced, size_t min_unions,
                                           Transformation3D const *trbase, UnplacedMultiUnion *munion)
{
  size_t nunion{0}, nintersection{0}, nsubtraction{0};
  CountBooleanNodes(unplaced, nunion, nintersection, nsubtraction);
  if (nunion < min_unions) {
    // If a multi-union is being built-up, add this volume
    if (munion) munion->AddNode(unplaced, *trbase);
    return nullptr;
  }
  VUnplacedVolume const *vol;
  BooleanStruct *bstruct = (BooleanStruct *)GetBooleanStruct(unplaced);
  if (bstruct->fOp == kUnion) {
    bool creator = munion == nullptr;
    if (!munion) munion = new UnplacedMultiUnion();
    Transformation3D transform;

    // Compute left transformation
    transform = (trbase) ? *trbase : Transformation3D();
    transform.MultiplyFromRight(*bstruct->fLeftVolume->GetTransformation());
    Flatten(bstruct->fLeftVolume->GetUnplacedVolume(), min_unions, &transform, munion);

    // Compute right transformation
    transform = (trbase) ? *trbase : Transformation3D();
    transform.MultiplyFromRight(*bstruct->fRightVolume->GetTransformation());
    Flatten(bstruct->fRightVolume->GetUnplacedVolume(), min_unions, &transform, munion);

    if (creator) {
      munion->Close();
      return munion;
    }
    return nullptr;
  }

  // Analyze branches in case of subtraction or intersection
  vol            = (VUnplacedVolume *)bstruct->fLeftVolume->GetUnplacedVolume();
  auto left_bool = Flatten(vol, min_unions);
  if (left_bool) {
    // Replace existing left volume with the new one
    auto lvol            = new LogicalVolume(left_bool);
    auto pvol            = lvol->Place(bstruct->fLeftVolume->GetTransformation());
    bstruct->fLeftVolume = pvol;
  }
  vol             = (VUnplacedVolume *)bstruct->fRightVolume->GetUnplacedVolume();
  auto right_bool = Flatten(vol, min_unions);
  if (right_bool) {
    // Replace existing right volume with the new one
    auto lvol             = new LogicalVolume(right_bool);
    auto pvol             = lvol->Place(bstruct->fRightVolume->GetTransformation());
    bstruct->fRightVolume = pvol;
  }
  if (munion) munion->AddNode(unplaced, *trbase);
  return nullptr;
}
#endif

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kSubtraction>::Create(
    LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement)
{
  return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kSubtraction>>(logical_volume, transformation,
#ifdef VECCORE_CUDA
                                                                                id, copy_no, child_id,
#endif
                                                                                placement); // TODO: add bounding box?
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kUnion>::Create(LogicalVolume const *const logical_volume,
                                                                        Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                                                        const int id, const int copy_no,
                                                                        const int child_id,
#endif
                                                                        VPlacedVolume *const placement)
{
  return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kUnion>>(logical_volume, transformation,
#ifdef VECCORE_CUDA
                                                                          id, copy_no, child_id,
#endif
                                                                          placement); // TODO: add bounding box?
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kIntersection>::Create(
    LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement)
{
  return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kIntersection>>(logical_volume, transformation,
#ifdef VECCORE_CUDA
                                                                                 id, copy_no, child_id,
#endif
                                                                                 placement); // TODO: add bounding box?
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kSubtraction>::SpecializedVolume(
    LogicalVolume const *const volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement) const
{
#ifndef VECCORE_CUDA

  return VolumeFactory::CreateByTransformation<UnplacedBooleanVolume<kSubtraction>>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                                    id, copy_no, child_id,
#endif
                                                                                    placement);

#else
  // Compiling the above code with nvcc 6.5 faile with the error:
  // nvcc error   : 'ptxas' died due to signal 11 (Invalid memory reference)
  // at least when optimized.
  return nullptr;
#endif
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kUnion>::SpecializedVolume(
    LogicalVolume const *const volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement) const
{
#ifndef VECCORE_CUDA

  return VolumeFactory::CreateByTransformation<UnplacedBooleanVolume<kUnion>>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                              id, copy_no, child_id,
#endif
                                                                              placement);

#else
  // Compiling the above code with nvcc 6.5 faile with the error:
  // nvcc error   : 'ptxas' died due to signal 11 (Invalid memory reference)
  // at least when optimized.
  return nullptr;
#endif
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kIntersection>::SpecializedVolume(
    LogicalVolume const *const volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement) const
{
#ifndef VECCORE_CUDA

  return VolumeFactory::CreateByTransformation<UnplacedBooleanVolume<kIntersection>>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                                     id, copy_no, child_id,
#endif
                                                                                     placement);

#else
  // Compiling the above code with nvcc 6.5 faile with the error:
  // nvcc error   : 'ptxas' died due to signal 11 (Invalid memory reference)
  // at least when optimized.
  return nullptr;
#endif
}

VECCORE_ATT_HOST_DEVICE
void TransformedExtent(VPlacedVolume const *pvol, Vector3D<Precision> &aMin, Vector3D<Precision> &aMax)
{
// CUDA does not have min and max in std:: namespace
#ifndef VECCORE_CUDA
  using std::max;
  using std::min;
#endif
  Vector3D<Precision> lower, upper;
  pvol->Extent(lower, upper);
  Vector3D<Precision> delta = upper - lower;
  Precision minx, miny, minz, maxx, maxy, maxz;
  minx         = kInfLength;
  miny         = kInfLength;
  minz         = kInfLength;
  maxx         = -kInfLength;
  maxy         = -kInfLength;
  maxz         = -kInfLength;
  auto *transf = pvol->GetTransformation();
  for (int x = 0; x <= 1; ++x)
    for (int y = 0; y <= 1; ++y)
      for (int z = 0; z <= 1; ++z) {
        Vector3D<Precision> corner;
        corner.x()                            = lower.x() + x * delta.x();
        corner.y()                            = lower.y() + y * delta.y();
        corner.z()                            = lower.z() + z * delta.z();
        Vector3D<Precision> transformedcorner = transf->InverseTransform(corner);
        minx                                  = min(minx, transformedcorner.x());
        miny                                  = min(miny, transformedcorner.y());
        minz                                  = min(minz, transformedcorner.z());
        maxx                                  = max(maxx, transformedcorner.x());
        maxy                                  = max(maxy, transformedcorner.y());
        maxz                                  = max(maxz, transformedcorner.z());
      }
  aMin.x() = minx;
  aMin.y() = miny;
  aMin.z() = minz;
  aMax.x() = maxx;
  aMax.y() = maxy;
  aMax.z() = maxz;
}

template <>
VECCORE_ATT_HOST_DEVICE void UnplacedBooleanVolume<kSubtraction>::Extent(Vector3D<Precision> &aMin,
                                                                         Vector3D<Precision> &aMax) const
{
  Vector3D<Precision> minLeft, maxLeft;
  // NOTE: VPlacedVolume::Extent returns now the placed volume extent
  // We ignore the subtracted volume extent since we miss the extent cutoff functionality
  fBoolean.fLeftVolume->Extent(aMin, aMax);
}

template <>
VECCORE_ATT_HOST_DEVICE void UnplacedBooleanVolume<kUnion>::Extent(Vector3D<Precision> &aMin,
                                                                   Vector3D<Precision> &aMax) const
{
  Vector3D<Precision> minLeft, maxLeft, minRight, maxRight;
  // NOTE: VPlacedVolume::Extent returns now the placed volume extent
  fBoolean.fLeftVolume->Extent(minLeft, maxLeft);
  fBoolean.fRightVolume->Extent(minRight, maxRight);
  aMin = Vector3D<Precision>(Min(minLeft.x(), minRight.x()), Min(minLeft.y(), minRight.y()),
                             Min(minLeft.z(), minRight.z()));
  aMax = Vector3D<Precision>(Max(maxLeft.x(), maxRight.x()), Max(maxLeft.y(), maxRight.y()),
                             Max(maxLeft.z(), maxRight.z()));
}

template <>
VECCORE_ATT_HOST_DEVICE void UnplacedBooleanVolume<kIntersection>::Extent(Vector3D<Precision> &aMin,
                                                                          Vector3D<Precision> &aMax) const
{
  Vector3D<Precision> minLeft, maxLeft, minRight, maxRight;
  // NOTE: VPlacedVolume::Extent returns now the placed volume extent
  fBoolean.fLeftVolume->Extent(minLeft, maxLeft);
  fBoolean.fRightVolume->Extent(minRight, maxRight);
  aMin = Vector3D<Precision>(Max(minLeft.x(), minRight.x()), Max(minLeft.y(), minRight.y()),
                             Max(minLeft.z(), minRight.z()));
  aMax = Vector3D<Precision>(Min(maxLeft.x(), maxRight.x()), Min(maxLeft.y(), maxRight.y()),
                             Min(maxLeft.z(), maxRight.z()));
}

template <>
VECCORE_ATT_HOST_DEVICE bool UnplacedBooleanVolume<kSubtraction>::Normal(Vector3D<Precision> const &point,
                                                                         Vector3D<Precision> &normal) const
{
  // Compute normal vector to closest surface
  bool valid = false;
  BooleanImplementation<kSubtraction>::NormalKernel(GetStruct(), point, normal, valid);
  return valid;
}

template <>
VECCORE_ATT_HOST_DEVICE bool UnplacedBooleanVolume<kUnion>::Normal(Vector3D<Precision> const &point,
                                                                   Vector3D<Precision> &normal) const
{
  // Compute normal vector to closest surface
  bool valid = false;
  BooleanImplementation<kUnion>::NormalKernel(GetStruct(), point, normal, valid);
  return valid;
}

template <>
VECCORE_ATT_HOST_DEVICE bool UnplacedBooleanVolume<kIntersection>::Normal(Vector3D<Precision> const &point,
                                                                          Vector3D<Precision> &normal) const
{
  // Compute normal vector to closest surface
  bool valid = false;
  BooleanImplementation<kIntersection>::NormalKernel(GetStruct(), point, normal, valid);
  return valid;
}

#ifdef VECGEOM_CUDA_INTERFACE

// functions to copy data structures to GPU
template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kUnion>::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  // here we have our recursion:
  // since UnplacedBooleanVolume has pointer members we need to copy/construct those members too
  // very brute force; because this might have been copied already
  // TODO: integrate this into CUDA MGR?

  // use CUDA Manager to lookup GPU pointer
  DevicePtr<cuda::VPlacedVolume> leftgpuptr  = CudaManager::Instance().LookupPlaced(GetLeft());
  DevicePtr<cuda::VPlacedVolume> rightgpuptr = CudaManager::Instance().LookupPlaced(GetRight());

  return CopyToGpuImpl<UnplacedBooleanVolume<kUnion>>(in_gpu_ptr, GetOp(), leftgpuptr, rightgpuptr);
}
template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kUnion>::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedBooleanVolume<kUnion>>();
}

// functions to copy data structures to GPU
template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kIntersection>::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  // here we have our recursion:
  // since UnplacedBooleanVolume has pointer members we need to copy/construct those members too
  // very brute force; because this might have been copied already
  // TODO: integrate this into CUDA MGR?

  // use CUDA Manager to lookup GPU pointer
  DevicePtr<cuda::VPlacedVolume> leftgpuptr  = CudaManager::Instance().LookupPlaced(GetLeft());
  DevicePtr<cuda::VPlacedVolume> rightgpuptr = CudaManager::Instance().LookupPlaced(GetRight());

  return CopyToGpuImpl<UnplacedBooleanVolume<kIntersection>>(in_gpu_ptr, GetOp(), leftgpuptr, rightgpuptr);
}

template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kIntersection>::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedBooleanVolume<kIntersection>>();
}

template <>
// functions to copy data structures to GPU
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kSubtraction>::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  // here we have our recursion:
  // since UnplacedBooleanVolume has pointer members we need to copy/construct those members too
  // very brute force; because this might have been copied already
  // TODO: integrate this into CUDA MGR?

  // use CUDA Manager to lookup GPU pointer
  DevicePtr<cuda::VPlacedVolume> leftgpuptr  = CudaManager::Instance().LookupPlaced(GetLeft());
  DevicePtr<cuda::VPlacedVolume> rightgpuptr = CudaManager::Instance().LookupPlaced(GetRight());

  return CopyToGpuImpl<UnplacedBooleanVolume<kSubtraction>>(in_gpu_ptr, GetOp(), leftgpuptr, rightgpuptr);
}

template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kSubtraction>::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedBooleanVolume<kSubtraction>>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedBooleanVolume<kUnion>>::SizeOf();
template void DevicePtr<cuda::UnplacedBooleanVolume<kUnion>>::Construct(BooleanOperation op,
                                                                        DevicePtr<cuda::VPlacedVolume> left,
                                                                        DevicePtr<cuda::VPlacedVolume> right) const;
template size_t DevicePtr<cuda::UnplacedBooleanVolume<kIntersection>>::SizeOf();
template void DevicePtr<cuda::UnplacedBooleanVolume<kIntersection>>::Construct(
    BooleanOperation op, DevicePtr<cuda::VPlacedVolume> left, DevicePtr<cuda::VPlacedVolume> right) const;
template size_t DevicePtr<cuda::UnplacedBooleanVolume<kSubtraction>>::SizeOf();
template void DevicePtr<cuda::UnplacedBooleanVolume<kSubtraction>>::Construct(
    BooleanOperation op, DevicePtr<cuda::VPlacedVolume> left, DevicePtr<cuda::VPlacedVolume> right) const;

} // namespace cxx

#endif

} // End namespace vecgeom
