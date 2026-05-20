#pragma once

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/volumes/SurfaceHitDispatch.h"

#include <algorithm>
#include "VecGeom/base/Assert.h"

#ifdef VECGEOM_DISTANCE_DEBUG
#include "VecGeom/volumes/utilities/ResultComparator.h"
#endif

namespace vecgeom {

// putting a forward declaration by hand
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, SpecializedVolImplHelper, typename);
// VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_1t_2v(class, SpecializedVolImplHelper, typename, TranslationCode,
//                                            translation::kGeneric, RotationCode, rotation::kGeneric);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <class Specialization>
class SpecializedVolImplHelper : public Specialization::PlacedShape_t {

  using PlacedShape_t    = typename Specialization::PlacedShape_t;
  using UnplacedVolume_t = typename Specialization::UnplacedVolume_t;

public:
#ifndef VECCORE_CUDA
  SpecializedVolImplHelper(VPlacedVolume const *other)
      : PlacedShape_t(other->GetName(), other->GetLogicalVolume(), other->GetTransformation())
  {
  }

  SpecializedVolImplHelper(char const *const label, LogicalVolume const *const logical_volume,
                           Transformation3D const *const transformation)
      : PlacedShape_t(label, logical_volume, transformation)
  {
  }

  SpecializedVolImplHelper(char const *const label, LogicalVolume *const logical_volume,
                           Transformation3D const *const transformation)
      : PlacedShape_t(label, logical_volume, transformation)
  {
  }

  SpecializedVolImplHelper(LogicalVolume const *const logical_volume, Transformation3D const *const transformation)
      : SpecializedVolImplHelper("", logical_volume, transformation)
  {
  }

  // this constructor mimics the constructor from the Unplaced solid
  // it ensures that placed volumes can be constructed just like ordinary Geant4/ROOT solids
  template <typename... ArgTypes>
  SpecializedVolImplHelper(char const *const label, ArgTypes... params)
      : SpecializedVolImplHelper(label, new LogicalVolume(new UnplacedVolume_t(params...)),
                                 &Transformation3D::kIdentity)
  {
  }

#else // Compiling for CUDA
  VECCORE_ATT_DEVICE SpecializedVolImplHelper(LogicalVolume const *const logical_volume,
                                              Transformation3D const *const transformation, const unsigned int id,
                                              const int copy_no, const int child_id)
      : PlacedShape_t(logical_volume, transformation, id, copy_no, child_id)
  {
  }
#endif
  using PlacedShape_t::Contains;
  using PlacedShape_t::DistanceToIn;
  using PlacedShape_t::DistanceToOut;
  using PlacedShape_t::Inside;
  using PlacedShape_t::PlacedShape_t;
  using PlacedShape_t::SafetyToIn;
  using PlacedShape_t::SafetyToOut;
  using PlacedShape_t::UnplacedContains;

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual EnumInside Inside(Vector3D<Precision> const &point) const override
  {
    Inside_t output;
    Transformation3D const *tr = this->GetTransformation();
    Specialization::Inside(*this->GetUnplacedStruct(), tr->Transform<Precision>(point), output);
    return (EnumInside)output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point) const override
  {
    bool output(false);
    Transformation3D const *tr = this->GetTransformation();
    Vector3D<Precision> lp     = tr->Transform<Precision>(point);
    Specialization::Contains(*this->GetUnplacedStruct(), lp, output);
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const override
  {
    bool output(false);
    Transformation3D const *tr = this->GetTransformation();
    localPoint                 = tr->Transform<Precision>(point);
    Specialization::Contains(*this->GetUnplacedStruct(), localPoint, output);
#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareUnplacedContains(this, output, localPoint);
#endif
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                 const Precision stepMax = kInfLength) const override
  {
    VECGEOM_ASSERT(direction.IsNormalized() && " direction not normalized in call to DistanceToIn ");
    Precision output(kInfLength);
    Transformation3D const *tr = this->GetTransformation();
    SurfaceHitDispatch::DistanceToIn<Specialization>(*this->GetUnplacedStruct(), tr->Transform(point),
                                                     tr->TransformDirection(direction), stepMax, output,
                                                     static_cast<SurfaceHitView<Precision> *>(nullptr));
#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToIn(this, output, point, direction, stepMax);
#endif
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                 const Precision stepMax, SurfaceHitView<Precision> *hit_info) const override
  {
    VECGEOM_ASSERT(direction.IsNormalized() && " direction not normalized in call to DistanceToIn ");
    Precision output(kInfLength);
    Transformation3D const *tr         = this->GetTransformation();
    Vector3D<Precision> localPoint     = tr->Transform(point);
    Vector3D<Precision> localDirection = tr->TransformDirection(direction);
    Vector3D<Precision> localNormal;
    SurfaceHitView<Precision> localHitInfo;
    SurfaceHitView<Precision> *kernelHitInfo = nullptr;
    if (hit_info) {
      localHitInfo.fNormal = hit_info->WantsNormal() ? &localNormal : nullptr;
      kernelHitInfo        = &localHitInfo;
    }
    SurfaceHitDispatch::DistanceToIn<Specialization>(*this->GetUnplacedStruct(), localPoint, localDirection, stepMax,
                                                     output, kernelHitInfo);
    if (hit_info) {
      hit_info->fSurface = localHitInfo.fSurface;
      if (hit_info->WantsNormal()) hit_info->SetNormal(tr->InverseTransformDirection(localNormal));
    }
#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToIn(this, output, point, direction, stepMax);
#endif
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                        const Precision stepMax = kInfLength) const override
  {
    VECGEOM_ASSERT(direction.IsNormalized() && " direction not normalized in call to PlacedDistanceToOut ");
    Transformation3D const *tr = this->GetTransformation();
    Precision output(-1.);
    SurfaceHitDispatch::DistanceToOut<Specialization>(*this->GetUnplacedStruct(), tr->Transform(point),
                                                      tr->TransformDirection(direction), stepMax, output,
                                                      static_cast<SurfaceHitView<Precision> *>(nullptr));

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToOut(this, output, this->GetTransformation()->Transform(point),
                                             this->GetTransformation()->TransformDirection(direction), stepMax);
#endif
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                        const Precision stepMax, SurfaceHitView<Precision> *hit_info) const override
  {
    VECGEOM_ASSERT(direction.IsNormalized() && " direction not normalized in call to PlacedDistanceToOut ");
    Transformation3D const *tr         = this->GetTransformation();
    Vector3D<Precision> localPoint     = tr->Transform(point);
    Vector3D<Precision> localDirection = tr->TransformDirection(direction);
    Vector3D<Precision> localNormal;
    SurfaceHitView<Precision> localHitInfo;
    SurfaceHitView<Precision> *kernelHitInfo = nullptr;
    if (hit_info) {
      localHitInfo.fNormal = hit_info->WantsNormal() ? &localNormal : nullptr;
      kernelHitInfo        = &localHitInfo;
    }
    Precision output(-1.);
    SurfaceHitDispatch::DistanceToOut<Specialization>(*this->GetUnplacedStruct(), localPoint, localDirection, stepMax,
                                                      output, kernelHitInfo);
    if (hit_info) {
      hit_info->fSurface = localHitInfo.fSurface;
      if (hit_info->WantsNormal()) hit_info->SetNormal(tr->InverseTransformDirection(localNormal));
    }

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToOut(this, output, localPoint, localDirection, stepMax);
#endif
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToIn(Vector3D<Precision> const &point) const override
  {
    Precision output(kInfLength);
    Transformation3D const *tr     = this->GetTransformation();
    Vector3D<Precision> localPoint = tr->Transform(point);
    Specialization::SafetyToIn(*this->GetUnplacedStruct(), localPoint, output);
    return output;
  }

#ifdef VECGEOM_CUDA_INTERFACE
  using ThisClass_t = SpecializedVolImplHelper<Specialization>;
  virtual size_t DeviceSizeOf() const override { return DevicePtr<CudaType_t<ThisClass_t>>::SizeOf(); }

  DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                           DevicePtr<cuda::Transformation3D> const transform,
                                           DevicePtr<cuda::VPlacedVolume> const in_gpu_ptr) const override
  {
    DevicePtr<CudaType_t<ThisClass_t>> gpu_ptr(in_gpu_ptr);
    gpu_ptr.Construct(logical_volume, transform, this->id(), this->GetCopyNo(), this->GetChildId());
    VECGEOM_DEVICE_API_CALL(GetLastError());
    // Need to go via the void* because the regular c++ compilation
    // does not actually see the declaration for the cuda version
    // (and thus can not determine the inheritance).
    return DevicePtr<cuda::VPlacedVolume>((void *)gpu_ptr);
  }

  DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                           DevicePtr<cuda::Transformation3D> const transform) const override
  {
    DevicePtr<CudaType_t<ThisClass_t>> gpu_ptr;
    gpu_ptr.Allocate();
    return CopyToGpu(logical_volume, transform, DevicePtr<cuda::VPlacedVolume>((void *)gpu_ptr));
  }

  /**
   * Copy many instances of this class to the GPU.
   * \param host_volumes Host volumes to be copied. These should all be of the same type as the class that this function is called with.
   * \param logical_volumes GPU addresses of the logical volumes corresponding to the placed volumes.
   * \param transforms GPU addresses of the transformations corresponding to the placed volumes.
   * \param in_gpu_ptrs GPU addresses where the GPU instances of the host volumes should be placed.
   * \note This requires an explicit template instantiation of ConstructManyOnGpu<ThisClass_t>().
   * \see VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL and its multi-argument versions.
   */
  void CopyManyToGpu(std::vector<VPlacedVolume const *> const &host_volumes,
                     std::vector<DevicePtr<cuda::LogicalVolume>> const &logical_volumes,
                     std::vector<DevicePtr<cuda::Transformation3D>> const &transforms,
                     std::vector<DevicePtr<cuda::VPlacedVolume>> const &in_gpu_ptrs) const override
  {
    VECGEOM_ASSERT(host_volumes.size() == logical_volumes.size());
    VECGEOM_ASSERT(host_volumes.size() == transforms.size());
    VECGEOM_ASSERT(host_volumes.size() == in_gpu_ptrs.size());

    std::vector<decltype(std::declval<ThisClass_t>().id())> ids;
    std::vector<decltype(std::declval<ThisClass_t>().GetCopyNo())> copyNos;
    std::vector<decltype(std::declval<ThisClass_t>().GetChildId())> childIds;
    for (auto placedVol : host_volumes) {
      ids.push_back(placedVol->id());
      copyNos.push_back(placedVol->GetCopyNo());
      childIds.push_back(placedVol->GetChildId());
    }

    ConstructManyOnGpu<CudaType_t<ThisClass_t>>(in_gpu_ptrs.size(), in_gpu_ptrs.data(), logical_volumes.data(),
                                                transforms.data(), ids.data(), copyNos.data(), childIds.data());
  }

#endif // VECGEOM_CUDA_INTERFACE

}; // End class SpecializedVolImplHelper

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
