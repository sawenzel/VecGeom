// LICENSING INFORMATION TBD

#ifndef VECGEOM_PLACEDASSEMBLY_H
#define VECGEOM_PLACEDASSEMBLY_H

#include "VecGeom/base/Cuda.h"
#include "VecGeom/volumes/UnplacedAssembly.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/navigation/NavStateFwd.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedAssembly;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedAssembly);

inline namespace VECGEOM_IMPL_NAMESPACE {

// class NavigationState;

// placed version of an assembly
// simple and unspecialized implementation
class PlacedAssembly : public VPlacedVolume {

private:
public:
#ifndef VECCORE_CUDA
  VECCORE_ATT_HOST_DEVICE
  PlacedAssembly(char const *const label, LogicalVolume const *const logicalVolume,
                 Transformation3D const *const transformation)
      : VPlacedVolume(label, logicalVolume, transformation)
  {
  } // the constructor
#else
  VECCORE_ATT_DEVICE PlacedAssembly(char const *const label, LogicalVolume const *const logical_volume,
                                    Transformation3D const *const transformation, const int id, const int copy_no,
                                    const int child_id)
      : VPlacedVolume(logical_volume, transformation, id, copy_no, child_id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedAssembly() {}

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

  // the VPlacedVolume Interfaces -----
  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &p) const override
  {
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())
        ->UnplacedAssembly::Contains(GetTransformation()->Transform(p));
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const override
  {
    localPoint = GetTransformation()->Transform(point);
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())->UnplacedAssembly::Contains(localPoint);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const override
  {
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())->UnplacedAssembly::Contains(point);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual EnumInside Inside(Vector3D<Precision> const & /*point*/) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return vecgeom::kOutside; // dummy return
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                 const Precision step_max = kInfLength) const override
  {
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())
        ->UnplacedAssembly::DistanceToIn(GetTransformation()->Transform(position),
                                         GetTransformation()->TransformDirection(direction), step_max);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                 const Precision step_max, SurfaceHitView<Precision> *hit_info) const override
  {
    Vector3D<Precision> localNormal;
    SurfaceHitView<Precision> localHitInfo;
    SurfaceHitView<Precision> *localHitInfoPtr = nullptr;
    if (hit_info) {
      localHitInfo.fNormal = hit_info->WantsNormal() ? &localNormal : nullptr;
      localHitInfoPtr      = &localHitInfo;
    }
    Precision distance = static_cast<UnplacedAssembly const *>(GetUnplacedVolume())
                             ->UnplacedAssembly::DistanceToIn(GetTransformation()->Transform(position),
                                                              GetTransformation()->TransformDirection(direction),
                                                              step_max, localHitInfoPtr);
    if (hit_info) {
      hit_info->fSurface = localHitInfo.fSurface;
      if (hit_info->WantsNormal()) hit_info->SetNormal(GetTransformation()->InverseTransformDirection(localNormal));
    }
    return distance;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToOut(Vector3D<Precision> const & /*position*/, Vector3D<Precision> const & /*direction*/,
                                  Precision const /*stepMax*/ = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return -1.; // dummy return
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                  Precision const stepMax, SurfaceHitView<Precision> *hit_info) const override
  {
    if (hit_info) hit_info->Clear();
    return DistanceToOut(position, direction, stepMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const & /*position*/,
                                        Vector3D<Precision> const & /*direction*/,
                                        Precision const /*stepMax*/ = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return -1.;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                        Precision const stepMax, SurfaceHitView<Precision> *hit_info) const override
  {
    if (hit_info) hit_info->Clear();
    return PlacedDistanceToOut(position, direction, stepMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToOut(Vector3D<Precision> const &position) const override
  {
    return GetUnplacedVolume()->SafetyToOut(position);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const override
  {
    Precision safety = static_cast<UnplacedAssembly const *>(GetUnplacedVolume())
                           ->UnplacedAssembly::SafetyToIn(GetTransformation()->Transform(position));
    return safety;
  }

  Precision SurfaceArea() const override
  {
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())->SurfaceArea();
  }

#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override { return this; }
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override { throw std::runtime_error("unimplemented function called"); }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override
  {
    throw std::runtime_error("unimplemented function called");
  }
#endif
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  // TBD properly
  virtual size_t DeviceSizeOf() const override { return 0; /*DevicePtr<cuda::PlacedAssembly>::SizeOf();*/ }
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const /*logical_volume*/,
                                                   DevicePtr<cuda::Transformation3D> const /*transform*/,
                                                   DevicePtr<cuda::VPlacedVolume> const /*gpu_ptr*/) const override
  {
    return DevicePtr<cuda::VPlacedVolume>(nullptr);
  }
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const /*logical_volume*/,
                                                   DevicePtr<cuda::Transformation3D> const /*transform*/) const override
  {
    return DevicePtr<cuda::VPlacedVolume>(nullptr);
  }

  /// Not implemented.
  virtual void CopyManyToGpu(std::vector<VPlacedVolume const *> const &host_volumes,
                             std::vector<DevicePtr<cuda::LogicalVolume>> const &logical_volumes,
                             std::vector<DevicePtr<cuda::Transformation3D>> const &transforms,
                             std::vector<DevicePtr<cuda::VPlacedVolume>> const &in_gpu_ptrs) const override
  {
  }
#endif

  // specific PlacedAssembly Interfaces ---------

  // an extended contains functions needed for navigation
  // if this function returns true it modifies the navigation state to point to the first non-assembly volume
  // the point is contained in
  // this function is not part of the generic UnplacedVolume interface but we could consider doing so
  // N.B To work correctly, the input state must be initialized to point to the parent volume of this placed assembly
  //     otherwise the Push/Pop cannot work correctly since they do not match a valid navigation state
  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<Precision> const &p, Vector3D<Precision> &lp, NavigationState &state) const
  {
    state.Push(this);
    // call unplaced variant with transformed point
    auto indaughter = static_cast<UnplacedAssembly const *>(GetUnplacedVolume())
                          ->UnplacedAssembly::Contains(GetTransformation()->Transform(p), lp, state);
    if (!indaughter) state.Pop();
    return indaughter;
  }

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // PLACEDASSEMBLY_H
