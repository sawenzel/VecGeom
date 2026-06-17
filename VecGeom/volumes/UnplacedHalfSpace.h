/// @file UnplacedHalfSpace.h
/// @brief Unplaced half-space primitive for Boolean CSG cutters.

#ifndef VECGEOM_VOLUMES_UNPLACEDHALFSPACE_H_
#define VECGEOM_VOLUMES_UNPLACEDHALFSPACE_H_

#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/HalfSpaceStruct.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"
#include "VecGeom/volumes/kernel/HalfSpaceImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedHalfSpace;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedHalfSpace);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// @brief Infinite half-space represented by a plane point and outward normal.
/// @details The primitive is intended for Boolean CSG trees. It is infinite and
/// must not be installed directly as a world or daughter volume. The material
/// side is the half-space where `(p - point).Dot(normal) <= 0`.
class UnplacedHalfSpace : public UnplacedVolumeImplHelper<HalfSpaceImplementation>, public AlignedBase {
private:
  HalfSpaceStruct<Precision> fHalfSpace;

public:
  using Kernel = HalfSpaceImplementation;

  /// @brief Construct a half-space from a plane point and outward normal.
  /// @param point Point lying on the limiting plane.
  /// @param normal Non-zero normal pointing outside; normalized on storage.
  VECCORE_ATT_HOST_DEVICE
  UnplacedHalfSpace(Vector3D<Precision> const &point, Vector3D<Precision> const &normal);

  /// @brief Construct a half-space from scalar point and normal components.
  VECCORE_ATT_HOST_DEVICE
  UnplacedHalfSpace(Precision px, Precision py, Precision pz, Precision nx, Precision ny, Precision nz)
      : UnplacedHalfSpace(Vector3D<Precision>(px, py, pz), Vector3D<Precision>(nx, ny, nz))
  {
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::halfspace; }

  /// @brief Return the normalized half-space runtime data.
  VECCORE_ATT_HOST_DEVICE
  HalfSpaceStruct<Precision> const &GetStruct() const { return fHalfSpace; }

  /// @brief Return one point on the limiting plane.
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetPoint() const { return fHalfSpace.fPoint; }

  /// @brief Return the unit outward normal of the limiting plane.
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetNormal() const { return fHalfSpace.fNormal; }

  /// @brief Return zero for the infinite primitive placeholder capacity.
  /// @details Half-space capacity is not finite; Boolean results containing a
  /// subtracted half-space estimate their own finite capacity from their extent.
  Precision Capacity() const override { return 0.; }

  /// @brief Return zero for the infinite primitive placeholder surface area.
  /// @details The plane surface is infinite until clipped by a finite Boolean
  /// result, so standalone half-space surface area is intentionally unavailable.
  Precision SurfaceArea() const override { return 0.; }

  /// @brief Return the unbounded extent of the standalone half-space.
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override;

  /// @brief Return a deterministic point on the limiting plane.
  /// @details Standalone uniform plane sampling is undefined for an infinite
  /// plane. Boolean samplers clip half-space planes to their finite result box.
  Vector3D<Precision> SamplePointOnSurface() const override;

  /// @brief Return the outward plane normal at a surface point.
  /// @param p Query point in half-space local coordinates.
  /// @param[out] normal Unit outward normal.
  /// @return True when @p p is within tolerance of the limiting plane.
  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = HalfSpaceImplementation::NormalKernel(fHalfSpace, p, valid);
    return valid;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  /// @brief Stream a compact textual representation of the half-space data.
  virtual void Print(std::ostream &os) const override;

#ifndef VECCORE_CUDA
  /// @brief Return an empty mesh because the primitive is unbounded.
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  /// @brief Return the CUDA-side allocation size for this unplaced volume.
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedHalfSpace>::SizeOf(); }
  /// @brief Copy this half-space to newly allocated GPU memory.
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  /// @brief Construct this half-space at an existing GPU allocation.
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECCORE_CUDA
  /// @brief Create a placed half-space instance for Boolean tree operands.
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  /// @brief Create the specialized placed volume for the requested transform.
  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   VPlacedVolume *const placement) const override;
#else
  /// @brief Device-side placed half-space factory.
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, const int copy_no, const int child_id,
                               VPlacedVolume *const placement = NULL);

  /// @brief Device-side specialized placed volume factory.
  VECCORE_ATT_DEVICE
  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   const int id, const int copy_no, const int child_id,
                                   VPlacedVolume *const placement) const override;
#endif

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
  /// @brief Convert to ROOT's `TGeoHalfSpace`.
  TGeoShape const *ConvertToRoot(char const *label = "") const;
#endif
#endif
};

using GenericUnplacedHalfSpace = UnplacedHalfSpace;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDHALFSPACE_H_
