/// @file UnplacedGenericPolycone.h
/// @brief Unplaced generic polycone volume.
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDGENERICPOLYCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDGENERICPOLYCONE_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/GenericPolyconeStruct.h"
#include "VecGeom/volumes/kernel/GenericPolyconeImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"
#include "VecGeom/volumes/ReducedPolycone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedGenericPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedGenericPolycone);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// @brief Unplaced volume for generic polycones.
/// @details Stores the original `(r,z)` contour and a reduced section
/// decomposition used by `GenericPolyconeImplementation`. This shape supports
/// arbitrary radial contours that cannot be represented by the specialized
/// polycone implementation.
class UnplacedGenericPolycone : public UnplacedVolumeImplHelper<GenericPolyconeImplementation>, public AlignedBase {

private:
  GenericPolyconeStruct<Precision> fGenericPolycone;

  // Original Polycone Parameters
  Precision fSPhi;
  Precision fDPhi;
  int fNumRZ;
  Vector<Precision> fR;
  Vector<Precision> fZ;

  // Used for Extent
  Vector3D<Precision> fAMin;
  Vector3D<Precision> fAMax;

public:
  /// @brief Construct an empty generic polycone.
  VECCORE_ATT_HOST_DEVICE
  UnplacedGenericPolycone();

  /// @brief Construct a generic polycone from an `(r,z)` contour.
  /// @param phiStart Initial phi angle.
  /// @param phiTotal Total phi opening.
  /// @param numRZ Number of contour vertices in `(r,z)`; must be even.
  /// @param r Radial coordinates of the contour vertices.
  /// @param z Z coordinates of the contour vertices.
  VECCORE_ATT_HOST_DEVICE
  UnplacedGenericPolycone(Precision phiStart, Precision phiTotal, int numRZ, Precision const *r, Precision const *z);

  /// @brief Get the solid type identifier.
  /// @return `ESolidType::genericpolycone`.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::genericpolycone; }

  /// @brief Get the reduced runtime structure.
  /// @return Generic polycone storage used by kernels.
  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeStruct<Precision> const &GetStruct() const { return fGenericPolycone; }

  /// @brief Get the start phi angle.
  /// @return Start phi.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSPhi() const { return fSPhi; }

  /// @brief Get the total phi opening.
  /// @return Delta phi.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDPhi() const { return fDPhi; }

  /// @brief Get the number of input `(r,z)` contour vertices.
  /// @return Number of contour vertices.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetNumRz() const { return fNumRZ; }

  /// @brief Get the original input radial contour.
  /// @return Vector of radial coordinates.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Precision> GetR() const { return fR; }

  /// @brief Get the original input z contour.
  /// @return Vector of z coordinates.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Precision> GetZ() const { return fZ; }

  /// @brief Compute the axis-aligned bounding box.
  /// @param aMin Output lower extent.
  /// @param aMax Output upper extent.
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override;

  /// @brief Compute the outward normal at a surface point.
  /// @param point Local query point.
  /// @param norm Output normal.
  /// @return True if a normal was found.
  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const override;

  /// @brief Get the cached volume.
  /// @return Cubic volume.
  Precision Capacity() const override { return fGenericPolycone.fCubicVolume; }

  /// @brief Estimate the surface area using the generic unplaced-volume implementation.
  /// @return Estimated surface area.
  Precision SurfaceArea() const override { return EstimateSurfaceArea(); }

  /// @brief Sample a point on the solid surface.
  /// @return Local point on the surface.
  Vector3D<Precision> SamplePointOnSurface() const override;

  /// @brief Get the entity type name.
  /// @return Static entity type string.
  std::string GetEntityType() const { return "GenericPolycone"; }

public:
  /// @brief Get the host memory size of this unplaced volume.
  /// @return `sizeof(*this)`.
  virtual int MemorySize() const final { return sizeof(*this); }

  /// @brief Print the shape parameters to the default stream.
  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  /// @brief Print the shape parameters to a stream.
  /// @param os Output stream.
  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  /// @brief Get the CUDA device object size.
  /// @return Device allocation size.
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedGenericPolycone>::SizeOf(); }

  /// @brief Copy this unplaced volume to GPU memory.
  /// @return Device pointer to the copied volume.
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;

  /// @brief Copy this unplaced volume to an existing GPU allocation.
  /// @param gpu_ptr Destination device pointer.
  /// @return Device pointer to the copied volume.
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECCORE_CUDA
  /// @brief Construct a placed volume for the generic polycone.
  /// @param logical_volume Logical volume owning this solid.
  /// @param transformation Placement transform.
  /// @param placement Optional placement storage.
  /// @return Placed volume pointer.
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  /// @brief Construct the specialized placed-volume implementation.
  /// @param volume Logical volume owning this solid.
  /// @param transformation Placement transform.
  /// @param placement Optional placement storage.
  /// @return Placed volume pointer.
  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   VPlacedVolume *const placement) const override;
#else
  /// @brief Construct a placed volume on the device.
  /// @param logical_volume Logical volume owning this solid.
  /// @param transformation Placement transform.
  /// @param id Placed-volume id.
  /// @param copy_no Placed-volume copy number.
  /// @param child_id Child id.
  /// @param placement Optional placement storage.
  /// @return Placed volume pointer.
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, const int copy_no, const int child_id,
                               VPlacedVolume *const placement = NULL);

  /// @brief Construct the specialized placed-volume implementation on the device.
  /// @param volume Logical volume owning this solid.
  /// @param transformation Placement transform.
  /// @param id Placed-volume id.
  /// @param copy_no Placed-volume copy number.
  /// @param child_id Child id.
  /// @param placement Optional placement storage.
  /// @return Placed volume pointer.
  VECCORE_ATT_DEVICE VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation, const int id,
                                                      const int copy_no, const int child_id,
                                                      VPlacedVolume *const placement) const override;

#endif
};

using GenericUnplacedGenericPolycone = UnplacedGenericPolycone;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
