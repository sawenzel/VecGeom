// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the unplaced elliptical tube shape
/// @file volumes/UnplacedEllipticalTube.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_UNPLACEDELLIPTICALTUBE_H_
#define VECGEOM_VOLUMES_UNPLACEDELLIPTICALTUBE_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/EllipticalTubeStruct.h" // the pure EllipticalTube struct
#include "VecGeom/volumes/kernel/EllipticalTubeImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedEllipticalTube;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedEllipticalTube);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for elliptical tube shape primitive
///
/// Elliptical tube is a whole cylinder with elliptic cross section.
/// The shape is centered at the origin, its half length is equal to dz.
/// The equation of the ellipse in cross section is:
///
/// (x/dx)^2 + (y/dy)^2 = 1
///
class UnplacedEllipticalTube : public UnplacedVolumeImplHelper<EllipticalTubeImplementation>, public AlignedBase {

private:
  EllipticalTubeStruct<Precision> fEllipticalTube;

  /// Check correctness of the parameters and set the data members
  VECCORE_ATT_HOST_DEVICE
  void CheckParameters();

public:
  /// Constructor
  /// @param dx Length of x semi-axis
  /// @param dy Length of y semi-axis
  /// @param dz Half length in z
  VECCORE_ATT_HOST_DEVICE
  UnplacedEllipticalTube(Precision dx, Precision dy, Precision dz);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::ellipticaltube; }

  /// Getter for the structure storing elliptical cone data
  VECCORE_ATT_HOST_DEVICE
  EllipticalTubeStruct<Precision> const &GetStruct() const { return fEllipticalTube; }

  /// Getter for x semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx() const { return fEllipticalTube.fDx; }

  /// Getter for y semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDy() const { return fEllipticalTube.fDy; }

  /// Getter for the half length in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fEllipticalTube.fDz; }

  /// Setter for the elliptical tube parameters
  /// @param dx Length of x semi-axis
  /// @param dy Length of y semi-axis
  /// @param dz Half length in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetParameters(Precision dx, Precision dy, Precision dz)
  {
    fEllipticalTube.fDx = dx;
    fEllipticalTube.fDy = dy;
    fEllipticalTube.fDz = dz;
    CheckParameters();
  };

  /// Set x semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDx(Precision dx)
  {
    fEllipticalTube.fDx = dx;
    CheckParameters();
  };

  /// Set y semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDy(Precision dy)
  {
    fEllipticalTube.fDy = dy;
    CheckParameters();
  };

  /// Set half length in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDz(Precision dz)
  {
    fEllipticalTube.fDz = dz;
    CheckParameters();
  };

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fEllipticalTube.fCubicVolume; }

  Precision SurfaceArea() const override { return fEllipticalTube.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = EllipticalTubeImplementation::NormalKernel(fEllipticalTube, p, valid);
    return valid;
  }

  /// Get the solid type as string
  /// @return Name of the solid type
  std::string GetEntityType() const { return "EllipticalTube"; }

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedEllipticalTube>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

  /// Templated factory for creating a placed volume
#ifndef VECCORE_CUDA
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   VPlacedVolume *const placement) const override;
#else
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, const int copy_no, const int child_id,
                               VPlacedVolume *const placement = NULL);
  VECCORE_ATT_DEVICE VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation, const int id,
                                                      const int copy_no, const int child_id,
                                                      VPlacedVolume *const placement) const override;

#endif
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
