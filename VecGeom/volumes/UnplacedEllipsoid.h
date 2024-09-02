// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the unplaced ellipsoid shape
/// @file volumes/UnplacedEllipsoid.h
/// @author Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_UNPLACEDELLIPSOID_H_
#define VECGEOM_VOLUMES_UNPLACEDELLIPSOID_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/EllipsoidStruct.h"
#include "VecGeom/volumes/kernel/EllipsoidImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedEllipsoid;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedEllipsoid);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for ellipsoid shape primitive
///
/// Ellipsoid shape is limited by a quadratic surface given by the equation:
///
/// (x/dx)^2 + (y/dy)^2 + (z/dz)^2 = 1
///
/// where dx, dy and dz are lengths of the semi-axes. The shape is centered at the origin
/// and and optionally may have cuts in z.
///
class UnplacedEllipsoid : public UnplacedVolumeImplHelper<EllipsoidImplementation>, public AlignedBase {

private:
  EllipsoidStruct<Precision> fEllipsoid;

  /// Check correctness of the parameters and set the data members
  VECCORE_ATT_HOST_DEVICE
  void CheckParameters();

  /// Compute area of lateral surface
  VECCORE_ATT_HOST_DEVICE
  Precision LateralSurfaceArea() const;

public:
  /// Default constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedEllipsoid();

  /// Constructor
  /// @param dx Length of x semi-axis
  /// @param dy Length of y semi-axis
  /// @param dz Length of z semi-axis
  /// @param zBottomCut Bottom cut in z, shape lies above this plane
  /// @param zTopCut Top cut in z, shape lies below this plane
  VECCORE_ATT_HOST_DEVICE
  UnplacedEllipsoid(Precision dx, Precision dy, Precision dz, Precision zBottomCut = 0., Precision zTopCut = 0.);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::ellipsoid; }

  /// Getter for the structure storing ellipsoid data
  VECCORE_ATT_HOST_DEVICE
  EllipsoidStruct<Precision> const &GetStruct() const { return fEllipsoid; }

  /// Getter for x semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx() const { return fEllipsoid.fDx; }

  /// Getter for y semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDy() const { return fEllipsoid.fDy; }

  /// Getter for z semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fEllipsoid.fDz; }

  /// Getter for bottom cut in z, return -dz if the cut is not set
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZBottomCut() const { return fEllipsoid.fZBottomCut; }

  /// Getter for top cut in z, return +dz if the cut is not set
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZTopCut() const { return fEllipsoid.fZTopCut; }

  /// Set ellipsoid dimensions
  /// @param dx Length of x semi-axis
  /// @param dy Length of y semi-axis
  /// @param dy Length of z semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetSemiAxes(Precision dx, Precision dy, Precision dz)
  {
    fEllipsoid.fDx = dx;
    fEllipsoid.fDy = dy;
    fEllipsoid.fDz = dz;
    CheckParameters();
  };

  /// Set cuts in z
  /// @param zBottomCut Bottom cut in z, shape lies above this plane
  /// @param zTopCut Top cut in z, shape lies below this plane
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetZCuts(Precision zBottomCut, Precision zTopCut)
  {
    fEllipsoid.fZBottomCut = zBottomCut;
    fEllipsoid.fZTopCut    = zTopCut;
    CheckParameters();
  };

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fEllipsoid.fCubicVolume; }

  Precision SurfaceArea() const override { return fEllipsoid.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = EllipsoidImplementation::NormalKernel(fEllipsoid, p, valid);
    return valid;
  }

  /// Return type of the solid as a string
  std::string GetEntityType() const { return "Ellipsoid"; }

  /// Dump volume parameters to an output stream
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
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedEllipsoid>::SizeOf(); }
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
