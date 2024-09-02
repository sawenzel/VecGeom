// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @brief This file contains the declaration of the UnplacedParaboloid class
/// @file volumes/UnplacedParaboloid.h
/// @author Marilena Bandieramonte

#ifndef VECGEOM_VOLUMES_UNPLACEDPARABOLOID_H_
#define VECGEOM_VOLUMES_UNPLACEDPARABOLOID_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/ParaboloidStruct.h" // the pure Paraboloid struct
#include "VecGeom/volumes/kernel/ParaboloidImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedParaboloid;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedParaboloid);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for paraboloid shape primitive
///
/// A paraboloid is the solid bounded by the following surfaces:
/// - 2 planes parallel with XY cutting the Z axis at z = -dz and z = +dz
/// - the surface of revolution of a parabola described by: z = a * (x^2 + y^2) + b
///
/// The parameters a and b are automatically computed from:
/// - rlo - radius of the circle of intersection between the
/// parabolic surface and the plane z = -dz
/// - rhi - the radius of the circle of intersection between the
/// parabolic surface and the plane z = +dz
/// - dz = a * rhi^2 + b and  -dz = a * rlo^2 + b, where rhi > rlo, both >= 0
/// - a = 2 * dz * dd and b = -dz * (rlo^2 + rhi^2) * dd, where dd = 1 / (rhi^2 - rlo^2)
///
class UnplacedParaboloid : public UnplacedVolumeImplHelper<ParaboloidImplementation>, public AlignedBase {

private:
  ParaboloidStruct<Precision> fParaboloid; ///< The paraboloid structure

  Precision fCubicVolume; ///< Cached value of the volume
  Precision fSurfaceArea; ///< Cached value of the surface area

public:
  /// Default constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedParaboloid();

  /// Constructor
  /// @param rlo Radius of the circle at z = -dz
  /// @param rhi Radius of the circle at z = +dz
  /// @param dz Half size in z
  VECCORE_ATT_HOST_DEVICE
  UnplacedParaboloid(const Precision rlo, const Precision rhi, const Precision dz);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::paraboloid; }

  /// Getter for the structure storing the paraboloid data
  VECCORE_ATT_HOST_DEVICE
  ParaboloidStruct<Precision> const &GetStruct() const { return fParaboloid; }

  /// Getter for the raduis of the circle at z = -dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRlo() const { return fParaboloid.fRlo; }

  /// Getter for the raduis of the circle at z = +dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRhi() const { return fParaboloid.fRhi; }

  /// Getter for the half size in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fParaboloid.fDz; }

  /// Returns the parameter a of the paraboloid surface
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetA() const { return fParaboloid.fA; }

  /// Returns the parameter b of the paraboloid surface
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetB() const { return fParaboloid.fB; }

  /// Sets the raduis of the circle at z = -dz
  /// @param val Value of the radius
  VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetRlo(Precision val)
  {
    fParaboloid.SetRlo(val);
    CalcCapacity();
    CalcSurfaceArea();
  }

  /// Sets the raduis of the circle at z = +dz
  /// @param val Value of the radius
  VECCORE_ATT_HOST_DEVICE
  void SetRhi(Precision val)
  {
    fParaboloid.SetRhi(val);
    CalcCapacity();
    CalcSurfaceArea();
  }

  /// Sets the half size in z
  /// @param val Value of the half size in z
  VECCORE_ATT_HOST_DEVICE
  void SetDz(Precision val)
  {
    fParaboloid.SetDz(val);
    CalcCapacity();
    CalcSurfaceArea();
  }

  /// Sets all parameters of the paraboloid
  /// @param rlo Radius of the circle at z = -dz
  /// @param rhi Radius of the circle at z = +dz
  /// @param dz Half size in z
  VECCORE_ATT_HOST_DEVICE
  void SetRloAndRhiAndDz(Precision rlo, Precision rhi, Precision dz)
  {
    fParaboloid.SetRloAndRhiAndDz(rlo, rhi, dz);
    CalcCapacity();
    CalcSurfaceArea();
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  /// Calculate the volume
  VECCORE_ATT_HOST_DEVICE
  void CalcCapacity();

  /// Calculate the surface area
  VECCORE_ATT_HOST_DEVICE
  void CalcSurfaceArea();

  Precision Capacity() const override { return fCubicVolume; }

  Precision SurfaceArea() const override { return fSurfaceArea; }

  virtual Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid = false;
    normal     = ParaboloidImplementation::NormalKernel(fParaboloid, p, valid);
    return valid;
  }

  /// Get the solid type as string
  /// @return Name of the solid type
  std::string GetEntityType() const;

  /// Get list of the paraboloid parameters as an array. Not implemented !!!
  VECCORE_ATT_HOST_DEVICE
  void GetParametersList(int aNumber, Precision *aArray) const;

  VECCORE_ATT_HOST_DEVICE
  UnplacedParaboloid *Clone() const;

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
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedParaboloid>::SizeOf(); }
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
