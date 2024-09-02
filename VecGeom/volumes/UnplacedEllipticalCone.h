// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the unplaced elliptical cone shape
/// @file volumes/UnplacedEllipticalCone.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_UNPLACEDELLIPTICALCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDELLIPTICALCONE_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/EllipticalConeStruct.h" // the pure EllipticalCone struct
#include "VecGeom/volumes/kernel/EllipticalConeImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedEllipticalCone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedEllipticalCone);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for elliptical cone shape primitive.
///
/// Elliptical cone is a full cone with elliptical base which can be cut in z.
/// The shape is centered at the origin: one base is at z=-zcut, another at z=zcut.
/// If there is no cut then the base is at z=-h the apex is at z=h.
/// Lateral surface of an elliptical cone is defined by the equation:
///
/// (x / a)^2 + (y / b)^2 = (h - z)^2
///
/// - a - undimensioned value, it specifies the inclination of the surface in x
/// - b - undimensioned value, it specifies the inclination of the surface in y
/// - h - height, z-coordinate where the uncut surface hits z axis
///
/// The semi-major axes at the z=0 plane are equal to a*h and b*h
///
class UnplacedEllipticalCone : public UnplacedVolumeImplHelper<EllipticalConeImplementation>, public AlignedBase {

private:
  EllipticalConeStruct<Precision> fEllipticalCone; ///< Structure holding the data for Elliptical Cone

private:
  /// Check correctness of the parameters and set the data members
  VECCORE_ATT_HOST_DEVICE
  void CheckParameters();

  /// Generates random point on the lateral surface of the elliptical cone
  Vector3D<Precision> SamplePointOnLateralSurface() const;

public:
  /// Constructor
  /// @param a Inclination of the conical surface in x
  /// @param b Inclination of the conical surface in y
  /// @param h Height
  /// @param zcut cut in z
  VECCORE_ATT_HOST_DEVICE
  UnplacedEllipticalCone(Precision a, Precision b, Precision h, Precision zcut);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::ellipticalcone; }

  /// Getter for the structure storing elliptical cone data.
  VECCORE_ATT_HOST_DEVICE
  EllipticalConeStruct<Precision> const &GetStruct() const { return fEllipticalCone; }

  /// Getter for the parameter that specifies inclination of the conical surface in x
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSemiAxisX() const { return fEllipticalCone.fDx; }

  /// Getter for that parameter that specifies inclination of the conical surface in y
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSemiAxisY() const { return fEllipticalCone.fDy; }

  /// Getter for the height
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZMax() const { return fEllipticalCone.fDz; }

  /// Getter for the cut in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZTopCut() const { return fEllipticalCone.fZCut; }

  /// Setter for the elliptical cone parameters
  /// @param a Inclination of the conical surface in x
  /// @param b Inclination of the conical surface in y
  /// @param h Height
  /// @param zcut cut in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetParameters(Precision a, Precision b, Precision h, Precision zcut)
  {
    fEllipticalCone.fDx   = a;
    fEllipticalCone.fDy   = b;
    fEllipticalCone.fDz   = h;
    fEllipticalCone.fZCut = zcut;
    CheckParameters();
  };

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fEllipticalCone.fCubicVolume; }

  Precision SurfaceArea() const override { return fEllipticalCone.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = EllipticalConeImplementation::NormalKernel(fEllipticalCone, p, valid);
    return valid;
  }

  /// Get the solid type as string.
  /// @return Name of the solid type
  std::string GetEntityType() const { return "EllipticalCone"; }

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
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedEllipticalCone>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECCORE_CUDA
  /// Templated factory for creating a placed volume
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   VPlacedVolume *const placement) const override;
#else
  VECCORE_ATT_DEVICE static VPlacedVolume *Create(LogicalVolume const *const logical_volume,
                                                  Transformation3D const *const transformation, const int id,
                                                  const int copy_no, const int child_id,
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
