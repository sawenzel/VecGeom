// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the unplaced parallelepiped shape.
/// @file volumes/UnplacedParallelepiped.h
/// @author: Johannes de Fine Licht, Mihaela Gheata

#ifndef VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "ParallelepipedStruct.h"
#include "VecGeom/volumes/kernel/ParallelepipedImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedParallelepiped;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedParallelepiped);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for parallelepiped shape primitive.
///
/// Parallepiped is a `skewed' box with half lengths dx, dy, dz.
/// Angles theta & phi are the polar and azimuthal angles of the line
/// joining centres of the faces at +/- dz.
/// Angle alpha is formed by the y-axis and the line joining
/// centres of the faces at +/- dy.

class UnplacedParallelepiped : public UnplacedVolumeImplHelper<ParallelepipedImplementation>, public AlignedBase {

private:
  ParallelepipedStruct<Precision> fPara; ///< The parallelepiped structure

public:
  using Kernel = ParallelepipedImplementation;

  /// Constructor from a vector of dimensions and three angles
  /// @param dim 3D vector with dx, dy, dz
  /// @param alpha Angle between y-axis and the line joining centres of the faces at +/- dy
  /// @param theta Polar angle
  /// @param phi Azimuthal angle
  VECCORE_ATT_HOST_DEVICE
  UnplacedParallelepiped(Vector3D<Precision> const &dimensions, const Precision alpha, const Precision theta,
                         const Precision phi)
      : fPara(dimensions, alpha, theta, phi)
  {
    fGlobalConvexity = true;
    ComputeBBox();
  }

  /// Constructor from three dimensions and three angles
  /// @param dx Half length in x
  /// @param dy Half length in y
  /// @param dz Half length in z
  /// @param alpha Angle between y-axis and the line joining centres of the faces at +/- dy
  /// @param theta Polar angle
  /// @param phi Azimuthal angle
  VECCORE_ATT_HOST_DEVICE
  UnplacedParallelepiped(const Precision dx, const Precision dy, const Precision dz, const Precision alpha,
                         const Precision theta, const Precision phi)
      : fPara(dx, dy, dz, alpha, theta, phi)
  {
    fGlobalConvexity = true;
    ComputeBBox();
  }

  /// Default constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedParallelepiped() : fPara(0., 0., 0., 0., 0., 0.) { fGlobalConvexity = true; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::parallelepiped; }

  /// Getter for the structure storing parallepiped data.
  VECCORE_ATT_HOST_DEVICE
  ParallelepipedStruct<Precision> const &GetStruct() const { return fPara; }

  /// Getter for parallelepiped dimensions
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetDimensions() const { return fPara.fDimensions; }

  /// Getter for parallelepiped normals
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetNormal(int i) const { return fPara.fNormals[i]; }

  /// Getter for dx
  VECCORE_ATT_HOST_DEVICE
  Precision GetX() const { return fPara.fDimensions[0]; }

  /// Getter for dy
  VECCORE_ATT_HOST_DEVICE
  Precision GetY() const { return fPara.fDimensions[1]; }

  /// Getter for dz
  VECCORE_ATT_HOST_DEVICE
  Precision GetZ() const { return fPara.fDimensions[2]; }

  /// Getter for alpha
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlpha() const { return fPara.fAlpha; }

  /// Getter for theta
  VECCORE_ATT_HOST_DEVICE
  Precision GetTheta() const { return fPara.fTheta; }

  /// Getter for phi
  VECCORE_ATT_HOST_DEVICE
  Precision GetPhi() const { return fPara.fPhi; }

  /// Getter for tan(alpha)
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanAlpha() const { return fPara.fTanAlpha; }

  /// Getter for tan(th)*sin(phi)
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanThetaSinPhi() const { return fPara.fTanThetaSinPhi; }

  /// Getter for tan(th)*cos(phi)
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanThetaCosPhi() const { return fPara.fTanThetaCosPhi; }

  /// Getter for scale factor fCtx
  VECCORE_ATT_HOST_DEVICE
  Precision GetCtx() const { return fPara.fCtx; }

  /// Getter for scale factor fCty
  VECCORE_ATT_HOST_DEVICE
  Precision GetCty() const { return fPara.fCty; }

  /// Setter for dimensions in x, y, z
  /// @param dimension Vector with length in x, y, z
  VECCORE_ATT_HOST_DEVICE
  void SetDimensions(Vector3D<Precision> const &dimensions) { fPara.fDimensions = dimensions; }

  /// Setter for dimensions in x, y, z
  /// @param dx Half length in x
  /// @param dy Half length in y
  /// @param dz Half length in z
  VECCORE_ATT_HOST_DEVICE
  void SetDimensions(const Precision dx, const Precision dy, const Precision dz) { fPara.fDimensions.Set(dx, dy, dz); }

  /// Setter for alpha
  /// @param alpha Angle between y-axis and the line joining centres of the faces at +/- dy
  VECCORE_ATT_HOST_DEVICE
  void SetAlpha(const Precision alpha) { fPara.SetAlpha(alpha); }

  /// Setter for theta
  /// @param theta Polar angle
  VECCORE_ATT_HOST_DEVICE
  void SetTheta(const Precision theta) { fPara.SetTheta(theta); }

  /// Setter for phi
  /// @param phi Azimuthal angle
  VECCORE_ATT_HOST_DEVICE
  void SetPhi(const Precision phi) { fPara.SetPhi(phi); }

  /// Setter for theta and phi
  /// @param theta Polar angle
  /// @param phi Azimuthal angle
  VECCORE_ATT_HOST_DEVICE
  void SetThetaAndPhi(const Precision theta, const Precision phi) { fPara.SetThetaAndPhi(theta, phi); }

  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;

  virtual void Print(std::ostream &os) const final;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  /// Implementation of capacity computation
  Precision volume() const { return 8.0 * fPara.fDimensions[0] * fPara.fDimensions[1] * fPara.fDimensions[2]; }

  Precision Capacity() const override { return volume(); }

  Vector3D<Precision> SamplePointOnSurface() const override;

  Precision SurfaceArea() const override { return 2. * (fPara.fAreas[0] + fPara.fAreas[1] + fPara.fAreas[2]); }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = ParallelepipedImplementation::NormalKernel(fPara, point, valid);
    return valid;
  }

  /// Get the solid type as string.
  /// @return Name of the solid type
  std::string GetEntityType() const { return "parallelepiped"; }

  /// Templated factory for creating a placed volume
#ifdef VECCORE_CUDA
  VECCORE_ATT_DEVICE
#endif
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id, const int copy_no, const int child_id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedParallelepiped>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifdef VECCORE_CUDA
  VECCORE_ATT_DEVICE
#endif
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                           const int id, const int copy_no, const int child_id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
