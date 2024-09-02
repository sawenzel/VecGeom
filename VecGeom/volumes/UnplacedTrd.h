// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the unplaced Trd shape
/// @file volumes/UnplacedTrd.h
/// @author Georgios Bitzes

#ifndef VECGEOM_VOLUMES_UNPLACEDTRD_H_
#define VECGEOM_VOLUMES_UNPLACEDTRD_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "TrdStruct.h"
#include "VecGeom/volumes/kernel/TrdImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedTrd;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedTrd);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, SUnplacedTrd, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for the Trd shape primitive
///
/// Trd is a trapezoid with x and y dimensions varying along z:
/// - bases at z = -dz and z = +dz are rectangles
/// - lateral surface consists of four isosceles trapeziums
class UnplacedTrd : public VUnplacedVolume {
private:
  TrdStruct<Precision> fTrd; ///< Structure with trapezoid parameters

public:
  /// Default constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd() : fTrd()
  {
    fGlobalConvexity = true;
    ComputeBBox();
  }

  /// Constructor, special case where Trd is a box
  /// @param x Half-length in x
  /// @param y Half-length in y
  /// @param z Half-length in z
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd(const Precision x, const Precision y, const Precision z) : fTrd(x, y, z)
  {
    fGlobalConvexity = true;
    ComputeBBox();
  }

  /// Constructor, special case where y dimension remains constant
  /// @param x1 Half-length along x at the surface positioned at -dz
  /// @param x2 Half-length along x at the surface positioned at +dz
  /// @param y Half-length in y
  /// @param z Half-length in z
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd(const Precision x1, const Precision x2, const Precision y, const Precision z) : fTrd(x1, x2, y, z)
  {
    fGlobalConvexity = true;
    ComputeBBox();
  }

  /// Constructor
  /// @param x1 Half-length along x at the surface positioned at -dz
  /// @param x2 Half-length along x at the surface positioned at +dz
  /// @param y1 Half-length along y at the surface positioned at -dz
  /// @param y2 Half-length along y at the surface positioned at +dz
  /// @param z Half-length along z axis
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd(const Precision x1, const Precision x2, const Precision y1, const Precision y2, const Precision z)
      : fTrd(x1, x2, y1, y2, z)
  {
    fGlobalConvexity = true;
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::trd; }

  /// Getter for the structure storing Trd data
  VECCORE_ATT_HOST_DEVICE
  TrdStruct<Precision> const &GetStruct() const { return fTrd; }

  /// Setter for all parameters
  /// @param x1 Half-length along x at the surface positioned at -dz
  /// @param x2 Half-length along x at the surface positioned at +dz
  /// @param y1 Half-length along y at the surface positioned at -dz
  /// @param y2 Half-length along y at the surface positioned at +dz
  /// @param z Half-length along z axis
  VECCORE_ATT_HOST_DEVICE
  void SetAllParameters(Precision x1, Precision x2, Precision y1, Precision y2, Precision z)
  {
    fTrd.SetAllParameters(x1, x2, y1, y2, z);
  }

  /// Setter for half-length along x at -dz
  VECCORE_ATT_HOST_DEVICE
  void SetXHalfLength1(Precision arg)
  {
    fTrd.fDX1 = arg;
    fTrd.CalculateCached();
  }

  /// Setter for half-length along x at +dz
  VECCORE_ATT_HOST_DEVICE
  void SetXHalfLength2(Precision arg)
  {
    fTrd.fDX2 = arg;
    fTrd.CalculateCached();
  }

  /// Setter for half-length along y at -dz
  VECCORE_ATT_HOST_DEVICE
  void SetYHalfLength1(Precision arg)
  {
    fTrd.fDY1 = arg;
    fTrd.CalculateCached();
  }

  /// Setter for half-length along y at +dz
  VECCORE_ATT_HOST_DEVICE
  void SetYHalfLength2(Precision arg)
  {
    fTrd.fDY2 = arg;
    fTrd.CalculateCached();
  }

  /// Setter for half-length along z
  VECCORE_ATT_HOST_DEVICE
  void SetZHalfLength(Precision arg)
  {
    fTrd.fDZ = arg;
    fTrd.CalculateCached();
  }

  /// Getter for half-length along x at -dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx1() const { return fTrd.fDX1; }

  /// Getter for half-length along x at +dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx2() const { return fTrd.fDX2; }

  /// Getter for half-length along y at -dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy1() const { return fTrd.fDY1; }

  /// Getter for half-length along y at +dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy2() const { return fTrd.fDY2; }

  /// Getter for half-length along z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dz() const { return fTrd.fDZ; }

  /// Return difference between half-legths along x at +dz and -dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision x2minusx1() const { return fTrd.fX2minusX1; }

  /// Return difference between half-legths along y at +dz and -dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision y2minusy1() const { return fTrd.fY2minusY1; }

  /// Return half-length along x at z = 0
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision halfx1plusx2() const { return fTrd.fHalfX1plusX2; }

  /// Return half-length along y at z = 0
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision halfy1plusy2() const { return fTrd.fHalfY1plusY2; }

  /// Return tangent of inclination angle along x
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision fx() const { return fTrd.fFx; }

  /// Return tangent of inclination angle along x
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision fy() const { return fTrd.fFy; }

  /// Return absolute value of cosine of inclination angle along x
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision calfx() const { return fTrd.fCalfX; }

  /// Return absolute value of cosine of inclination angle along y
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision calfy() const { return fTrd.fCalfY; }

  /// Return corrected tolerance for Inside checks on x
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision ToleranceX() const { return fTrd.fToleranceX; }

  /// Return corrected tolerance for Inside checks on y
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision ToleranceY() const { return fTrd.fToleranceY; }

  // virtual int MemorySize() const final { return sizeof(*this); }
  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    aMin = Vector3D<Precision>(-Max(fTrd.fDX1, fTrd.fDX2), -Max(fTrd.fDY1, fTrd.fDY2), -fTrd.fDZ);
    aMax = Vector3D<Precision>(Max(fTrd.fDX1, fTrd.fDX2), Max(fTrd.fDY1, fTrd.fDY2), fTrd.fDZ);
  }

  // Computes capacity of the shape in [length^3]
  Precision Capacity() const override;

  // Computes surface are in [length^2]
  Precision SurfaceArea() const override;

  Vector3D<Precision> SamplePointOnSurface() const override;

  /// Return area of face at +x
  Precision GetPlusXArea() const { return 2 * fTrd.fDZ * (fTrd.fDY1 + fTrd.fDY2) * fTrd.fSecxz; }

  /// Return area of face at -x
  Precision GetMinusXArea() const { return GetPlusXArea(); }

  /// Return area of face at +y
  Precision GetPlusYArea() const { return 2 * fTrd.fDZ * (fTrd.fDX1 + fTrd.fDX2) * fTrd.fSecyz; }

  /// Return area of face at -y
  Precision GetMinusYArea() const { return GetPlusYArea(); }

  /// Return area of face at +z
  Precision GetPlusZArea() const { return 4 * fTrd.fDX2 * fTrd.fDY2; }

  /// Return area of face at -z
  Precision GetMinusZArea() const { return 4 * fTrd.fDX1 * fTrd.fDY1; }

  /// Select surface for sampling point
  int ChooseSurface() const;

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;

  virtual void Print(std::ostream &os) const final;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

  /// Get the solid type as string
  /// @return Name of the solid type
  std::string GetEntityType() const { return "Trd"; }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override
  {
    return DevicePtr<cuda::SUnplacedTrd<cuda::TrdTypes::UniversalTrd>>::SizeOf();
  }

  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

  std::ostream &StreamInfo(std::ostream &os) const;

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
  TGeoShape const *ConvertToRoot(char const *label) const;
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4(char const *label) const;
#endif
#endif
};

template <>
struct Maker<UnplacedTrd> {
  template <typename... ArgTypes>
  static UnplacedTrd *MakeInstance(const Precision x1, const Precision x2, const Precision y1, const Precision y2,
                                   const Precision z);
  template <typename... ArgTypes>
  static UnplacedTrd *MakeInstance(const Precision x1, const Precision x2, const Precision y1, const Precision z);
};

template <typename TrdType = TrdTypes::UniversalTrd>
class SUnplacedTrd : public UnplacedVolumeImplHelper<TrdImplementation<TrdType>, UnplacedTrd>, public AlignedBase {
public:
  using Kernel     = TrdImplementation<TrdType>;
  using BaseType_t = UnplacedVolumeImplHelper<TrdImplementation<TrdType>, UnplacedTrd>;
  using BaseType_t::BaseType_t;

  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id, const int copy_no, const int child_id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifndef VECCORE_CUDA
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedTrd<TrdType>>(volume, transformation, placement);
  }

#else
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation, const int id,
                                           const int copy_no, const int child_id,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedTrd<TrdType>>(volume, transformation, id, copy_no, child_id,
                                                                        placement);
  }
#endif
};

using GenericUnplacedTrd = SUnplacedTrd<TrdTypes::UniversalTrd>;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#include "VecGeom/volumes/SpecializedTrd.h"

#endif // VECGEOM_VOLUMES_UNPLACEDTRD_H_
