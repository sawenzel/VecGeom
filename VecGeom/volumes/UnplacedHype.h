// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Declaration of the unplaced hyperboloid shape
/// \file volumes/UnplacedHype.h
/// \author First version created by Marilena Bandieramonte (CERN).

#ifndef VECGEOM_VOLUMES_UNPLACEDHYPE_H_
#define VECGEOM_VOLUMES_UNPLACEDHYPE_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/HypeStruct.h"
#include "VecGeom/volumes/kernel/HypeImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedHype;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedHype);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, SUnplacedHype, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

/** Class for hyperboloid shape primitive.

  Hyperboloid class is defined by 5 parameters
  A Hype is the solid bounded by the following surfaces:
  - 2 planes parallel with XY cutting the Z axis at Z=-dz and Z=+dz
  - Inner and outer lateral surfaces. These represent the surfaces
  described by the revolution of 2 hyperbolas about the Z axis:
  r^2 - (t*z)^2 = a^2 where:
  r = distance between hyperbola and Z axis at coordinate z
  t = tangent of the stereo angle (angle made by hyperbola asimptotic lines and Z axis). t=0 means cylindrical
  surface.
  a = distance between hyperbola and Z axis at z=0
*/
class UnplacedHype : public VUnplacedVolume {

private:
  HypeStruct<Precision> fHype; ///< Structure holding the data for Hype

public:
  /// Default constructor for the unplaced hyperboloid.
  /** The constructor takes 5 parameters: inner and outer radius, stereo angles and half length in Z.
      @param rMin  Inner radius.
      @param rMax  Outer radius.
      @param stIn  Stereo angle for inner surface.
      @param stOut Stereo angle for outer surface.
      @param dz    Half length in Z.
  */
  VECCORE_ATT_HOST_DEVICE
  UnplacedHype(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut,
               const Precision dz)
      : fHype(rMin, rMax, stIn, stOut, dz)
  {
    DetectConvexity();
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::hyperboloid; }

  /// Getter for the structure storing hyperboloid data.
  VECCORE_ATT_HOST_DEVICE
  HypeStruct<Precision> const &GetStruct() const { return fHype; }

  /// Getter for tolerance relative to the Z half-length.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZToleranceLevel() const { return fHype.zToleranceLevel; }

  /// Getter for tolerance relative to the inner radius.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInnerRadToleranceLevel() const { return fHype.innerRadToleranceLevel; }

  /// Getter for tolerance relative to the outer radius.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetOuterRadToleranceLevel() const { return fHype.outerRadToleranceLevel; }

  /// Getter for the inner radius.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin() const { return fHype.fRmin; }

  /// Getter for the outer radius.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax() const { return fHype.fRmax; }

  /// Getter for the squared inner radius.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin2() const { return fHype.fRmin2; }

  /// Getter for the squared outer radius.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax2() const { return fHype.fRmax2; }

  /// Getter for the inner angle of the inner surface.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStIn() const { return fHype.fStIn; }

  /// Getter for the inner angle of the outer surface.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStOut() const { return fHype.fStOut; }

  /// Getter for the tangent of the inner stereo angle.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn() const { return fHype.fTIn; }

  /// Getter for the tangent of the outer stereo angle.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut() const { return fHype.fTOut; }

  /// Getter for the squared tangent of the inner stereo angle.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn2() const { return fHype.fTIn2; }

  /// Getter for the squared tangent of the outer stereo angle.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut2() const { return fHype.fTOut2; }

  /// Getter for the inverse of the tangent of the inner stereo angle.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn2Inv() const { return fHype.fTIn2Inv; }

  /// Getter for the inverse of the tangent of the outer stereo angle.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut2Inv() const { return fHype.fTOut2Inv; }

  /// Getter for the half-length in Z.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fHype.fDz; }

  /// Getter for the squared half-length in Z.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz2() const { return fHype.fDz2; }

  /// Getter for the inner radius of endcaps.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndInnerRadius() const { return fHype.fEndInnerRadius; }

  /// Getter for the squared inner radius of endcaps.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndInnerRadius2() const { return fHype.fEndInnerRadius2; }

  /// Getter for the outer radius of endcaps.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndOuterRadius() const { return fHype.fEndOuterRadius; }

  /// Getter for the squared outer radius of endcaps.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndOuterRadius2() const { return fHype.fEndOuterRadius2; }

  /// Getter for the side of the square inscribed in the inner circle.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInSqSide() const { return fHype.fInSqSide; }

  /// Method to set the parameters of the hyperboloid, used by the constructor.
  /** @param rMin  Inner radius.
      @param rMax  Outer radius.
      @param stIn  Stereo angle for inner surface.
      @param stOut Stereo angle for outer surface.
      @param dz    Half length in Z.
  */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetParameters(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut,
                     const Precision dz)
  {
    fHype.SetParameters(rMin, rMax, stIn, stOut, dz);
    DetectConvexity();
  }

  /// Method to compute the volume of the inner/outer hyperboloids.
  /** @param  outer Flag if it is about outer surface or inner one.
      @return The value of the volume for the inner/outer hype.
  */
  VECCORE_ATT_HOST_DEVICE
  Precision Volume(bool outer);

  /// Method to compute the surface area of the inner/outer hyperboloids.
  /** @param  outer Flag if it is about outer surface or inner one.
      @return The value of the area for the inner/outer hype.
  */
  VECCORE_ATT_HOST_DEVICE
  Precision Area(bool outer);

  /// Method to compute the area of hyperboloid endcaps.
  /** @return The value of the area for endcaps.*/
  VECCORE_ATT_HOST_DEVICE
  Precision AreaEndCaps();

  /// Method to compute and cache the capacity of the hyperboloid.
  VECCORE_ATT_HOST_DEVICE
  void CalcCapacity();

  /// Method to compute and cache the surface area of the hyperboloid.
  VECCORE_ATT_HOST_DEVICE
  void CalcSurfaceArea();

  /// Method to detect and cache the convexity of the hyperboloid.
  VECCORE_ATT_HOST_DEVICE
  void DetectConvexity();

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fHype.fCubicVolume; }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override { return fHype.fSurfaceArea; }

  /// Method to determine the squared hyperboloid radius at a given Z, for either the inner or the outer surfaces
  /// (template parameter).
  /** @param  dz Value of the Z coordinate.
      @return Squared radius of the hyperboloid surface.
  */
  template <bool ForInnerSurface>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Precision GetHypeRadius2(Precision dz) const
  {
    if (ForInnerSurface)
      return GetRmin2() + GetTIn2() * dz * dz;
    else
      return GetRmax2() + GetTOut2() * dz * dz;
  }

  /// Method to check if a point has a Z coordinate compatible with one of the Z surfaces.
  /** @param  p Point for which the check is made.
      @return True if point is within Z tolerance.
  */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool PointOnZSurface(Vector3D<Precision> const &p) const
  {
    return (p.z() > (GetDz() - GetZToleranceLevel())) && (p.z() < (GetDz() + GetZToleranceLevel()));
  }

  /// Method to check if a point is on the inner/outer hype surface (template parameter).
  /** @param  p Point for which the check is made.
      @return True if point is on the hype surface within the outer radius tolerance.
  */
  template <bool ForInnerSurface>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool PointOnHyperbolicSurface(Vector3D<Precision> const &p) const
  {
    Precision hypeR2    = 0.;
    hypeR2              = GetHypeRadius2<ForInnerSurface>(p.z());
    Precision pointRad2 = p.Perp2();
    return ((pointRad2 > (hypeR2 - GetOuterRadToleranceLevel())) &&
            (pointRad2 < (hypeR2 + GetOuterRadToleranceLevel())));
  }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {

    bool valid = true;

    Precision absZ(std::fabs(p.z()));
    Precision distZ(absZ - GetDz());
    Precision dist2Z(distZ * distZ);

    Precision xR2 = p.Perp2();
    Precision dist2Outer(std::fabs(xR2 - GetHypeRadius2<false>(absZ)));
    Precision dist2Inner(std::fabs(xR2 - GetHypeRadius2<true>(absZ)));

    // EndCap
    if (PointOnZSurface(p) || ((dist2Z < dist2Inner) && (dist2Z < dist2Outer)))
      normal = Vector3D<Precision>(0.0, 0.0, p.z() < 0 ? -1.0 : 1.0);

    // OuterHyperbolic Surface
    if (PointOnHyperbolicSurface<false>(p) || ((dist2Outer < dist2Inner) && (dist2Outer < dist2Z)))
      normal = Vector3D<Precision>(p.x(), p.y(), -p.z() * GetTOut2()).Unit();

    // InnerHyperbolic Surface
    if (PointOnHyperbolicSurface<true>(p) || ((dist2Inner < dist2Outer) && (dist2Inner < dist2Z)))
      normal = Vector3D<Precision>(-p.x(), -p.y(), p.z() * GetTIn2()).Unit();

    return valid;
  }

  Vector3D<Precision> SamplePointOnSurface() const override;

  /// Get the solid type as string.
  /** @return Name of the solid type.*/
  std::string GetEntityType() const;

  /// Get list of hyperboloid parameters as an array.
  /** @param[in]  aNumber Not used.
      @param[out] aArray User array to be filled (rMin, stIn, rMax, stOut, dz)
  */
  VECCORE_ATT_HOST_DEVICE
  void GetParametersList(int aNumber, Precision *aArray) const;

  VECCORE_ATT_HOST_DEVICE
  UnplacedHype *Clone() const;

  std::ostream &StreamInfo(std::ostream &os) const;

  /// Method to determine of the inner surface exists.
  /** @return True if the inner radius is zero. */
  VECCORE_ATT_HOST_DEVICE
  bool InnerSurfaceExists() const;

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override
  {
    return DevicePtr<cuda::SUnplacedHype<cuda::HypeTypes::UniversalHype>>::SizeOf();
  }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

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
struct Maker<UnplacedHype> {
  template <typename... ArgTypes>
  static UnplacedHype *MakeInstance(const Precision rMin, const Precision rMax, const Precision stIn,
                                    const Precision stOut, const Precision dz);
};

/** Specialized version of the unplaced hyperboloid, supporting universal/hollow/non-hollow types.*/
template <typename HypeType = HypeTypes::UniversalHype>
class SUnplacedHype : public UnplacedVolumeImplHelper<HypeImplementation<HypeType>, UnplacedHype>, public AlignedBase {
public:
  using BaseType_t = UnplacedVolumeImplHelper<HypeImplementation<HypeType>, UnplacedHype>;
  using BaseType_t::BaseType_t;

  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id, const int copy_no, const int child_id,
#endif
                               VPlacedVolume *const placement = NULL);

private:
#ifndef VECCORE_CUDA
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedHype<HypeType>>(volume, transformation, placement);
  }

#else
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation, const int id,
                                           const int copy_no, const int child_id,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedHype<HypeType>>(volume, transformation, id, copy_no, child_id,
                                                                          placement);
  }
#endif
};

using GenericUnplacedHype = SUnplacedHype<HypeTypes::UniversalHype>;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#include "VecGeom/volumes/SpecializedHype.h"

#endif // VECGEOM_VOLUMES_UNPLACEDHYPE_H_
