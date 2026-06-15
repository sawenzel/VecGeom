///===-- volumes/UnplacedTrapezoid.h ----------------------------------*- C++ -*-===//
///
/// \file volumes/UnplacedTrapezoid.h
/// \author Guilherme Lima (lima@fnal.gov)
/// \brief This file contains the declaration of the UnplacedTrapezoid class
///
/// _____________________________________________________________________________
/// A trapezoid is the solid bounded by the following surfaces:
/// - 2 XY-parallel quadrilaterals (trapezoids) cutting the Z axis at Z=-dz and Z=+dz
/// - 4 additional *planar* quadrilaterals intersecting the faces at +/-dz at their edges
///
/// The easiest way to think about the trapezoid is starting from a box with faces at +/-dz,
/// and moving its eight corners in such a way as to define the two +/-dz trapezoids, without
/// destroying the coplanarity of the other four faces.  Then the (x,y,z) coordinates of the
/// eight corners can be used to build this trapezoid.
///
/// If the four side faces are not coplanar, a generic trapezoid must be used (GenTrap).
//===------------------------------------------------------------------------===//
///
/// 140520 G. Lima   Created based on USolids algorithms and vectorized types
/// 160722 G. Lima   Migration to new helpers and VecCore-based scheme

#ifndef VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

#include "VecGeom/volumes/TrapezoidStruct.h" // the pure Trapezoid struct
#include "VecGeom/volumes/kernel/TrapezoidImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedTrapezoid;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedTrapezoid);

inline namespace VECGEOM_IMPL_NAMESPACE {

typedef Vector3D<Precision> TrapCorners[8];

class UnplacedTrapezoid : public UnplacedVolumeImplHelper<TrapezoidImplementation>, public AlignedBase {

private:
  TrapezoidStruct<Precision> fTrap;

  // variables to store cached values for Capacity and SurfaceArea
  // Precision fCubicVolume, fSurfaceArea;

public:
  // full constructor
  // Note: theta, phi are assumed to be in radians, for compatibility with Geant4
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrapezoid(const Precision dz, const Precision theta, const Precision phi, const Precision dy1,
                    const Precision dx1, const Precision dx2, const Precision Alpha1, const Precision dy2,
                    const Precision dx3, const Precision dx4, const Precision Alpha2)
      : fTrap(dz, theta, phi, dy1, dx1, dx2, vecgeom::Tan(Alpha1), dy2, dx3, dx4, vecgeom::Tan(Alpha2))
  {
    fGlobalConvexity = true;
    MakePlanes();
    ComputeBBox();
  }

  // default constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrapezoid() : fTrap(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)
  {
    fGlobalConvexity = true;
    ComputeBBox();
  }

  /// \brief Fast constructor: all parameters from one array
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrapezoid(Precision const *params)
      : UnplacedTrapezoid(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
                          params[8], params[9], params[10])
  {
    ComputeBBox();
  }

  /// \brief Constructor based on 8 corner points
  // convention: p0(---); p1(+--); p2(-+-); p3(++-); p4(--+); p5(+-+); p6(-++); p7(+++)
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrapezoid(TrapCorners const corners);

  /// \brief Constructor for masquerading a box (test purposes)
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrapezoid(Precision xbox, Precision ybox, Precision zbox);

  /// \brief Constructor required by Geant4
  VECCORE_ATT_HOST_DEVICE
  // Constructor corresponding to Trd1
  UnplacedTrapezoid(Precision dx1, Precision dx2, Precision dy, Precision dz);

  // Constructor corresponding to Trd2
  /// \brief Constructor for a Trd-like trapezoid
  UnplacedTrapezoid(Precision dx1, Precision dx2, Precision dy1, Precision dy2, Precision dz);

  /// \brief Constructor for a Parallelepiped-like trapezoid (Note: still to be validated)
  UnplacedTrapezoid(Precision dx, Precision dy, Precision dz, Precision alpha, Precision theta, Precision phi);

  /// \brief Accessors
  /// @{
  // VECCORE_ATT_HOST_DEVICE
  // TrapParameters const& GetParameters() const { return _params; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dz() const { return fTrap.fDz; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision theta() const { return fTrap.fTheta; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision phi() const { return fTrap.fPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy1() const { return fTrap.fDy1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy2() const { return fTrap.fDy2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx1() const { return fTrap.fDx1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx2() const { return fTrap.fDx2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx3() const { return fTrap.fDx3; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx4() const { return fTrap.fDx4; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision tanAlpha1() const { return fTrap.fTanAlpha1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision tanAlpha2() const { return fTrap.fTanAlpha2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision alpha1() const { return GetAlpha1(); } // note: slow, avoid using it

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision alpha2() const { return GetAlpha2(); } // note: slow, avoid using it

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision tanThetaCosPhi() const { return fTrap.fTthetaCphi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision tanThetaSinPhi() const { return fTrap.fTthetaSphi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fTrap.fDz; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTheta() const { return this->theta(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetPhi() const { return this->phi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDy1() const { return fTrap.fDy1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx1() const { return fTrap.fDx1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx2() const { return fTrap.fDx2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanAlpha1() const { return fTrap.fTanAlpha1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDy2() const { return fTrap.fDy2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx3() const { return fTrap.fDx3; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx4() const { return fTrap.fDx4; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanAlpha2() const { return fTrap.fTanAlpha2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanThetaSinPhi() const { return fTrap.fTthetaSphi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTanThetaCosPhi() const { return fTrap.fTthetaCphi; }
  /// @}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDz(Precision val) { fTrap.fDz = val; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetTheta(Precision val)
  {
    fTrap.fTheta = val;
    fTrap.CalculateCached();
    this->MakePlanes();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetPhi(Precision val)
  {
    fTrap.fPhi = val;
    fTrap.CalculateCached();
    this->MakePlanes();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDy1(Precision val) { fTrap.fDy1 = val; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDy2(Precision val) { fTrap.fDy2 = val; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDx1(Precision val) { fTrap.fDx1 = val; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDx2(Precision val) { fTrap.fDx2 = val; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDx3(Precision val) { fTrap.fDx3 = val; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDx4(Precision val) { fTrap.fDx4 = val; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetTanAlpha1(Precision val) { fTrap.fTanAlpha1 = val; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetTanAlpha2(Precision val) { fTrap.fTanAlpha2 = val; }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  // VECCORE_ATT_HOST_DEVICE
  // void CalcCapacity();

  // VECCORE_ATT_HOST_DEVICE
  // void CalcSurfaceArea();

  // VECCORE_ATT_HOST_DEVICE
  Precision Capacity() const override;
  // {
  //   Assert(!fOutdated);
  //   return fCubicVolume;
  // }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override;
  // {
  //   Assert(!fOutdated);
  //   return fSurfaceArea;
  // }

  Vector3D<Precision> SamplePointOnSurface() const override;

  Vector3D<Precision> GetPointOnPlane(Vector3D<Precision> const &p0, Vector3D<Precision> const &p1,
                                      Vector3D<Precision> const &p2, Vector3D<Precision> const &p3) const;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    bool valid = false;
    normal     = TrapezoidImplementation::NormalKernel(fTrap, point, valid);
    return valid;
  }

  std::string GetEntityType() const { return "Trapezoid"; }

  template <typename T>
  VECCORE_ATT_HOST_DEVICE void GetParametersList(int, T *aArray) const;

  VECCORE_ATT_HOST_DEVICE
  UnplacedTrapezoid *Clone() const;

public:
  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedTrapezoid>::SizeOf(); }

  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;

  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECCORE_CUDA
  // this is the function called from the VolumeFactory, it may be specific to the trapezoid
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

  // Note: use of ATan() makes this one slow -- to be avoided
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetAlpha1() const { return vecgeom::ATan(fTrap.fTanAlpha1); }

  // Note: use of Atan() makes this one slow -- to be avoided
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetAlpha2() const { return vecgeom::ATan(fTrap.fTanAlpha2); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::trapezoid; }

  // The next functions force upon the user insider knowledge about how the side planes should be used
  VECCORE_ATT_HOST_DEVICE
  TrapezoidStruct<Precision> const &GetStruct() const { return fTrap; }

  // #ifdef VECGEOM_PLANESHELL
  //   VECCORE_ATT_HOST_DEVICE
  //   VECGEOM_FORCE_INLINE
  //   PlaneShell<4,Precision> const *GetPlanes() const { return fTrap.GetPlanes(); }

  // #else
  //   using TrapSidePlane = TrapezoidStruct<Precision>::TrapSidePlane;
  //   VECCORE_ATT_HOST_DEVICE
  //   TrapSidePlane const *GetPlanes() const { return fTrap.GetPlanes(); }
  // #endif

  /// \brief Calculate trapezoid parameters when user provides the 8 corners
  VECCORE_ATT_HOST_DEVICE
  void fromCornersToParameters(TrapCorners const pt);

private:
  /// \brief Calculate the 8 corner points using pre-stored parameters, then use corners to build planes
  VECCORE_ATT_HOST_DEVICE
  void FromParametersToCorners(TrapCorners pt) const;

  // \brief Determine corner points using intersections of the pre-calculated planes
  VECCORE_ATT_HOST_DEVICE
  void fromPlanesToCorners(TrapCorners pt) const;

  /// \brief Construct the four side planes from input corner points
  VECCORE_ATT_HOST_DEVICE
  bool MakePlanes(TrapCorners const corners);

  /// \brief Construct the four side planes by converting stored parameters into TrapCorners object
  VECCORE_ATT_HOST_DEVICE
  bool MakePlanes();

/// \brief Construct the four side planes from input corner points
#ifdef VECGEOM_PLANESHELL
  VECCORE_ATT_HOST_DEVICE
  bool MakeAPlane(Vector3D<Precision> const &p1, Vector3D<Precision> const &p2, Vector3D<Precision> const &p3,
                  Vector3D<Precision> const &p4, unsigned int iplane);
#else
  VECCORE_ATT_HOST_DEVICE
  bool MakeAPlane(Vector3D<Precision> const &p1, Vector3D<Precision> const &p2, Vector3D<Precision> const &p3,
                  Vector3D<Precision> const &p4, TrapezoidStruct<Precision>::TrapSidePlane &plane);
#endif

public:
#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
  TGeoShape const *ConvertToRoot(char const *label) const;
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4(char const *label) const;
#endif
#endif
};

// Adding specialized factory.
template <>
struct Maker<UnplacedTrapezoid> {
  template <typename... ArgTypes>
  static UnplacedTrapezoid *MakeInstance(const Precision dz, const Precision theta, const Precision phi,
                                         const Precision dy1, const Precision dx1, const Precision dx2,
                                         const Precision Alpha1, const Precision dy2, const Precision dx3,
                                         const Precision dx4, const Precision Alpha2);

  template <typename... ArgTypes>
  static UnplacedTrapezoid *MakeInstance(TrapCorners const pt);
};

// Helper function to be used by all the Factories of Trapezoid
#ifndef VECGEOM_NO_SPECIALIZATION
UnplacedTrapezoid *GetSpecialized(const Precision dz, const Precision theta, const Precision phi, const Precision dy1,
                                  const Precision dx1, const Precision dx2, const Precision Alpha1, const Precision dy2,
                                  const Precision dx3, const Precision dx4, const Precision Alpha2);
#endif

using GenericUnplacedTrapezoid = UnplacedTrapezoid;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
