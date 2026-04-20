/*
 * UnplacedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_UNPLACEDCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDCONE_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/ConeStruct.h"
#include "VecGeom/volumes/kernel/ConeImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedCone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedCone);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, SUnplacedCone, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Unplaced cone or conical tube section.
 *
 * Stores the intrinsic cone parameters and geometry queries independently of
 * any placement transform.
 *
 * Main parameters:
 * - `fCone.fDz`: half length in z; the full cone height is `2 * fCone.fDz`
 * - `fCone.fRmin1`: inner radius at `-fDz`
 * - `fCone.fRmin2`: inner radius at `+fDz`
 * - `fCone.fRmax1`: outer radius at `-fDz`
 * - `fCone.fRmax2`: outer radius at `+fDz`
 * - `fCone.fSPhi`: starting angle in radians
 * - `fCone.fDPhi`: angular span in radians
 */
class UnplacedCone : public VUnplacedVolume {

private:
  // cone parameters
  ConeStruct<Precision> fCone;

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedCone(Precision rmin1, Precision rmax1, Precision rmin2, Precision rmax2, Precision dz, Precision phimin,
               Precision deltaphi)
      : fCone(rmin1, rmax1, rmin2, rmax2, dz, phimin, deltaphi)
  {
    DetectConvexity();
    ComputeBBox();
  }

  // Constructor needed by specialization when Cone becomes Tube
  UnplacedCone(Precision rmin, Precision rmax, Precision dz, Precision phimin, Precision deltaphi)
      : fCone(rmin, rmax, rmin, rmax, dz, phimin, deltaphi)
  {
    DetectConvexity();
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::cone; }

  VECCORE_ATT_HOST_DEVICE
  ConeStruct<Precision> const &GetStruct() const { return fCone; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetInvSecRMax() const { return fCone.fInvSecRMax; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetInvSecRMin() const { return fCone.fInvSecRMin; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetTolIz() const { return fCone.fTolIz; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTolOz() const { return fCone.fTolOz; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmin1() const { return fCone.fSqRmin1; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmin2() const { return fCone.fSqRmin2; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmax1() const { return fCone.fSqRmax1; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSqRmax2() const { return fCone.fSqRmax2; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanRmax() const { return fCone.fTanRMax; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanRmin() const { return fCone.fTanRMin; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSecRmax() const { return fCone.fSecRMax; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSecRmin() const { return fCone.fSecRMin; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetZNormInner() const { return fCone.fZNormInner; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetZNormOuter() const { return fCone.fZNormOuter; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetInnerConeApex() const { return fCone.fInnerConeApex; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTInner() const { return fCone.fTanInnerApexAngle; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetOuterConeApex() const { return fCone.fOuterConeApex; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetTOuter() const { return fCone.fTanOuterApexAngle; }

  VECCORE_ATT_HOST_DEVICE
  void DetectConvexity();
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmin1() const { return fCone.fRmin1; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmax1() const { return fCone.fOriginalRmax1; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmin2() const { return fCone.fRmin2; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetRmax2() const { return fCone.fOriginalRmax2; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetDz() const { return fCone.fDz; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetSPhi() const { return fCone.fSPhi; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetDPhi() const { return fCone.fDPhi; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetInnerSlope() const { return fCone.fInnerSlope; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetOuterSlope() const { return fCone.fOuterSlope; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetInnerOffset() const { return fCone.fInnerOffset; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetOuterOffset() const { return fCone.fOuterOffset; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlongPhi1X() const { return fCone.fAlongPhi1x; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlongPhi1Y() const { return fCone.fAlongPhi1y; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlongPhi2X() const { return fCone.fAlongPhi2x; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlongPhi2Y() const { return fCone.fAlongPhi2y; }
  VECCORE_ATT_HOST_DEVICE
  evolution::Wedge const &GetWedge() const { return fCone.fPhiWedge; }

  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckSPhiAngle(Precision sPhi);

  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckDPhiAngle(Precision dPhi);

  void SetRmin1(Precision const &arg)
  {
    fCone.fRmin1 = arg;
    fCone.CalculateCached();
  }
  void SetRmax1(Precision const &arg)
  {
    fCone.fRmax1 = arg;
    fCone.CalculateCached();
  }
  void SetRmin2(Precision const &arg)
  {
    fCone.fRmin2 = arg;
    fCone.CalculateCached();
  }
  void SetRmax2(Precision const &arg)
  {
    fCone.fRmax2 = arg;
    fCone.CalculateCached();
  }
  void SetDz(Precision const &arg)
  {
    fCone.fDz = arg;
    fCone.CalculateCached();
  }
  void SetSPhi(Precision const &arg)
  {
    fCone.fSPhi = arg;
    fCone.SetAndCheckSPhiAngle(fCone.fSPhi);
    DetectConvexity();
  }
  void SetDPhi(Precision const &arg)
  {
    fCone.fDPhi = arg;
    fCone.SetAndCheckDPhiAngle(fCone.fDPhi);
    DetectConvexity();
  }

  VECCORE_ATT_HOST_DEVICE
  bool IsFullPhi() const { return fCone.fDPhi == kTwoPi; }

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;
  virtual void Print(std::ostream &os) const final;

  std::string GetEntityType() const { return "Cone"; }
  std::ostream &StreamInfo(std::ostream &os) const;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id, const int copy_no, const int child_id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override
  {
    return DevicePtr<cuda::SUnplacedCone<cuda::ConeTypes::UniversalCone>>::SizeOf();
  }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

  Precision Capacity() const override
  {
    return (fCone.fDz * fCone.fDPhi / 3.) *
           (fCone.fRmax1 * fCone.fRmax1 + fCone.fRmax2 * fCone.fRmax2 + fCone.fRmax1 * fCone.fRmax2 -
            fCone.fRmin1 * fCone.fRmin1 - fCone.fRmin2 * fCone.fRmin2 - fCone.fRmin1 * fCone.fRmin2);
  }

  Precision SurfaceArea() const override
  {
    Precision mmin, mmax, dmin, dmax;
    mmin = (fCone.fRmin1 + fCone.fRmin2) * 0.5;
    mmax = (fCone.fRmax1 + fCone.fRmax2) * 0.5;
    dmin = (fCone.fRmin2 - fCone.fRmin1);
    dmax = (fCone.fRmax2 - fCone.fRmax1);

    return fCone.fDPhi * (mmin * std::sqrt(dmin * dmin + 4 * fCone.fDz * fCone.fDz) +
                          mmax * std::sqrt(dmax * dmax + 4 * fCone.fDz * fCone.fDz) +
                          0.5 * (fCone.fRmax1 * fCone.fRmax1 - fCone.fRmin1 * fCone.fRmin1 +
                                 fCone.fRmax2 * fCone.fRmax2 - fCone.fRmin2 * fCone.fRmin2));
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override;

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;

  Vector3D<Precision> SamplePointOnSurface() const override;

  // Helper function to detect edge points.
  template <bool top>
  bool IsOnZPlane(Vector3D<Precision> const &point) const;
  template <bool start>
  bool IsOnPhiWedge(Vector3D<Precision> const &point) const;
  template <bool inner>
  bool IsOnConicalSurface(Vector3D<Precision> const &point) const;
  template <bool inner>
  Precision GetRadiusOfConeAtPoint(Precision const pointZ) const;

  bool IsOnEdge(Vector3D<Precision> &point) const;

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
struct Maker<UnplacedCone> {
  template <typename... ArgTypes>
  static UnplacedCone *MakeInstance(Precision const &_rmin1, Precision const &_rmax1, Precision const &_rmin2,
                                    Precision const &_rmax2, Precision const &_dz, Precision const &_phimin,
                                    Precision const &_deltaphi);
};

// this class finishes the implementation

template <typename ConeType = ConeTypes::UniversalCone>
class SUnplacedCone : public UnplacedVolumeImplHelper<ConeImplementation<ConeType>, UnplacedCone>,
                      public vecgeom::AlignedBase {
public:
  using Kernel     = ConeImplementation<ConeType>;
  using BaseType_t = UnplacedVolumeImplHelper<ConeImplementation<ConeType>, UnplacedCone>;
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
    return VolumeFactory::CreateByTransformation<SUnplacedCone<ConeType>>(volume, transformation, placement);
  }

#else
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation, const int id,
                                           const int copy_no, const int child_id,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedCone<ConeType>>(volume, transformation, id, copy_no, child_id,
                                                                          placement);
  }
#endif
};

using GenericUnplacedCone = SUnplacedCone<ConeTypes::UniversalCone>;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

// we include this header here because SpecializedCone
// implements the Create function of SUnplacedCone<> (and to avoid a circular dependency)
#include "VecGeom/volumes/SpecializedCone.h"

#endif // VECGEOM_VOLUMES_UNPLACEDCONE_H_
