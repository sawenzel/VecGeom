#ifndef VECGEOM_TUBESTRUCT_H_
#define VECGEOM_TUBESTRUCT_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/Wedge_Evolution.h"

namespace vecgeom {
// VECGEOM_DEVICE_FORWARD_DECLARE( struct TubeStruct; )
// VECGEOM_DEVICE_DECLARE_BoxStruct )

inline namespace VECGEOM_IMPL_NAMESPACE {

// a plain and lightweight struct to encapsulate data members of a tube
template <typename T = double>
struct TubeStruct {
  // tube defining parameters
  T fRmin{0.}; //< inner radius
  T fRmax{0.}; //< outer radius
  T fZ{0.};    //< half-length in +z and -z direction
  T fSphi{0.}; //< starting phi value (in radians)
  T fDphi{0.}; //< delta phi value of tube segment (in radians)

  // cached complex values (to avoid recomputation during usage)
  T fRmin2;
  T fRmax2;
  T fAlongPhi1x;
  T fAlongPhi1y;
  T fAlongPhi2x;
  T fAlongPhi2y;
  T fTolIrmin2;
  T fTolOrmin2;
  T fTolIrmax2;
  T fTolOrmax2;
  T fTolIz;
  T fTolOz;
  T fTolIrmin;
  T fTolOrmin;
  T fTolIrmax;
  T fTolOrmax;
  T fMaxVal;
  evolution::Wedge fPhiWedge;

  VECCORE_ATT_HOST_DEVICE
  void UpdatedZ()
  {
    fTolIz  = fZ - kHalfTolerance;
    fTolOz  = fZ + kHalfTolerance;
    fMaxVal = vecCore::math::Max(fRmax, fZ);
  }

  VECCORE_ATT_HOST_DEVICE
  void UpdatedRMin()
  {
    fRmin2     = fRmin * fRmin;
    fTolOrmin  = (fRmin - kHalfTolerance);
    fTolIrmin  = (fRmin + kHalfTolerance);
    fTolOrmin2 = fTolOrmin * fTolOrmin;
    fTolIrmin2 = fTolIrmin * fTolIrmin;
  }

  VECCORE_ATT_HOST_DEVICE
  void UpdatedRMax()
  {
    fRmax2     = fRmax * fRmax;
    fTolOrmax  = (fRmax + kHalfTolerance);
    fTolIrmax  = (fRmax - kHalfTolerance);
    fTolOrmax2 = fTolOrmax * fTolOrmax;
    fTolIrmax2 = fTolIrmax * fTolIrmax;
    fMaxVal    = vecCore::math::Max(fRmax, fZ);
  }

public:
  VECCORE_ATT_HOST_DEVICE
  T rmin() const { return fRmin; }
  VECCORE_ATT_HOST_DEVICE
  T z() const { return fZ; }

private:
  // can go into source
  VECCORE_ATT_HOST_DEVICE
  static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y)
  {
    x = std::cos(phi);
    y = std::sin(phi);
  }

public:
  VECCORE_ATT_HOST_DEVICE
  void SetRMin(Precision const &_rmin)
  {
    fRmin = _rmin;
    // Update cached values;
    UpdatedRMin();
  }

  VECCORE_ATT_HOST_DEVICE
  void SetRMax(Precision const &_rmax)
  {
    fRmax = _rmax;
    // Update cached values;
    UpdatedRMax();
  }

  VECCORE_ATT_HOST_DEVICE
  void SetDz(Precision const &_z)
  {
    fZ = _z;
    // Update cached values;
    UpdatedZ();
  }

  // can go into source
  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckSPhiAngle(T sPhi)
  {
    // Ensure fSphi in 0-2PI or -2PI-0 range if shape crosses 0
    if (sPhi < 0) {
      fSphi = kTwoPi - std::fmod(std::fabs(sPhi), kTwoPi);
    } else {
      fSphi = std::fmod(sPhi, kTwoPi);
    }
    if (fSphi + fDphi > kTwoPi) {
      fSphi -= kTwoPi;
    }
    // Update cached values.
    GetAlongVectorToPhiSector(fSphi, fAlongPhi1x, fAlongPhi1y);
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);
  }

  // can go into source
  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckDPhiAngle(T dPhi)
  {
    if (dPhi >= kTwoPi - 0.5 * kAngTolerance) {
      fDphi = kTwoPi;
      fSphi = 0;
    } else {
      if (dPhi > 0) {
        fDphi = dPhi;
      } else {
        //        std::ostringstream message;
        //        message << "Invalid dphi.\n"
        //                << "Negative or zero delta-Phi (" << dPhi << ")\n";
        //        std::cerr<<"UnplacedTube::CheckDPhiAngle(): Fatal error: "<< message.str().c_str() <<"\n";
      }
    }
    // Update cached values.
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);
  }

  // can go into source
  VECCORE_ATT_HOST_DEVICE
  void CalculateCached()
  {
    UpdatedZ();
    UpdatedRMin();
    UpdatedRMax();

    GetAlongVectorToPhiSector(fSphi, fAlongPhi1x, fAlongPhi1y);
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);

    fMaxVal = vecCore::math::Max(fRmax, fZ);
  }

public:
  // constructors
  TubeStruct() = default;

  VECCORE_ATT_HOST_DEVICE
  TubeStruct(T const &_rmin, T const &_rmax, T const &_z, T const &_sphi, T const &_dphi)
      : fRmin(_rmin < 0.0 ? 0.0 : _rmin), fRmax(_rmax), fZ(_z), fSphi(_sphi), fDphi(_dphi), fRmin2(0), fRmax2(0),
        fAlongPhi1x(0), fAlongPhi1y(0), fAlongPhi2x(0), fAlongPhi2y(0), fTolIrmin2(0), fTolOrmin2(0), fTolIrmax2(0),
        fTolOrmax2(0), fTolIz(0), fTolOz(0), fPhiWedge(_dphi, _sphi)
  {
    CalculateCached();
    // DetectConvexity();
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
