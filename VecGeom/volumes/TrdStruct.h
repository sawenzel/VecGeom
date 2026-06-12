// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of a struct with data members for the Trd class
/// @file volumes/TrdStruct.h
/// @author Mihaela Gheata

#ifndef VECGEOM_VOLUMES_TRDSTRUCT_H_
#define VECGEOM_VOLUMES_TRDSTRUCT_H_
#include "VecGeom/base/Global.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Struct encapsulating data members of the unplaced Trd
template <typename T = double>
struct TrdStruct {
  T fDX1; ///< Half-length along x at the surface positioned at -dz
  T fDX2; ///< Half-length along x at the surface positioned at +dz
  T fDY1; ///< Half-length along y at the surface positioned at -dz
  T fDY2; ///< Half-length along y at the surface positioned at +dz
  T fDZ;  ///< Half-length along z axis

  // cached values
  T fX2minusX1;    ///< Difference between half-legths along x at +dz and -dz
  T fY2minusY1;    ///< Difference between half-legths along y at +dz and -dz
  T fHalfX1plusX2; ///< Half-length along x at z = 0
  T fHalfY1plusY2; ///< Half-length along y at z = 0
  T fCalfX;        ///< Absolute value of cosine of inclination angle along x
  T fCalfY;        ///< Absolute value of cosine of inclination angle along y
  T fSecxz;        ///< Reciprocal of fCalfX
  T fSecyz;        ///< Reciprocal of fCalfY
  T fToleranceX;   ///< Corrected tolerance for Inside checks on X
  T fToleranceY;   ///< Corrected tolerance for Inside checks on Y

  T fFx; ///< Tangent of inclination angle along x
  T fFy; ///< Tangent of inclination angle along y

  /// Default constructor, it just allocates memory
  VECCORE_ATT_HOST_DEVICE
  TrdStruct() {}

  /// Constructor
  /// @param x1 Half-length along x at the surface positioned at -dz
  /// @param x2 Half-length along x at the surface positioned at +dz
  /// @param y Half-length along y axis
  /// @param z Half-length along z axis
  VECCORE_ATT_HOST_DEVICE
  TrdStruct(const T x1, const T x2, const T y, const T z)
      : fDX1(x1), fDX2(x2), fDY1(y), fDY2(y), fDZ(z), fX2minusX1(0), fY2minusY1(0), fHalfX1plusX2(0), fHalfY1plusY2(0),
        fCalfX(0), fCalfY(0), fFx(0), fFy(0)
  {
    CalculateCached();
  }

  /// Constructor
  /// @param x1 Half-length along x at the surface positioned at -dz
  /// @param x2 Half-length along x at the surface positioned at +dz
  /// @param y1 Half-length along y at the surface positioned at -dz
  /// @param y2 Half-length along y at the surface positioned at +dz
  /// @param z Half-length along z axis
  VECCORE_ATT_HOST_DEVICE
  TrdStruct(const T x1, const T x2, const T y1, const T y2, const T z)
      : fDX1(x1), fDX2(x2), fDY1(y1), fDY2(y2), fDZ(z), fX2minusX1(0), fY2minusY1(0), fHalfX1plusX2(0),
        fHalfY1plusY2(0), fCalfX(0), fCalfY(0), fFx(0), fFy(0)
  {
    CalculateCached();
  }

  /// Constructor
  /// @param x Half-length along x axis
  /// @param y Half-length along y axis
  /// @param z Half-length along z axis
  VECCORE_ATT_HOST_DEVICE
  TrdStruct(const T x, const T y, const T z)
      : fDX1(x), fDX2(x), fDY1(y), fDY2(y), fDZ(z), fX2minusX1(0), fY2minusY1(0), fHalfX1plusX2(0), fHalfY1plusY2(0),
        fCalfX(0), fCalfY(0), fFx(0), fFy(0)
  {
    CalculateCached();
  }

  /// Set all data members
  /// @param x1 Half-length along x at the surface positioned at -dz
  /// @param x2 Half-length along x at the surface positioned at +dz
  /// @param y1 Half-length along y at the surface positioned at -dz
  /// @param y2 Half-length along y at the surface positioned at +dz
  /// @param z Half-length along z axis
  VECCORE_ATT_HOST_DEVICE
  void SetAllParameters(T x1, T x2, T y1, T y2, T z)
  {
    fDX1 = x1;
    fDX2 = x2;
    fDY1 = y1;
    fDY2 = y2;
    fDZ  = z;
    CalculateCached();
  }

  /// Calculate cached values
  VECCORE_ATT_HOST_DEVICE
  void CalculateCached()
  {
    fX2minusX1    = fDX2 - fDX1;
    fY2minusY1    = fDY2 - fDY1;
    fHalfX1plusX2 = 0.5 * (fDX1 + fDX2);
    fHalfY1plusY2 = 0.5 * (fDY1 + fDY2);

    fFx    = 0.5 * (fDX1 - fDX2) / fDZ;
    fFy    = 0.5 * (fDY1 - fDY2) / fDZ;
    fSecxz = sqrt(1 + fFx * fFx);
    fSecyz = sqrt(1 + fFy * fFy);

    fCalfX      = 1. / Sqrt(1.0 + fFx * fFx);
    fCalfY      = 1. / Sqrt(1.0 + fFy * fFy);
    fToleranceX = kTolerance * Sqrt(fX2minusX1 * fX2minusX1 + 4 * fDZ * fDZ);
    fToleranceY = kTolerance * Sqrt(fY2minusY1 * fY2minusY1 + 4 * fDZ * fDZ);
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
