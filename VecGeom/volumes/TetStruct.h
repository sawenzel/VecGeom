// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of a struct with data members for the UnplacedTet class
/// @file volumes/TetStruct.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_TETSTRUCT_H_
#define VECGEOM_VOLUMES_TETSTRUCT_H_
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Struct encapsulating data members of the unplaced tetrahedron
template <typename T = double>
struct TetStruct {

  Vector3D<T> fVertex[4]; ///< Array of the tetrahedron vertices
  struct {
    Vector3D<T> n; ///< Normal of the plane
    T d;           ///< Distance from  origin to the plane
  } fPlane[4];     ///< The tetrahedron face planes

  Precision fCubicVolume; ///< Volume of the tetrahedron
  Precision fSurfaceArea; ///< Surface area of the tetrahedron

  /// Empty constructor
  VECCORE_ATT_HOST_DEVICE
  TetStruct() {}

  /// Constructor from four points
  /// @param p0 Point given as array
  /// @param p1 Point given as array
  /// @param p2 Point given as array
  /// @param p3 Point given as array
  VECCORE_ATT_HOST_DEVICE
  TetStruct(const T p0[], const T p1[], const T p2[], const T p3[])
  {
    Vector3D<T> vertices[4];
    vertices[0].Set(p0[0], p0[1], p0[2]);
    vertices[1].Set(p1[0], p1[1], p1[2]);
    vertices[2].Set(p2[0], p2[1], p2[2]);
    vertices[3].Set(p3[0], p3[1], p3[2]);

    CalculateCached(vertices[0], vertices[1], vertices[2], vertices[3]);
  }

  /// Constructor from four points
  /// @param p0 Point given as 3D vector
  /// @param p1 Point given as 3D vector
  /// @param p2 Point given as 3D vector
  /// @param p3 Point given as 3D vector
  VECCORE_ATT_HOST_DEVICE
  TetStruct(const Vector3D<T> p0, const Vector3D<T> p1, const Vector3D<T> p2, const Vector3D<T> p3)
      : fCubicVolume(0.), fSurfaceArea(0.)
  {

    CalculateCached(p0, p1, p2, p3);
  }

  /// Set the tetrahedron data members
  /// @param p0 Point given as array
  /// @param p1 Point given as array
  /// @param p2 Point given as array
  /// @param p3 Point given as array
  VECCORE_ATT_HOST_DEVICE
  void CalculateCached(const Vector3D<T> p0, const Vector3D<T> p1, const Vector3D<T> p2, const Vector3D<T> p3)
  {
    // Fill all the cached values
    fVertex[0] = p0;
    fVertex[1] = p1;
    fVertex[2] = p2;
    fVertex[3] = p3;

    // if (CheckDegeneracy()) std::cerr << "DeGenerate Tetrahedron not allowed" << std::endl;
    CheckDegeneracy();

    Vector3D<Precision> n0 = (fVertex[1] - fVertex[0]).Cross(fVertex[2] - fVertex[0]).Unit();
    Vector3D<Precision> n1 = (fVertex[2] - fVertex[1]).Cross(fVertex[3] - fVertex[1]).Unit();
    Vector3D<Precision> n2 = (fVertex[3] - fVertex[2]).Cross(fVertex[0] - fVertex[2]).Unit();
    Vector3D<Precision> n3 = (fVertex[0] - fVertex[3]).Cross(fVertex[1] - fVertex[3]).Unit();

    if (n0.Dot(fVertex[3] - fVertex[0]) > 0) n0 = -n0;
    if (n1.Dot(fVertex[0] - fVertex[1]) > 0) n1 = -n1;
    if (n2.Dot(fVertex[1] - fVertex[2]) > 0) n2 = -n2;
    if (n3.Dot(fVertex[2] - fVertex[3]) > 0) n3 = -n3;

    fPlane[0].n = n0;
    fPlane[0].d = -n0.Dot(fVertex[0]);
    // fPlane[0].d = n0.Dot(fVertex[0]);

    fPlane[1].n = n1;
    fPlane[1].d = -n1.Dot(fVertex[1]);
    // fPlane[1].d = n1.Dot(fVertex[1]);

    fPlane[2].n = n2;
    fPlane[2].d = -n2.Dot(fVertex[2]);
    // fPlane[2].d = n2.Dot(fVertex[2]);

    fPlane[3].n = n3;
    fPlane[3].d = -n3.Dot(fVertex[3]);
    // fPlane[3].d = n3.Dot(fVertex[3]);

    for (int i = 0; i < 4; i++) {
      // std::cerr << "Plane[" << i << "] = " << fPlane[i].n << "  " << fPlane[i].d << std::endl;
    }
  }

  /// Set volume of the tetrahedron
  VECCORE_ATT_HOST_DEVICE
  void CalcCapacity()
  {
    fCubicVolume =
        vecCore::math::Abs((fVertex[1] - fVertex[0]).Dot((fVertex[2] - fVertex[0]).Cross(fVertex[3] - fVertex[0]))) /
        6.;
  }

  /// Set surface area of the tetrahedron
  VECCORE_ATT_HOST_DEVICE
  void CalcSurfaceArea()
  {
    fSurfaceArea = ((fVertex[1] - fVertex[0]).Cross(fVertex[2] - fVertex[0]).Mag() +
                    (fVertex[2] - fVertex[1]).Cross(fVertex[3] - fVertex[1]).Mag() +
                    (fVertex[3] - fVertex[2]).Cross(fVertex[0] - fVertex[2]).Mag() +
                    (fVertex[0] - fVertex[3]).Cross(fVertex[1] - fVertex[3]).Mag()) *
                   0.5;
  }

  /// Set the tetrahedron
  /// @param p0 Point given as array
  /// @param p1 Point given as array
  /// @param p2 Point given as array
  /// @param p3 Point given as array
  VECCORE_ATT_HOST_DEVICE
  void SetParameters(const Vector3D<T> p0, const Vector3D<T> p1, const Vector3D<T> p2, const Vector3D<T> p3)
  {
    CalculateCached(p0, p1, p2, p3);
  }

  /// Check correctness of the tetrahedron data
  /// @return false if tetrahedron is degenerate
  VECCORE_ATT_HOST_DEVICE
  bool CheckDegeneracy()
  {
    CalcCapacity();
    CalcSurfaceArea();
    if (fCubicVolume < kTolerance * kTolerance * kTolerance) return true;
    Precision s0 = (fVertex[1] - fVertex[0]).Cross(fVertex[2] - fVertex[0]).Mag() * 0.5;
    Precision s1 = (fVertex[2] - fVertex[1]).Cross(fVertex[3] - fVertex[1]).Mag() * 0.5;
    Precision s2 = (fVertex[3] - fVertex[2]).Cross(fVertex[0] - fVertex[2]).Mag() * 0.5;
    Precision s3 = (fVertex[0] - fVertex[3]).Cross(fVertex[1] - fVertex[3]).Mag() * 0.5;
    return (vecCore::math::Max(vecCore::math::Max(vecCore::math::Max(s0, s1), s2), s3) < 2 * kTolerance);
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
