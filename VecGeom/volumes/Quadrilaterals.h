/// \file Quadrilaterals.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_QUADRILATERALS_H_
#define VECGEOM_VOLUMES_QUADRILATERALS_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/base/Array.h"
#include "VecGeom/base/AOS3D.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/Planes.h"

// Switches on/off explicit vectorization of algorithms using Vc
// #define VECGEOM_QUADRILATERAL_ACCELERATION --> now done in CMakeFile

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Quadrilaterals;);
VECGEOM_DEVICE_DECLARE_CONV(class, Quadrilaterals);

inline namespace VECGEOM_IMPL_NAMESPACE {

class Quadrilaterals {

private:
  Planes fPlanes;               ///< The planes in which the quadrilaterals lie.
  Planes fSideVectors[4];       ///< Vectors pointing from a side constructed from two
                                ///  corners to the origin, equivalent to
                                ///  normal x (c1 - c0)
                                ///  Used to check if an intersection is in bounds.
  AOS3D<Precision> fCorners[4]; ///< Four corners of the quadrilaterals. Used
                                ///  for bounds checking.

public:
  typedef Planes Sides_t[4];
  typedef AOS3D<Precision> Corners_t[4];

  Quadrilaterals() = default;

  VECCORE_ATT_HOST_DEVICE
  Quadrilaterals(int size, bool convex = true);

  /// @brief Construct aligned data using specialized allocator.
  /// @details To allocate the full object, call this constructor with placement new.
  /// @param size Size of the internal arrays
  /// @param a Aligned allocator, pre-initialized to fit the content
  /// @param convex Convexity of the quadrilaterals
  VECCORE_ATT_HOST_DEVICE
  Quadrilaterals(int size, AlignedAllocator &a, bool convex = true);

  VECCORE_ATT_HOST_DEVICE
  ~Quadrilaterals();

  VECCORE_ATT_HOST_DEVICE
  Quadrilaterals(Quadrilaterals const &other);

  VECCORE_ATT_HOST_DEVICE
  Quadrilaterals &operator=(Quadrilaterals const &other);

  // returns the number of quadrilaterals ( planes ) stored in this container
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int size() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static size_t aligned_sizeof_data(const size_t initSize);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Planes const &GetPlanes() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SOA3D<Precision> const &GetNormals() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetNormal(int i) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array<Precision> const &GetDistances() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDistance(int i) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Sides_t const &GetSideVectors() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Corners_t const &GetCorners() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTriangleArea(int index, int iCorner1, int iCorner2) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetQuadrilateralArea(int index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetPointOnTriangle(int index, int iCorner0, int iCorner1, int iCorner2) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetPointOnFace(int index) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool RayHitsQuadrilateral(int index, Vector3D<Precision> const &intersection) const
  {
    bool valid = true;
    for (int j = 0; j < 4; ++j) {
      valid &=
          intersection.Dot(fSideVectors[j].GetNormal(index)) + fSideVectors[j].GetDistances()[index] >= -kTolerance;
      if (vecCore::MaskEmpty(valid)) break;
    }
    return valid;
  }

  /// Sets the corners of a pre-existing quadrilateral.
  /// \param corner0 First corner in counterclockwise order.
  /// \param corner1 Second corner in counterclockwise order.
  /// \param corner2 Third corner in counterclockwise order.
  /// \param corner3 Fourth corner in counterclockwise order.
  VECCORE_ATT_HOST_DEVICE
  void Set(int index, Vector3D<Precision> const &corner0, Vector3D<Precision> const &corner1,
           Vector3D<Precision> const &corner2, Vector3D<Precision> const &corner3);

  /// Flips the sign of the normal and distance of the specified quadrilateral.
  VECCORE_ATT_HOST_DEVICE
  void FlipSign(int index);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE vecCore::Mask_v<Real_v> Contains(Vector3D<Real_v> const &point) const;

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Inside_v Inside(Vector3D<Real_v> const &point) const;

  template <typename Real_v, typename Inside_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Inside_v Inside(Vector3D<Real_v> const &point, int i) const;

  template <typename Real_v, bool behindPlanesT>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v DistanceToIn(Vector3D<Real_v> const &point,
                                                                   Vector3D<Real_v> const &direction) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v DistanceToOut(Vector3D<Real_v> const &point,
                                                                    Vector3D<Real_v> const &direction, Precision zMin,
                                                                    Precision zMax) const;

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v DistanceToOut(Vector3D<Real_v> const &point,
                                                                    Vector3D<Real_v> const &direction) const;

  /// \param index Quadrilateral to compute distance to.
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Precision ScalarDistanceSquared(int index, Vector3D<Precision> const &point) const;

  VECCORE_ATT_HOST_DEVICE
  void Print() const;

}; // end of class declaration

VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
int Quadrilaterals::size() const { return fPlanes.size(); }

VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
size_t Quadrilaterals::aligned_sizeof_data(const size_t initSize)
{
  return (5 * Planes::aligned_sizeof_data(initSize) + 4 * AOS3D<Precision>::aligned_sizeof_data(initSize));
}

VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Planes const &Quadrilaterals::GetPlanes() const { return fPlanes; }

VECCORE_ATT_HOST_DEVICE
SOA3D<Precision> const &Quadrilaterals::GetNormals() const { return fPlanes.GetNormals(); }

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> Quadrilaterals::GetNormal(int i) const { return fPlanes.GetNormal(i); }

VECCORE_ATT_HOST_DEVICE
Array<Precision> const &Quadrilaterals::GetDistances() const { return fPlanes.GetDistances(); }

VECCORE_ATT_HOST_DEVICE
Precision Quadrilaterals::GetDistance(int i) const { return fPlanes.GetDistance(i); }

VECCORE_ATT_HOST_DEVICE
Quadrilaterals::Sides_t const &Quadrilaterals::GetSideVectors() const { return fSideVectors; }

VECCORE_ATT_HOST_DEVICE
Quadrilaterals::Corners_t const &Quadrilaterals::GetCorners() const { return fCorners; }

VECCORE_ATT_HOST_DEVICE
Precision Quadrilaterals::GetTriangleArea(int index, int iCorner1, int iCorner2) const
{
  Precision fArea          = 0.;
  Vector3D<Precision> vec1 = fCorners[iCorner1][index] - fCorners[0][index];
  Vector3D<Precision> vec2 = fCorners[iCorner2][index] - fCorners[0][index];

  fArea = 0.5 * (vec1.Cross(vec2)).Mag();
  return fArea;
}

VECCORE_ATT_HOST_DEVICE
Precision Quadrilaterals::GetQuadrilateralArea(int index) const
{
  Precision fArea = 0.;

  fArea = GetTriangleArea(index, 1, 2) + GetTriangleArea(index, 2, 3);
  return fArea;
}

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> Quadrilaterals::GetPointOnTriangle(int index, int iCorner0, int iCorner1, int iCorner2) const
{
  Precision r1 = RNG::Instance().uniform(0.0, 1.0);
  Precision r2 = RNG::Instance().uniform(0.0, 1.0);
  if (r1 + r2 > 1.) {
    r1 = 1. - r1;
    r2 = 1. - r2;
  }
  Vector3D<Precision> vec1 = fCorners[iCorner1][index] - fCorners[iCorner0][index];
  Vector3D<Precision> vec2 = fCorners[iCorner2][index] - fCorners[iCorner0][index];
  return fCorners[iCorner0][index] + r1 * vec1 + r2 * vec2;
}

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> Quadrilaterals::GetPointOnFace(int index) const
{

  // Avoid degenerated surfaces
  int nvert = 0;
  int iCorners[4];
  for (int i = 0; i < 4; ++i) {
    if ((fCorners[(i + 1) % 4][index] - fCorners[i % 4][index]).Mag2() > kTolerance) {
      iCorners[nvert++] = i;
    }
  }
  if (nvert == 1) return fCorners[0][index];
  if (nvert == 2) return GetPointOnTriangle(index, iCorners[0], iCorners[0], iCorners[1]);
  if (nvert == 3) return GetPointOnTriangle(index, iCorners[0], iCorners[1], iCorners[2]);
  Precision choice = RNG::Instance().uniform(0, 1.0);
  if (choice < 0.5) {
    return GetPointOnTriangle(index, 0, 1, 2);
  } else {
    return GetPointOnTriangle(index, 0, 2, 3);
  }
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE vecCore::Mask_v<Real_v> Quadrilaterals::Contains(Vector3D<Real_v> const &point) const
{
  return fPlanes.Contains<Real_v>(point);
}

template <typename Real_v, typename Inside_v>
VECCORE_ATT_HOST_DEVICE Inside_v Quadrilaterals::Inside(Vector3D<Real_v> const &point) const
{
  return fPlanes.Inside<Real_v, Inside_v>(point);
}

template <typename Real_v, typename Inside_v>
VECCORE_ATT_HOST_DEVICE Inside_v Quadrilaterals::Inside(Vector3D<Real_v> const &point, int i) const
{
  return fPlanes.Inside<Real_v, Inside_v>(point, i);
}

namespace {

template <class Real_v>
struct AcceleratedDistanceToIn {
  template <bool behindPlanesT>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void VectorLoop(
      int & /*i*/, const int /*n*/, Planes const & /*planes*/, Planes const (& /*sideVectors*/)[4],
      Vector3D<Real_v> const & /*point*/, Vector3D<Real_v> const & /*direction*/, Real_v & /*distance*/)
  {
    // Do nothing if not scalar backend
    return;
  }
};

#if defined(VECGEOM_VC) && defined(VECGEOM_QUADRILATERAL_ACCELERATION)
template <>
struct AcceleratedDistanceToIn<Precision> {

  template <bool behindPlanesT>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void VectorLoop(int &i, const int n, Planes const &planes,
                                                                      Planes const (&sideVectors)[4],
                                                                      Vector3D<Precision> const &point,
                                                                      Vector3D<Precision> const &direction,
                                                                      Precision &distance)
  {

    // Explicitly vectorize over quadrilaterals using Vc
    for (; i <= n - kVectorSize; i += kVectorSize) {
      Vector3D<VcPrecision> plane(VcPrecision(planes.GetNormals().x() + i), VcPrecision(planes.GetNormals().y() + i),
                                  VcPrecision(planes.GetNormals().z() + i));
      VcPrecision dPlane(&planes.GetDistances()[0] + i);
      VcPrecision distanceTest = plane.Dot(point) + dPlane;

      // Check if the point is in front of/behind the plane according to the template parameter
      VcBool valid = Flip<behindPlanesT>::FlipSign(distanceTest) > -kTolerance;
      if (vecCore::MaskEmpty(valid)) continue;

      VcPrecision directionProjection = plane.Dot(direction);
      valid &= Flip<!behindPlanesT>::FlipSign(directionProjection) > 0;
      if (vecCore::MaskEmpty(valid)) continue;
      VcPrecision tiny = Vc::copysign(VcPrecision(1E-20), directionProjection);
      distanceTest /= -(directionProjection + tiny);
      Vector3D<VcPrecision> intersection = Vector3D<VcPrecision>(direction) * distanceTest + point;

      for (int j = 0; j < 4; ++j) {
        Vector3D<VcPrecision> sideVector(VcPrecision(sideVectors[j].GetNormals().x() + i),
                                         VcPrecision(sideVectors[j].GetNormals().y() + i),
                                         VcPrecision(sideVectors[j].GetNormals().z() + i));
        VcPrecision dSide(&sideVectors[j].GetDistances()[i]);
        valid &= sideVector.Dot(intersection) + dSide >= -kTolerance;
        // Where is your god now
        if (vecCore::MaskEmpty(valid)) goto distanceToInVcContinueOuter;
      }
      // If a hit is found, the algorithm can return, since only one side can
      // be hit for a convex set of quadrilaterals
      distanceTest(!valid) = InfinityLength<Precision>();
      distance             = Max(distanceTest.min(), Precision(0.));
      i                    = n;
      return;
    // Continue label of outer loop
    distanceToInVcContinueOuter:;
    }
    return;
  }
};
#endif

} // End anonymous namespace

template <typename Real_v, bool behindPlanesT>
VECCORE_ATT_HOST_DEVICE Real_v Quadrilaterals::DistanceToIn(Vector3D<Real_v> const &point,
                                                            Vector3D<Real_v> const &direction) const
{

  // Looks for the shortest distance to one of the quadrilaterals.
  // The algorithm projects the position and direction onto the plane of the
  // quadrilateral, determines the intersection point, then checks if this point
  // is within the bounds of the quadrilateral. There are many opportunities to
  // perform early returns along the way, and the speed of this algorithm relies
  // heavily on this property.
  //
  // The code below is optimized for the Polyhedron, and will return as soon as
  // a valid intersection is found, since only one intersection will ever occur
  // per Z-segment in Polyhedron case. If used in other contexts, a template
  // parameter would have to be added to make a distinction.

  using Bool_v = vecCore::Mask_v<Real_v>;

  Real_v bestDistance = InfinityLength<Real_v>();

  int i       = 0;
  const int n = size();
  AcceleratedDistanceToIn<Real_v>::template VectorLoop<behindPlanesT>(i, n, fPlanes, fSideVectors, point, direction,
                                                                      bestDistance);

  // TODO: IN CASE QUADRILATERALS ARE PERPENDICULAR TO Z WE COULD SAVE MANY DIVISIONS
  for (; i < n; ++i) {
    Vector3D<Precision> normal = fPlanes.GetNormal(i);
    Real_v distance            = point.Dot(normal) + fPlanes.GetDistance(i);
    // Check if the point is in front of/behind the plane according to the
    // template parameter
    Bool_v valid = Flip<behindPlanesT>::FlipSign(distance) > -kTolerance;
    if (vecCore::MaskEmpty(valid)) continue;
    Real_v directionProjection = direction.Dot(normal);
    valid &= Flip<!behindPlanesT>::FlipSign(directionProjection) > kTolerance;
    if (vecCore::MaskEmpty(valid)) continue;
    distance /= -(directionProjection + CopySign(Real_v(1E-20), directionProjection));
    Vector3D<Real_v> intersection = point + direction * distance;
    for (int j = 0; j < 4; ++j) {
      valid &= intersection.Dot(fSideVectors[j].GetNormal(i)) + fSideVectors[j].GetDistances()[i] >= -kTolerance;
      if (vecCore::MaskEmpty(valid)) break;
    }
    vecCore::MaskedAssign(bestDistance, valid, distance);
    // If all hits are found, the algorithm can return, since only one side can
    // be hit for a convex set of quadrilaterals
    if (vecCore::MaskFull(bestDistance < InfinityLength<Real_v>())) break;
  }

  return Max(Real_v(0.), bestDistance);
}

namespace {

template <typename Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AcceleratedDistanceToOut(
    int & /*i*/, const int /*n*/, Planes const & /*planes*/, Planes const (& /*sideVectors*/)[4],
    const Precision /*zMin*/, const Precision /*zMax*/, Vector3D<Real_v> const & /*point*/,
    Vector3D<Real_v> const & /*direction*/, Real_v & /*distance*/)
{
  // Do nothing if the backend is not scalar
  return;
}

#if defined(VECGEOM_VC) && defined(VECGEOM_QUADRILATERAL_ACCELERATION)
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void AcceleratedDistanceToOut<Precision>(
    int &i, const int n, Planes const &planes, Planes const (&sideVectors)[4], const Precision zMin,
    const Precision zMax, Vector3D<Precision> const &point, Vector3D<Precision> const &direction, Precision &distance)
{

  // Explicitly vectorize over quadrilaterals using Vc
  for (; i <= n - kVectorSize; i += kVectorSize) {
    Vector3D<VcPrecision> plane(VcPrecision(planes.GetNormals().x() + i), VcPrecision(planes.GetNormals().y() + i),
                                VcPrecision(planes.GetNormals().z() + i));
    VcPrecision dPlane(&planes.GetDistances()[0] + i);
    VcPrecision distanceTest = plane.Dot(point) + dPlane;
    // Check if the point is behind the plane
    VcBool valid = distanceTest < kTolerance;
    if (vecCore::MaskEmpty(valid)) continue;
    VcPrecision directionProjection = plane.Dot(direction);
    // Because the point is behind the plane, the direction must be along the
    // normal
    valid &= directionProjection > 0;
    if (vecCore::MaskEmpty(valid)) continue;
    distanceTest /= -NonZero(directionProjection);
    valid &= distanceTest < distance;
    if (vecCore::MaskEmpty(valid)) continue;

    if (zMin == zMax) { // need a careful treatment in case of degenerate Z planes
      // in this case need proper hit detection
      Vector3D<VcPrecision> intersection = Vector3D<VcPrecision>(direction) * distanceTest + point;
      for (int j = 0; j < 4; ++j) {
        Vector3D<VcPrecision> sideVector(VcPrecision(sideVectors[j].GetNormals().x() + i),
                                         VcPrecision(sideVectors[j].GetNormals().y() + i),
                                         VcPrecision(sideVectors[j].GetNormals().z() + i));
        VcPrecision dSide(&sideVectors[j].GetDistances()[i]);
        valid &= sideVector.Dot(intersection) + dSide >= -kTolerance;
        // Where is your god now
        if (vecCore::MaskEmpty(valid)) goto distanceToOutVcContinueOuter;
      }
    } else {
      VcPrecision zProjection = distanceTest * direction[2] + point[2];
      valid &= zProjection >= zMin && zProjection < zMax;
    }
  distanceToOutVcContinueOuter:
    if (vecCore::MaskEmpty(valid)) continue;
    distanceTest(!valid) = InfinityLength<Precision>();
    distance             = distanceTest.min();
  }
  distance = Max(Precision(0.), distance);
  return;
}
#endif

} // End anonymous namespace

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE Real_v Quadrilaterals::DistanceToOut(Vector3D<Real_v> const &point,
                                                             Vector3D<Real_v> const &direction, Precision zMin,
                                                             Precision zMax) const
{

  // The below computes the distance to the quadrilaterals similar to
  // DistanceToIn, but is optimized for Polyhedron, and as such can assume that
  // the quadrilaterals form a convex shell, and that the shortest distance to
  // one of the quadrilaterals will indeed be an intersection. The exception to
  // this is if the point leaves the Z-bounds specified in the input parameters.
  // If used for another purpose than Polyhedron, DistanceToIn should be used if
  // the set of quadrilaterals is not convex.

  using Bool_v = vecCore::Mask_v<Real_v>;

  Real_v bestDistance = InfinityLength<Real_v>();

  int i       = 0;
  const int n = size();
  AcceleratedDistanceToOut<Real_v>(i, n, fPlanes, fSideVectors, zMin, zMax, point, direction, bestDistance);

  for (; i < n; ++i) {
    Vector3D<Precision> normal = fPlanes.GetNormal(i);
    Real_v distanceTest        = point.Dot(normal) + fPlanes.GetDistance(i);
    // Check if the point is behind the plane
    Bool_v valid = distanceTest < kTolerance;
    if (vecCore::MaskEmpty(valid)) continue;
    Real_v directionProjection = direction.Dot(normal);
    // Because the point is behind the plane, the direction must be along the
    // normal
    valid &= directionProjection > kTolerance;
    if (vecCore::MaskEmpty(valid)) continue;
    distanceTest /= -directionProjection;
    valid &= distanceTest < bestDistance && distanceTest > -kTolerance / directionProjection;
    if (vecCore::MaskEmpty(valid)) continue;

    // this is a tricky test when zMin == zMax ( degenerate planes )
    if (zMin == zMax) {
      // in this case need proper hit detection
      // valid &= zProjection >= zMin-1E-10 && zProjection <= zMax+1E-10;
      Vector3D<Real_v> intersection = point + distanceTest * direction;

      valid = RayHitsQuadrilateral(i, intersection);

    } else {
      Real_v zProjection = point[2] + distanceTest * direction[2];
      valid &= (zProjection >= zMin - kTolerance) && (zProjection < zMax + kTolerance);
    }
    if (vecCore::MaskEmpty(valid)) continue;
    vecCore::MaskedAssign(bestDistance, valid, distanceTest);
  }

  if (bestDistance > -kTolerance) bestDistance = Max(bestDistance, Precision(0.));
  return bestDistance;
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE Real_v Quadrilaterals::DistanceToOut(Vector3D<Real_v> const &point,
                                                             Vector3D<Real_v> const &direction) const
{
  return DistanceToOut<Real_v>(point, direction, -InfinityLength<Real_v>(), InfinityLength<Real_v>());
}

VECCORE_ATT_HOST_DEVICE
Precision Quadrilaterals::ScalarDistanceSquared(int i, Vector3D<Precision> const &point) const
{

  // This function is used by the safety algorithms to return the exact distance
  // to the quadrilateral specified.
  // The algorithm has three stages, trying first to return the shortest
  // distance to the plane, then to the closest line segment, then to the
  // closest corner.
  assert(i < size());

  Vector3D<Precision> planeNormal = fPlanes.GetNormal(i);
  Precision distance              = point.Dot(planeNormal) + fPlanes.GetDistance(i);
  // Find the projection of the point on the quadrilateral "i". There was
  // a bug below by adding a distance along the plane normal, while the
  // correct version should subtract.
  Vector3D<Precision> intersection = point - distance * planeNormal;

  bool withinBound[4];
  for (int j = 0; j < 4; ++j) {
    // TODO: check if this autovectorizes. Otherwise it should be explicitly
    //       vectorized.
    withinBound[j] = intersection[0] * fSideVectors[j].GetNormals().x(i) +
                         intersection[1] * fSideVectors[j].GetNormals().y(i) +
                         intersection[2] * fSideVectors[j].GetNormals().z(i) + fSideVectors[j].GetDistances()[i] >=
                     0;
  }
  if (withinBound[0] && withinBound[1] && withinBound[2] && withinBound[3]) {
    return distance * distance;
  }

  // If the closest point is not on the plane itself, it must either be the
  // distance to the closest line segment or to the closest corner.
  // Since it is already known whether the point is to the left or right of
  // each line, only one side and its corners have to be checked.
  //
  //           above
  //   corner3_______corner0
  //         |       |
  //   left  |       |  right
  //         |_______|
  //   corner2       corner1
  //           below
  //
  // The assumption above that only one side has to be checked is not always
  // true. If withinBound is false for 2 connected segments making an angle > 90
  // and if the point projection on each of these segments is outside bounds,
  // it can happen that the closest point is not the same depending on the
  // checked segment. Something like below:
  //
  //    Point  x
  //
  //     corner3 ___corner0
  //            |   -
  //            |     -
  //            |_______- corner1
  //
  // If the "above" segment is checked, corner3 will be selected closest, which
  // is correct, but if the "right" segment gets checked, corner0 will be
  // wrongly selected.

  Precision distancesq = InfinityLength<Precision>();
  for (int j = 0; j < 4; ++j) {
    if (!withinBound[j]) {
      distance = DistanceToLineSegmentSquared1<Precision>(fCorners[j][i], fCorners[(j + 1) % 4][i], point);
      if (distance < distancesq) distancesq = distance;
    }
  }
  return distancesq;
}

std::ostream &operator<<(std::ostream &os, Quadrilaterals const &quads);

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif // VECGEOM_VOLUMES_QUADRILATERALS_H_
