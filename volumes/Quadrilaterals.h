/// \file Quadrilaterals.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_QUADRILATERALS_H_
#define VECGEOM_VOLUMES_QUADRILATERALS_H_

#include "base/Global.h"

#include "backend/Backend.h"
#include "base/Array.h"
#include "base/AOS3D.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"
#include "base/RNG.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/Planes.h"

// Switches on/off explicit vectorization of algorithms using Vc
#define VECGEOM_QUADRILATERALS_VC

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class Quadrilaterals; )
VECGEOM_DEVICE_DECLARE_CONV( Quadrilaterals );

inline namespace VECGEOM_IMPL_NAMESPACE {

class Quadrilaterals {

private:

  Planes fPlanes; ///< The planes in which the quadrilaterals lie.
  Planes fSideVectors[4]; ///< Vectors pointing from a side constructed from two
                          ///  corners to the origin, equivalent to
                          ///  normal x (c1 - c0)
                          ///  Used to check if an intersection is in bounds.
  AOS3D<Precision> fCorners[4]; ///< Four corners of the quadrilaterals. Used
                                ///  for bounds checking.

public:

  typedef Planes Sides_t[4];
  typedef AOS3D<Precision> Corners_t[4];

  VECGEOM_CUDA_HEADER_BOTH
  Quadrilaterals(int size);

  VECGEOM_CUDA_HEADER_BOTH
  ~Quadrilaterals();

  VECGEOM_CUDA_HEADER_BOTH
  Quadrilaterals(Quadrilaterals const &other);

  VECGEOM_CUDA_HEADER_BOTH
  Quadrilaterals& operator=(Quadrilaterals const &other);

  // returns the number of quadrilaterals ( planes ) stored in this container
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Planes const& GetPlanes() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SOA3D<Precision> const& GetNormals() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> GetNormal(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Array<Precision> const& GetDistances() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDistance(int i) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Sides_t const& GetSideVectors() const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Corners_t const& GetCorners() const;

   VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTriangleArea(int index,int iCorner1, int iCorner2) const;
 
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetQuadrilateralArea(int index) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> GetPointOnTriangle(int index, 
					 int iCorner1, int iCorner2) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> GetPointOnFace(int index) const;



  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool RayHitsQuadrilateral( int index, Vector3D<Precision> const & intersection ) const {
      bool valid = true;
      for (int j = 0; j < 4; ++j) {
        valid &= intersection.Dot(fSideVectors[j].GetNormal(index)) +
                 fSideVectors[j].GetDistances()[index] >= 0;
        if (IsEmpty(valid)) break;
      }
     return valid;
  }

  /// \param corner0 First corner in counterclockwise order.
  /// \param corner1 Second corner in counterclockwise order.
  /// \param corner2 Third corner in counterclockwise order.
  /// \param corner3 Fourth corner in counterclockwise order.
  VECGEOM_CUDA_HEADER_BOTH
  void Set(
      int index,
      Vector3D<Precision> const &corner0,
      Vector3D<Precision> const &corner1,
      Vector3D<Precision> const &corner2,
      Vector3D<Precision> const &corner3);

  /// Flips the sign of the normal and distance of the specified quadrilateral.
  VECGEOM_CUDA_HEADER_BOTH
  void FlipSign(int index);

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::bool_v Contains(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::inside_v Inside(
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend, bool behindPlanesT>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToIn(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToOut(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      Precision zMin,
      Precision zMax) const;

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToOut(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

  /// \param index Quadrilateral to compute distance to.
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  Precision ScalarDistanceSquared(
      int index,
      Vector3D<Precision> const &point) const;

  VECGEOM_CUDA_HEADER_BOTH
  void Print() const;

}; // end of class declaration

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
int Quadrilaterals::size() const { return fPlanes.size(); }

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Planes const& Quadrilaterals::GetPlanes() const { return fPlanes; }

VECGEOM_CUDA_HEADER_BOTH
SOA3D<Precision> const& Quadrilaterals::GetNormals() const {
  return fPlanes.GetNormals();
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Quadrilaterals::GetNormal(int i) const {
  return fPlanes.GetNormal(i);
}

VECGEOM_CUDA_HEADER_BOTH
Array<Precision> const& Quadrilaterals::GetDistances() const {
  return fPlanes.GetDistances();
}

VECGEOM_CUDA_HEADER_BOTH
Precision Quadrilaterals::GetDistance(int i) const {
  return fPlanes.GetDistance(i);
}

VECGEOM_CUDA_HEADER_BOTH
Quadrilaterals::Sides_t const& Quadrilaterals::GetSideVectors() const {
  return fSideVectors;
}

VECGEOM_CUDA_HEADER_BOTH
Quadrilaterals::Corners_t const& Quadrilaterals::GetCorners() const {
  return fCorners;
}

VECGEOM_CUDA_HEADER_BOTH
  Precision Quadrilaterals::GetTriangleArea(int index,
           int iCorner1, int iCorner2) const {
  Precision fArea=0.;
  Vector3D<Precision> vec1 = fCorners[iCorner1][index]-fCorners[0][index];
  Vector3D<Precision> vec2 = fCorners[iCorner2][index]-fCorners[0][index];
 
  fArea = 0.5 * (vec1.Cross(vec2)).Mag();
  return fArea;
}

VECGEOM_CUDA_HEADER_BOTH
Precision Quadrilaterals::GetQuadrilateralArea(int index) const {
  Precision fArea=0.;
 
  fArea = GetTriangleArea(index,1,2)+GetTriangleArea(index,2,3);
  return fArea;
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Quadrilaterals::GetPointOnTriangle(int index, 
                           int iCorner1, int iCorner2) const{

   Precision alpha = RNG::Instance().uniform(0.0, 1.0);
   Precision beta = RNG::Instance().uniform(0.0, 1.0);
   Precision lambda1 = alpha * beta;
   Precision lambda0 = alpha - lambda1;
   Vector3D<Precision> vec1 = fCorners[iCorner1][index]-fCorners[0][index];
   Vector3D<Precision> vec2 = fCorners[iCorner2][index]-fCorners[0][index];
   return fCorners[0][index] + lambda0 * vec1 + lambda1 * vec2;
 
}

VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision> Quadrilaterals::GetPointOnFace(int index) const{

   Precision choice =RNG::Instance().uniform(0, 1.0);
   if (choice < 0.5)
     {
       return GetPointOnTriangle(index,1,2);
     }
   else
     {
       return GetPointOnTriangle(index,2,3);
     }
}


template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Quadrilaterals::Contains(
    Vector3D<typename Backend::precision_v> const &point) const {
  return fPlanes.Contains<Backend>(point);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Quadrilaterals::Inside(
    Vector3D<typename Backend::precision_v> const &point) const {
  return fPlanes.Inside<Backend>(point);
}

namespace {

template <class Backend>
struct AcceleratedDistanceToIn {

  template <bool behindPlanesT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void VectorLoop(
      int &i,
      const int n,
      Planes const &planes,
      Planes const (&sideVectors)[4],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v &distance) {
    // Do nothing if not scalar backend
    return;
  }

};

template <>
struct AcceleratedDistanceToIn<kScalar> {

  template <bool behindPlanesT>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void VectorLoop(
      int &i,
      const int n,
      Planes const &planes,
      Planes const (&sideVectors)[4],
      Vector3D<Precision> const &point,
      Vector3D<Precision> const &direction,
      Precision &distance) {
  #if defined(VECGEOM_VC) && defined(VECGEOM_QUADRILATERALS_VC)
    // Explicitly vectorize over quadrilaterals using Vc
    for (; i <= n-kVectorSize; i += kVectorSize) {
      Vector3D<VcPrecision> plane(
          VcPrecision(planes.GetNormals().x()+i),
          VcPrecision(planes.GetNormals().y()+i),
          VcPrecision(planes.GetNormals().z()+i));
      VcPrecision dPlane(&planes.GetDistances()[0]+i);
      VcPrecision distanceTest = plane.Dot(point) + dPlane;
      // Check if the point is in front of/behind the plane according to the
      // template parameter
      VcBool valid = Flip<behindPlanesT>::FlipSign(distanceTest) >= 0;
      if (IsEmpty(valid)) continue;
      VcPrecision directionProjection = plane.Dot(direction);
      valid &= Flip<!behindPlanesT>::FlipSign(directionProjection) >= 0;
      if (IsEmpty(valid)) continue;
      VcPrecision tiny = VcPrecision(1E-20).copySign(directionProjection);
      distanceTest /= -(directionProjection + tiny);
      Vector3D<VcPrecision> intersection =
          Vector3D<VcPrecision>(direction)*distanceTest + point;
      for (int j = 0; j < 4; ++j) {
        Vector3D<VcPrecision> sideVector(
            VcPrecision(sideVectors[j].GetNormals().x()+i),
            VcPrecision(sideVectors[j].GetNormals().y()+i),
            VcPrecision(sideVectors[j].GetNormals().z()+i));
        VcPrecision dSide(&sideVectors[j].GetDistances()[i]);
        valid &= sideVector.Dot(intersection) + dSide >= 0;
        // Where is your god now
        if (IsEmpty(valid)) goto distanceToInVcContinueOuter;
      }
      // If a hit is found, the algorithm can return, since only one side can
      // be hit for a convex set of quadrilaterals
      distanceTest(!valid) = kInfinity;
      distance = distanceTest.min();
      i = n;
      return;
      // Continue label of outer loop
      distanceToInVcContinueOuter:;
    }
  #endif
    return;
  }

};

} // End anonymous namespace

template <class Backend, bool behindPlanesT>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToIn(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {

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

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;

  int i = 0;
  const int n = size();
  AcceleratedDistanceToIn<Backend>::template VectorLoop<behindPlanesT>(
      i, n, fPlanes, fSideVectors, point, direction, bestDistance);

  // TODO: IN CASE QUADRILATERALS ARE PERPENDICULAR TO Z WE COULD SAVE MANY DIVISIONS
  for (; i < n; ++i) {
    Vector3D<Precision> normal = fPlanes.GetNormal(i);
    Float_t distance = point.Dot(normal) + fPlanes.GetDistance(i);
    // Check if the point is in front of/behind the plane according to the
    // template parameter
    Bool_t valid = Flip<behindPlanesT>::FlipSign(distance) >= 0;
    if (IsEmpty(valid)) continue;
    Float_t directionProjection = direction.Dot(normal);
    valid &= Flip<!behindPlanesT>::FlipSign(directionProjection) >= 0;
    if (IsEmpty(valid)) continue;
    distance /= -(directionProjection + CopySign(1E-20, directionProjection));
    Vector3D<Float_t> intersection = point + direction*distance;
    for (int j = 0; j < 4; ++j) {
      valid &= intersection.Dot(fSideVectors[j].GetNormal(i)) +
               fSideVectors[j].GetDistances()[i] >= 0;
      if (IsEmpty(valid)) break;
    }
    MaskedAssign(valid, distance, &bestDistance);
    // If all hits are found, the algorithm can return, since only one side can
    // be hit for a convex set of quadrilaterals
    if (IsFull(bestDistance < kInfinity)) break;
  }

  return bestDistance;
}

namespace {

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AcceleratedDistanceToOut(
    int &i,
    const int n,
    Planes const &planes,
    Planes const (&sideVectors)[4],
    const Precision zMin,
    const Precision zMax,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v &distance) {
  // Do nothing if the backend is not scalar
  return;
}

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AcceleratedDistanceToOut<kScalar>(
    int &i,
    const int n,
    Planes const &planes,
    Planes const (&sideVectors)[4],
    const Precision zMin,
    const Precision zMax,
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    Precision &distance) {
#if defined(VECGEOM_VC) && defined(VECGEOM_QUADRILATERALS_VC)
  // Explicitly vectorize over quadrilaterals using Vc
  for (;i <= n-kVectorSize; i += kVectorSize) {
    Vector3D<VcPrecision> plane(
        VcPrecision(planes.GetNormals().x()+i),
        VcPrecision(planes.GetNormals().y()+i),
        VcPrecision(planes.GetNormals().z()+i));
    VcPrecision dPlane(&planes.GetDistances()[0]+i);
    VcPrecision distanceTest = plane.Dot(point) + dPlane;
    // Check if the point is behind the plane
    VcBool valid = distanceTest < 0;
    if (IsEmpty(valid)) continue;
    VcPrecision directionProjection = plane.Dot(direction);
    // Because the point is behind the plane, the direction must be along the
    // normal
    valid &= directionProjection >= 0;
    if (IsEmpty(valid)) continue;
    distanceTest /= -directionProjection;
    valid &= distanceTest < distance;
    if (IsEmpty(valid)) continue;

    if(zMin==zMax) { // need a careful treatment in case of degenerate Z planes
        // in this case need proper hit detection
        Vector3D<VcPrecision> intersection =
                  Vector3D<VcPrecision>(direction)*distanceTest + point;
        for (int j = 0; j < 4; ++j) {
            Vector3D<VcPrecision> sideVector(
                VcPrecision(sideVectors[j].GetNormals().x()+i),
                VcPrecision(sideVectors[j].GetNormals().y()+i),
                VcPrecision(sideVectors[j].GetNormals().z()+i));
           VcPrecision dSide(&sideVectors[j].GetDistances()[i]);
           valid &= sideVector.Dot(intersection) + dSide >= 0;
          // Where is your god now
           if (IsEmpty(valid)) goto distanceToOutVcContinueOuter;
        }
    }
    else
    {
        VcPrecision zProjection = distanceTest*direction[2] + point[2];
        valid &= zProjection >= zMin && zProjection < zMax;
    }
distanceToOutVcContinueOuter:
    if (IsEmpty(valid)) continue;
    distanceTest(!valid) = kInfinity;
    distance = distanceTest.min();
  }
#endif
  return;
}

} // End anonymous namespace

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    Precision zMin,
    Precision zMax) const {

  // The below computes the distance to the quadrilaterals similar to
  // DistanceToIn, but is optimized for Polyhedron, and as such can assume that
  // the quadrilaterals form a convex shell, and that the shortest distance to
  // one of the quadrilaterals will indeed be an intersection. The exception to
  // this is if the point leaves the Z-bounds specified in the input parameters.
  // If used for another purpose than Polyhedron, DistanceToIn should be used if
  // the set of quadrilaterals is not convex.

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;

  int i = 0;
  const int n = size();
  AcceleratedDistanceToOut<Backend>(
      i, n, fPlanes, fSideVectors, zMin, zMax, point, direction, bestDistance);

  for (; i < n; ++i) {
    Vector3D<Precision> normal = fPlanes.GetNormal(i);
    Float_t distanceTest = point.Dot(normal) + fPlanes.GetDistance(i);
    // Check if the point is behind the plane
    Bool_t valid = distanceTest < 0;
    if (IsEmpty(valid)) continue;
    Float_t directionProjection = direction.Dot(normal);
    // Because the point is behind the plane, the direction must be along the
    // normal
    valid &= directionProjection >= 0;
    if (IsEmpty(valid)) continue;
    distanceTest /= -directionProjection;
    valid &= distanceTest < bestDistance;
    if (IsEmpty(valid)) continue;


    // this is a tricky test when zMin == zMax ( degenerate planes )
    if( zMin == zMax ){
        // in this case need proper hit detection
        //valid &= zProjection >= zMin-1E-10 && zProjection <= zMax+1E-10;
        Vector3D<Float_t> intersection = point + distanceTest*direction;

        valid = RayHitsQuadrilateral( i, intersection );

    }
    else
    {
        Float_t zProjection = point[2] + distanceTest*direction[2];
        valid &= zProjection >= zMin && zProjection < zMax;
    }
    if (IsEmpty(valid)) continue;
    MaskedAssign(valid, distanceTest, &bestDistance);
  }

  return bestDistance;
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceToOut<Backend>(point, direction, -kInfinity, kInfinity);
}

VECGEOM_CUDA_HEADER_BOTH
Precision Quadrilaterals::ScalarDistanceSquared(
    int i,
    Vector3D<Precision> const &point) const {

  // This function is used by the safety algorithms to return the exact distance
  // to the quadrilateral specified.
  // The algorithm has three stages, trying first to return the shortest
  // distance to the plane, then to the closest line segment, then to the
  // closest corner.
  assert( i < size() );

  Vector3D<Precision> planeNormal = fPlanes.GetNormal(i);
  Precision distance = point.Dot(planeNormal) + fPlanes.GetDistance(i);
  Vector3D<Precision> intersection = point + distance*planeNormal;

  bool withinBound[4];
  for (int j = 0; j < 4; ++j) {
    // TODO: check if this autovectorizes. Otherwise it should be explicitly
    //       vectorized.
    withinBound[j] = intersection[0]*fSideVectors[j].GetNormals().x(i) +
                     intersection[1]*fSideVectors[j].GetNormals().y(i) +
                     intersection[2]*fSideVectors[j].GetNormals().z(i) +
                     fSideVectors[j].GetDistances()[i] >= 0;
  }
  if (withinBound[0] && withinBound[1] && withinBound[2] && withinBound[3]) {
    return distance*distance;
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

  Vector3D<Precision> corner0, corner1;

  if (!withinBound[0]) {
    // To the right of right side
    corner0 = fCorners[0][i];
    corner1 = fCorners[1][i];
  } else if (!withinBound[2]) {
    // To the left of left side
    corner0 = fCorners[2][i];
    corner1 = fCorners[3][i];
  } else if (!withinBound[1]) {
    // Below in the middle
    corner0 = fCorners[1][i];
    corner1 = fCorners[2][i];
  } else {
    // Above in the middle
    corner0 = fCorners[3][i];
    corner1 = fCorners[0][i];
  }

  return DistanceToLineSegmentSquared<kScalar>(corner0, corner1, point);
}

std::ostream& operator<<(std::ostream &os, Quadrilaterals const &quads);

} // End inline namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_QUADRILATERALS_H_
