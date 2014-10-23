/// \file Quadrilaterals.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_QUADRILATERALS_H_
#define VECGEOM_VOLUMES_QUADRILATERALS_H_

#include "base/Global.h"

#include "backend/Backend.h"
#include "base/Array.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/Planes.h"

// Switches on/off explicit vectorization of algorithms using Vc
#define VECGEOM_QUADRILATERALS_VC

namespace VECGEOM_NAMESPACE {

class Quadrilaterals {

private:

  Planes fPlanes;
  Planes fSideVectors[4];

public:

  typedef Planes Sides_t[4];

#ifdef VECGEOM_STD_CXX11
  Quadrilaterals(int size);
#endif

  ~Quadrilaterals();

  Quadrilaterals(Quadrilaterals const &other);

  Quadrilaterals& operator=(Quadrilaterals const &other);

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

#ifdef VECGEOM_STD_CXX11
  /// \param corner0 First corner in counterclockwise order.
  /// \param corner1 Second corner in counterclockwise order.
  /// \param corner2 Third corner in counterclockwise order.
  /// \param corner3 Fourth corner in counterclockwise order.
  void Set(
      int index,
      Vector3D<Precision> const &corner0,
      Vector3D<Precision> const &corner1,
      Vector3D<Precision> const &corner2,
      Vector3D<Precision> const &corner3);
#endif

  /// Flips the sign of the normal and distance of the specified quadrilateral.
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

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::bool_v Contains(
      int begin,
      int end,
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::inside_v Inside(
      int begin,
      int end,
      Vector3D<typename Backend::precision_v> const &point) const;

  template <class Backend, bool behindPlanesT>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToIn(
      int begin,
      int end,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

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
      int begin,
      int end,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      Precision zMin,
      Precision zMax) const;

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

  template <class Backend, bool behindPlanesT>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToInKernel(
      int begin,
      const int end,
      Planes const &planes,
      Planes const (&sideVectors)[4],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToOutKernel(
      int begin,
      const int end,
      Planes const &planes,
      const Precision zMin,
      const Precision zMax,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

};

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

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::bool_v Quadrilaterals::Contains(
    int begin,
    int end,
    Vector3D<typename Backend::precision_v> const &point) const {
  return fPlanes.Contains<Backend>(begin, end, point);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::inside_v Quadrilaterals::Inside(
    int begin,
    int end,
    Vector3D<typename Backend::precision_v> const &point) const {
  return fPlanes.Inside<Backend>(begin, end, point);
}

template <class Backend, bool behindPlanesT>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToIn(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceToInKernel<Backend, behindPlanesT>(
      0, size(), fPlanes, fSideVectors, point, direction);
}

template <class Backend, bool behindPlanesT>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToIn(
    int begin,
    int end,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceToInKernel<Backend, behindPlanesT>(
      begin, end, fPlanes, fSideVectors, point, direction);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToOut(
    int begin,
    int end,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    Precision zMin,
    Precision zMax) const {
  return DistanceToOutKernel<Backend>(
      begin, end, GetPlanes(), zMin, zMax, point, direction);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    Precision zMin,
    Precision zMax) const {
  return DistanceToOutKernel<Backend>(
      0, size(), GetPlanes(), zMin, zMax, point, direction);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceToOutKernel<Backend>(
      0, size(), GetPlanes(), -kInfinity, kInfinity, point, direction);
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
      VcBool valid = FlipSign<behindPlanesT>::Flip(distanceTest) >= 0;
      if (IsEmpty(valid)) continue;
      VcPrecision directionProjection = plane.Dot(direction);
      valid &= FlipSign<!behindPlanesT>::Flip(directionProjection) >= 0;
      if (IsEmpty(valid)) continue;
      distanceTest /= -directionProjection;
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
typename Backend::precision_v Quadrilaterals::DistanceToInKernel(
    int i,
    const int n,
    Planes const &planes,
    Planes const (&sideVectors)[4],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;

  AcceleratedDistanceToIn<Backend>::template VectorLoop<behindPlanesT>(
      i, n, planes, sideVectors, point, direction, bestDistance);

  for (; i < n; ++i) {
    Float_t distance = 
        planes.GetNormals().x(i)*point[0] +
        planes.GetNormals().y(i)*point[1] +
        planes.GetNormals().z(i)*point[2] + planes.GetDistances()[i];
    // Check if the point is in front of/behind the plane according to the
    // template parameter
    Bool_t valid = vecgeom::FlipSign<behindPlanesT>::Flip(distance) >= 0;
    if (IsEmpty(valid)) continue;
    Float_t directionProjection =
        planes.GetNormals().x(i)*direction[0] +
        planes.GetNormals().y(i)*direction[1] +
        planes.GetNormals().z(i)*direction[2];
    valid &= vecgeom::FlipSign<!behindPlanesT>::Flip(directionProjection) >= 0;
    if (IsEmpty(valid)) continue;
    distance /= -directionProjection;
    Vector3D<Float_t> intersection = point + direction*distance;
    for (int j = 0; j < 4; ++j) {
      valid &= sideVectors[j].GetNormals().x(i)*intersection[0] +
               sideVectors[j].GetNormals().y(i)*intersection[1] +
               sideVectors[j].GetNormals().z(i)*intersection[2] +
               sideVectors[j].GetDistances()[i] >= 0;
      // Where is your god now
      if (IsEmpty(valid)) goto distanceToInContinueOuterLoop;
    }
    MaskedAssign(valid, distance, &bestDistance);
    // If all hits are found, the algorithm can return, since only one side can
    // be hit for a convex set of quadrilaterals
    if (IsFull(bestDistance < kInfinity)) break;
    distanceToInContinueOuterLoop:;
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
    VcPrecision zProjection = distanceTest*direction[2] + point[2];
    valid &= zProjection >= zMin && zProjection < zMax;
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
typename Backend::precision_v Quadrilaterals::DistanceToOutKernel(
    int i,
    const int n,
    Planes const &planes,
    const Precision zMin,
    const Precision zMax,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;

  AcceleratedDistanceToOut<Backend>(
      i, n, planes, zMin, zMax, point, direction, bestDistance);

  for (; i < n; ++i) {
    Float_t distanceTest = 
        planes.GetNormals().x(i)*point[0] +
        planes.GetNormals().y(i)*point[1] +
        planes.GetNormals().z(i)*point[2] + planes.GetDistances()[i];
    // Check if the point is behind the plane
    Bool_t valid = distanceTest < 0;
    if (IsEmpty(valid)) continue;
    Float_t directionProjection =
        planes.GetNormals().x(i)*direction[0] +
        planes.GetNormals().y(i)*direction[1] +
        planes.GetNormals().z(i)*direction[2];
    // Because the point is behind the plane, the direction must be along the
    // normal
    valid &= directionProjection >= 0;
    if (IsEmpty(valid)) continue;
    distanceTest /= -directionProjection;
    valid &= distanceTest < bestDistance;
    if (IsEmpty(valid)) continue;
    Float_t zProjection = point[2] + distanceTest*direction[2];
    valid &= zProjection >= zMin && zProjection < zMax;
    if (IsEmpty(valid)) continue;
    MaskedAssign(valid, distanceTest, &bestDistance);
  }

  return bestDistance;
}

std::ostream& operator<<(std::ostream &os, Quadrilaterals const &quads);

} // End global namespace

#endif // VECGEOM_VOLUMES_QUADRILATERALS_H_