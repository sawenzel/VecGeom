/// \file Quadrilaterals.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_QUADRILATERALS_H_
#define VECGEOM_VOLUMES_QUADRILATERALS_H_

#include "base/Global.h"

#include "backend/Backend.h"
#include "base/Array.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"
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
  int size() const { return fPlanes.size(); }

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

  /// Set the sign of the specified component of the normal to positive or
  /// negative by flipping the sign of all components to the desired sign.
  /// \param component Which component of the normal to fix, [0: x, 1: y, 2: z].
  /// \param positive Whether the component should be set to positive (true) or
  ///                 negative (false).
  void FixNormalSign(int component, bool positive);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToIn(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::precision_v DistanceToOut(
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      Precision zMin = -kInfinity,
      Precision zMax = kInfinity) const;

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToInKernel(
      const int n,
      Planes const &planes,
      Planes const (&sideVectors)[4],
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static typename Backend::precision_v DistanceToOutKernel(
      const int n,
      __restrict__ Precision const a[],
      __restrict__ Precision const b[],
      __restrict__ Precision const c[],
      __restrict__ Precision const d[],
      const Precision zMin,
      const Precision zMax,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction);

};

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
typename Backend::precision_v Quadrilaterals::DistanceToIn(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) const {
  return DistanceToInKernel<Backend>(
      size(), fPlanes, fSideVectors, point, direction);
}

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToOut(
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    Precision zMin,
    Precision zMax) const {
  return DistanceToOutKernel<Backend>(
      size(), GetNormals().x(), GetNormals().y(), GetNormals().z(),
      &GetDistances()[0], zMin, zMax, point, direction);
}

namespace {

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AcceleratedDistanceToIn(
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

template <>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void AcceleratedDistanceToIn<kScalar>(
    int &i,
    const int n,
    Planes const &planes,
    Planes const (&sideVectors)[4],
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
    VcPrecision distanceTest = -(plane.Dot(point) + dPlane) /
                                plane.Dot(direction);
    Vector3D<VcPrecision> intersection =
        Vector3D<VcPrecision>(direction)*distanceTest + point;
    VcBool valid = distanceTest >= 0 && distanceTest < distance;
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
    distanceTest(!valid) = kInfinity;
    distance = Min<Precision>(distance, distanceTest.min());
    distanceToInVcContinueOuter:;
  }
#endif
  return;
}

} // End anonymous namespace

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToInKernel(
    const int n,
    Planes const &planes,
    Planes const (&sideVectors)[4],
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

  Float_t bestDistance = kInfinity;

  int i = 0;
  AcceleratedDistanceToIn<Backend>(
      i, n, planes, sideVectors, point, direction, bestDistance);

  for (; i < n; ++i) {
    Float_t distance = 
        -(planes.GetNormals().x(i)*point[0] +
          planes.GetNormals().y(i)*point[1] +
          planes.GetNormals().z(i)*point[2] + planes.GetDistances()[i])
       / (planes.GetNormals().x(i)*direction[0] +
          planes.GetNormals().y(i)*direction[1] +
          planes.GetNormals().z(i)*direction[2]);
    Vector3D<Float_t> intersection = point + direction*distance;
    Bool_t inBounds[4];
    for (int j = 0; j < 4; ++j) {
      inBounds[j] = sideVectors[j].GetNormals().x(i)*intersection[0] +
                    sideVectors[j].GetNormals().y(i)*intersection[1] +
                    sideVectors[j].GetNormals().z(i)*intersection[2] +
                    sideVectors[j].GetDistances()[i] >= 0;
    }
    MaskedAssign(distance >= 0 && distance < bestDistance &&
                 inBounds[0] && inBounds[1] && inBounds[2] && inBounds[3],
                 distance, &bestDistance);
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
    __restrict__ Precision const a[],
    __restrict__ Precision const b[],
    __restrict__ Precision const c[],
    __restrict__ Precision const d[],
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
    __restrict__ Precision const a[],
    __restrict__ Precision const b[],
    __restrict__ Precision const c[],
    __restrict__ Precision const d[],
    const Precision zMin,
    const Precision zMax,
    Vector3D<Precision> const &point,
    Vector3D<Precision> const &direction,
    Precision &distance) {
#if defined(VECGEOM_VC) && defined(VECGEOM_QUADRILATERALS_VC)
  // Explicitly vectorize over quadrilaterals using Vc
  for (;i <= n-kVectorSize; i += kVectorSize) {
    VcPrecision aVc(&a[i]), bVc(&b[i]), cVc(&c[i]), dVc(&d[i]);
    VcPrecision distanceTest =
        -(aVc*point[0] + bVc*point[1] + cVc*point[2] + dVc)
       / (aVc*direction[0] + bVc*direction[1] + cVc*direction[2]);
    VcPrecision zProjection = distanceTest*direction[2] + point[2];
    distanceTest(distanceTest < 0 || zProjection < zMin ||
                 zProjection >= zMax) = kInfinity;
    distance = Min<Precision>(distance, distanceTest.min());
  }
#endif
  return;
}

} // End anonymous namespace

template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v Quadrilaterals::DistanceToOutKernel(
    const int n,
    __restrict__ Precision const a[],
    __restrict__ Precision const b[],
    __restrict__ Precision const c[],
    __restrict__ Precision const d[],
    const Precision zMin,
    const Precision zMax,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction) {

  typedef typename Backend::precision_v Float_t;

  Float_t bestDistance = kInfinity;

  int i = 0;
  AcceleratedDistanceToOut<Backend>(
      i, n, a, b, c, d, zMin, zMax, point, direction, bestDistance);

  for (; i < n; ++i) {
    Float_t distance = -(a[i]*point[0] + b[i]*point[1] +
                         c[i]*point[2] + d[i])
                      / (a[i]*direction[0] + b[i]*direction[1] +
                         c[i]*direction[2]);
    Float_t zProjection = point[2] + distance*direction[2];
    MaskedAssign(distance >= 0 && zProjection >= zMin && zProjection <= zMax &&
                 distance < bestDistance, distance, &bestDistance);
  }

  return bestDistance;
}

std::ostream& operator<<(std::ostream &os, Quadrilaterals const &quads);

} // End global namespace

#endif // VECGEOM_VOLUMES_QUADRILATERALS_H_