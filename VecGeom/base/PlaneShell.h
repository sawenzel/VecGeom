//===-- base/PlaneShell.h ----------------------------*- C++ -*-===//
/// \file PlaneShell.h
/// \author Guilherme Lima (lima at fnal dot gov)

#ifndef VECGEOM_BASE_SIDEPLANES_H_
#define VECGEOM_BASE_SIDEPLANES_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"

// namespace vecgeom::cuda { template <typename Real_v, int N> class PlaneShell; }
#include <VecCore/VecCore>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * @brief Uses SoA layout to store arrays of N (plane) parameters,
 *        representing a set of planes defining a volume or shape.
 *
 * For some volumes, e.g. trapezoid, when two of the planes are
 * parallel, they should be set perpendicular to the Z-axis, and then
 * the inside/outside calculations become trivial.  Therefore those planes
 * should NOT be included in this class.
 *
 * @details If vector acceleration is enabled, the scalar template
 *        instantiation will use vector instructions for operations
 *        when possible.
 */

template <int N, typename Type>
struct PlaneShell {

  // Using a SOA-like data structure for vectorization
  Precision fA[N];
  Precision fB[N];
  Precision fC[N];
  Precision fD[N];

public:
  /**
   * Initializes the SOA with existing data arrays, performing no allocation.
   */
  VECCORE_ATT_HOST_DEVICE
  PlaneShell(Precision *const a, Precision *const b, Precision *const c, Precision *const d)
  {
    memcpy(&(this->fA), a, N * sizeof(Type));
    memcpy(&(this->fB), b, N * sizeof(Type));
    memcpy(&(this->fC), c, N * sizeof(Type));
    memcpy(&(this->fD), d, N * sizeof(Type));
  }

  /**
   * Initializes the SOA with a fixed size, allocating an aligned array for each
   * coordinate of the specified size.
   */
  VECCORE_ATT_HOST_DEVICE
  PlaneShell()
  {
    memset(&(this->fA), 0, N * sizeof(Type));
    memset(&(this->fB), 0, N * sizeof(Type));
    memset(&(this->fC), 0, N * sizeof(Type));
    memset(&(this->fD), 0, N * sizeof(Type));
  }

  /**
   * Copy constructor
   */
  VECCORE_ATT_HOST_DEVICE
  PlaneShell(PlaneShell const &other)
  {
    memcpy(&(this->fA), &(other.fA), N * sizeof(Type));
    memcpy(&(this->fB), &(other.fB), N * sizeof(Type));
    memcpy(&(this->fC), &(other.fC), N * sizeof(Type));
    memcpy(&(this->fD), &(other.fD), N * sizeof(Type));
  }

  /**
   * assignment operator
   */
  VECCORE_ATT_HOST_DEVICE
  PlaneShell &operator=(PlaneShell const &other)
  {
    memcpy(this->fA, other.fA, N * sizeof(Type));
    memcpy(this->fB, other.fB, N * sizeof(Type));
    memcpy(this->fC, other.fC, N * sizeof(Type));
    memcpy(this->fD, other.fD, N * sizeof(Type));
    return *this;
  }

  VECCORE_ATT_HOST_DEVICE
  void Set(int i, Precision a, Precision b, Precision c, Precision d)
  {
    fA[i] = a;
    fB[i] = b;
    fC[i] = c;
    fD[i] = d;
  }

  VECCORE_ATT_HOST_DEVICE
  unsigned int size() { return N; }

  VECCORE_ATT_HOST_DEVICE
  ~PlaneShell() {}

  /// \return the distance from point to each plane.  The type returned is float, double, or various SIMD vector types.
  /// Distances are negative (positive) for points in same (opposite) side from plane as the normal vector.
  template <typename Type2>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void DistanceToPoint(Vector3D<Type2> const &point, Type2 *distances) const
  {
    for (int i = 0; i < N; ++i) {
      distances[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
    }
  }

  /// \return the projection of a (Vector3D) direction into each plane's normal vector.
  /// The type returned is float, double, or various SIMD vector types.
  template <typename Type2>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void ProjectionToNormal(Vector3D<Type2> const &dir, Type2 *projection) const
  {
    for (int i = 0; i < N; ++i) {
      projection[i] = this->fA[i] * dir.x() + this->fB[i] * dir.y() + this->fC[i] * dir.z();
    }
  }

  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void GenericKernelForContainsAndInside(Vector3D<Real_v> const &point, vecCore::Mask_v<Real_v> &completelyInside,
                                         vecCore::Mask_v<Real_v> &completelyOutside) const
  {
    // auto-vectorizable loop for Backend==scalar
    Real_v dist[N];
    for (unsigned int i = 0; i < N; ++i) {
      dist[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
    }

    // analysis loop - not auto-vectorizable
    for (unsigned int i = 0; i < N; ++i) {
      // is it outside of this side plane?
      completelyOutside = completelyOutside || (dist[i] > Real_v(MakePlusTolerant<ForInside>(0.)));
      if (ForInside) {
        completelyInside = completelyInside && (dist[i] < Real_v(MakeMinusTolerant<ForInside>(0.)));
      }
      // if (vecCore::EarlyReturnMaxLength(completelyOutside,1) && vecCore::MaskFull(completelyOutside)) return;
    }
  }

  /// \return the distance to the planar shell when the point is located outside.
  /// The type returned is the type corresponding to the backend given.
  /// For some special cases, the value returned is:
  ///     (1) +inf, if point+dir is outside & moving AWAY FROM OR PARALLEL TO any plane,
  ///     (2) -1, if point+dir crosses out a plane BEFORE crossing in ALL other planes (wrong-side)
  ///
  /// Note: smin,smax parameters are needed here, to flag shape-missing tracks.
  ///
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Real_v DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir, Real_v &smin, Real_v &smax) const
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    Real_v distIn(kInfLength); // set for earlier returns

    // hope for a vectorization of this part for Backend==scalar!!
    Real_v pdist[N];
    Real_v proj[N];
    Real_v vdist[N];
    // vectorizable part
    for (int i = 0; i < N; ++i) {
      pdist[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
      proj[i]  = this->fA[i] * dir.x() + this->fB[i] * dir.y() + this->fC[i] * dir.z();

      // note(SW): on my machine it was better to keep vdist[N] instead of a local variable vdist below
      vdist[i] = -pdist[i] / NonZero(proj[i]);
    }

    // wrong-side check: if (inside && smin<0) return -1
    for (int i = 0; i < N; ++i) {
      done = done || (pdist[i] > Real_v(MakePlusTolerant<true>(0.)) && proj[i] >= Real_v(0.));
      done = done || (pdist[i] > Real_v(MakeMinusTolerant<true>(0.)) && proj[i] > Real_v(0.));
    }
    if (vecCore::EarlyReturnMaxLength(done, 1) && vecCore::MaskFull(done)) return distIn;

    // analysis loop
    for (int i = 0; i < N; ++i) {
      // if outside and moving away, return infinity
      Bool_v posPoint = pdist[i] > Real_v(MakeMinusTolerant<true>(0.));
      Bool_v posDir   = proj[i] > 0;

      // check if trajectory will intercept plane within current range (smin,smax)
      Bool_v interceptFromInside = (!posPoint && posDir);
      done                       = done || (interceptFromInside && vdist[i] < smin);

      Bool_v interceptFromOutside = (posPoint && !posDir);
      done                        = done || (interceptFromOutside && vdist[i] > smax);
      if (vecCore::EarlyReturnMaxLength(done, 1) && vecCore::MaskFull(done)) return distIn;

      // update smin,smax
      vecCore__MaskedAssignFunc(smin, interceptFromOutside && vdist[i] > smin, vdist[i]);
      vecCore__MaskedAssignFunc(smax, interceptFromInside && vdist[i] < smax, vdist[i]);
    }

    // Survivors will return smin, which is the maximum distance in an interceptFromOutside situation
    // (SW: not sure this is true since smin is initialized from outside and can have any arbitrary value)
    vecCore::MaskedAssign(distIn, !done && smin <= smax, smin);
    return distIn;
  }

  /// \return the distance to the planar shell when the point is located within the shell itself.
  /// The type returned is the type corresponding to the backend given.
  /// For some special cases, the value returned is:
  ///     (1) -1, if point is outside (wrong-side)
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Real_v DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {
    // using Bool_v = vecCore::Mask_v<Real_v>;
    // Bool_v done(false);
    Real_v distOut(kInfLength);
    // Real_v distOut1(kInfLength);

    // hope for a vectorization of this part for Backend==scalar !!
    // the idea is to put vectorizable things into this loop
    // and separate the analysis into a separate loop if need be
    Real_v pdist[N];
    Real_v proj[N];
    Real_v vdist[N];
    for (int i = 0; i < N; ++i) {
      pdist[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
      proj[i]  = this->fA[i] * dir.x() + this->fB[i] * dir.y() + this->fC[i] * dir.z();
      vdist[i] = -pdist[i] / NonZero(proj[i]);
    }

    // early return if point is outside of plane
    // for (int i = 0; i < N; ++i) {
    //   done = done || (pdist[i] > kHalfTolerance);
    // }
    // vecCore__MaskedAssignFunc(distOut, done, Real_v(-1.0));
    // // if (vecCore::EarlyReturnMaxLength(done,1) && vecCore::MaskFull(done)) return distOut;

    // std::cerr<<"=== point="<< point <<", dir="<< dir <<"\n";
    for (int i = 0; i < N; ++i) {
      vecCore__MaskedAssignFunc(distOut, pdist[i] > kHalfTolerance, Real_v(-1.));
      vecCore__MaskedAssignFunc(distOut, proj[i] > Real_v(0.) && vdist[i] < distOut, vdist[i]);
      // std::cerr<<"i="<< i <<", pdist="<< pdist[i] <<", proj="<< proj[i] <<", vdist="<< vdist[i] <<" "<< vdist1[i] <<"
      // --> dist="<< distOut <<", "<< distOut1 <<"\n";
    }

    return distOut;
  }

  /// \return the safety distance to the planar shell when the point is located within the shell itself.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SafetyToIn(Vector3D<Real_v> const &point, Real_v &safety) const
  {
    // vectorizable loop
    Real_v dist[N];
    for (int i = 0; i < N; ++i) {
      dist[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
    }

    // non-vectorizable part
    for (int i = 0; i < N; ++i) {
      vecCore__MaskedAssignFunc(safety, dist[i] > safety, dist[i]);
    }
  }

  /// \return the distance to the planar shell when the point is located within the shell itself.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SafetyToOut(Vector3D<Real_v> const &point, Real_v &safety) const
  {
    // vectorizable loop
    Real_v dist[N];
    for (int i = 0; i < N; ++i) {
      dist[i] = -(this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i]);
    }

    // non-vectorizable part
    for (int i = 0; i < N; ++i) {
      vecCore__MaskedAssignFunc(safety, dist[i] < safety, dist[i]);
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  size_t ClosestFace(Vector3D<Real_v> const &point, Real_v &safety) const
  {
    // vectorizable loop
    Real_v dist[N];
    for (int i = 0; i < N; ++i) {
      dist[i] = Abs(this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i]);
    }

    // non-vectorizable part
    using Bool_v    = vecCore::Mask_v<Real_v>;
    using Index_v   = vecCore::Index<Real_v>;
    Index_v closest = static_cast<Index_v>(-1);
    for (size_t i = 0; i < N; ++i) {
      Bool_v closer = dist[i] < safety;
      vecCore__MaskedAssignFunc(safety, closer, dist[i]);
      vecCore::MaskedAssign(closest, closer, i);
    }

    return closest;
  }

  /// \return a *non-normalized* vector normal to the plane containing (or really close) to point.
  /// In most cases the resulting normal is either (0,0,0) when point is not close to any plane,
  /// or the normal of that single plane really close to the point (in this case, it *is* normalized).
  ///
  /// Note: If the point is really close to more than one plane, those planes' normals are added,
  /// and in this case the vector returned *is not* normalized.  This can be used as a flag, so that
  /// the callee knows that nsurf==2 (the maximum value possible).
  ///
  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Real_v NormalKernel(Vector3D<Real_v> const &point, Vector3D<Real_v> &normal) const
  {
    Real_v safety = -InfinityLength<Real_v>();

    // vectorizable loop
    Real_v dist[N];
    for (int i = 0; i < N; ++i) {
      dist[i] = (this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i]);
    }

    // non-vectorizable part
    for (int i = 0; i < 4; ++i) {
      Real_v saf_i = dist[i] - safety;

      // if more planes found as far (within tolerance) as the best one so far *and not fully inside*, add its normal
      vecCore__MaskedAssignFunc(normal, Abs(saf_i) < kHalfTolerance && dist[i] >= -kHalfTolerance,
                                normal + Vector3D<Real_v>(this->fA[i], this->fB[i], this->fC[i]));

      // this one is farther than our previous one -- update safety and normal
      vecCore__MaskedAssignFunc(normal, saf_i > Real_v(0.), Vector3D<Real_v>(this->fA[i], this->fB[i], this->fC[i]));
      vecCore__MaskedAssignFunc(safety, saf_i > Real_v(0.), dist[i]);
      // std::cerr<<"dist["<< i <<"]="<< dist[i] <<", saf_i="<< saf_i <<", safety="<< safety <<", normal="<< normal
      // <<"\n";
    }

    // Note: this could be (rarely) a non-normalized normal vector (when point is close to 2 planes)
    // std::cerr<<"Return from PlaneShell::Normal: safety="<< safety <<", normal="<< normal <<"\n";
    return safety;
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BASE_SIDEPLANES_H_
