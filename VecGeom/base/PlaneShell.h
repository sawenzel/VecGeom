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
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DistanceToPoint(Vector3D<Type2> const &point,
                                                                    Type2 *distances) const
  {
    for (int i = 0; i < N; ++i) {
      distances[i] = this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i];
    }
  }

  /// \return the projection of a (Vector3D) direction into each plane's normal vector.
  /// The type returned is float, double, or various SIMD vector types.
  template <typename Type2>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void ProjectionToNormal(Vector3D<Type2> const &dir,
                                                                       Type2 *projection) const
  {
    for (int i = 0; i < N; ++i) {
      projection[i] = this->fA[i] * dir.x() + this->fB[i] * dir.y() + this->fC[i] * dir.z();
    }
  }

  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void GenericKernelForContainsAndInside(
      Vector3D<Real_v> const &point, vecCore::Mask_v<Real_v> &completelyInside,
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
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v DistanceToIn(Vector3D<Real_v> const &point,
                                                                   Vector3D<Real_v> const &dir, Real_v &smin,
                                                                   Real_v &smax) const
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
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v DistanceToOut(Vector3D<Real_v> const &point,
                                                                    Vector3D<Real_v> const &dir) const
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
      vecCore__MaskedAssignFunc(distOut, proj[i] > kTolerance && vdist[i] < distOut, vdist[i]);
      // std::cerr<<"i="<< i <<", pdist="<< pdist[i] <<", proj="<< proj[i] <<", vdist="<< vdist[i] <<" "<< vdist1[i] <<"
      // --> dist="<< distOut <<", "<< distOut1 <<"\n";
    }

    return distOut;
  }

  /// @brief Update a `DistanceToOut` result from a tolerated set of planes.
  /// @tparam Real_v Floating-point scalar type.
  /// @param point Local start point.
  /// @param dir Unit local direction.
  /// @param distanceTolerance Distance tolerance for the plane shell.
  /// @param directionTolerance Projection threshold for an outward crossing.
  /// @param[in,out] outside Set when @p point is outside any plane beyond tolerance.
  /// @param[in,out] distance Current exit distance, updated with a nearer side-plane exit.
  ///
  /// @details Plane distances are negative inside and positive outside. A
  /// tolerated surface point moving outward contributes a zero exit; a strictly
  /// inside point contributes the nearest forward plane crossing.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void DistanceToOut(Vector3D<Real_v> const &point,
                                                                  Vector3D<Real_v> const &dir,
                                                                  Real_v const &distanceTolerance,
                                                                  Real_v const &directionTolerance, bool &outside,
                                                                  Real_v &distance) const
  {
    int i = 0;

#if defined(VECGEOM_VC) && defined(VECGEOM_QUADRILATERAL_ACCELERATION)
    if constexpr (vecCore::VectorSize<Real_v>() == 1) {
      using VecReal          = vecgeom::VectorBackend::Real_v;
      constexpr int kVecSize = static_cast<int>(vecCore::VectorSize<VecReal>());
      for (; i <= N - kVecSize; i += kVecSize) {
        VecReal pdist =
            VecReal(fA + i) * point.x() + VecReal(fB + i) * point.y() + VecReal(fC + i) * point.z() + VecReal(fD + i);
        VecReal proj = VecReal(fA + i) * dir.x() + VecReal(fB + i) * dir.y() + VecReal(fC + i) * dir.z();

        auto outsidePlanes = pdist > distanceTolerance;
        outside |= !vecCore::MaskEmpty(outsidePlanes);

        auto exitingSurface = pdist >= -distanceTolerance;
        exitingSurface &= pdist <= distanceTolerance;
        exitingSurface &= proj > directionTolerance;
        if (!vecCore::MaskEmpty(exitingSurface)) distance = Real_v(0.);

        auto candidate = pdist < -distanceTolerance;
        candidate &= proj > directionTolerance;
        if (!vecCore::MaskEmpty(candidate)) {
          VecReal denom            = proj;
          denom(!candidate)        = Real_v(1.);
          VecReal vdist            = -pdist / denom;
          vdist(!candidate)        = InfinityLength<Real_v>();
          Real_v candidateDistance = vdist.min();
          if (candidateDistance < distance) distance = candidateDistance;
        }
      }
    }
#endif

    for (; i < N; ++i) {
      Real_v pdist = fA[i] * point.x() + fB[i] * point.y() + fC[i] * point.z() + fD[i];
      Real_v proj  = fA[i] * dir.x() + fB[i] * dir.y() + fC[i] * dir.z();

      outside |= pdist > distanceTolerance;
      if (pdist < -distanceTolerance) {
        if (proj > directionTolerance) {
          Real_v vdist = -pdist / NonZero(proj);
          if (vdist < distance) distance = vdist;
        }
      } else if (pdist <= distanceTolerance && proj > directionTolerance) {
        distance = Real_v(0.);
      }
    }
  }

  /// \return the safety distance to the planar shell when the point is located within the shell itself.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void SafetyToIn(Vector3D<Real_v> const &point, Real_v &safety) const
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
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE void SafetyToOut(Vector3D<Real_v> const &point, Real_v &safety) const
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
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE size_t ClosestFace(Vector3D<Real_v> const &point, Real_v &safety) const
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

  /// @param point Point position in local coordinates
  /// @param[out] normal A vector normal to the plane closest to point.
  /// If the point has kSurface condition for more than one plane, the un-normalized sum is returned
  /// @param[out] edge Point on edge condition. The normal vector needs to be normalized by the user
  /// @return Distance to closest surface
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v NormalKernel(Vector3D<Real_v> const &point,
                                                                   Vector3D<Real_v> &normal, bool &edge) const
  {
    Real_v safety = InfinityLength<Real_v>();

    // vectorizable loop
    Real_v dist[N];
    Vector3D<Real_v> cornerNormal;
    unsigned char surfaces = 0;
    edge                   = false;
    for (int i = 0; i < N; ++i) {
      dist[i] = Abs(this->fA[i] * point.x() + this->fB[i] * point.y() + this->fC[i] * point.z() + this->fD[i]);
      // If closest update normal
      if (dist[i] < safety) {
        normal.Set(this->fA[i], this->fB[i], this->fC[i]);
        safety = dist[i];
      }
      // If on surface add to separate vector
      if (dist[i] < kTolerance) {
        surfaces++;
        cornerNormal += Vector3D<Real_v>(this->fA[i], this->fB[i], this->fC[i]);
      }
    }
    if (surfaces > 1) {
      // The point is on the edge - do not normalize the vector
      normal = cornerNormal;
      edge   = true;
    }

    return safety;
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BASE_SIDEPLANES_H_
