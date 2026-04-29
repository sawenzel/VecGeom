/*
 * PolyconeImplementation.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

/// @file PolyconeImplementation.h
/// @brief Geometry kernels for the classic polycone solid.

/// History notes:
/// Jan-March 2017: revision + moving to use new Cone Kernels (Raman Sehgal)
/// May-June 2017: revision + moving to new Structure (Raman Sehgal)

#ifndef VECGEOM_VOLUMES_KERNEL_POLYCONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYCONEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/PolyconeStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>
#include "VecGeom/volumes/kernel/ConeImplementation.h"
#include "VecGeom/volumes/kernel/shapetypes/ConeTypes.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(struct, PolyconeImplementation, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T>
class SPlacedPolycone;
template <typename T>
class SUnplacedPolycone;

template <typename polyconeTypeT>
/**
 * @brief Kernel entry points for the classic polycone.
 *
 * The classic polycone is implemented as a union of shifted cone sections plus
 * an optional phi wedge. Most point and distance queries are delegated to the
 * section cones, while shared-plane hand-off and contour-based safety are
 * resolved at the polycone layer.
 */
struct PolyconeImplementation {

  using UnplacedStruct_t = PolyconeStruct<Precision>;
  using UnplacedVolume_t = SUnplacedPolycone<polyconeTypeT>;
  using PlacedShape_t    = SPlacedPolycone<UnplacedVolume_t>;

  /**
   * @brief Return the squared minimum `z` separation between a query `z` and an
   * `(r,z)` segment.
   *
   * This is the cheap directional pruning bound used by the contour walk in
   * `SafetyToIn`.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v SegmentMinDeltaZSquared(
      Real_v const &z, Vector3D<Precision> const &a, Vector3D<Precision> const &b)
  {
    const Real_v z0 = Min(Real_v(a.y()), Real_v(b.y()));
    const Real_v z1 = Max(Real_v(a.y()), Real_v(b.y()));
    if (z < z0) {
      const Real_v dz = z0 - z;
      return dz * dz;
    }
    if (z > z1) {
      const Real_v dz = z - z1;
      return dz * dz;
    }
    return Real_v(0.);
  }

  /**
   * @brief Return the squared distance from `(rho,z)` to one `(r,z)` segment.
   *
   * The projection is clamped to the segment end points. The result stays in
   * squared form so the contour search can avoid intermediate square roots.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v DistanceSquaredToRZSegment(
      Real_v const &rho, Real_v const &z, Vector3D<Precision> const &a, Vector3D<Precision> const &b)
  {
    const Real_v ax = Real_v(a.x());
    const Real_v az = Real_v(a.y());
    const Real_v bx = Real_v(b.x());
    const Real_v bz = Real_v(b.y());
    const Real_v dx = bx - ax;
    const Real_v dz = bz - az;
    const Real_v l2 = dx * dx + dz * dz;

    if (l2 <= Real_v(0.)) {
      const Real_v qx = rho - ax;
      const Real_v qz = z - az;
      return qx * qx + qz * qz;
    }

    Real_v t = ((rho - ax) * dx + (z - az) * dz) / l2;
    t        = Max(Real_v(0.), Min(Real_v(1.), t));

    const Real_v rx = rho - (ax + t * dx);
    const Real_v rz = z - (az + t * dz);
    return rx * rx + rz * rz;
  }

  /**
   * @brief Evaluate one cone section for containment or inside classification.
   *
   * @param unplaced Polycone data model.
   * @param isect Section index. Negative indices are treated as fully outside.
   * @param polyconePoint Query point in polycone coordinates.
   * @param secFullyInside Output flag for section containment.
   * @param secFullyOutside Output flag for section exclusion.
   */
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForASection(
      UnplacedStruct_t const &unplaced, int isect, Vector3D<Real_v> const &polyconePoint, bool &secFullyInside,
      bool &secFullyOutside)
  {

    // using namespace PolyconeTypes;
    using namespace ConeTypes;

    if (isect < 0) {
      secFullyInside  = false;
      secFullyOutside = true;
      return;
    }

    PolyconeSection const &sec    = unplaced.GetSection(isect);
    Vector3D<Precision> secLocalp = polyconePoint - Vector3D<Precision>(0, 0, sec.fShift);

    ConeHelpers<Real_v, polyconeTypeT>::template GenericKernelForContainsAndInside<ForInside>(
        sec.fSolid, secLocalp, secFullyInside, secFullyOutside);
  }

  /**
   * @brief Check whether a point belongs to the polycone.
   *
   * This is the boundary-inclusive predicate used by `Contains`.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &polycone,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    bool unused  = false;
    bool outside = false;
    GenericKernelForContainsAndInside<Real_v, false>(polycone, point, unused, outside);
    inside = !outside;
  }

  /**
   * @brief Classify a point as `kInside`, `kSurface`, or `kOutside`.
   *
   * Shared section planes are resolved at the polycone level before mapping the
   * result to `EInside`.
   */
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &polycone,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyinside  = false;
    bool completelyoutside = false;
    GenericKernelForContainsAndInside<Real_v, true>(polycone, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    if (completelyoutside) inside = EInside::kOutside;
    if (completelyinside) inside = EInside::kInside;
  }

  /**
   * @brief Shared kernel for `Contains` and `Inside`.
   *
   * The point is matched against the section owning `z - kTolerance` and
   * `z + kTolerance`. This lets the polycone distinguish ordinary section
   * ownership from shared-plane cases near repeated-`z` boundaries.
   */
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &localPoint, bool &completelyInside,
      bool &completelyOutside)
  {
    int indexLow  = unplaced.GetSectionIndex(localPoint.z() - kTolerance);
    int indexHigh = unplaced.GetSectionIndex(localPoint.z() + kTolerance);
    if (indexLow < 0 && indexHigh < 0) {
      completelyOutside = true;
      return;
    }
    if (indexLow < 0 && indexHigh == 0) {
      // Check location in section 0 and return
      GenericKernelForASection<Real_v, ForInside>(unplaced, 0, localPoint, completelyInside, completelyOutside);
      return;
    }
    if (indexHigh < 0 && indexLow == (unplaced.GetNSections() - 1)) {
      // Check location in section N-1 and return
      GenericKernelForASection<Real_v, ForInside>(unplaced, (unplaced.GetNSections() - 1), localPoint, completelyInside,
                                                  completelyOutside);

      return;
    }
    if (indexLow >= 0 && indexHigh >= 0) {
      if (indexLow == indexHigh) {
        // Check location in section indexLow and return
        GenericKernelForASection<Real_v, ForInside>(unplaced, indexLow, localPoint, completelyInside,
                                                    completelyOutside);

        return;
      } else {

        bool secInLow = false, secOutLow = false;
        bool secInHigh = false, secOutHigh = false;
        GenericKernelForASection<Real_v, ForInside>(unplaced, indexLow, localPoint, secInLow, secOutLow);
        GenericKernelForASection<Real_v, ForInside>(unplaced, indexHigh, localPoint, secInHigh, secOutHigh);
        bool surfLow  = !secInLow && !secOutLow;
        bool surfHigh = !secInHigh && !secOutHigh;

        if (surfLow && surfHigh) {
          // A point can be surface for both adjacent sections on a shared
          // plane. Containment keeps the cheap "inside the union" answer here;
          // entry/exit hand-off is refined later by the distance kernels using
          // the two section end annuli.
          completelyInside = true;
          return;
        } else {
          // else if point is on surface of only one of the two sections then point is actually on surface , the default
          // case,
          // so no need to check

          // What needs to check is if it is outside both ie. Outside indexLow section and Outside indexHigh section
          // then it is certainly outside
          if (secOutLow && secOutHigh) {
            completelyOutside = true;
            return;
          }
        }
      }
    }
  }

  /**
   * @brief Compute the distance from an outside point to the next polycone
   * entry.
   *
   * The ray walks section-by-section in the `z` direction. Finite section hits
   * landing on a section end plane are reclassified at the polycone level so
   * shared internal annuli can be rejected while exposed boundary hits are
   * preserved.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &polycone,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    // using namespace PolyconeTypes;
    Vector3D<Real_v> p = point;
    Vector3D<Real_v> v = direction;

    // TODO: add bounding box check maybe??

    distance      = kInfLength;
    int increment = (v.z() > 0) ? 1 : -1;
    if (std::fabs(v.z()) < kTolerance) increment = 0;
    int index = polycone.GetSectionIndex(p.z());
    if (index == -1) index = 0;
    if (index == -2) index = polycone.GetNSections() - 1;

    do {
      // now we have to find a section
      PolyconeSection const &sec = polycone.GetSection(index);

      ConeImplementation<polyconeTypeT>::template DistanceToIn<Real_v>(
          sec.fSolid, p - Vector3D<Precision>(0, 0, sec.fShift), v, stepMax, distance);

      if (distance < kInfLength && increment != 0) {
        const int nextIndex = index + increment;
        if (nextIndex >= 0 && nextIndex < polycone.GetNSections()) {
          const Real_v localHitZ = p.z() + distance * v.z() - Real_v(sec.fShift);
          const Real_v touchedEnd = increment > 0 ? Real_v(sec.fSolid.fDz) : Real_v(-sec.fSolid.fDz);
          if (Abs(localHitZ - touchedEnd) <= Real_v(kTolerance)) {
            // Reclassify end-plane hits at the polycone level. A section cone
            // cannot know whether the touched end annulus is exposed or only a
            // shared internal hand-off to the adjacent section.
            const Vector3D<Real_v> hit = p + distance * v;
            const Real_v rho2          = hit.x() * hit.x() + hit.y() * hit.y();

            const Precision curRMin = increment > 0 ? sec.fSolid._frmin2 : sec.fSolid._frmin1;
            const Precision curRMax = increment > 0 ? sec.fSolid._frmax2 : sec.fSolid._frmax1;

            const PolyconeSection &nextSec = polycone.GetSection(nextIndex);
            const Precision nextRMin       = increment > 0 ? nextSec.fSolid._frmin1 : nextSec.fSolid._frmin2;
            const Precision nextRMax       = increment > 0 ? nextSec.fSolid._frmax1 : nextSec.fSolid._frmax2;

            const Real_v curMinSq = Real_v(curRMin) * Real_v(curRMin);
            const Real_v curMaxSq = Real_v(curRMax) * Real_v(curRMax);

            const bool onCurrentEnd =
                (rho2 >= curMinSq - Real_v(kTolerance)) && (rho2 <= curMaxSq + Real_v(kTolerance));

            if (!onCurrentEnd) {
              distance = kInfLength;
            } else {
              const Precision overlapRMin = Max(curRMin, nextRMin);
              const Precision overlapRMax = Min(curRMax, nextRMax);
              if (overlapRMin <= overlapRMax) {
                const Real_v overlapMinSq = Real_v(overlapRMin) * Real_v(overlapRMin);
                const Real_v overlapMaxSq = Real_v(overlapRMax) * Real_v(overlapRMax);
                const bool onSharedInternal =
                    (rho2 >= overlapMinSq - Real_v(kTolerance)) && (rho2 <= overlapMaxSq + Real_v(kTolerance));
                if (onSharedInternal) distance = kInfLength;
              }
            }
          }
        }
      }

      if (distance < kInfLength || !increment) break;
      index += increment;
    } while (index >= 0 && index < polycone.GetNSections());
    return;
  }

  /**
   * @brief Compute the distance from an inside point to the next polycone exit.
   *
   * The starting section is selected from the local point and direction. The
   * query is then delegated to section cones while the polycone layer manages
   * shared-plane transitions between neighboring sections.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &polycone,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    // using namespace PolyconeTypes;
    distance            = kInfLength;
    Vector3D<Real_v> pn = point;

    // specialization for N==1??? It should be a cone in the first place
    if (polycone.GetNSections() == 1) {
      const PolyconeSection &section = polycone.GetSection(0);

      ConeImplementation<polyconeTypeT>::template DistanceToOut<Real_v>(
          section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), dir, stepMax, distance);

      return;
    }

    int indexLow  = polycone.GetSectionIndex(point.z() - kTolerance);
    int indexHigh = polycone.GetSectionIndex(point.z() + kTolerance);
    int index     = 0;

    // section index is -1 when out of left-end
    // section index is -2 when beyond right-end

    if (indexLow < 0 && indexHigh < 0) {
      distance = -1;
      return;
    } else if (indexLow < 0 && indexHigh >= 0) {
      index                          = indexHigh;
      const PolyconeSection &section = polycone.GetSection(index);

      Inside_t inside;
      //      ConeImplementation<ConeTypes::UniversalCone>::Inside<Real_v>(
      //          section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);
      ConeImplementation<polyconeTypeT>::template Inside<Real_v>(
          section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);
      if (inside == EInside::kOutside) {
        distance = -1;
        return;
      }
    } else if (indexLow != indexHigh && (indexLow >= 0)) {
      // we are close to an intermediate Surface, section has to be identified
      if (indexHigh < 0) {
        if (dir.z() > kTolerance) {
          index = indexHigh;
        } else {
          index = indexLow;
        }
      } else {
        const PolyconeSection &sectionLow  = polycone.GetSection(indexLow);
        const PolyconeSection &sectionHigh = polycone.GetSection(indexHigh);

        Inside_t insideLow;
        ConeImplementation<polyconeTypeT>::template Inside<Real_v>(
            sectionLow.fSolid, point - Vector3D<Precision>(0, 0, sectionLow.fShift), insideLow);
        const bool lowOwns = (insideLow != EInside::kOutside);

        Inside_t insideHigh;
        ConeImplementation<polyconeTypeT>::template Inside<Real_v>(
            sectionHigh.fSolid, point - Vector3D<Precision>(0, 0, sectionHigh.fShift), insideHigh);
        const bool highOwns = (insideHigh != EInside::kOutside);

        // On repeated-z transitions an exposed end annulus can belong to only
        // one neighboring section. Resolve unique ownership first and only use
        // the ray direction when both sections legitimately own the boundary.
        if (lowOwns && !highOwns) {
          index = indexLow;
        } else if (!lowOwns && highOwns) {
          index = indexHigh;
        } else if (dir.z() < -kTolerance) {
          index = indexLow;
        } else if (dir.z() > kTolerance) {
          index = indexHigh;
        } else {
          index = indexLow;
        }
      }
    } else {
      index = indexLow;
      if (index < 0) index = polycone.GetSectionIndex(point.z());
    }
    if (index < 0) {
      distance = 0.;
      return;
    }
    // Added
    else {
      const PolyconeSection &section = polycone.GetSection(index);

      Inside_t inside;
      //      ConeImplementation<ConeTypes::UniversalCone>::Inside<Real_v>(
      //          section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);
      ConeImplementation<polyconeTypeT>::template Inside<Real_v>(
          section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);
      if (inside == EInside::kOutside) {
        distance = -1;
        return;
      }
    }

    Precision totalDistance = 0.;
    Precision dist;
    int increment = (dir.z() > 0) ? 1 : -1;
    if (std::fabs(dir.z()) < kTolerance) increment = 0;

    // What is the relevance of istep?
    int istep = 0;
    do {
      const PolyconeSection &section = polycone.GetSection(index);

      if ((totalDistance != 0) || (istep < 2)) {
        pn = point + totalDistance * dir; // point must be shifted, so it could eventually get into another solid
        pn.z() -= section.fShift;
        Inside_t inside;
        //        ConeImplementation<ConeTypes::UniversalCone>::Inside<Real_v>(section.fSolid, pn, inside);
        ConeImplementation<polyconeTypeT>::template Inside<Real_v>(section.fSolid, pn, inside);

        if (inside == EInside::kOutside) {
          break;
        }
      } else
        pn.z() -= section.fShift;

      istep++;

      // ConeImplementation<ConeTypes::UniversalCone>::DistanceToOut<Real_v>(section.fSolid, pn, dir, stepMax, dist);
      ConeImplementation<polyconeTypeT>::template DistanceToOut<Real_v>(section.fSolid, pn, dir, stepMax, dist);
      if (dist == -1) return;

      // Section Surface case
      if (std::fabs(dist) < 0.5 * kTolerance) {
        int index1 = index;
        if ((index > 0) && (index < polycone.GetNSections() - 1)) {
          index1 += increment;
        } else {
          if ((index == 0) && (increment > 0)) index1 += increment;
          if ((index == polycone.GetNSections() - 1) && (increment < 0)) index1 += increment;
        }

        Vector3D<Precision> pte         = point + (totalDistance + dist) * dir;
        const PolyconeSection &section1 = polycone.GetSection(index1);
        pte.z() -= section1.fShift;
        Vector3D<Precision> localp;
        Inside_t inside22;
        // ConeImplementation<ConeTypes::UniversalCone>::Inside<Real_v>(section1.fSolid, pte, inside22);
        ConeImplementation<polyconeTypeT>::template Inside<Real_v>(section1.fSolid, pte, inside22);
        if (inside22 == 3 || (increment == 0)) {
          break;
        }
      } // end if surface case

      totalDistance += dist;
      index += increment;
    } while (increment != 0 && index >= 0 && index < polycone.GetNSections());

    distance = totalDistance;

    return;
  }

  /**
   * @brief Compute the outside safety for the polycone.
   *
   * The method starts from the seeded section-cone safety and refines it with a
   * local search on the relevant `(r,z)` contour. This is required for sharp
   * jumps and repeated-`z` connectors where the nearest polycone boundary is not
   * owned by the seeded section cone.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &polycone,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {

    using namespace ConeUtilities;

    Vector3D<Real_v> p = point;
    int index          = polycone.GetSectionIndex(p.z());

    bool needZ = false;
    if (index < 0) {
      needZ = true;
      if (index == -1) index = 0;
      if (index == -2) index = polycone.GetNSections() - 1;
    }

    PolyconeSection const &sec = polycone.GetSection(index);
    Vector3D<Real_v> localp    = p - Vector3D<Precision>(0, 0, sec.fShift);
    ConeImplementation<polyconeTypeT>::template SafetyToIn<Real_v>(sec.fSolid, localp, safety);

    if (safety < kTolerance) return;

    Real_v best = safety;

    const Real_v rho2 = p.x() * p.x() + p.y() * p.y();

    Real_v clampedLocalZ = localp.z();
    clampedLocalZ        = Max(Real_v(-sec.fSolid.fDz), Min(Real_v(sec.fSolid.fDz), clampedLocalZ));
    const Real_v outerRadius = GetRadiusOfConeAtPoint<Real_v, false>(sec.fSolid, clampedLocalZ);
    Real_v innerRadius(0.);
    if (sec.fSolid.fRmin1 > 0. || sec.fSolid.fRmin2 > 0.) {
      innerRadius = GetRadiusOfConeAtPoint<Real_v, true>(sec.fSolid, clampedLocalZ);
    }

    const Vector<Vector3D<Precision>> *contour = nullptr;
    const Real_v outerLimit = outerRadius + Real_v(kTolerance);
    if (rho2 > outerLimit * outerLimit) {
      contour = &polycone.fRMaxTwoDVec;
    } else if (sec.fSolid.fRmin1 > 0. || sec.fSolid.fRmin2 > 0.) {
      const Real_v innerLimit = innerRadius - Real_v(kTolerance);
      if (innerLimit > Real_v(0.) && rho2 < innerLimit * innerLimit) {
        contour = &polycone.fRMinTwoDVec;
      }
    } else {
      return;
    }
    if (!contour) return;

    const int npts = static_cast<int>(contour->size());
    if (npts < 2) return;

    // For outside safety the nearest polycone boundary can be a profile jump or
    // repeated-z connector that the seeded cone section does not own. Search
    // the relevant rz contour locally, pruning by the monotone dz^2 bound.
    Real_v bestSq = (best < Real_v(kInfLength)) ? best * best : Real_v(kInfLength);
    const Real_v rho = vecCore::math::Sqrt(rho2);

    int seedSegment = 2 * index;
    if (needZ && p.z() < polycone.fZs[0]) seedSegment = 0;
    if (needZ && p.z() > polycone.fZs[polycone.fZs.size() - 1]) seedSegment = npts - 2;
    seedSegment = Max(0, Min(npts - 2, seedSegment));

    for (int i = seedSegment; i >= 0; --i) {
      const auto &a = (*contour)[i];
      const auto &b = (*contour)[i + 1];
      const Real_v minDzSq = SegmentMinDeltaZSquared(p.z(), a, b);
      if (bestSq < Real_v(kInfLength) && minDzSq >= bestSq) break;

      const Real_v distSq = DistanceSquaredToRZSegment(rho, p.z(), a, b);
      if (distSq < bestSq) bestSq = distSq;
    }

    for (int i = seedSegment + 1; i < (npts - 1); ++i) {
      const auto &a = (*contour)[i];
      const auto &b = (*contour)[i + 1];
      const Real_v minDzSq = SegmentMinDeltaZSquared(p.z(), a, b);
      if (bestSq < Real_v(kInfLength) && minDzSq >= bestSq) break;

      const Real_v distSq = DistanceSquaredToRZSegment(rho, p.z(), a, b);
      if (distSq < bestSq) bestSq = distSq;
    }

    const Real_v contourSafety = vecCore::math::Sqrt(bestSq);
    if (contourSafety < safety) safety = contourSafety;
    return;
  }

  /**
   * @brief Compute the inside safety for the polycone.
   *
   * The nearest exit is found from the closed `(r,z)` contour and then
   * minimized with the phi-wedge safety when the solid is phi-limited.
   */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &polycone,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    safety = Real_v(kInfLength);
    bool compIn = false, compOut = false;
    GenericKernelForContainsAndInside<Real_v, true>(polycone, point, compIn, compOut);
    if (compOut) {
      safety = Real_v(-1.);
      return;
    }

    const Real_v rho = point.Perp();
    const Real_v z   = point.z();
    Real_v bestSq    = Real_v(kInfLength);
    for (unsigned int currSegIndex = 0; currSegIndex < polycone.fTwoDVec.size(); currSegIndex++) {
      unsigned int nextSegIndex = currSegIndex + 1;
      if (currSegIndex == polycone.fTwoDVec.size() - 1) {
        nextSegIndex = 0;
      }

      if ((polycone.fTwoDVec[currSegIndex].x() == 0 && polycone.fTwoDVec[nextSegIndex].x() == 0) ||
          (polycone.fTwoDVec[currSegIndex].x() == polycone.fTwoDVec[nextSegIndex].x() &&
           polycone.fTwoDVec[currSegIndex].y() == polycone.fTwoDVec[nextSegIndex].y()))
        continue;

      const Real_v distanceSq =
          DistanceSquaredToRZSegment(rho, z, polycone.fTwoDVec[currSegIndex], polycone.fTwoDVec[nextSegIndex]);
      if (distanceSq < bestSq) bestSq = distanceSq;
    }

    safety = vecCore::math::Sqrt(bestSq);

    if (polycone.fDeltaPhi < 2 * kPi) {
      Real_v safetyPhi = polycone.fPhiWedge.SafetyToOut(point);
      if (safetyPhi < safety) safety = safetyPhi;
    }
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_polyconeIMPLEMENTATION_H_
