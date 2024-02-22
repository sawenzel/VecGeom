/*
 * PolyconeImplementation.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

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
struct PolyconeImplementation {

  using UnplacedStruct_t = PolyconeStruct<Precision>;
  using UnplacedVolume_t = SUnplacedPolycone<polyconeTypeT>;
  using PlacedShape_t    = SPlacedPolycone<UnplacedVolume_t>;

  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForASection(
      UnplacedStruct_t const &unplaced, int isect, Vector3D<Real_v> const &polyconePoint,
      typename vecCore::Mask_v<Real_v> &secFullyInside, typename vecCore::Mask_v<Real_v> &secFullyOutside)
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
#ifdef POLYCONEDEBUG
    std::cerr << " isect=" << isect << "/" << unplaced.GetNSections() << " secLocalP=" << secLocalp
              << ", secShift=" << sec.fShift << " sec.fSolid=" << sec.fSolid << std::endl;
    if (sec.fSolid) sec.fSolid->Print();
#endif

    ConeHelpers<Real_v, polyconeTypeT>::template GenericKernelForContainsAndInside<ForInside>(
        *sec.fSolid, secLocalp, secFullyInside, secFullyOutside);
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &polycone,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {

    Bool_v unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(polycone, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &polycone,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(polycone, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &localPoint, Bool_v &completelyInside,
      Bool_v &completelyOutside)
  {

    typedef Bool_v Bool_t;

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

        Bool_t secInLow = false, secOutLow = false;
        Bool_t secInHigh = false, secOutHigh = false;
        GenericKernelForASection<Real_v, ForInside>(unplaced, indexLow, localPoint, secInLow, secOutLow);
        GenericKernelForASection<Real_v, ForInside>(unplaced, indexHigh, localPoint, secInHigh, secOutHigh);
        Bool_t surfLow  = !secInLow && !secOutLow;
        Bool_t surfHigh = !secInHigh && !secOutHigh;

        if (surfLow && surfHigh) {
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

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &polycone,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    // using namespace PolyconeTypes;
    Vector3D<Real_v> p = point;
    Vector3D<Real_v> v = direction;

#ifdef POLYCONEDEBUG
    std::cerr << "Polycone::DistToIn() (spot 1): point=" << point << ", dir=" << direction << ", localPoint=" << p
              << ", localDir=" << v << "\n";
#endif

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

#ifdef POLYCONEDEBUG
      std::cerr << "Polycone::DistToIn() (spot 2):"
                << " index=" << index << " NSec=" << polycone.GetNSections() << " &sec=" << &sec << " - secPars:"
                << " secOffset=" << sec.fShift << " Dz=" << sec.fSolid->GetDz() << " Rmin1=" << sec.fSolid->GetRmin1()
                << " Rmin2=" << sec.fSolid->GetRmin2() << " Rmax1=" << sec.fSolid->GetRmax1()
                << " Rmax2=" << sec.fSolid->GetRmax2() << " -- calling Cone::DistToIn()...\n";
#endif

      ConeImplementation<polyconeTypeT>::template DistanceToIn<Real_v>(
          *sec.fSolid, p - Vector3D<Precision>(0, 0, sec.fShift), v, stepMax, distance);

#ifdef POLYCONEDEBUG
      std::cerr << "Polycone::DistToIn() (spot 3):"
                << " distToIn() = " << distance << "\n";
#endif

      if (distance < kInfLength || !increment) break;
      index += increment;
    } while (index >= 0 && index < polycone.GetNSections());
    return;
  }

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
          *section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), dir, stepMax, distance);

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
      //          *section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);
      ConeImplementation<polyconeTypeT>::template Inside<Real_v>(
          *section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);
      if (inside == EInside::kOutside) {
        distance = -1;
        return;
      }
    } else if (indexLow != indexHigh && (indexLow >= 0)) {
      // we are close to an intermediate Surface, section has to be identified
      const PolyconeSection &section = polycone.GetSection(indexLow);

      Inside_t inside;
      //      ConeImplementation<ConeTypes::UniversalCone>::Inside<Real_v>(
      //        *section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);

      ConeImplementation<polyconeTypeT>::template Inside<Real_v>(
          *section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);

      if (inside == EInside::kOutside) {
        index = indexHigh;
      } else {
        index = indexLow;
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
      //          *section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);
      ConeImplementation<polyconeTypeT>::template Inside<Real_v>(
          *section.fSolid, point - Vector3D<Precision>(0, 0, section.fShift), inside);
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
        //        ConeImplementation<ConeTypes::UniversalCone>::Inside<Real_v>(*section.fSolid, pn, inside);
        ConeImplementation<polyconeTypeT>::template Inside<Real_v>(*section.fSolid, pn, inside);

        if (inside == EInside::kOutside) {
          break;
        }
      } else
        pn.z() -= section.fShift;

      istep++;

      // ConeImplementation<ConeTypes::UniversalCone>::DistanceToOut<Real_v>(*section.fSolid, pn, dir, stepMax, dist);
      ConeImplementation<polyconeTypeT>::template DistanceToOut<Real_v>(*section.fSolid, pn, dir, stepMax, dist);
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
        // ConeImplementation<ConeTypes::UniversalCone>::Inside<Real_v>(*section1.fSolid, pte, inside22);
        ConeImplementation<polyconeTypeT>::template Inside<Real_v>(*section1.fSolid, pte, inside22);
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

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &polycone,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {

    Vector3D<Real_v> p = point;
    int index          = polycone.GetSectionIndex(p.z());

    bool needZ = false;
    if (index < 0) {
      needZ = true;
      if (index == -1) index = 0;
      if (index == -2) index = polycone.GetNSections() - 1;
    }
    Precision minSafety        = 0; //= SafetyFromOutsideSection(index, p);
    PolyconeSection const &sec = polycone.GetSection(index);
    // safety to current segment
    if (needZ) {
      //      ConeImplementation<ConeTypes::UniversalCone>::SafetyToIn<Real_v>(
      //          *sec.fSolid, p - Vector3D<Precision>(0, 0, sec.fShift), safety);
      ConeImplementation<polyconeTypeT>::template SafetyToIn<Real_v>(*sec.fSolid,
                                                                     p - Vector3D<Precision>(0, 0, sec.fShift), safety);
    } else {

      //      ConeImplementation<ConeTypes::UniversalCone>::SafetyToIn<Real_v>(
      //          *sec.fSolid, p - Vector3D<Precision>(0, 0, sec.fShift), safety);
      ConeImplementation<polyconeTypeT>::template SafetyToIn<Real_v>(*sec.fSolid,
                                                                     p - Vector3D<Precision>(0, 0, sec.fShift), safety);

      if (safety < kTolerance) return;
      minSafety       = safety;
      Precision zbase = polycone.fZs[index + 1];
      // going right
      for (int i = index + 1; i < polycone.GetNSections(); ++i) {
        Precision dz = polycone.fZs[i] - zbase;
        if (dz >= minSafety) break;

        PolyconeSection const &sect = polycone.GetSection(i);

        //      ConeImplementation<ConeTypes::UniversalCone>::SafetyToIn<Real_v>(
        //          *sect.fSolid, p - Vector3D<Precision>(0, 0, sect.fShift), safety);

        ConeImplementation<polyconeTypeT>::template SafetyToIn<Real_v>(
            *sect.fSolid, p - Vector3D<Precision>(0, 0, sect.fShift), safety);

        if (safety < minSafety) minSafety = safety;
      }

      // going left if this is possible
      if (index > 0) {
        zbase = polycone.fZs[index - 1];
        for (int i = index - 1; i >= 0; --i) {
          Precision dz = zbase - polycone.fZs[i];
          if (dz >= minSafety) break;
          PolyconeSection const &sect = polycone.GetSection(i);

          //        ConeImplementation<ConeTypes::UniversalCone>::SafetyToIn<Real_v>(
          //            *sect.fSolid, p - Vector3D<Precision>(0, 0, sect.fShift), safety);

          ConeImplementation<polyconeTypeT>::template SafetyToIn<Real_v>(
              *sect.fSolid, p - Vector3D<Precision>(0, 0, sect.fShift), safety);

          if (safety < minSafety) minSafety = safety;
        }
      }
      safety = minSafety;
    }
    return;
  }
  /* A function to calculation the shortest distance of a point from a segment */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Real_v DistanceToSeg(Vector3D<Real_v> const &point,
                                                                           Vector3D<Precision> segment_start,
                                                                           Vector3D<Precision> segment_end)
  {
    Vector3D<Real_v> segment_vector   = segment_end - segment_start;
    Vector3D<Real_v> point_vector     = point - segment_start;
    Real_v projection_scalar          = point_vector.Dot(segment_vector) / segment_vector.Mag2();
    Vector3D<Real_v> projection_point = segment_start + projection_scalar * segment_vector;
    vecCore__MaskedAssignFunc(projection_point, projection_scalar < Real_v(0.), Vector3D<Real_v>(segment_start));
    vecCore__MaskedAssignFunc(projection_point, projection_scalar > Real_v(1.), Vector3D<Real_v>(segment_end));
    Vector3D<Real_v> distVec = point - projection_point;
    return distVec.Mag();
  }

  /*
  ** New definition that uses Segments and Single Wedge, instead of SafetyToOut from ConeImplementation
  */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &polycone,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    typedef typename vecCore::Mask_v<Real_v> Bool_v;
    safety = Real_v(kInfLength);
    Bool_v compIn(false), compOut(false), done(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(polycone, point, compIn, compOut);
    vecCore__MaskedAssignFunc(safety, compOut && !done, Real_v(-1.));
    done |= compOut;
    if (vecCore::MaskFull(done)) return;

    Vector3D<Real_v> twoDPoint = Vector3D<Real_v>(point.Perp(), point.z(), 0.);
    for (unsigned int currSegIndex = 0; currSegIndex < polycone.fTwoDVec.size(); currSegIndex++) {
      unsigned int nextSegIndex = currSegIndex + 1;
      if (currSegIndex == polycone.fTwoDVec.size() - 1) {
        nextSegIndex = 0;
      }

      Real_v distance(kInfLength);
      if ((polycone.fTwoDVec[currSegIndex].x() == 0 && polycone.fTwoDVec[nextSegIndex].x() == 0) ||
          (polycone.fTwoDVec[currSegIndex].x() == polycone.fTwoDVec[nextSegIndex].x() &&
           polycone.fTwoDVec[currSegIndex].y() == polycone.fTwoDVec[nextSegIndex].y()))
        distance = Real_v(kInfLength);
      else
        distance = DistanceToSeg(twoDPoint, polycone.fTwoDVec[currSegIndex], polycone.fTwoDVec[nextSegIndex]);

      vecCore__MaskedAssignFunc(safety, (distance < safety) && !done, distance);
    }

    if (polycone.fDeltaPhi < 2 * kPi) {
      Real_v safetyPhi = polycone.fPhiWedge.SafetyToOut<Real_v>(point);
      vecCore__MaskedAssignFunc(safety, safetyPhi < safety, safetyPhi);
    }
  }

#if (0)
  /* Retaining the old definition for the time being */
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut_Old(UnplacedStruct_t const &polycone,
                                                                           Vector3D<Real_v> const &point,
                                                                           Real_v &safety)
  {
    typedef typename vecCore::Mask_v<Real_v> Bool_v;
    Bool_v compIn(false), compOut(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(polycone, point, compIn, compOut);
    if (compOut) {
      safety = -1;
      return;
    }

    int index = polycone.GetSectionIndex(point.z());
    if (index < 0) {
      safety = -1;
      return;
    }

    PolyconeSection const &sec = polycone.GetSection(index);

    Vector3D<Real_v> p = point - Vector3D<Precision>(0, 0, sec.fShift);
    // ConeImplementation<ConeTypes::UniversalCone>::SafetyToOut<Real_v>(*sec.fSolid, p, safety);
    ConeImplementation<polyconeTypeT>::template SafetyToOut<Real_v>(*sec.fSolid, p, safety);

    Precision minSafety = safety;
    if (minSafety == kInfLength) {
      safety = 0.;
      return;
    }
    if (minSafety < kTolerance) {
      safety = 0.;
      return;
    }

    Precision zbase = polycone.fZs[index + 1];
    for (int i = index + 1; i < polycone.GetNSections(); ++i) {
      Precision dz = polycone.fZs[i] - zbase;
      if (dz >= minSafety) break;
      PolyconeSection const &sect = polycone.GetSection(i);
      p                           = point - Vector3D<Precision>(0, 0, sect.fShift);

      // ConeImplementation<ConeTypes::UniversalCone>::SafetyToIn<Real_v>(*sect.fSolid, p, safety);
      ConeImplementation<polyconeTypeT>::template SafetyToIn<Real_v>(*sect.fSolid, p, safety);

      if (safety < minSafety) minSafety = safety;
    }

    if (index > 0) {
      zbase = polycone.fZs[index - 1];
      for (int i = index - 1; i >= 0; --i) {
        Precision dz = zbase - polycone.fZs[i];
        if (dz >= minSafety) break;
        PolyconeSection const &sect = polycone.GetSection(i);
        p                           = point - Vector3D<Precision>(0, 0, sect.fShift);

        // ConeImplementation<ConeTypes::UniversalCone>::SafetyToIn<Real_v>(*sect.fSolid, p, safety);
        ConeImplementation<polyconeTypeT>::template SafetyToIn<Real_v>(*sect.fSolid, p, safety);

        if (safety < minSafety) minSafety = safety;
      }
    }

    safety = minSafety;
    return;
  }
#endif
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_polyconeIMPLEMENTATION_H_
