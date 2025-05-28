/*
 * SimpleABBoxNavigator.h
 *
 *  Created on: Nov 23, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLEABBOXNAVIGATOR_H_
#define NAVIGATION_SIMPLEABBOXNAVIGATOR_H_

#include "VNavigator.h"
#include "SimpleABBoxSafetyEstimator.h"
#include "VecGeom/management/ABBoxManager.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// A basic implementation of a navigator which is SIMD accelerated by using a flat aligned bounding box list
template <bool MotherIsConvex = false>
class SimpleABBoxNavigator : public VNavigatorHelper<SimpleABBoxNavigator<MotherIsConvex>, MotherIsConvex> {

private:
  ABBoxManager<Precision> &fABBoxManager;
  SimpleABBoxNavigator()
      : VNavigatorHelper<SimpleABBoxNavigator<MotherIsConvex>, MotherIsConvex>(SimpleABBoxSafetyEstimator::Instance()),
        fABBoxManager(ABBoxManager<Precision>::Instance())
  {
  }

  // convert index to physical daugher
  VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int id) const
  {
    VECGEOM_VALIDATE(id >= 0, << "access with negative index");
    VECGEOM_VALIDATE(size_t(id) < lvol->GetDaughtersp()->size(), << "access beyond size of daughterlist ");
    return lvol->GetDaughtersp()->operator[](id);
  }

  // a simple sort class (based on insertionsort)
  // template <typename T, typename Cmp>
  static void insertionsort(ABBoxManager<Precision>::BoxIdDistancePair_t *arr, unsigned int N)
  {
    for (unsigned short i = 1; i < N; ++i) {
      ABBoxManager<Precision>::BoxIdDistancePair_t value = arr[i];
      short hole                                         = i;

      for (; hole > 0 && value.second < arr[hole - 1].second; --hole)
        arr[hole] = arr[hole - 1];

      arr[hole] = value;
    }
  }

  // vector version
  size_t GetHitCandidates_v(LogicalVolume const * /*lvol*/, Vector3D<Precision> const &point,
                            Vector3D<Precision> const &dir, ABBoxManager<Precision>::ABBoxContainer_v const &corners,
                            size_t size, ABBoxManager<Precision>::BoxIdDistancePair_t *hitlist) const
  {
    size_t vecsize  = size;
    size_t hitcount = 0;
    Vector3D<float> invdirfloat(1.f / (float)dir.x(), 1.f / (float)dir.y(), 1.f / (float)dir.z());
    Vector3D<float> pfloat((float)point.x(), (float)point.y(), (float)point.z());
    int sign[3];
    sign[0]       = invdirfloat.x() < 0;
    sign[1]       = invdirfloat.y() < 0;
    sign[2]       = invdirfloat.z() < 0;
    using Float_v = ABBoxManager<Precision>::Float_v;
    using Bool_v  = vecCore::Mask_v<Float_v>;
    for (size_t box = 0; box < vecsize; ++box) {
      Float_v distance = BoxImplementation::IntersectCachedKernel2<Float_v, float>(
          &corners[2 * box], pfloat, invdirfloat, sign[0], sign[1], sign[2], 0, InfinityLength<float>());
      Bool_v hit = distance < InfinityLength<float>();
      if (!vecCore::MaskEmpty(hit)) {
        constexpr auto kVS = vecCore::VectorSize<Float_v>();

        // VecCore does not have firstOne() function; iterating from zero
        // consider putting a firstOne into vecCore or in VecGeom
        for (size_t i = 0; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(hit, i)) {
            VECGEOM_ASSERT(hitcount < VECGEOM_MAXDAUGHTERS);
            hitlist[hitcount] =
                (ABBoxManager<Precision>::BoxIdDistancePair_t(box * kVS + i, vecCore::LaneAt(distance, i)));
            hitcount++;
          }
        }
      }
    }
    return hitcount;
  }

public:
  // we provide hit detection on the local level and reuse the generic implementations from
  // VNavigatorHelper<SimpleABBoxNavigator>

  VECGEOM_FORCE_INLINE
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, NavigationState const *in_state,
                                          NavigationState * /*out_state*/, Precision &step,
                                          VPlacedVolume const *&hitcandidate) const override
  {
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = ABBoxManager<Precision>::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
    IdDistPair_t *hitlist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    if (lvol->GetDaughtersp()->size() == 0) return false;

    int size;
    ABBoxManager<Precision>::ABBoxContainer_v bboxes = fABBoxManager.GetABBoxes_v(lvol, size);
    auto ncandidates = GetHitCandidates_v(lvol, localpoint, localdir, bboxes, size, hitlist);

    // sort candidates according to their bounding volume hit distance
    insertionsort(hitlist, ncandidates);

    for (size_t index = 0; index < ncandidates; ++index) {
      auto &hitbox                   = hitlist[index];
      VPlacedVolume const *candidate = LookupDaughter(lvol, hitbox.first);
      if (in_state && in_state->GetLastExited() == candidate) continue;

      // only consider those hitboxes which are within potential reach of this step
      if (!(step < hitbox.second)) {
        //      std::cerr << "checking id " << hitbox.first << " at box distance " << hitbox.second << "\n";
        //           if( hitbox.second < 0 ){
        //            //   std::cerr << "funny2\n";
        //              bool checkindaughter = candidate->Contains( localpoint );
        //              if( checkindaughter == true ){
        //                  std::cerr << "funny\n";
        //
        //                  // need to relocate
        //                  step = 0;
        //                  hitcandidate = candidate;
        //                  // THE ALTERNATIVE WOULD BE TO PUSH THE CURRENT STATE AND RETURN DIRECTLY
        //                  break;
        //              }
        //          }
        Precision ddistance = candidate->DistanceToIn(localpoint, localdir, step);
#ifdef VERBOSE
        std::cerr << "distance to " << candidate->GetLabel() << " is " << ddistance << "\n";
#endif
        const auto valid = !IsInf(ddistance) && ddistance < step && ddistance > -kTolerance;
        hitcandidate     = valid ? candidate : hitcandidate;
        step             = valid ? ddistance : step;
      } else {
        break;
      }
    }
    return false;
  }

  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, VPlacedVolume const *blocked,
                                          Precision &step, VPlacedVolume const *&hitcandidate) const override
  {
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = ABBoxManager<Precision>::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
    IdDistPair_t *hitlist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    if (lvol->GetDaughtersp()->size() == 0) return false;

    int size;
    ABBoxManager<Precision>::ABBoxContainer_v bboxes = fABBoxManager.GetABBoxes_v(lvol, size);
    auto ncandidates = GetHitCandidates_v(lvol, localpoint, localdir, bboxes, size, hitlist);

    // sort candidates according to their bounding volume hit distance
    insertionsort(hitlist, ncandidates);

    for (size_t index = 0; index < ncandidates; ++index) {
      auto &hitbox                   = hitlist[index];
      VPlacedVolume const *candidate = LookupDaughter(lvol, hitbox.first);

      // only consider those hitboxes which are within potential reach of this step
      if (hitbox.second <= step) { // !(step < hitbox.second)) {
#ifdef VERBOSE
        std::cerr << "SimpleABBoxNavigator: checking id " << hitbox.first << " at box distance " << hitbox.second
                  << "\n";
#endif
        //           if( hitbox.second < 0 ){
        //            //   std::cerr << "hit dist < 0. \n";
        //              bool checkindaughter = candidate->Contains( localpoint );
        //              if( checkindaughter == true ){
        //                  std::cerr << " -- unexpected : in daughter vol \n";
        //
        //                  // need to relocate
        //                  step = 0;
        //                  hitcandidate = candidate;
        //                  // THE ALTERNATIVE WOULD BE TO PUSH THE CURRENT STATE AND RETURN DIRECTLY
        //                  break;
        //              }
        //          }
        Precision ddistance = candidate->DistanceToIn(localpoint, localdir, step);
        Vector3D<Precision> normal; // To reuse in printing below - else move it into 'if'
#ifdef VERBOSE
        std::cerr << " . Distance to " << candidate->GetLabel() << " is " << ddistance;
#endif
        if (ddistance <= 0. /* && candidate == blocked */) {
          candidate->Normal(localpoint, normal);
#ifdef VERBOSE
          std::cerr << "  SimpleABBoxNavigator:  Negative daughter dist = " << ddistance
                    << (blocked == candidate ? " Is " : " Not ") << "Blocked. "
                    << " normal= " << normal << "  normal.dir= " << normal.Dot(localdir) << "\n";
#endif
        }
        const auto valid = !IsInf(ddistance) && ddistance < step &&
                           !((ddistance <= 0.) && (blocked == candidate || normal.Dot(localdir) > 0.0));

        hitcandidate = valid ? candidate : hitcandidate;
        step         = valid ? ddistance : step;
      } else {
        break;
      }
    }
    return false;
  }

  static VNavigator *Instance()
  {
    static SimpleABBoxNavigator instance;
    return &instance;
  }

  static constexpr const char *gClassNameString = "SimpleABBoxNavigator";
  typedef SimpleABBoxSafetyEstimator SafetyEstimator_t;
}; // end of class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_SIMPLEABBOXNAVIGATOR_H_ */
