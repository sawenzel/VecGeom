/*
 * HybridNavigator2.h
 *
 *  Created on: 27.08.2015
 *      Author: yang.zhang@cern.ch and sandro.wenzel@cern.ch
 */

#ifndef VECGEOM_HYBRIDNAVIGATOR
#define VECGEOM_HYBRIDNAVIGATOR

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include "VecGeom/management/HybridManager2.h"
#include "VecGeom/navigation/VNavigator.h"
#include "VecGeom/navigation/HybridSafetyEstimator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"

#include <vector>
#include <cmath>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// A navigator using a shallow tree of aligned bounding boxes (hybrid approach) to quickly exclude
// potential hit targets.
// This navigator goes into the direction of "voxel" navigators used in Geant4
// and ROOT. Checking single-rays against a set of aligned bounding boxes can be done
// in a vectorized fashion.
template <bool MotherIsConvex = false>
class HybridNavigator : public VNavigatorHelper<HybridNavigator<MotherIsConvex>, MotherIsConvex> {

private:
  HybridManager2 &fAccelerationManager;
  HybridNavigator()
      : VNavigatorHelper<HybridNavigator<MotherIsConvex>, MotherIsConvex>(SimpleABBoxSafetyEstimator::Instance()),
        fAccelerationManager(HybridManager2::Instance())
  {
  }

  static VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int const daughterIndex)
  {
    return lvol->GetDaughters()[daughterIndex];
  }

  // a simple sort class (based on insertionsort)
  template <typename T> //, typename Cmp>
  static void insertionsort(T *arr, unsigned int N)
  {
    for (unsigned short i = 1; i < N; ++i) {
      T value    = arr[i];
      short hole = i;

      for (; hole > 0 && value.second < arr[hole - 1].second; --hole)
        arr[hole] = arr[hole - 1];

      arr[hole] = value;
    }
  }

  /**
   * Returns list of daughter candidates containing the point.
   */
  size_t GetContainingCandidates_v(HybridManager2::HybridBoxAccelerationStructure const &accstructure,
                                   Vector3D<Precision> const &point, size_t *hitlist) const
  {
    using Float_v      = HybridManager2::Float_v;
    using Bool_v       = vecCore::Mask<Float_v>;
    constexpr auto kVS = vecCore::VectorSize<Float_v>();
    size_t count       = 0;
    int numberOfNodes, size;
    auto boxes_v                = fAccelerationManager.GetABBoxes_v(accstructure, size, numberOfNodes);
    auto const *nodeToDaughters = accstructure.fNodeToDaughters;
    for (size_t index = 0, nodeindex = 0; index < size_t(size) * 2; index += 2 * (kVS + 1), nodeindex += kVS) {
      Bool_v inside, inside_d;
      Vector3D<Float_v> *corners = &boxes_v[index];
      ABBoxImplementation::ABBoxContainsKernelGeneric<HybridManager2::Float_v, Precision, Bool_v>(
          corners[0], corners[1], point, inside);
      if (!vecCore::MaskEmpty(inside)) {
        // loop lanes
        for (size_t i = 0; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(inside, i)) {
            corners = &boxes_v[index + 2 * (i + 1)];
            ABBoxImplementation::ABBoxContainsKernelGeneric<HybridManager2::Float_v, Precision, Bool_v>(
                corners[0], corners[1], point, inside_d);
            if (!vecCore::MaskEmpty(inside_d)) {
              // loop lanes at second level
              for (size_t j = 0; j < kVS; ++j) {
                if (vecCore::MaskLaneAt(inside_d, j)) {
                  assert(count < VECGEOM_MAXFACETS);
                  hitlist[count++] = nodeToDaughters[nodeindex + i][j];
                }
              }
            }
          }
        }
      }
    }
    return count;
  }

  /**
   * Returns hitlist of daughter candidates (pairs of [daughter index, step to bounding box]) crossed by ray.
   */
  size_t GetHitCandidates_v(HybridManager2::HybridBoxAccelerationStructure const &accstructure,
                            Vector3D<Precision> const &point, Vector3D<Precision> const &dir, float maxstep,
                            HybridManager2::BoxIdDistancePair_t *hitlist) const
  {
    size_t count = 0;
    Vector3D<Precision> invdir(1. / NonZero(dir.x()), 1. / NonZero(dir.y()), 1. / NonZero(dir.z()));
    Vector3D<int> sign;
    sign[0] = invdir.x() < 0;
    sign[1] = invdir.y() < 0;
    sign[2] = invdir.z() < 0;
    int numberOfNodes, size;
    auto boxes_v                = fAccelerationManager.GetABBoxes_v(accstructure, size, numberOfNodes);
    constexpr auto kVS          = vecCore::VectorSize<HybridManager2::Float_v>();
    auto const *nodeToDaughters = accstructure.fNodeToDaughters;
    for (size_t index = 0, nodeindex = 0; index < size_t(size) * 2; index += 2 * (kVS + 1), nodeindex += kVS) {
      HybridManager2::Float_v distance = BoxImplementation::IntersectCachedKernel2<HybridManager2::Float_v, float>(
          &boxes_v[index], point, invdir, sign.x(), sign.y(), sign.z(), 0, maxstep);
      auto hit = distance < maxstep;
      if (!vecCore::MaskEmpty(hit)) {
        for (size_t i = 0 /*hit.firstOne()*/; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(hit, i)) {
            distance = BoxImplementation::IntersectCachedKernel2<HybridManager2::Float_v, float>(
                &boxes_v[index + 2 * (i + 1)], point, invdir, sign.x(), sign.y(), sign.z(), 0, maxstep);
            auto hit1 = distance < maxstep;
            if (!vecCore::MaskEmpty(hit1)) {
              for (size_t j = 0 /*hit1.firstOne()*/; j < kVS; ++j) { // leaf node
                if (vecCore::MaskLaneAt(hit1, j)) {
                  assert(count < VECGEOM_MAXFACETS);
                  hitlist[count] = HybridManager2::BoxIdDistancePair_t(nodeToDaughters[nodeindex + i][j],
                                                                       vecCore::LaneAt(distance, j));
                  count++;
                }
              }
            }
          }
        }
      }
    }
    return count;
  }

public:
  // we provide hit detection on the local level and reuse the generic implementations from
  // VNavigatorHelper<SimpleABBoxNavigator>

  // a generic looper function that
  // given an acceleration structure (an aligned bounding box hierarchy),
  // a hit-query will be performed, the intersected boxes sorted, looped over
  // and a user hook called for processing
  // the user hook needs to indicate with a boolean return value whether to continue looping (false)
  // or whether we are done (true) and can exit

  // FIXME: might be generic enough to work for all possible kinds of BVH structures
  // FIXME: offer various sorting directions, etc.
  template <typename AccStructure, typename Func>
  VECGEOM_FORCE_INLINE void BVHSortedIntersectionsLooper(AccStructure const &accstructure,
                                                         Vector3D<Precision> const &localpoint,
                                                         Vector3D<Precision> const &localdir, Precision maxstep,
                                                         Func &&userhook) const
  {
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = HybridManager2::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXFACETS * sizeof(IdDistPair_t)];
    IdDistPair_t *hitlist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    // it could be that someone passed InfinityLength<double> to this function
    // which we need to reduce to the float version since GetHitCandidates processes floats
    const float maxstep_float = std::min((float)maxstep, InfinityLength<float>());
    auto ncandidates          = GetHitCandidates_v(accstructure, localpoint, localdir, maxstep_float, hitlist);
    // sort candidates according to their bounding volume hit distance
    insertionsort(hitlist, ncandidates);

    for (size_t index = 0; index < ncandidates; ++index) {
      auto hitbox = hitlist[index];
      // here we got the hit candidates
      // now we execute user specific code to process this "hitbox"
      auto done = userhook(hitbox);
      if (done) break;
    }
  }

  template <typename AccStructure, typename Func>
  VECGEOM_FORCE_INLINE void BVHContainsLooper(AccStructure const &accstructure, Vector3D<Precision> const &localpoint,
                                              Func &&userhook) const
  {
    size_t hitlist[VECGEOM_MAXFACETS];
    auto ncandidates = GetContainingCandidates_v(accstructure, localpoint, hitlist);
    for (size_t index = 0; index < ncandidates; ++index) {
      auto hitbox = hitlist[index];
      auto done   = userhook(hitbox);
      if (done) break;
    }
  }

  VECGEOM_FORCE_INLINE
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, NavigationState const *in_state,
                                          NavigationState * /*out_state*/, Precision &step,
                                          VPlacedVolume const *&hitcandidate) const override
  {
    if (lvol->GetDaughtersp()->size() == 0) return false;
    auto &accstructure = *fAccelerationManager.GetAccStructure(lvol);

    float maxstep = static_cast<float>(step);
    BVHSortedIntersectionsLooper(
        accstructure, localpoint, localdir, maxstep, [&](HybridManager2::BoxIdDistancePair_t hitbox) {
          // only consider those hitboxes which are within potential reach of this step
          if (!(step < hitbox.second)) {
            VPlacedVolume const *candidate = LookupDaughter(lvol, hitbox.first);
            Precision ddistance            = candidate->DistanceToIn(localpoint, localdir, step);
            const auto valid               = !IsInf(ddistance) && ddistance < step &&
                               !((ddistance <= 0.) && in_state && in_state->GetLastExited() == candidate);
            hitcandidate = valid ? candidate : hitcandidate;
            step         = valid ? ddistance : step;
            return false; // not yet done; need to continue in looper
          }
          return true; // mark done in this case
        });
    return false;
  }

  VECGEOM_FORCE_INLINE
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, VPlacedVolume const *blocked,
                                          Precision &step, VPlacedVolume const *&hitcandidate) const override
  {
    if (lvol->GetDaughtersp()->size() == 0) return false;
    auto &accstructure = *fAccelerationManager.GetAccStructure(lvol);

    const float maxstep = static_cast<float>(step);
    BVHSortedIntersectionsLooper(
        accstructure, localpoint, localdir, maxstep, [&](HybridManager2::BoxIdDistancePair_t hitbox) {
          // only consider those hitboxes which are within potential reach of this step
          if (!(step < hitbox.second)) {
            // To reuse in printing below - else move it into 'if'
            Vector3D<Precision> normal;
            VPlacedVolume const *candidate = LookupDaughter(lvol, hitbox.first);
            if (candidate == blocked) {
              // return false; // return early and go on in the looper
              candidate->Normal(localpoint, normal);
              if (normal.Dot(localdir) >= 0.0) {
                std::cerr << "HybridNav2> blocked " << candidate << " has normal.dir = " << normal.Dot(localdir)
                          << " and distToIn = " << candidate->DistanceToIn(localpoint, localdir, step) << "\n";
              }
            }
            const Precision ddistance = candidate->DistanceToIn(localpoint, localdir, step);
            const auto valid          = !IsInf(ddistance) && ddistance < step &&
                               !((ddistance <= 0.) && blocked == candidate); // && normal.Dot(localdir) > 0.0);
            hitcandidate = valid ? candidate : hitcandidate;
            step         = valid ? ddistance : step;
#if 0 // enable for debugging
        if ( ddistance<=0 ) {
           std::cerr << "HybridNav2> negative distance found for " << candidate->GetName() << "\n"; 
           auto inside = candidate->Inside(localpoint);
           static std::string InsideCode[4] = { "N/A", "Inside", "Surface", "Outside" } ;
           std::cerr << InsideCode[inside];
           const auto transf = candidate->GetTransformation();
           const auto unpl = candidate->GetUnplacedVolume();
           Vector3D<Precision> normalDg;
           const auto testdaughterlocal = transf->Transform(localpoint);      
           auto valid = unpl->Normal(testdaughterlocal, normalDg);
           
           const auto directiondaughterlocal = transf->TransformDirection(localdir);
           const auto dot = normalDg.Dot(directiondaughterlocal);
           std::cerr << " normal.dir = " << dot;
           if (dot >= 0) {
             std::cerr << " exiting " << valid << "\n";
           }
           else {
             std::cerr << " entering " << valid << "\n";
           }
        }
#endif
            return false; // not yet done; need to continue in looper
          }
          return true; // mark done in this case
        });
    return false;
  }

  static VNavigator *Instance()
  {
    static HybridNavigator instance;
    return &instance;
  }

  static constexpr const char *gClassNameString = "HybridNavigator";
  typedef SimpleABBoxSafetyEstimator SafetyEstimator_t;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
