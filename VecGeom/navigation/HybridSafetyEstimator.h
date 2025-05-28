/*
 *  HybridSafetyEstimator.h
 *
 *  Created on: 22.11.2015
 *      Author: sandro.wenzel@cern.ch
 *
 *  (based on prototype implementation by Yang Zhang (Sep 2015)
 */

#ifndef NAVIGATION_HYBRIDSAFETYESTIMATOR_H_
#define NAVIGATION_HYBRIDSAFETYESTIMATOR_H_

#include "VecGeom/navigation/VSafetyEstimator.h"
#include "VecGeom/management/HybridManager2.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//! a safety estimator using a (vectorized) search through bounding boxes to exclude certain daughter volumes
//! to talk to
class HybridSafetyEstimator : public VSafetyEstimatorHelper<HybridSafetyEstimator> {

private:
  // we keep a reference to the ABBoxManager ( avoids calling Instance() on this guy all the time )
  HybridManager2 &fAccelerationStructureManager;

  HybridSafetyEstimator()
      : VSafetyEstimatorHelper<HybridSafetyEstimator>(), fAccelerationStructureManager(HybridManager2::Instance())
  {
  }

  // convert index to physical daugher
  VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int id) const
  {
    VECGEOM_VALIDATE(id >= 0, << "access with negative index");
    VECGEOM_VALIDATE(size_t(id) < lvol->GetDaughtersp()->size(), << "access beyond size of daughterlist ");
    return lvol->GetDaughtersp()->operator[](id);
  }

public:
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

  // helper structure to find the candidate set for safety calculations
  size_t GetSafetyCandidates_v(HybridManager2::HybridBoxAccelerationStructure const &accstructure,
                               Vector3D<Precision> const &point, HybridManager2::BoxIdDistancePair_t *boxsafetypairs,
                               Precision upper_squared_limit) const
  {
    size_t count = 0;
    Vector3D<float> pointfloat((float)point.x(), (float)point.y(), (float)point.z());
    int halfvectorsize, numberOfNodes;
    auto boxes_v = fAccelerationStructureManager.GetABBoxes_v(accstructure, halfvectorsize, numberOfNodes);
    auto const *nodeToDaughters = accstructure.fNodeToDaughters;
    constexpr auto kVS          = vecCore::VectorSize<HybridManager2::Float_v>();

    for (int index = 0, nodeindex = 0; index < halfvectorsize * 2; index += 2 * (kVS + 1), nodeindex += kVS) {
      HybridManager2::Float_v safetytoboxsqr =
          ABBoxImplementation::ABBoxSafetySqr(boxes_v[index], boxes_v[index + 1], pointfloat);
      auto closer = safetytoboxsqr < HybridManager2::Float_v(upper_squared_limit);
      if (!vecCore::MaskEmpty(closer)) {
        for (size_t i = 0 /*closer.firstOne()*/; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(closer, i)) {
            safetytoboxsqr =
                ABBoxImplementation::ABBoxSafetySqr(boxes_v[index + 2 * i + 2], boxes_v[index + 2 * i + 3], pointfloat);
            auto closer1 = safetytoboxsqr < HybridManager2::Float_v(upper_squared_limit);
            if (!vecCore::MaskEmpty(closer1)) {
              for (size_t j = 0 /*closer.firstOne()*/; j < kVS; ++j) { // leaf node
                if (vecCore::MaskLaneAt(closer1, j)) {
                  VECGEOM_ASSERT(count < VECGEOM_MAXFACETS);
                  boxsafetypairs[count] = HybridManager2::BoxIdDistancePair_t(nodeToDaughters[nodeindex + i][j],
                                                                              vecCore::LaneAt(safetytoboxsqr, j));
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

  // Improved safety estimator that can dynamically reduce the upper search limit.
  // Internally calls ABBoxSafetyRangeSqr that besides validating a candidate
  // if closer than the limit, it also gives the upper limit for the range where
  // the candidate can be surely found. This allows to update the search limit after
  // each check, giving much less final candidates.
  //
  //         current checked
  //         box  _________
  // point       |         |          upper_limit
  //   x---------|candidate|----------|
  //             |_________|
  //   |-------------------|
  //                       updated upper_limit

  size_t GetSafetyCandidates2_v(HybridManager2::HybridBoxAccelerationStructure const &accstructure,
                                Vector3D<Precision> const &point, HybridManager2::BoxIdDistancePair_t *hitlist,
                                Precision upper_squared_limit) const
  {
    using Float_v = vecgeom::VectorBackend::Float_v;
    size_t count  = 0;
    int numberOfNodes, size;
    auto boxes_v                = fAccelerationStructureManager.GetABBoxes_v(accstructure, size, numberOfNodes);
    constexpr auto kVS          = vecCore::VectorSize<HybridManager2::Float_v>();
    auto const *nodeToDaughters = accstructure.fNodeToDaughters;

    Vector3D<float> pointfloat((float)point.x(), (float)point.y(), (float)point.z());

    // Loop internal nodes (internal node = kVS clusters)
    for (size_t index = 0, nodeindex = 0; index < size_t(size) * 2; index += 2 * (kVS + 1), nodeindex += kVS) {
      // index = start index for internal node
      // nodeindex = cluster index
      Float_v distmaxsqr; // Maximum distance that may still touch the box
      Float_v safetytonodesqr =
          ABBoxImplementation::ABBoxSafetyRangeSqr(boxes_v[index], boxes_v[index + 1], pointfloat, distmaxsqr);
      // Find minimum of distmaxsqr
      for (size_t i = 0; i < kVS; ++i) {
        Precision distmaxsqr_s = vecCore::LaneAt(distmaxsqr, i);
        if (distmaxsqr_s < upper_squared_limit) upper_squared_limit = distmaxsqr_s;
      }
      auto hit = safetytonodesqr < ABBoxManager<Precision>::Real_s(upper_squared_limit);
      if (!vecCore::MaskEmpty(hit)) {
        for (size_t i = 0; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(hit, i)) {
            Float_v safetytoboxsqr = ABBoxImplementation::ABBoxSafetyRangeSqr(
                boxes_v[index + 2 * (i + 1)], boxes_v[index + 2 * (i + 1) + 1], pointfloat, distmaxsqr);

            auto hit1 = safetytoboxsqr < ABBoxManager<Precision>::Real_s(upper_squared_limit);
            if (!vecCore::MaskEmpty(hit1)) {
              // loop bounding boxes in the cluster
              for (size_t j = 0; j < kVS; ++j) {
                if (vecCore::MaskLaneAt(hit1, j)) {
                  VECGEOM_ASSERT(count < VECGEOM_MAXFACETS);
                  hitlist[count]         = HybridManager2::BoxIdDistancePair_t(nodeToDaughters[nodeindex + i][j],
                                                                               vecCore::LaneAt(safetytoboxsqr, j));
                  Precision distmaxsqr_s = vecCore::LaneAt(distmaxsqr, j);
                  // Reduce the upper limit
                  if (distmaxsqr_s < upper_squared_limit) upper_squared_limit = distmaxsqr_s;
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

  template <typename AccStructure, typename Func>
  VECGEOM_FORCE_INLINE void BVHSortedSafetyLooper(AccStructure const &accstructure,
                                                  Vector3D<Precision> const &localpoint, Func &&userhook,
                                                  Precision upper_squared_limit) const
  {
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = HybridManager2::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXFACETS * sizeof(IdDistPair_t)];
    IdDistPair_t *hitlist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    // Get candidates using HybridSafetyEstimator
    //    HybridSafetyEstimator *hse = static_cast<HybridSafetyEstimator*>(HybridSafetyEstimator::Instance());
    //    auto ncandidates = hse->GetSafetyCandidates_v(accstructure, localpoint, hitlist, upper_squared_limit);

    // Get candidates in vectorized mode
    auto ncandidates = GetSafetyCandidates2_v(accstructure, localpoint, hitlist, upper_squared_limit);

    // sort candidates according to their bounding volume safety distance
    insertionsort(hitlist, ncandidates);

    for (size_t index = 0; index < ncandidates; ++index) {
      auto hitbox = hitlist[index];
      // here we got the hit candidates
      // now we execute user specific code to process this "hitbox"
      auto done = userhook(hitbox);
      if (done) break;
    }
  }

  static constexpr const char *gClassNameString = "HybridSafetyEstimator";

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Precision TreatSafetyToIn(Vector3D<Precision> const &localpoint, LogicalVolume const *lvol, Precision outsafety) const
  {
    // a stack based workspace array
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = HybridManager2::BoxIdDistancePair_t;
    char unitstackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
    IdDistPair_t *boxsafetylist = reinterpret_cast<IdDistPair_t *>(&unitstackspace);

    double safety    = outsafety; // we use the outsafety estimate as starting point
    double safetysqr = safety * safety;

    // safety to bounding boxes
    if (safety > 0. && lvol->GetDaughtersp()->size() > 0) {
      // calculate squared bounding box safeties in vectorized way
      auto ncandidates = GetSafetyCandidates_v(*fAccelerationStructureManager.GetAccStructure(lvol), localpoint,
                                               boxsafetylist, safetysqr);
      // not sorting the candidate list ( which one could do )
      for (unsigned int candidate = 0; candidate < ncandidates; ++candidate) {
        auto boxsafetypair = boxsafetylist[candidate];
        if (boxsafetypair.second < safetysqr) {
          VPlacedVolume const *cand = LookupDaughter(lvol, boxsafetypair.first);
          if (size_t(boxsafetypair.first) > lvol->GetDaughtersp()->size()) break;
          auto candidatesafety = cand->SafetyToIn(localpoint);
#ifdef VERBOSE
          if (candidatesafety * candidatesafety > boxsafetypair.second && boxsafetypair.second > 0)
            std::cerr << "real safety smaller than boxsafety \n";
#endif
          if (candidatesafety < safety) {
            safety    = candidatesafety;
            safetysqr = safety * safety;
          }
        }
      }
    }
    return safety;
  }

  // this is (almost) the same code as in SimpleABBoxSafetyEstimator --> avoid this
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const &localpoint,
                                               VPlacedVolume const *pvol) const override
  {
    // safety to mother
    double safety = pvol->SafetyToOut(localpoint);
    if (safety <= 0.) {
      return 0.;
    }
    return TreatSafetyToIn(localpoint, pvol->GetLogicalVolume(), safety);
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyToDaughtersForLocalPoint(Vector3D<Precision> const &localpoint,
                                                          LogicalVolume const *lvol) const override
  {
    return TreatSafetyToIn(localpoint, lvol, kInfLength);
  }

  static VSafetyEstimator *Instance()
  {
    static HybridSafetyEstimator instance;
    return &instance;
  }

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_SIMPLEABBOXSAFETYESTIMATOR_H_ */
