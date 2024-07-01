/*
 * SimpleABBoxSafetyEstimator.h
 *
 *  Created on: Aug 28, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLEABBOXSAFETYESTIMATOR_H_
#define NAVIGATION_SIMPLEABBOXSAFETYESTIMATOR_H_

#include "VecGeom/navigation/VSafetyEstimator.h"
#include "VecGeom/management/ABBoxManager.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//! a safety estimator using a (vectorized) search through bounding boxes to exclude certain daughter volumes
//! to talk to
class SimpleABBoxSafetyEstimator : public VSafetyEstimatorHelper<SimpleABBoxSafetyEstimator> {

private:
  // we keep a reference to the ABBoxManager ( avoids calling Instance() on this guy all the time )
  ABBoxManager<Precision> &fABBoxManager;

  SimpleABBoxSafetyEstimator()
      : VSafetyEstimatorHelper<SimpleABBoxSafetyEstimator>(), fABBoxManager(ABBoxManager<Precision>::Instance())
  {
  }

  // convert index to physical daugher
  VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int id) const
  {
    assert(id >= 0 && "access with negative index");
    assert(size_t(id) < lvol->GetDaughtersp()->size() && "access beyond size of daughterlist ");
    return lvol->GetDaughtersp()->operator[](id);
  }

public:
  // helper function calculating some candidate volumes
  VECCORE_ATT_HOST_DEVICE
  static size_t GetSafetyCandidates_v(Vector3D<Precision> const &point,
                                      ABBoxManager<Precision>::ABBoxContainer_v const &corners, size_t size,
                                      ABBoxManager<Precision>::BoxIdDistancePair_t *boxsafetypairs,
                                      Precision upper_squared_limit)
  {
    size_t count = 0;
    Vector3D<float> pointfloat((float)point.x(), (float)point.y(), (float)point.z());
    size_t vecsize = size;
    for (size_t box = 0; box < vecsize; ++box) {
      ABBoxManager<Precision>::Float_v safetytoboxsqr =
          ABBoxImplementation::ABBoxSafetySqr(corners[2 * box], corners[2 * box + 1], pointfloat);

      auto hit           = safetytoboxsqr < ABBoxManager<Precision>::Real_s(upper_squared_limit);
      constexpr auto kVS = vecCore::VectorSize<ABBoxManager<Precision>::Float_v>();
      if (!vecCore::MaskEmpty(hit)) {
        for (size_t i = 0; i < kVS; ++i) {
          if (vecCore::MaskLaneAt(hit, i)) {
            boxsafetypairs[count] =
                ABBoxManager<Precision>::BoxIdDistancePair_t(box * kVS + i, vecCore::LaneAt(safetytoboxsqr, i));
            count++;
          }
        }
      }
    }
    return count;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Precision TreatSafetyToIn(Vector3D<Precision> const &localpoint, LogicalVolume const *lvol, Precision outsafety) const
  {
    // a stack based workspace array
    // The following construct reserves stackspace for objects
    // of type IdDistPair_t WITHOUT initializing those objects
    using IdDistPair_t = ABBoxManager<Precision>::BoxIdDistancePair_t;
    char stackspace[VECGEOM_MAXDAUGHTERS * sizeof(IdDistPair_t)];
    IdDistPair_t *boxsafetylist = reinterpret_cast<IdDistPair_t *>(&stackspace);

    double safety    = outsafety; // we use the outsafety estimate as starting point
    double safetysqr = safety * safety;

    // safety to bounding boxes
    if (safety > 0. && lvol->GetDaughtersp()->size() > 0) {
      int size;

      ABBoxManager<Precision>::ABBoxContainer_v bboxes = fABBoxManager.GetABBoxes_v(lvol, size);
      // calculate squared bounding box safeties in vectorized way
      auto ncandidates = GetSafetyCandidates_v(localpoint, bboxes, size, boxsafetylist, safetysqr);
      // not sorting the candidate list ( which one could do )
      for (unsigned int candidate = 0; candidate < ncandidates; ++candidate) {
        auto boxsafetypair = boxsafetylist[candidate];
        if (boxsafetypair.second < safetysqr) {
          VPlacedVolume const *cand = LookupDaughter(lvol, boxsafetypair.first);
          if (boxsafetypair.first > lvol->GetDaughtersp()->size()) break;
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

public:
  static constexpr const char *gClassNameString = "SimpleABBoxSafetyEstimator";

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const &localpoint,
                                               VPlacedVolume const *pvol) const override
  {

    // safety to mother
    double safety = pvol->SafetyToOut(localpoint);
    return TreatSafetyToIn(localpoint, pvol->GetLogicalVolume(), safety);
  }

  // estimate just the safety to daughters for a local point with respect to a logical volume
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyToDaughtersForLocalPoint(Vector3D<Precision> const &localpoint,
                                                          LogicalVolume const *lvol) const override
  {
    return TreatSafetyToIn(localpoint, lvol, kInfLength);
  }

  static VSafetyEstimator *Instance()
  {
    static SimpleABBoxSafetyEstimator instance;
    return &instance;
  }

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_SIMPLEABBOXSAFETYESTIMATOR_H_ */
