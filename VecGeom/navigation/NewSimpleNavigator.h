/*
 * NewSimpleNavigator.h
 *
 *  Created on: 17.09.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_NEWSIMPLENAVIGATOR_H_
#define NAVIGATION_NEWSIMPLENAVIGATOR_H_

#include "VNavigator.h"
#include "SimpleSafetyEstimator.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// A very basic implementation of a navigator ( brute force which scales linearly with the number of daughters )
template <bool MotherIsConvex = false>
class NewSimpleNavigator : public VNavigatorHelper<class NewSimpleNavigator<MotherIsConvex>, MotherIsConvex> {

private:
  VECCORE_ATT_DEVICE
  NewSimpleNavigator()
      : VNavigatorHelper<class NewSimpleNavigator<MotherIsConvex>, MotherIsConvex>(SimpleSafetyEstimator::Instance()) {
  } VECCORE_ATT_DEVICE virtual ~NewSimpleNavigator() {}

public:
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, NavigationState const *in_state,
                                          NavigationState * /*out_state*/, Precision &step,
                                          VPlacedVolume const *&hitcandidate) const override
  {
    // iterate over all daughters
    auto *daughters = lvol->GetDaughtersp();
    auto ndaughters = daughters->size();
    for (decltype(ndaughters) d = 0; d < ndaughters; ++d) {
      auto daughter = daughters->operator[](d);
      if (in_state && in_state->GetLastExited() == daughter) continue;
//    previous distance becomes step estimate, distance to daughter returned in workspace
// SW: this makes the navigation more robust and it appears that I have to
// put this at the moment since not all shapes respond yet with a negative distance if
// the point is actually inside the daughter
#ifdef CHECKCONTAINS
      bool contains = daughter->Contains(localpoint);
      if (!contains) {
#endif
        Precision ddistance = daughter->DistanceToIn(localpoint, localdir, step);

        // if distance is negative; we are inside that daughter and should relocate
        // unless distance is minus infinity
        const bool valid = (ddistance < step && !IsInf(ddistance) && ddistance > -kTolerance);
        hitcandidate     = valid ? daughter : hitcandidate;
        step             = valid ? ddistance : step;
#ifdef CHECKCONTAINS
      } else {
        std::cerr << " INDA "
                  << " contained in daughter " << daughter << " - inside = " << daughter->Inside(localpoint)
                  << " , distToIn(p,v,s) = " << daughter->DistanceToIn(localpoint, localdir, step) << " \n";

        std::cerr << " INDA ";
        step         = -1.;
        hitcandidate = daughter;
        break;
      }
#endif
    }
    return false;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual bool CheckDaughterIntersections(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint,
                                          Vector3D<Precision> const &localdir, VPlacedVolume const *blocked,
                                          Precision &step, VPlacedVolume const *&hitcandidate) const override
  {
    //  New Implementation JA 2021.03.18
    static const double kMinExitingCos = 1.e-3;
    VPlacedVolume const *excludedVol   = nullptr;
    if (blocked) {
      Vector3D<Precision> normal;
      blocked->Normal(localpoint, normal);
      if (normal.Dot(localdir) >= kMinExitingCos) {
        excludedVol = blocked;
      }
    }

    // iterate over all daughters
    auto *daughters = lvol->GetDaughtersp();
    auto ndaughters = daughters->size();
    for (decltype(ndaughters) d = 0; d < ndaughters; ++d) {
      auto daughter = daughters->operator[](d);
//    previous distance becomes step estimate, distance to daughter returned in workspace
// SW: this makes the navigation more robust and it appears that I have to
// put this at the moment since not all shapes respond yet with a negative distance if
// the point is actually inside the daughter
#ifdef CHECKCONTAINS
      bool contains = daughter->Contains(localpoint);
      if (!contains) {
#endif
        if (daughter != excludedVol) {
          Precision ddistance = daughter->DistanceToIn(localpoint, localdir, step);

          // if distance is negative; we are inside that daughter and should relocate
          // unless distance is minus infinity
          const bool valid = (ddistance < step && !IsInf(ddistance));
          hitcandidate     = valid ? daughter : hitcandidate;
          step             = valid ? ddistance : step;
        }
#ifdef CHECKCONTAINS
      } else {
        std::cerr << " INDA: contained in daughter " << daughter << " - inside = " << daughter->Inside(localpoint)
                  << " , distToIn(p,v,s) = " << daughter->DistanceToIn(localpoint, localdir, step) << " \n";
        step         = -1.;
        hitcandidate = daughter;
        break;
      }
#endif
    }
    // assert(false); --- Was not implemented before
    return false;
  }

#ifndef VECCORE_CUDA
  static VNavigator *Instance()
  {
    static NewSimpleNavigator instance;
    return &instance;
  }
#else
  VECCORE_ATT_DEVICE
  static VNavigator *Instance();
#endif

  static constexpr const char *gClassNameString = "NewSimpleNavigator";
  typedef SimpleSafetyEstimator SafetyEstimator_t;
}; // end of class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_NEWSIMPLENAVIGATOR_H_ */
