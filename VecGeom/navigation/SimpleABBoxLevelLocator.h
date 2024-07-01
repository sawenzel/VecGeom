/*
 * SimpleABBoxLevelLocator.h
 *
 *  Created on: Aug 27, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLEABBOXLEVELLOCATOR_H_
#define NAVIGATION_SIMPLEABBOXLEVELLOCATOR_H_

#include "VecGeom/navigation/VLevelLocator.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/management/ABBoxManager.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"

#include "VecGeom/navigation/SimpleLevelLocator.h"

#include <exception>
#include <stdexcept>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// a LevelLocator using a flat list of bounding boxes
template <bool IsAssemblyAware = false>
class TSimpleABBoxLevelLocator : public VLevelLocator {

private:
  ABBoxManager<Precision> &fAccelerationStructure;
  TSimpleABBoxLevelLocator() : fAccelerationStructure(ABBoxManager<Precision>::Instance()) {}

  // the actual implementation kernel
  // the template "ifs" should be optimized away
  // arguments are pointers to allow for nullptr
  template <bool ExclV, bool ModifyState>
  __attribute__((always_inline)) bool LevelLocateKernel(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                                        Vector3D<Precision> const &localpoint, NavigationState *state,
                                                        VPlacedVolume const *&pvol,
                                                        Vector3D<Precision> &daughterlocalpoint) const
  {
    int size;
    ABBoxManager<Precision>::ABBoxContainer_v alignedbboxes = fAccelerationStructure.GetABBoxes_v(lvol, size);

    auto daughters = lvol->GetDaughtersp();
    // here the loop is over groups of bounding boxes
    // it is basically linear but vectorizable search
    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
      using Bool_v = vecCore::Mask_v<ABBoxManager<Precision>::Float_v>;
      Bool_v inBox;
      ABBoxImplementation::ABBoxContainsKernel(alignedbboxes[2 * boxgroupid], alignedbboxes[2 * boxgroupid + 1],
                                               localpoint, inBox);
      if (!vecCore::MaskEmpty(inBox)) {
        constexpr auto kVS = vecCore::VectorSize<ABBoxManager<Precision>::Float_v>();
        // TODO: could start directly at first 1 in inBox
        for (size_t ii = 0; ii < kVS; ++ii) {
          auto daughterid = boxgroupid * kVS + ii;
          if (daughterid < daughters->size() && vecCore::MaskLaneAt(inBox, ii)) {

            VPlacedVolume const *daughter = (*daughters)[daughterid];
            if (ExclV) {
              if (daughter == exclvol) continue;
            }
            // call final treatment of candidate
            if (CheckCandidateVol<IsAssemblyAware, ModifyState>(daughter, localpoint, state, pvol, daughterlocalpoint))
              return true;
          }
        }
      }
    }
    return false;
  }

  template <bool ExclV, bool ModifyState>
  __attribute__((always_inline)) bool LevelLocateKernelWithDirection(LogicalVolume const *lvol,
                                                                     VPlacedVolume const *exclvol,
                                                                     Vector3D<Precision> const &localpoint,
                                                                     Vector3D<Precision> const &localdir,
                                                                     NavigationState *state, VPlacedVolume const *&pvol,
                                                                     Vector3D<Precision> &daughterlocalpoint) const
  {

    int size;
    ABBoxManager<Precision>::ABBoxContainer_v alignedbboxes = fAccelerationStructure.GetABBoxes_v(lvol, size);

    auto daughters = lvol->GetDaughtersp();
    // here the loop is over groups of bounding boxes
    // it is basically linear but vectorizable search
    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
      using Bool_v = vecCore::Mask_v<ABBoxManager<Precision>::Float_v>;
      Bool_v inBox;
      ABBoxImplementation::ABBoxContainsKernel(alignedbboxes[2 * boxgroupid], alignedbboxes[2 * boxgroupid + 1],
                                               localpoint, inBox);
      if (!vecCore::MaskEmpty(inBox)) {
        constexpr auto kVS = vecCore::VectorSize<ABBoxManager<Precision>::Float_v>();
        // TODO: could start directly at first 1 in inBox
        for (size_t ii = 0; ii < kVS; ++ii) {
          auto daughterid = boxgroupid * kVS + ii;
          if (daughterid < daughters->size() && vecCore::MaskLaneAt(inBox, ii)) {

            VPlacedVolume const *nextvolume = (*daughters)[daughterid];
            if (ExclV) {
              if (exclvol == nextvolume) continue;
            }
            if (CheckCandidateVolWithDirection<IsAssemblyAware, ModifyState>(nextvolume, localpoint, localdir, state,
                                                                             pvol, daughterlocalpoint)) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

public:
  VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                           Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<false, false>(lvol, nullptr, localpoint, nullptr, pvol, daughterlocalpoint);
  } // end function

  // version that directly modifies the navigation state
  VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, NavigationState &outstate,
                           Vector3D<Precision> &daughterlocalpoint) const override
  {
    VPlacedVolume const *pvol;
    return LevelLocateKernel<false, true>(lvol, nullptr, localpoint, &outstate, pvol, daughterlocalpoint);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                                  Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<true, false>(lvol, exclvol, localpoint, nullptr, pvol, daughterlocalpoint);
  } // end function

  VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, Vector3D<Precision> const &localdir,
                                  VPlacedVolume const *&pvol, Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernelWithDirection<true, false>(lvol, exclvol, localpoint, localdir, nullptr, pvol,
                                                       daughterlocalpoint);
  }

  static std::string GetClassName() { return "SimpleABBoxLevelLocator"; }
  virtual std::string GetName() const override { return GetClassName(); }

  static VLevelLocator const *GetInstance()
  {
    static TSimpleABBoxLevelLocator instance;
    return &instance;
  }

}; // end class declaration

using SimpleABBoxLevelLocator              = TSimpleABBoxLevelLocator<>;
using SimpleAssemblyAwareABBoxLevelLocator = TSimpleABBoxLevelLocator<true>;

//// template specializations
// template <>
// inline
// bool TSimpleABBoxLevelLocator<true>::LevelLocate(LogicalVolume const * lvol, Vector3D<Precision> const & localpoint,
//                                                 NavigationState & outstate, Vector3D<Precision> & daughterlocalpoint)
//                                                 const
//{
//    int size;
//    ABBoxManager<Precision>::ABBoxContainer_v alignedbboxes = fAccelerationStructure.GetABBoxes_v(lvol, size);

//    auto daughters = lvol->GetDaughtersp();
//    // here the loop is over groups of bounding boxes
//    // it is basically linear but vectorizable search
//    for (int boxgroupid = 0; boxgroupid < size; ++boxgroupid) {
//      using Bool_v = vecCore::Mask_v<ABBoxManager<Precision>::Float_v>;
//      Bool_v inBox;
//      ABBoxImplementation::ABBoxContainsKernel(alignedbboxes[2 * boxgroupid], alignedbboxes[2 * boxgroupid + 1],
//                                               localpoint, inBox);
//      if (!vecCore::MaskEmpty(inBox)) {
//        constexpr auto kVS = vecCore::VectorSize<ABBoxManager<Precision>::Float_v>();
//        // TODO: could start directly at first 1 in inBox
//        for (size_t ii = 0; ii < kVS; ++ii) {
//          auto daughterid = boxgroupid * kVS + ii;
//          if (daughterid < daughters->size() && vecCore::MaskLaneAt(inBox, ii)) {
//            VPlacedVolume const *daughter = (*daughters)[daughterid];
//            if (daughter->Contains(localpoint, daughterlocalpoint)) {
//              outstate.Push(daughter);
//              // careful here: we also want to break on external loop
//              return true;
//            }
//          }
//        }
//      }
//    }
//    return false;
//}

template <>
inline std::string TSimpleABBoxLevelLocator<true>::GetClassName()
{
  return "SimpleAssemblyAwareABBoxLevelLocator";
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_SIMPLEABBOXLEVELLOCATOR_H_ */
