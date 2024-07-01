// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file navigation/VoxelLevelLocator.h
/// \author Sandro Wenzel (CERN)

#ifndef NAVIGATION_VOXELLEVELLOCATOR_H_
#define NAVIGATION_VOXELLEVELLOCATOR_H_

#include "VecGeom/navigation/VLevelLocator.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/management/FlatVoxelManager.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include "VecGeom/navigation/SimpleLevelLocator.h"
#include "VecGeom/management/ABBoxManager.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//! A LevelLocator using voxel hash-maps containing candidate volume ids to quickly
//! decide in which volume a point is.
template <bool IsAssemblyAware = false>
class TVoxelLevelLocator : public VLevelLocator {

private:
  FlatVoxelManager &fAccelerationStructure;

  TVoxelLevelLocator() : fAccelerationStructure(FlatVoxelManager::Instance()) {}

  // the actual implementation kernel
  // the template "ifs" should be optimized away
  // arguments are pointers to allow for nullptr
  template <bool ExclV, bool ModifyState>
  __attribute__((always_inline)) bool LevelLocateKernel(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                                        Vector3D<Precision> const &localpoint, NavigationState *state,
                                                        VPlacedVolume const *&pvol,
                                                        Vector3D<Precision> &daughterlocalpoint) const
  {
    const auto accstructure = fAccelerationStructure.GetStructure(lvol);
    // fetch the right voxels
    const auto locatevoxels = accstructure->fVoxelToLocateCandidates;
    // fetch the list of candidates to check (as id)
    Vector3D<float> floatlocalpoint(localpoint.x(), localpoint.y(), localpoint.z());

    int numbercandidates{0};
    const auto candidateidsptr = locatevoxels->getProperties(floatlocalpoint, numbercandidates);

    if (numbercandidates > 0) {
      // here we have something to do
      int numberboxes{0};
      const auto boxes = ABBoxManager<Precision>::Instance().GetABBoxes(lvol, numberboxes);

      for (int i = 0; i < numbercandidates; ++i) {
        const int daughterid = candidateidsptr[i];

        // discard blocked volume
        VPlacedVolume const *candidate = lvol->GetDaughters()[daughterid];
        if (ExclV) {
          if (candidate == exclvol) continue;
        }

        // quick aligned bounding box check (could be done in SIMD fashion if multiple candidates)
        const auto &lower = boxes[2 * daughterid];
        const auto &upper = boxes[2 * daughterid + 1];
        bool boxcontains{false};
        ABBoxImplementation::ABBoxContainsKernel(lower, upper, localpoint, boxcontains);
        if (!boxcontains) {
          continue;
        }

        // final candidate check
        if (CheckCandidateVol<IsAssemblyAware, ModifyState>(candidate, localpoint, state, pvol, daughterlocalpoint)) {
          return true;
        }
      }
    }
    return false;
  }

  // the actual implementation kernel
  // the template "ifs" should be optimized away
  // arguments are pointers to allow for nullptr
  template <bool ExclV, bool ModifyState>
  __attribute__((always_inline)) bool LevelLocateKernelWithDirection(LogicalVolume const *lvol,
                                                                     VPlacedVolume const *exclvol,
                                                                     Vector3D<Precision> const &localpoint,
                                                                     Vector3D<Precision> const &localdir,
                                                                     NavigationState *state, VPlacedVolume const *&pvol,
                                                                     Vector3D<Precision> &daughterlocalpoint) const
  {
    const auto accstructure = fAccelerationStructure.GetStructure(lvol);
    // fetch the right voxels
    const auto locatevoxels = accstructure->fVoxelToLocateCandidates;
    // fetch the list of candidates to check (as id)
    Vector3D<float> floatlocalpoint(localpoint.x(), localpoint.y(), localpoint.z());

    int numbercandidates{0};
    const auto candidateidsptr = locatevoxels->getProperties(floatlocalpoint, numbercandidates);

    if (numbercandidates > 0) {
      // here we have something to do
      int numberboxes{0};
      const auto boxes = ABBoxManager<Precision>::Instance().GetABBoxes(lvol, numberboxes);

      for (int i = 0; i < numbercandidates; ++i) {
        const int daughterid = candidateidsptr[i];

        // discard blocked volume
        VPlacedVolume const *candidate = lvol->GetDaughters()[daughterid];
        if (ExclV) {
          if (candidate == exclvol) continue;
        }

        // quick aligned bounding box check (could be done in SIMD fashion if multiple candidates)
        const auto &lower = boxes[2 * daughterid];
        const auto &upper = boxes[2 * daughterid + 1];
        bool boxcontains{false};
        ABBoxImplementation::ABBoxContainsKernel(lower, upper, localpoint, boxcontains);
        if (!boxcontains) {
          continue;
        }

        // final candidate check
        if (CheckCandidateVolWithDirection<IsAssemblyAware, ModifyState>(candidate, localpoint, localdir, state, pvol,
                                                                         daughterlocalpoint)) {
          return true;
        }
      }
    }
    return false;
  }

public:
  static std::string GetClassName() { return "VoxelLevelLocator"; }
  std::string GetName() const override { return GetClassName(); }

  bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                   Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<false, false>(lvol, nullptr, localpoint, nullptr, pvol, daughterlocalpoint);
  }

  bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, NavigationState &state,
                   Vector3D<Precision> &daughterlocalpoint) const override
  {
    VPlacedVolume const *pvol;
    return LevelLocateKernel<false, true>(lvol, nullptr, localpoint, &state, pvol, daughterlocalpoint);
  }

  bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                          Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                          Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<true, false>(lvol, exclvol, localpoint, nullptr, pvol, daughterlocalpoint);
  }

  bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                          Vector3D<Precision> const &localpoint, Vector3D<Precision> const &localdir,
                          VPlacedVolume const *&pvol, Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernelWithDirection<true, false>(lvol, exclvol, localpoint, localdir, nullptr, pvol,
                                                       daughterlocalpoint);
  }

  static VLevelLocator const *GetInstance()
  {
    static TVoxelLevelLocator instance;
    return &instance;
  }

}; // end class declaration

template <>
inline std::string TVoxelLevelLocator<true>::GetClassName()
{
  return "AssemblyAwareVoxelLevelLocator";
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_VOXELLEVELLOCATOR_H_ */
