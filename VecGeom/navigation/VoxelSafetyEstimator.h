// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file navigation/VoxelSafetyEstimator.h
/// \author Sandro Wenzel (CERN)

#ifndef NAVIGATION_VOXELSAFETYESTIMATOR_H_
#define NAVIGATION_VOXELSAFETYESTIMATOR_H_

#include "VecGeom/navigation/VSafetyEstimator.h"
#include "VecGeom/management/FlatVoxelManager.h"
#include <cmath>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//! a safety estimator using a voxel lookup table of precomputed safety candidates
//! (or in the extreme case of precomputed safety values)
class VoxelSafetyEstimator : public VSafetyEstimatorHelper<VoxelSafetyEstimator> {

private:
  const FlatVoxelManager &fAccStructureManager;

  VoxelSafetyEstimator()
      : VSafetyEstimatorHelper<VoxelSafetyEstimator>(), fAccStructureManager{FlatVoxelManager::Instance()}
  {
  }

  // convert index to physical daugher
  VPlacedVolume const *LookupDaughter(LogicalVolume const *lvol, int id) const
  {
    assert(id >= 0 && "access with negative index");
    assert(size_t(id) < lvol->GetDaughtersp()->size() && "access beyond size of daughterlist");
    return lvol->GetDaughtersp()->operator[](id);
  }

public:
  static constexpr const char *gClassNameString = "VoxelSafetyEstimator";

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Precision TreatSafetyToIn(Vector3D<Precision> const &localpoint, LogicalVolume const *lvol, Precision outsafety) const
  {
    throw std::runtime_error("unimplemented function called");
  }

  // #define CROSSCHECK

#ifdef CROSSCHECK
  void printProcedure(int const *safetycandidates, int length, LogicalVolume const *lvol,
                      Vector3D<Precision> const &localpoint) const
  {
    int size{0};
    auto abboxcorners = ABBoxManager<Precision>::Instance().GetABBoxes(lvol, size);
    Vector3D<float> lp(localpoint.x(), localpoint.y(), localpoint.z()); // in float
    const bool needmother        = (safetycandidates[0] == -1);
    const Precision safetymother = needmother ? lvol->GetUnplacedVolume()->SafetyToOut(localpoint) : 0.;
    std::cerr << "safetymother " << safetymother;
    Precision safetyToDaughters{1E20};
    // now talk to daughter candidates - with bounding box speedup
    // TODO: this could actually happen with SIMD speedup
    float bestSafetySqr{1E20};
    for (int i = 0; i < length; ++i) {
      if (safetycandidates[i] == -1) continue;
      const auto candidateid = safetycandidates[i];
      const auto &lower      = abboxcorners[2 * candidateid];
      const auto &upper      = abboxcorners[2 * candidateid + 1];
      const float ssqr       = ABBoxImplementation::ABBoxSafetySqr(lower, upper, lp);
      std::cerr << i << " boxsqr " << ssqr << "\t" << std::sqrt(ssqr) << "\n";
      if (ssqr > 0) {
        bestSafetySqr = std::min(bestSafetySqr, ssqr);
      } else { // inside
        const auto daughter = LookupDaughter(lvol, safetycandidates[i]);
        auto sin            = daughter->SafetyToIn(localpoint);
        std::cerr << i << " SI " << sin << "\n";
        safetyToDaughters = std::min(safetyToDaughters, daughter->SafetyToIn(localpoint));
      }
    }
    safetyToDaughters = std::min(safetyToDaughters, (Precision)std::sqrt(bestSafetySqr));
    std::cerr << "final safetyd " << safetyToDaughters << "\n";
    const float returnvalue = needmother ? std::min(safetyToDaughters, safetymother) : safetyToDaughters;
    std::cerr << "returning " << returnvalue << "\n";
  }
#endif

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeSafetyForLocalPoint(Vector3D<Precision> const &localpoint,
                                               VPlacedVolume const *pvol) const override
  {
    static int counter = 0;
    counter++;
    const auto lvol      = pvol->GetLogicalVolume();
    const auto structure = fAccStructureManager.GetStructure(lvol);

    int size{0};
    auto abboxcorners = ABBoxManager<Precision>::Instance().GetABBoxes(lvol, size);

    if (structure != nullptr) {
      const Vector3D<float> lp(localpoint.x(), localpoint.y(), localpoint.z()); // in float
      // fetch safety properties for localpoint;
      const int *safetycandidates{nullptr};

      int length{0};
      auto voxelHashMap = structure->fVoxelToCandidate;
      if (voxelHashMap == nullptr) {
        std::cerr << "ERROR> VoxelSafetyEstimator::ComputeSafetyForLocalPoint call# " << counter
                  << " no structure VoxelToCandidate structure found for volume "
                  << " phys: " << pvol->GetName() << "  physvol id = " << pvol->id() << " log : " << lvol->GetName()
                  << "  log vol id = " << lvol->id() << std::endl;
        return 0.0;
      }
      safetycandidates = voxelHashMap->getProperties(lp, length);
      if (length > 0) {
        const bool needmother        = true; //(safetycandidates[0] == -1);
        const Precision safetymother = needmother ? lvol->GetUnplacedVolume()->SafetyToOut(localpoint) : 0.;
        Precision safetyToDaughters{1E20};
        // now talk to daughter candidates - with bounding box speedup
        // TODO: this could actually happen with SIMD speedup
        float bestSafetySqr{1E20};
        for (int i = 0; i < length; ++i) {
          if (safetycandidates[i] == -1) continue;
          const auto candidateid = safetycandidates[i];
          const auto &lower      = abboxcorners[2 * candidateid];
          const auto &upper      = abboxcorners[2 * candidateid + 1];
          const float ssqr       = ABBoxImplementation::ABBoxSafetySqr(lower, upper, lp);
          if (ssqr > 0) {
            bestSafetySqr = std::min(bestSafetySqr, ssqr);
          } else { // inside
            const auto daughter = LookupDaughter(lvol, safetycandidates[i]);
            auto s              = daughter->SafetyToIn(localpoint);
            if (s <= 0.) {
              // can break here
              s = 0.;
            }
            safetyToDaughters = std::min(safetyToDaughters, s);
          }
        }
        safetyToDaughters       = std::min(safetyToDaughters, (Precision)std::sqrt(bestSafetySqr));
        const float returnvalue = needmother ? std::min(safetyToDaughters, safetymother) : safetyToDaughters;

#ifdef CROSSCHECK
        if (returnvalue > lvol->GetUnplacedVolume()->SafetyToOut(localpoint) + 1E-4) {
          std::cerr << returnvalue << " " << lvol->GetUnplacedVolume()->SafetyToOut(localpoint)
                    << "Problem in voxel safety for point " << lp << " voxelkey "
                    << structure->fVoxelToCandidate->getKey(lp.x(), lp.y(), lp.z()) << " ";
          std::cerr << "{ ";
          for (int i = 0; i < length; ++i) {
            std::cerr << safetycandidates[i] << " , ";
          }
          std::cerr << " }\n";
          printProcedure(safetycandidates, length, lvol, localpoint);
        }
#endif

        if (returnvalue < -1.0e-10) {
          std::cerr << "VoxelSafetyEstimator: returning negative value.  saf-to-daughters= " << safetyToDaughters
                    << " saf-to-mother = " << safetymother << "\n";
        }

        return returnvalue;
      } else {
        // no information for this voxel present
        std::cerr << "Warning> ComputeSafetyForLocalPoint call# " << counter
                  << " no information for this voxel present " << localpoint << " at key "
                  << structure->fVoxelToCandidate->getKey(lp.x(), lp.y(), lp.z()) << " \n ";
        return 0.;
      }
    }
    std::cerr << " No voxel information found; Cannot use voxel safety\n";
    return 0.;
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
    static VoxelSafetyEstimator instance;
    return &instance;
  }

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_VOXELSAFETYESTIMATOR_H_ */
