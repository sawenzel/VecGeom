/*
 * SimpleLevelLocator.h
 *
 *  Created on: Aug 27, 2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_SIMPLELEVELLOCATOR_H_
#define NAVIGATION_SIMPLELEVELLOCATOR_H_

#include "VecGeom/navigation/VLevelLocator.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/PlacedAssembly.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// shared kernel for many locators
// treats the actual final check (depending on which interface to serve)
template <bool IsAssemblyAware, bool ModifyState>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE
  static bool CheckCandidateVol(VPlacedVolume const *nextvolume,
				Vector3D<Precision> const &localpoint,
				NavigationState *state, VPlacedVolume const *&pvol,
				Vector3D<Precision> &daughterlocalpoint)
{
  if (IsAssemblyAware && ModifyState) {
    if (nextvolume->GetUnplacedVolume()->IsAssembly()) {
      // in this case we call a special version of Contains
      // offered by the assembly
      VECGEOM_ASSERT(ModifyState == true);
      if (((PlacedAssembly *)nextvolume)->Contains(localpoint, daughterlocalpoint, *state)) {
        return true;
      }
    } else {
      if (nextvolume->Contains(localpoint, daughterlocalpoint)) {
        state->Push(nextvolume);
        return true;
      }
    }
  } else {
    if (nextvolume->Contains(localpoint, daughterlocalpoint)) {
      if (ModifyState) {
        state->Push(nextvolume);
      } else {
        pvol = nextvolume;
      }
      return true;
    }
  }
  return false;
}

// shared kernel for many locators
// treats the actual final check (depending on which interface to serve)
template <bool IsAssemblyAware, bool ModifyState>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE
  static bool CheckCandidateVolWithDirection(
    VPlacedVolume const *nextvolume, Vector3D<Precision> const &localpoint, Vector3D<Precision> const &localdirection,
    NavigationState *state, VPlacedVolume const *&pvol, Vector3D<Precision> &daughterlocalpoint)
{
  if (IsAssemblyAware && ModifyState) {
    /* if (nextvolume->GetUnplacedVolume()->IsAssembly()) { */
    /*   // in this case we call a special version of Contains */
    /*   // offered by the assembly */
    /*   VECGEOM_ASSERT(ModifyState == true); */
    /*   if (((PlacedAssembly *)nextvolume)->Inside(localpoint, daughterlocalpoint, *state)) { */
    /*     return true; */
    /*   } */
    /* } else { */
    /*   if (nextvolume->Inside(localpoint, daughterlocalpoint)) { */
    /*     state->Push(nextvolume); */
    /*     return true; */
    /*   } */
    /* } */
    VECGEOM_VALIDATE(false, << "not implemented yet");
  } else {
    //
    const auto transf            = nextvolume->GetTransformation();
    const auto testdaughterlocal = transf->Transform(localpoint);
    const auto inside            = nextvolume->GetUnplacedVolume()->Inside(testdaughterlocal);

    auto CheckEntering = [&transf, &testdaughterlocal, &localdirection, &nextvolume]() {
      const auto unpl = nextvolume->GetUnplacedVolume();
      Vector3D<Precision> normal;
      unpl->Normal(testdaughterlocal, normal);
      const auto directiondaughterlocal = transf->TransformDirection(localdirection);
      const auto dot                    = normal.Dot(directiondaughterlocal);
      if (dot >= 0) {
        return false;
      }
      return true;
    };

    if (inside == kInside || ((inside == kSurface) && CheckEntering())) {
      if (ModifyState) {
        state->Push(nextvolume);
        daughterlocalpoint = testdaughterlocal;
      } else {
        pvol               = nextvolume;
        daughterlocalpoint = testdaughterlocal;
      }
      return true;
    }
  }
  return false;
}

// a simple version of a LevelLocator offering a generic brute force algorithm
template <bool IsAssemblyAware = false>
class TSimpleLevelLocator : public VLevelLocator {

public:
  VECGEOM_FORCE_INLINE  VECCORE_ATT_HOST_DEVICE
  TSimpleLevelLocator() {}

private:
  // the actual implementation kernel
  // the template "ifs" should be optimized away
  // arguments are pointers to allow for nullptr
  template <bool ExclV, bool ModifyState>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool LevelLocateKernel(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
			 Vector3D<Precision> const &localpoint, NavigationState *state,
			 VPlacedVolume const *&pvol,
			 Vector3D<Precision> &daughterlocalpoint) const
  {
    auto daughters = lvol->GetDaughtersp();
    for (size_t i = 0; i < daughters->size(); ++i) {
      VPlacedVolume const *nextvolume = (*daughters)[i];
      if (ExclV) {
        if (exclvol == nextvolume) continue;
      }
      if (CheckCandidateVol<IsAssemblyAware, ModifyState>(nextvolume, localpoint, state, pvol, daughterlocalpoint)) {
        return true;
      }
    }
    return false;
  }

  // the actual implementation kernel
  // the template "ifs" should be optimized away
  // arguments are pointers to allow for nullptr
  template <bool ExclV, bool ModifyState>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE
  bool LevelLocateKernelWithDirection(LogicalVolume const *lvol,
				      VPlacedVolume const *exclvol,
				      Vector3D<Precision> const &localpoint,
				      Vector3D<Precision> const &localdir,
				      NavigationState *state, VPlacedVolume const *&pvol,
				      Vector3D<Precision> &daughterlocalpoint) const
  {
    auto daughters = lvol->GetDaughtersp();
    for (size_t i = 0; i < daughters->size(); ++i) {
      VPlacedVolume const *nextvolume = (*daughters)[i];
      if (ExclV) {
        if (exclvol == nextvolume) continue;
      }
      if (CheckCandidateVolWithDirection<IsAssemblyAware, ModifyState>(nextvolume, localpoint, localdir, state, pvol,
                                                                       daughterlocalpoint)) {
        return true;
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
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocate(LogicalVolume const *lvol, Vector3D<Precision> const &localpoint, NavigationState &state,
                           Vector3D<Precision> &daughterlocalpoint) const override
  {
    VPlacedVolume const *pvol = nullptr;
    return LevelLocateKernel<false, true>(lvol, nullptr, localpoint, &state, pvol, daughterlocalpoint);
  }

  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, VPlacedVolume const *&pvol,
                                  Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernel<true, false>(lvol, exclvol, localpoint, nullptr, pvol, daughterlocalpoint);
  }

  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocateExclVol(LogicalVolume const *lvol, VPlacedVolume const *exclvol,
                                  Vector3D<Precision> const &localpoint, Vector3D<Precision> const &localdirection,
                                  VPlacedVolume const *&pvol, Vector3D<Precision> &daughterlocalpoint) const override
  {
    return LevelLocateKernelWithDirection<true, false>(lvol, exclvol, localpoint, localdirection, nullptr, pvol,
                                                       daughterlocalpoint);
  }

  static std::string GetClassName() { return "SimpleLevelLocator"; }
  virtual std::string GetName() const override { return GetClassName(); }

#ifndef VECCORE_CUDA
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE
  static VLevelLocator const *GetInstance()
  {
    static TSimpleLevelLocator instance;
    return &instance;
  }
#endif

}; // end class declaration

using SimpleLevelLocator = TSimpleLevelLocator<>;

template <>
inline std::string TSimpleLevelLocator<true>::GetClassName()
{
  return "SimpleAssemblyLevelLocator";
}
using SimpleAssemblyLevelLocator = TSimpleLevelLocator<true>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_SIMPLELEVELLOCATOR_H_ */
