/*
 * RaytraceBenchmark.cpp
 *
 *  Created on: May 8, 2020
 *      Author: andrei.gheata@cern.ch
 */

#include <iomanip>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/NavIndexTable.h>
#include <VecGeom/management/ReferenceNavState.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/base/Stopwatch.h>
#include "ArgParser.h"

#ifdef VECGEOM_GDML
#include "Frontend.h"
#endif

using namespace vecgeom;

int TestNavIndexCPU(vecgeom::cxx::VPlacedVolume const *const world, int maxdepth);

#ifdef VECGEOM_ENABLE_CUDA
int TestNavIndexGPU(vecgeom::cxx::VPlacedVolume const *const world, int maxdepth);
#endif

namespace visitorcxx {

template <typename EncodedNavState, typename EncodedState>
VECCORE_ATT_HOST_DEVICE int ReportValidationError(ReferenceNavValidationError error, ReferenceNavState const &reference,
                                                  EncodedState encoded_state)
{
  printf("=== EEE === TestNavIndex: %s\n", ToString(error));
  PrintValidationFailure<EncodedNavState>(error, reference, encoded_state);
  return static_cast<int>(error);
}

VECCORE_ATT_HOST_DEVICE
int ReportIncompatibleDaughter(VPlacedVolume const *parent, VPlacedVolume const *daughter)
{
  printf("=== EEE === TestNavIndex: %s\n", ToString(ReferenceNavValidationError::kIncompatibleDaughter));
  printf("    expected daughter child id >= 0 for descent from %d to %d, got %d\n", parent ? parent->id() : -1,
         daughter ? daughter->id() : -1, daughter ? daughter->GetChildId() : -1);
  return static_cast<int>(ReferenceNavValidationError::kIncompatibleDaughter);
}

template <typename EncodedNavState, typename EncodedState>
VECCORE_ATT_HOST_DEVICE int ReportSceneTransitionError(EncodedState parent_state, EncodedState child_state,
                                                       VPlacedVolume const *parent, VPlacedVolume const *daughter)
{
  printf("=== EEE === TestNavIndex: %s\n", ToString(ReferenceNavValidationError::kIncompatibleScene));
  PrintSceneTransitionFailure<EncodedNavState>(parent_state, child_state, parent, daughter);
  return static_cast<int>(ReferenceNavValidationError::kIncompatibleScene);
}

VECCORE_ATT_HOST_DEVICE
int visitAllPlacedVolumesPassNavIndex(VPlacedVolume const *currentvolume, ReferenceNavState &reference,
                                      NavIndex_t nav_ind)
{
  auto validation_error = ValidateEncodedState<NavStateIndex>(reference, nav_ind);
  if (validation_error != ReferenceNavValidationError::kNone) {
    return ReportValidationError<NavStateIndex>(validation_error, reference, nav_ind);
  }

  for (auto daughter : currentvolume->GetDaughters()) {
    if (daughter->GetChildId() < 0) {
      return ReportIncompatibleDaughter(currentvolume, daughter);
    }
    auto child_nav_ind = nav_ind;
    NavStateIndex::PushImpl(child_nav_ind, daughter);
    reference.Push(daughter);
    auto ierr = visitAllPlacedVolumesPassNavIndex(daughter, reference, child_nav_ind);
    reference.Pop();
    if (ierr) return ierr;
  }
  return 0;
}

VECCORE_ATT_HOST_DEVICE
int visitAllPlacedVolumesPassNavTuple(VPlacedVolume const *currentvolume, ReferenceNavState &reference,
                                      NavTuple_t nav_tuple)
{
  auto validation_error = ValidateEncodedState<NavStateTuple>(reference, nav_tuple);
  if (validation_error != ReferenceNavValidationError::kNone) {
    return ReportValidationError<NavStateTuple>(validation_error, reference, nav_tuple);
  }

  for (auto daughter : currentvolume->GetDaughters()) {
    if (daughter->GetChildId() < 0) {
      return ReportIncompatibleDaughter(currentvolume, daughter);
    }
    auto child_nav_tuple = nav_tuple;
    NavStateTuple::PushImpl(child_nav_tuple, daughter);
    auto scene_error = ValidateSceneTransition<NavStateTuple>(nav_tuple, child_nav_tuple);
    if (scene_error != ReferenceNavValidationError::kNone) {
      return ReportSceneTransitionError<NavStateTuple>(nav_tuple, child_nav_tuple, currentvolume, daughter);
    }
    reference.Push(daughter);
    auto ierr = visitAllPlacedVolumesPassNavTuple(daughter, reference, child_nav_tuple);
    reference.Pop();
    if (ierr) return ierr;
  }
  return 0;
}

} // namespace visitorcxx

int TestNavIndexCPU(vecgeom::cxx::VPlacedVolume const *const world, int maxdepth)
{
  // Check performance
  using namespace visitorcxx;
  (void)maxdepth;

  Stopwatch timer;
  auto reference    = ReferenceNavState::MakeWorld(world);
  NavIndex_t navind = 1; // The navigation index corresponding to the world

  timer.Start();
#ifdef VECGEOM_USE_NAVTUPLE
  auto ierr = visitAllPlacedVolumesPassNavTuple(world, reference, NavTuple_t{navind});
#else
  auto ierr = visitAllPlacedVolumesPassNavIndex(world, reference, navind);
#endif
  auto tvalidate = timer.Stop();

  if (!ierr) std::cout << "=== Info navigation table validation on CPU took: " << tvalidate << " sec.\n";
  return ierr;
}

int main(int argc, char *argv[])
{
  OPTION_STRING(gdml_name, "default.gdml");
  OPTION_INT(max_depth, 0);
  OPTION_INT(on_gpu, 0);
#ifndef VECGEOM_GDML
  (void)max_depth;
  std::cout << "### VecGeom must be compiled with GDML support to run this.\n";
  return 1;
#endif

  Stopwatch timer;
  // Try to open the input file
#ifdef VECGEOM_GDML
  GeoManager::Instance().SetTransformationCacheDepth(max_depth);
  auto load = vgdml::Frontend::Load(gdml_name.c_str(), false);
  if (!load) return 2;
#endif

  auto world = GeoManager::Instance().GetWorld();
  if (!world) return 3;
  int maxdepth = GeoManager::Instance().getMaxDepth();

  auto ierr = 0;
  timer.Start();
  if (on_gpu) {
#ifdef VECGEOM_ENABLE_CUDA
    VECGEOM_DEVICE_API_CALL(DeviceSetLimit(VECGEOM_DEVICE_API_SYMBOL(LimitStackSize), 8192));
    ierr = TestNavIndexGPU(GeoManager::Instance().GetWorld(), maxdepth);
#else
    std::cout << "=== Cannot run the test on GPU since VecGeom CUDA support not compiled.\n";
    return 1;
#endif
  } else {
    ierr = TestNavIndexCPU(GeoManager::Instance().GetWorld(), maxdepth);
  }
  auto validation_time = timer.Stop();
  if (ierr)
    std::cout << "TestNavIndex FAILED\n";
  else {
    std::cout << "Navigation index table validation took " << validation_time << " seconds\n";
    std::cout << "TestNavIndex PASSED\n";
  }

  return ierr;
}
