//------------------------------- -*- C++ -*- -------------------------------//
// Copyright G4VG contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TestBase.cpp
//---------------------------------------------------------------------------//
#include "TestBase.h"

#include "VecGeom/management/ABBoxManager.h"
#include "VecGeom/management/BVHManager.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/base/Assert.h"
#include "vg_test_config.h"

#ifdef VECGEOM_GDML
#include "Frontend.h"
#endif
#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/management/CudaManager.h"
#endif

#ifdef VECGEOM_ENABLE_CUDA
#define VG_CUDA_CALL(CODE) CODE
#else
#define VG_CUDA_CALL(CODE) VECGEOM_UNREACHABLE
#endif

namespace vecgeom {
namespace test {
//---------------------------------------------------------------------------//
// TestBase
//---------------------------------------------------------------------------//

/*!
 * Load Vecgeom geometry during setup.
 */
void TestBase::SetUp()
{
  auto &geo_manager = GeoManager::Instance();

  // Guard against loading multiple geometry in the same run
  static std::string loaded_basename{};
  std::string this_basename = this->GetBasename();

  if (this_basename == loaded_basename) {
    // The loaded file matches the current one; exit early
    return;
  } else if (!loaded_basename.empty()) {
    VECGEOM_LOG(info) << "Clearing geo manager due to geometry name change";
    geo_manager.Clear();
    loaded_basename.clear();
  }

  // Set the basename to a temporary value in case something goes wrong
  loaded_basename = "<FAILURE>";

  // Save world volume
  this->LoadWorld();
  ASSERT_TRUE(geo_manager.GetWorld() != nullptr) << "Failed to create world volume";

  // Set up tracking
  this->SetUpVolumeTracking();

  // Save the basename
  if (!this->HasFatalFailure()) {
    // Everything loaded successfully
    loaded_basename = this_basename;
  }
}

void TestBase::SetUpVolumeTracking()
{
  using ABBoxManager_t = ABBoxManager<Precision>;
  using BVHManager_t   = cxx::BVHManager;

  ABBoxManager_t::Instance().InitABBoxesForCompleteGeometry();
  BVHManager_t::Init();

  if (false) {
    // TODO: optional device tests
#ifdef VECGEOM_ENABLE_CUDA
    auto &cuda_manager = vecgeom::cxx::CudaManager::Instance();
    cuda_manager.LoadGeometry();
    // .. more ...
#endif
  }
}

//---------------------------------------------------------------------------//
// CustomTestBase
//---------------------------------------------------------------------------//

//! Use the GoogleTest harness to set a base name for repeable geometry
std::string CustomTestBase::GetBasename() const
{
  auto *ut = ::testing::UnitTest::GetInstance();
  VECGEOM_ASSERT(ut);
  auto *test = ut->current_test_info();
  return test->test_suite_name();
}

//! Dispatch to the custom test's load function
void CustomTestBase::LoadWorld()
{
  auto *world = this->MakeWorld();
  ASSERT_TRUE(world) << "Custom test did not create world volume";

  // Set world in VecGeom manager
  auto &vg_manager = vecgeom::GeoManager::Instance();
  vg_manager.RegisterPlacedVolume(world);
  vg_manager.SetWorldAndClose(world);
}

//---------------------------------------------------------------------------//
// GdmlTestBase
//---------------------------------------------------------------------------//

void GdmlTestBase::LoadWorld()
{
  // Construct absolute path to GDML input
  std::string filename = vecgeom_source_dir;
  filename += "/test/gdml/gdmls/";
  filename += this->GetBasename();
  filename += ".gdml";

  auto unit_system = this->GetUnitLength();
  auto mm_value    = (unit_system == UnitLength::mm ? 1.0 : unit_system == UnitLength::cm ? 0.1 : 0.0);

  VECGEOM_LOG(info) << "Loading GDML at " << filename;
#ifdef VECGEOM_GDML
  vgdml::Frontend::Load(filename,
                        /* validate_xml_schema = */ false,
                        /* mm_unit = */ mm_value,
                        /* verbose = */ false);
#else
  FAIL() << "VGDML is not enabled: cannot run test";
#endif
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace vecgeom
