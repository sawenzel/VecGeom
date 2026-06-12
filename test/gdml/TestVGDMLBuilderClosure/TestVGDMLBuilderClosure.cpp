//!    \file TestVGDMLBuilderClosure.cpp
//!    \brief Regression tests for GDML shapes requiring explicit finalization

#include "Frontend.h"

#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedMultiUnion.h"
#include "VecGeom/volumes/UnplacedTessellated.h"
#include "vg_test_config.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using vecgeom::GeoManager;
using vecgeom::UnplacedMultiUnion;
using vecgeom::UnplacedTessellated;

std::string GdmlFilename(std::string const &basename)
{
  std::string filename = vecgeom_source_dir;
  filename += "/test/gdml/gdmls/";
  filename += basename;
  filename += ".gdml";
  return filename;
}

class VGDMLBuilderClosure : public ::testing::Test {
protected:
  void SetUp() override { GeoManager::Instance().Clear(); }
  void TearDown() override { GeoManager::Instance().Clear(); }

  vecgeom::VUnplacedVolume const *LoadWorldUnplaced(std::string const &filename)
  {
    constexpr bool validate_xml_schema = false;
    constexpr double mm_unit           = 0.1;
    constexpr bool verbose             = false;
    if (!vgdml::Frontend::Load(filename, validate_xml_schema, mm_unit, verbose)) {
      return nullptr;
    }

    auto const *world = GeoManager::Instance().GetWorld();
    return world ? world->GetUnplacedVolume() : nullptr;
  }
};

TEST_F(VGDMLBuilderClosure, TessellatedSolidIsClosed)
{
  auto const filename  = GdmlFilename("oneTesselatedSolid");
  auto const *unplaced = LoadWorldUnplaced(filename);
  ASSERT_NE(unplaced, nullptr) << "failed to load tessellated GDML world from " << filename;

  auto const *tessellated = dynamic_cast<UnplacedTessellated const *>(unplaced);
  ASSERT_NE(tessellated, nullptr) << "world is not an UnplacedTessellated";

  auto const &runtime = tessellated->GetStruct();
  EXPECT_GT(runtime.fNFacets, 0);
  EXPECT_NE(runtime.fFacets, nullptr);
  EXPECT_NE(runtime.fBVH, nullptr);
}

TEST_F(VGDMLBuilderClosure, MultiUnionIsClosed)
{
  auto const filename  = GdmlFilename("solidMultiUnion");
  auto const *unplaced = LoadWorldUnplaced(filename);
  ASSERT_NE(unplaced, nullptr) << "failed to load multiunion GDML world from " << filename;

  auto const *multiunion = dynamic_cast<UnplacedMultiUnion const *>(unplaced);
  ASSERT_NE(multiunion, nullptr) << "world is not an UnplacedMultiUnion";

  auto const &structure = multiunion->GetStruct();
  EXPECT_GT(structure.fVolumes.size(), 0);
  EXPECT_NE(structure.fNavHelper, nullptr);
  EXPECT_NE(structure.fNeighbours, nullptr);
  EXPECT_NE(structure.fNneighbours, nullptr);
}

} // namespace
