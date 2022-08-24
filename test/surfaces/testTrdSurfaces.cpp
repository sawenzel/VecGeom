#include <iostream>
#include <fstream>
#include <string>

#include <VecGeom/volumes/Trd.h>
#include "BenchmarkSurfaceModel.h"

using BrepHelper = vgbrep::BrepHelper<vecgeom::Precision>;
using namespace vecgeom;

// Forward declarations of generation functions
void CreateSimpleTrd(double, double, double, double, double, double, double, double);
void CreateTwoNestedTrds(double, double, double, double, double, double, double, double, double);
void CreateConcatenatedTrds(double, double, double, double, double, double, double);
void CreateComplexGeometry(int, double, double, double, double, double, double, double);

int main(int argc, char *argv[])
{
  //
  // Testing options
  //
  OPTION_INT(generate, 0);
  OPTION_INT(test, 0);
  OPTION_INT(verbose, 0);
  // Determines the scale of generating volume relative to world:
  OPTION_DOUBLE(scale, 0.2);

  //
  // World and validation options
  //
  OPTION_DOUBLE(worldX, 40);
  OPTION_DOUBLE(worldY, 40);
  OPTION_DOUBLE(worldZ, 80);
  OPTION_INT(nvalidate, 10000);

  //
  // Volumes options
  //
  OPTION_DOUBLE(dx1, 20);
  OPTION_DOUBLE(dx2, 18);
  OPTION_DOUBLE(dy1, 20);
  OPTION_DOUBLE(dy2, 18);
  OPTION_DOUBLE(dz, 40);
  OPTION_INT(layers, 50);

  //
  // Benchmarking options:
  //
  OPTION_INT(nbench, 1000000);

  switch (generate) {
  case 0:
    CreateComplexGeometry(layers, worldX, worldY, worldZ, dx1, dy1, dy2, dz);
    break;
  case 1:
    CreateSimpleTrd(worldX, worldY, worldZ, dx1, dx2, dy1, dy2, dz);
    break;
  case 2:
    CreateTwoNestedTrds(worldX, worldY, worldZ, dx1, dx2, dy1, dy2, dz, 0.5 * dz);
    break;
  case 3:
    CreateTwoNestedTrds(worldX, worldY, worldZ, dx1, dx2, dy1, dy2, dz, dz);
    break;
  case 4:
    CreateConcatenatedTrds(worldX, worldY, worldZ, dx1, dy1, dy2, dz);
    break;
  }

  BrepHelper::Instance().SetVerbosity(verbose);

  if (!BrepHelper::Instance().CreateLocalSurfaces()) return 1;
  if (!BrepHelper::Instance().CreateCommonSurfacesFlatTop()) return 2;

  switch (test) {
  case 0:
    ValidateNavigation(nvalidate, BrepHelper::Instance().GetSurfData(), worldX, worldY, worldZ, scale);
    break;
  case 1:
    ShootOneParticle(0, -5, 1, 0, 1, 0, BrepHelper::Instance().GetSurfData());
    break;
  case 2:
    ValidateNavigation(nvalidate, BrepHelper::Instance().GetSurfData(), worldX, worldY, worldZ, scale);
    TestPerformance(worldX, worldY, worldZ, scale, nbench, layers, BrepHelper::Instance().GetSurfData());
    break;
  }
}

void CreateSimpleTrd(double worldX, double worldY, double worldZ, double dx1, double dx2, double dy1, double dy2,
                     double dz)
{
  auto worldSolid                     = new vecgeom::UnplacedBox(worldX, worldY, worldZ);
  auto worldLogic                     = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  auto const trdSolid = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTrd>(dx1, dx2, dy1, dy2, dz);
  auto trdLogic       = new vecgeom::LogicalVolume("Trd", trdSolid);
  vecgeom::Transformation3D origin;
  worldLogic->PlaceDaughter(trdLogic, &origin);

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

void CreateTwoNestedTrds(double worldX, double worldY, double worldZ, double dx1, double dx2, double dy1, double dy2,
                         double dz1, double dz2)
{
  // This test currently workds only if UP<DOWN
  auto worldSolid                     = new vecgeom::UnplacedBox(worldX, worldY, worldZ);
  auto worldLogic                     = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  auto const trdSolid = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTrd>(dx1, dx2, dy1, dy2, dz1);
  auto trdLogic       = new vecgeom::LogicalVolume("Trd", trdSolid);
  vecgeom::Transformation3D origin;
  worldLogic->PlaceDaughter(trdLogic, &origin);

  auto ratioUp                = (dz1 - dz2) / dz1 * 0.5;
  auto ratioDown              = (dz1 + dz2) / dz1 * 0.5;
  auto const trdDaughterSolid = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTrd>(
      dx2 + ratioDown * (dx1 - dx2), dx2 + ratioUp * (dx1 - dx2), dy2 + ratioDown * (dy1 - dy2),
      dy2 + ratioUp * (dy1 - dy2), dz2);
  auto trdDaughterLogic = new vecgeom::LogicalVolume("TrdDaughter", trdDaughterSolid);
  trdLogic->PlaceDaughter(trdDaughterLogic, &origin);

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

void CreateConcatenatedTrds(double worldX, double worldY, double worldZ, double dx1, double dy1, double dy2, double dz)
{
  auto worldSolid                     = new vecgeom::UnplacedBox(worldX, worldY, worldZ);
  auto worldLogic                     = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  auto const trdSolid1 = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTrd>(dx1, dx1, dy1, dy2, dz);
  auto trdLogic1       = new vecgeom::LogicalVolume("Trd1", trdSolid1);
  vecgeom::Transformation3D t1{-dx1, 0, 0};
  worldLogic->PlaceDaughter(trdLogic1, &t1);

  auto const trdSolid2 = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTrd>(dx1, dx1, dy1, dy2, dz);
  auto trdLogic2       = new vecgeom::LogicalVolume("Trd2", trdSolid2);
  vecgeom::Transformation3D t2{dx1, 0, 0};
  worldLogic->PlaceDaughter(trdLogic2, &t2);

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

void CreateComplexGeometry(int NbOfLayers, double worldX, double worldY, double worldZ, double dx1, double dy1,
                           double dy2, double dz)
{
  const double CalorSizeY        = 40;
  const double CalorSizeZ        = 40;
  const double GapThickness      = 2.3;
  const double AbsorberThickness = 5.7;

  const double LayerThickness = GapThickness + AbsorberThickness;
  const double CalorThickness = NbOfLayers * LayerThickness;

  const double WorldSizeX = 1.2 * CalorThickness + 1000;
  const double WorldSizeY = 1.2 * CalorSizeY + 1000;
  const double WorldSizeZ = 1.2 * CalorSizeZ + 1000;

  auto worldSolid                     = new vecgeom::UnplacedBox(0.5 * WorldSizeX, 0.5 * WorldSizeY, 0.5 * WorldSizeZ);
  auto worldLogic                     = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  //
  // Calorimeter
  //
  auto const calorSolid = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTrd>(
      0.5 * CalorThickness, 0.5 * CalorThickness, 0.5 * CalorSizeY, 0.4 * CalorSizeY, 0.5 * CalorSizeZ);
  auto calorLogic = new vecgeom::LogicalVolume("Calorimeter", calorSolid);
  vecgeom::Transformation3D origin;
  worldLogic->PlaceDaughter(calorLogic, &origin);

  //
  // Layers
  //
  auto layerSolid = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTrd>(
      0.5 * LayerThickness, 0.5 * LayerThickness, 0.5 * CalorSizeY, 0.4 * CalorSizeY, 0.5 * CalorSizeZ);

  //
  // Absorbers
  //
  auto gapSolid = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTrd>(
      0.5 * GapThickness, 0.5 * GapThickness, 0.5 * CalorSizeY, 0.4 * CalorSizeY, 0.5 * CalorSizeZ);
  auto gapLogic = new vecgeom::LogicalVolume("Gap", gapSolid);
  vecgeom::Transformation3D gapPlacement(-0.5 * LayerThickness + 0.5 * GapThickness, 0, 0);

  auto absorberSolid = vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedTrd>(
      0.5 * AbsorberThickness, 0.5 * AbsorberThickness, 0.5 * CalorSizeY, 0.4 * CalorSizeY, 0.5 * CalorSizeZ);
  auto absorberLogic = new vecgeom::LogicalVolume("Absorber", absorberSolid);
  vecgeom::Transformation3D absorberPlacement(0.5 * LayerThickness - 0.5 * AbsorberThickness, 0, 0);

  double xCenter = -0.5 * CalorThickness + 0.5 * LayerThickness;
  for (int i = 0; i < NbOfLayers; i++) {
    std::string name("Layer_");
    name += std::to_string(i);
    auto layerLogic = new vecgeom::LogicalVolume("Layer", layerSolid);
    vecgeom::Transformation3D placement(xCenter, 0, 0);
    calorLogic->PlaceDaughter(name.c_str(), layerLogic, &placement);

    layerLogic->PlaceDaughter(gapLogic, &gapPlacement);
    layerLogic->PlaceDaughter(absorberLogic, &absorberPlacement);

    xCenter += LayerThickness;
  }

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}
