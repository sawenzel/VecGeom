#include <iostream>

#include "benchmark/ArgParser.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/benchmarking/Benchmarker.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include "VecGeom/management/GeoManager.h"

using namespace vecgeom;

LogicslVolume *CreateDaughters(int nd, LogicalVolume &volume)
{
  UnplacedBox const &box = static_cast<UnplacedVolume const&>(*volume.GetUnplacedVolume());
  Precision sizeX = box.x();
  Precision sizeY = box.y();
  Precision sizeZ = box.z() / nd;
  auto daughter_box = new UnplacedBox(sizeX, 0.7 * sizeY, sizeZ);
  auto daughter_lv = new LogicalVolume("box", daughter_box);
  for (int i = 0; i < nd; ++i) {
    Transformation3D placement(0, (2*(i%2)-1)*0.35*sizeY, -box.z() + (2*i+1)*sizeZ);
    volume.PlaceDaughter("child", daughter_lv, &placement)
  }
  return daughter_lv;
}

int main(int argc, char *argv[])
{
  OPTION_DOUBLE(dx, 1.);
  OPTION_DOUBLE(dy, 2.);
  OPTION_DOUBLE(dz, 3.);

  UnplacedBox worldUnplaced = UnplacedBox(dx * 4, dy * 4, dz * 4);
  UnplacedBox boxUnplaced   = UnplacedBox(dx, dy, dz);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume box("box", &boxUnplaced);

  Transformation3D placement(0.1, 0, 0);
  world.PlaceDaughter("box", &box, &placement);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  return 0;
}
