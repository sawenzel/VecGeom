#include "volumes/LogicalVolume.h"
#include "volumes/Torus.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"
#include "base/Vector3D.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints);
  OPTION_INT(nrep);
  OPTION_DOUBLE(drmin);
  OPTION_DOUBLE(drmax);
  OPTION_DOUBLE(drtor);

  UnplacedBox worldUnplaced = UnplacedBox((drtor+drmax)*2, (drtor+drmax)*2 , (drtor+drmax)*2);
  UnplacedTorus torusUnplaced = UnplacedTorus(drmin, drmax, drtor, 0, 2.*M_PI);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume torus = LogicalVolume("torus", &torusUnplaced);

  Transformation3D placement(5, 5, 5);
  world.PlaceDaughter("torus", &torus, &placement);

  VPlacedVolume *pt = torus.Place(&placement);

  Vector3D<Precision> pos(5.853854, -2.729679, 2.817193);
  Vector3D<Precision> dir(-0.808899, -0.314328, -0.496871);

  pt->DistanceToOut( pos, dir, kInfinity );
  return 1;


  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.RunInsideBenchmark();

  tester.RunToOutBenchmark();

  return 0;
}
