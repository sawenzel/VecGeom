#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "volumes/Box.h"
#include "volumes/Tube.h"

#include <fstream>
#include <list>

using namespace vecgeom;

VPlacedVolume* SetupBoxGeometry();
// VPlacedVolume* SetupParallelepipedGeometry();
VPlacedVolume* SetupTubeGeometry();

int main() {
  Benchmarker benchmarker = Benchmarker();
  benchmarker.SetRepetitions(1<<10);
  benchmarker.SetVerbosity(1);

  std::ofstream outStream;

  auto runAndWrite = [&] (char const *const fileName) {
    for (int i = 1; i < 13; ++i) {
      benchmarker.SetPointCount(2<<i);
      benchmarker.RunBenchmark();
    }
    std::list<BenchmarkResult> results = benchmarker.PopResults();
    outStream.open(fileName, std::fstream::app);
    BenchmarkResult::WriteCsvHeader(outStream);
    for (auto i = results.begin(), i_end = results.end(); i != i_end; ++i) {
      i->WriteToCsv(outStream);
    }
    outStream.close();
  };

  benchmarker.SetWorld(SetupBoxGeometry());
  runAndWrite("box.csv");

  benchmarker.SetWorld(SetupTubeGeometry());
  runAndWrite("tube.csv");

  return 0;
}

VPlacedVolume* SetupBoxGeometry() {
  UnplacedBox *worldUnplaced = new UnplacedBox(10, 10, 10);
  UnplacedBox *boxUnplaced = new UnplacedBox(2.5, 2.5, 2.5);
  Transformation3D *placement1 = new Transformation3D( 2,  2,  2,  0,  0,  0);
  Transformation3D *placement2 = new Transformation3D(-2,  2,  2, 45,  0,  0);
  Transformation3D *placement3 = new Transformation3D( 2, -2,  2,  0, 45,  0);
  Transformation3D *placement4 = new Transformation3D( 2,  2, -2,  0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-2, -2,  2, 45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-2,  2, -2, 45,  0, 45);
  Transformation3D *placement7 = new Transformation3D( 2, -2, -2,  0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-2, -2, -2, 45, 45, 45);
  LogicalVolume *world = new LogicalVolume(worldUnplaced);
  LogicalVolume *box = new LogicalVolume(boxUnplaced);
  world->PlaceDaughter(box, placement1);
  world->PlaceDaughter(box, placement2);
  world->PlaceDaughter(box, placement3);
  world->PlaceDaughter(box, placement4);
  world->PlaceDaughter(box, placement5);
  world->PlaceDaughter(box, placement6);
  world->PlaceDaughter(box, placement7);
  world->PlaceDaughter(box, placement8);
  return world->Place();
}

VPlacedVolume* SetupTubeGeometry() {
  UnplacedBox *worldUnplaced = new UnplacedBox(10, 10, 10);
  UnplacedTube *tube1Unplaced = new UnplacedTube(0, 2.5, 2.5, 0, 2*kPi);
  UnplacedTube *tube2Unplaced = new UnplacedTube(1.5, 2.5, 2.5, 0, 2*kPi);
  UnplacedTube *tube3Unplaced = new UnplacedTube(0, 2.5, 2.5, 0, 1.5*kPi);
  Transformation3D *placement1 = new Transformation3D( 2,  2,  2,  0,  0,  0);
  Transformation3D *placement2 = new Transformation3D(-2,  2,  2, 45,  0,  0);
  Transformation3D *placement3 = new Transformation3D( 2, -2,  2,  0, 45,  0);
  Transformation3D *placement4 = new Transformation3D( 2,  2, -2,  0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-2, -2,  2, 45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-2,  2, -2, 45,  0, 45);
  Transformation3D *placement7 = new Transformation3D( 2, -2, -2,  0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-2, -2, -2, 45, 45, 45);
  LogicalVolume *world = new LogicalVolume(worldUnplaced);
  LogicalVolume *tube1 = new LogicalVolume(tube1Unplaced);
  LogicalVolume *tube2 = new LogicalVolume(tube2Unplaced);
  LogicalVolume *tube3 = new LogicalVolume(tube3Unplaced);
  world->PlaceDaughter(tube1, placement1);
  world->PlaceDaughter(tube2, placement2);
  world->PlaceDaughter(tube3, placement3);
  world->PlaceDaughter(tube1, placement4);
  world->PlaceDaughter(tube2, placement5);
  world->PlaceDaughter(tube3, placement6);
  world->PlaceDaughter(tube1, placement7);
  world->PlaceDaughter(tube2, placement8);
  return world->Place();
}
