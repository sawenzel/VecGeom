#include "volumes/LogicalVolume.h"
#include "volumes/Polyhedron.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

#include <fstream>

using namespace vecgeom;

UnplacedPolyhedron NoInnerRadii() {
  constexpr int nPlanes = 5;
  Precision zPlanes[nPlanes] = {-4, -2, 0, 2, 4};
  Precision rInner[nPlanes] = {0, 0, 0, 0, 0};
  Precision rOuter[nPlanes] = {2, 3, 2, 3, 2};
  return UnplacedPolyhedron(5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron WithInnerRadii() {
 constexpr int nPlanes = 5;
 Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
 Precision rInner[nPlanes] = {1, 0.75, 0.5, 0.75, 1};
 Precision rOuter[nPlanes] = {1.5, 1.5, 1.5, 1.5, 1.5};
 return UnplacedPolyhedron(5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron WithPhiSection() {
 constexpr int nPlanes = 5;
 Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
 Precision rInner[nPlanes] = {1, 0.75, 0.5, 0.75, 1};
 Precision rOuter[nPlanes] = {1.5, 1.5, 1.5, 1.5, 1.5};
 return UnplacedPolyhedron(15, 45, 5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron ManySegments() {
  constexpr int nPlanes = 17;
  Precision zPlanes[nPlanes] =
      {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  Precision rInner[nPlanes] =
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  Precision rOuter[nPlanes] =
      {2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2};
  return UnplacedPolyhedron(6, nPlanes, zPlanes, rInner, rOuter);
}

int main(int nArgs, char **args) {

  UnplacedBox worldUnplaced = UnplacedBox(10, 10, 10);

  auto RunBenchmark = [&worldUnplaced] (
      UnplacedPolyhedron shape,
      char const *label) {
    LogicalVolume logical(&shape);
    VPlacedVolume *placed = logical.Place();
    LogicalVolume worldLogical(&worldUnplaced);
    worldLogical.PlaceDaughter(placed);
    Transformation3D transformation(5, 5, 5);
    VPlacedVolume *world = worldLogical.Place(&transformation);
    Benchmarker benchmarker(world);
    benchmarker.SetVerbosity(3);
    benchmarker.SetPoolMultiplier(1);
    benchmarker.SetRepetitions(8192);
    benchmarker.SetPointCount(128);
    benchmarker.RunInsideBenchmark();
    benchmarker.RunToInBenchmark();
    benchmarker.RunToOutBenchmark();
    std::list<BenchmarkResult> results = benchmarker.PopResults();
    std::ofstream outStream;
    outStream.open(label, std::fstream::app);
    BenchmarkResult::WriteCsvHeader(outStream);
    for (auto i = results.begin(), iEnd = results.end(); i != iEnd; ++i) {
      i->WriteToCsv(outStream);
    }
    outStream.close();
  };

  RunBenchmark(NoInnerRadii(), "polyhedron_no-inner-radii.csv");
  // RunBenchmark(WithInnerRadii(), "polyhedron_with-inner-radii.csv");
  // RunBenchmark(ManySegments(), "polyhedron_many-segments.csv");
  // RunBenchmark(WithPhiSection(), "polyhedron_phi-section.csv");

  return 0;
}
