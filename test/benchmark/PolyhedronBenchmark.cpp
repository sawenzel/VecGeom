#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Polyhedron.h"
#include "VecGeom/volumes/Tessellated.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"

#include <fstream>

using namespace vecgeom;

UnplacedPolyhedron *NoInnerRadii()
{
  constexpr int nPlanes      = 5;
  Precision zPlanes[nPlanes] = {-4, -2, 0, 2, 4};
  Precision rInner[nPlanes]  = {0, 0, 0, 0, 0};
  Precision rOuter[nPlanes]  = {2, 3, 2, 3, 2};
  return new UnplacedPolyhedron(5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron *WithInnerRadii()
{
  constexpr int nPlanes      = 5;
  Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
  Precision rInner[nPlanes]  = {1, 0.75, 0.5, 0.75, 1};
  Precision rOuter[nPlanes]  = {1.5, 1.5, 1.5, 1.5, 1.5};
  return new UnplacedPolyhedron(5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron *WithPhiSectionConvex()
{
  constexpr int nPlanes      = 5;
  Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
  Precision rInner[nPlanes]  = {1, 0.75, 0.5, 0.75, 1};
  Precision rOuter[nPlanes]  = {1.5, 1.5, 1.5, 1.5, 1.5};
  return new UnplacedPolyhedron(15 * kDegToRad, 45 * kDegToRad, 5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron *WithPhiSectionNonConvex()
{
  constexpr int nPlanes      = 5;
  Precision zPlanes[nPlanes] = {-4, -1, 0, 1, 4};
  Precision rInner[nPlanes]  = {1, 0.75, 0.5, 0.75, 1};
  Precision rOuter[nPlanes]  = {1.5, 1.5, 1.5, 1.5, 1.5};
  return new UnplacedPolyhedron(15 * kDegToRad, 340 * kDegToRad, 5, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron *ManySegments()
{
  constexpr int nPlanes      = 17;
  Precision zPlanes[nPlanes] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  Precision rInner[nPlanes]  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  Precision rOuter[nPlanes]  = {2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2};
  return new UnplacedPolyhedron(6, nPlanes, zPlanes, rInner, rOuter);
}

UnplacedPolyhedron *SameZsection()
{
  constexpr int nPlanes      = 5;
  Precision zPlanes[nPlanes] = {-2, -1, 1, 1, 2};
  Precision rInner[nPlanes]  = {0, 1, 0.5, 1, 0};
  Precision rOuter[nPlanes]  = {1, 2, 2, 2.5, 1};
  return new UnplacedPolyhedron(15 * kDegToRad, 340 * kDegToRad, 5, nPlanes, zPlanes, rInner, rOuter);
};

vecgeom::UnplacedTessellated *toTessellated(UnplacedPolyhedron const &p)
{
  auto mesh    = p.CreateMesh3D(Transformation3D(), 1);
  auto &polies = mesh->GetPolygons();
  auto tsl     = new vecgeom::UnplacedTessellated();

  for (auto &p : polies) {
    auto &v0 = mesh->GetVertices()[p.fInd[0]];
    auto &v1 = mesh->GetVertices()[p.fInd[1]];
    auto &v2 = mesh->GetVertices()[p.fInd[2]];
    tsl->AddTriangularFacet(v0, v1, v2);
  }
  tsl->Close();
  return tsl;
}

int main(int argc, char *argv[])
{

  OPTION_INT(npoints, 10000);
  OPTION_INT(nrep, 4);
  // Polyhedron type:
  //   0=NoInnerRadii
  //   1=WithInnerRadii
  //   2=WithPhiSectionConvex
  //   3=WithPhiSectionNonConvex
  //   4=ManySegments
  //   5=SameZsection
  OPTION_INT(type, 3);
  OPTION_INT(tsl, 0); // benchmark polyhedron as tessellated implementation

  UnplacedBox worldUnplaced = UnplacedBox(5, 5, 10);

  auto RunBenchmark = [&worldUnplaced, tsl](UnplacedPolyhedron const *shape, char const *label, int npoints,
                                            int nrep) -> int {
    auto vol = tsl > 0 ? static_cast<const VUnplacedVolume *>(toTessellated(*shape))
                       : static_cast<const VUnplacedVolume *>(shape);
    LogicalVolume logical("pgon", vol);
    LogicalVolume worldLogical(&worldUnplaced);
    Transformation3D transformation(0, 0, 0);
    worldLogical.PlaceDaughter("pgonplaced", &logical, &transformation);
    GeoManager::Instance().SetWorldAndClose(worldLogical.Place());

    Benchmarker benchmarker(GeoManager::Instance().GetWorld());
    benchmarker.SetVerbosity(2);
    benchmarker.SetPoolMultiplier(1);
    benchmarker.SetRepetitions(nrep);
    benchmarker.SetPointCount(npoints);
    auto errcode                       = benchmarker.RunBenchmark();
    std::list<BenchmarkResult> results = benchmarker.PopResults();
    std::ofstream outStream;
    outStream.open(label, std::fstream::app);
    BenchmarkResult::WriteCsvHeader(outStream);
    for (auto i = results.begin(), iEnd = results.end(); i != iEnd; ++i) {
      i->WriteToCsv(outStream);
    }
    outStream.close();
    return errcode;
  };

  std::cout << "________________________________________________________________________________\n";
  switch (type) {
  case 0:
    std::cout << "Testing NoInnerRadii with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(NoInnerRadii(), "polyhedron_no-inner-radii.csv", npoints, nrep);
  case 1:
    std::cout << "Testing WithInnerRadii with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(WithInnerRadii(), "polyhedron_with-inner-radii.csv", npoints, nrep);
  case 2:
    std::cout << "Testing WithPhiSectionConvex with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(WithPhiSectionConvex(), "polyhedron_phi-section-conv.csv", npoints, nrep);
  case 3:
    std::cout << "Testing WithPhiSectionNonConvex with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(WithPhiSectionNonConvex(), "polyhedron_phi-section-non-conv.csv", npoints, nrep);
  case 4:
    std::cout << "Testing ManySegments with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(ManySegments(), "polyhedron_many-segments.csv", npoints, nrep);
  case 5:
    std::cout << "Testing SameZsection with npoints = " << npoints << " nrep = " << nrep << std::endl;
    std::cout << "________________________________________________________________________________\n";
    return RunBenchmark(SameZsection(), "polyhedron_sameZsection.csv", npoints, nrep);
  default:
    std::cout << "Unknown polyhedron type." << std::endl;
  }
  return 1;
}
