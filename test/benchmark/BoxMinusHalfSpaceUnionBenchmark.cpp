/// @file BoxMinusHalfSpaceUnionBenchmark.cpp
/// @brief Benchmark a finite Boolean subtraction containing intermediate half-space CSG.

// Geant4 has no half-space solid, so keep this benchmark to VecGeom/CUDA and
// ROOT comparisons when those backends are enabled.
#undef VECGEOM_GEANT4

#include "ArgParser.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/BooleanVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/HalfSpace.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeomTest/Benchmarker.h"

using namespace vecgeom;

void BuildBoxMinusHalfSpace(LogicalVolume &world)
{
  auto *hostBox   = new UnplacedBox(10., 10., 10.);
  auto *halfspace = new UnplacedHalfSpace(Vector3D<Precision>(0., 0., 0.), Vector3D<Precision>(0., 1., 1.));

  auto *hostLogical      = new LogicalVolume("root_supported_host", hostBox);
  auto *halfspaceLogical = new LogicalVolume("root_supported_halfspace", halfspace);
  Transformation3D hostTransform(0.25, -0.5, 0.5, 0., 0., 12.);
  auto *placedHost      = hostLogical->Place(&hostTransform);
  auto *placedHalfSpace = halfspaceLogical->Place();

  auto *booleanSolid   = new UnplacedBooleanVolume<kSubtraction>(kSubtraction, placedHost, placedHalfSpace);
  auto *booleanLogical = new LogicalVolume("box_minus_halfspace", booleanSolid);
  world.PlaceDaughter(booleanLogical, &Transformation3D::kIdentity);
}

void BuildBoxMinusHalfSpaceUnion(LogicalVolume &world)
{
  auto *hostBox   = new UnplacedBox(10., 10., 10.);
  auto *tube      = new GenericUnplacedTube(0., 3., 10., 0., kTwoPi);
  auto *halfspace = new UnplacedHalfSpace(Vector3D<Precision>(0.5, -0.5, 0.), Vector3D<Precision>(0., 1., 1.));

  auto *hostLogical      = new LogicalVolume("host", hostBox);
  auto *tubeLogical      = new LogicalVolume("tube", tube);
  auto *halfspaceLogical = new LogicalVolume("halfspace", halfspace);
  auto *placedHost       = hostLogical->Place();
  Transformation3D tubeTransform(0.5, -0.5, 0., 0., 25., 20.);
  auto *placedTube      = tubeLogical->Place(&tubeTransform);
  auto *placedHalfSpace = halfspaceLogical->Place();

  auto *hostCut        = new UnplacedBooleanVolume<kSubtraction>(kSubtraction, placedHost, placedHalfSpace);
  auto *hostCutLogical = new LogicalVolume("host_cut", hostCut);
  auto *placedHostCut  = hostCutLogical->Place();

  auto *booleanSolid   = new UnplacedBooleanVolume<kSubtraction>(kSubtraction, placedHostCut, placedTube);
  auto *booleanLogical = new LogicalVolume("box_minus_halfspace_union", booleanSolid);
  world.PlaceDaughter(booleanLogical, &Transformation3D::kIdentity);
}

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 1024);
  OPTION_STRING(case_name, "nested_union");

  UnplacedBox worldUnplaced(15., 15., 15.);
  LogicalVolume world("world", &worldUnplaced);

  if (case_name == "box_minus_halfspace") {
    BuildBoxMinusHalfSpace(world);
  } else {
    BuildBoxMinusHalfSpaceUnion(world);
  }

  GeoManager::Instance().SetWorldAndClose(world.Place());

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPoolMultiplier(1);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);
  return tester.RunBenchmark();
}
