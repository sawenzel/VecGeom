#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Trapezoid.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints, 1024);
  OPTION_INT(nreps, 1024);

  UnplacedBox worldUnplaced = UnplacedBox(20., 20., 20.);

  // validate construtor for input corner points -- add an xy-offset for non-zero theta,phi
  TrapCorners_t xyz;
  Precision xoffset = 9;
  Precision yoffset = -6;

  // define corner points
  // convention: p0(---); p1(+--); p2(-+-); p3(++-); p4(--+); p5(+-+); p6(-++); p7(+++)
  xyz[0] = Vector3D<Precision>( -2+xoffset, -5+yoffset, -15 );
  xyz[1] = Vector3D<Precision>(  2+xoffset, -5+yoffset, -15 );
  xyz[2] = Vector3D<Precision>( -3+xoffset,  5+yoffset, -15 );
  xyz[3] = Vector3D<Precision>(  3+xoffset,  5+yoffset, -15 );
  xyz[4] = Vector3D<Precision>( -4-xoffset,-10-yoffset,  15 );
  xyz[5] = Vector3D<Precision>(  4-xoffset,-10-yoffset,  15 );
  xyz[6] = Vector3D<Precision>( -6-xoffset, 10-yoffset,  15 );
  xyz[7] = Vector3D<Precision>(  6-xoffset, 10-yoffset,  15 );

  // create trapezoid
  UnplacedTrapezoid trapUnplaced(xyz);

//  UnplacedTrapezoid trapUnplaced2(1,0,0, 1,1,1,0, 1,1,1,0);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume trap("trap", &trapUnplaced);

  // Transformation3D *transf = new Transformation3D(5,2,3, 15,30,45);
  // world.PlaceDaughter(&trap, transf);
  world.PlaceDaughter(&trap, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nreps);
  tester.SetPoolMultiplier(1);
  tester.SetTolerance(2.e-12);

  tester.RunBenchmark();

  // cleanup
  //delete transf;

  return 0;
}
