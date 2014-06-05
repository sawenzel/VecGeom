#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Trapezoid.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

using namespace vecgeom;

int main() {

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
  UnplacedTrapezoid trapUnplaced = UnplacedTrapezoid(xyz);

//  UnplacedTrapezoid trapUnplaced2 = UnplacedTrapezoid(1,0,0, 1,1,1,0, 1,1,1,0);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume trap = LogicalVolume("trap", &trapUnplaced);

  // world.PlaceDaughter(&trap, new Transformation3D(5,6,7));
  world.PlaceDaughter(&trap, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetPointCount(1<<13);

  tester.RunBenchmark();

  // // if running BenchmarkerGL tester()
  // tester.RunInsideDebug();

  return 0;
}
