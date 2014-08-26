#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Trapezoid.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"

using namespace vecgeom;

Precision pmax(Precision p1, Precision p2) {
    if(p1 > p2) return p1;
    return p2;
}

int main(int argc, char* argv[]) {
  OPTION_INT(npoints);
  OPTION_INT(nrep);
  OPTION_DOUBLE(dz);

  OPTION_DOUBLE(p1x);
  OPTION_DOUBLE(p2x);
  OPTION_DOUBLE(p3x);
  OPTION_DOUBLE(p4x);
  OPTION_DOUBLE(p5x);
  OPTION_DOUBLE(p6x);
  OPTION_DOUBLE(p7x);
  OPTION_DOUBLE(p8x);

  OPTION_DOUBLE(p1y);
  OPTION_DOUBLE(p2y);
  OPTION_DOUBLE(p3y);
  OPTION_DOUBLE(p4y);
  OPTION_DOUBLE(p5y);
  OPTION_DOUBLE(p6y);
  OPTION_DOUBLE(p7y);
  OPTION_DOUBLE(p8y);


  // UnplacedBox worldUnplaced = UnplacedBox(pmax(p1x, pmax(p2x, pmax(p3x, p4x)))*4, pmax(p1y, pmax(p2y, pmax(p3y, p4y)))*4, dz*4);
  UnplacedBox worldUnplaced = UnplacedBox(20., 20., 20.);
  
  // validate construtor for input corner points -- add an xy-offset for non-zero theta,phi
  TrapCorners_t xyz;
  Precision xoffset = 9;
  Precision yoffset = -6;

  // define corner points
  // convention: p0(---); p1(+--); p2(-+-); p3(++-); p4(--+); p5(+-+); p6(-++); p7(+++)
  xyz[0] = Vector3D<Precision>( p1x+xoffset, p1y+yoffset, -dz );
  xyz[1] = Vector3D<Precision>( p2x+xoffset, p2y+yoffset, -dz );
  xyz[2] = Vector3D<Precision>( p3x+xoffset, p3y+yoffset, -dz );
  xyz[3] = Vector3D<Precision>( p4x+xoffset, p4y+yoffset, -dz );
  xyz[4] = Vector3D<Precision>( p5x-xoffset, p5y-yoffset, dz );
  xyz[5] = Vector3D<Precision>( p6x-xoffset, p6y-yoffset, dz );
  xyz[6] = Vector3D<Precision>( p7x-xoffset, p7y-yoffset, dz );
  xyz[7] = Vector3D<Precision>( p8x-xoffset, p8y-yoffset, dz );

  // xyz[0] = Vector3D<Precision>( -2+xoffset, -5+yoffset, -15 );
  // xyz[1] = Vector3D<Precision>(  2+xoffset, -5+yoffset, -15 );
  // xyz[2] = Vector3D<Precision>( -3+xoffset,  5+yoffset, -15 );
  // xyz[3] = Vector3D<Precision>(  3+xoffset,  5+yoffset, -15 );
  // xyz[4] = Vector3D<Precision>( -4-xoffset,-10-yoffset,  15 );
  // xyz[5] = Vector3D<Precision>(  4-xoffset,-10-yoffset,  15 );
  // xyz[6] = Vector3D<Precision>( -6-xoffset, 10-yoffset,  15 );
  // xyz[7] = Vector3D<Precision>(  6-xoffset, 10-yoffset,  15 );
  //
  // create trapezoid
  UnplacedTrapezoid trapUnplaced(xyz);

//  UnplacedTrapezoid trapUnplaced2(1,0,0, 1,1,1,0, 1,1,1,0);

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume trap("trap", &trapUnplaced);

  Transformation3D transf(5,5,5);
  world.PlaceDaughter(&trap, &transf);
  //world.PlaceDaughter(&trap, &Transformation3D::kIdentity);

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().set_world(worldPlaced);

  Benchmarker tester(GeoManager::Instance().world());
  tester.SetVerbosity(3);
  tester.SetPointCount(npoints);
  tester.SetRepetitions(nrep);

  tester.RunBenchmark();

  // cleanup
  // delete transf;

  return 0;
}
