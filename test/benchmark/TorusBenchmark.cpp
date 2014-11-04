#include "volumes/LogicalVolume.h"
#include "volumes/Torus.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"
#include "ArgParser.h"
#include "base/Vector3D.h"
#include "base/Global.h"

using namespace vecgeom;

int main(int argc, char* argv[]) {
  OPTION_INT(npoints,1024);
  OPTION_INT(nrep,1024);
  OPTION_DOUBLE(drmin,1.2);
  OPTION_DOUBLE(drmax,3.1);
  OPTION_DOUBLE(drtor,5.);
  OPTION_DOUBLE(dsphi,0.);
  OPTION_DOUBLE(ddphi,kTwoPi);

  UnplacedBox worldUnplaced = UnplacedBox((drtor+drmax)*2, (drtor+drmax)*2 , (drtor+drmax)*2);
  UnplacedTorus torusUnplaced = UnplacedTorus(drmin, drmax, drtor, dsphi, ddphi);

  LogicalVolume world = LogicalVolume("world", &worldUnplaced);
  LogicalVolume torus = LogicalVolume("torus", &torusUnplaced);

  //Transformation3D placement(5, 5, 5);
  Transformation3D placement(0, 0, 0);
  world.PlaceDaughter("torus", &torus, &placement);

  VPlacedVolume *pt = torus.Place(&placement);

  // {
  // Vector3D<Precision> pos(5.853854, -2.729679, 2.817193);
  // Vector3D<Precision> dir(0.121540, -0.179785, -0.976169);

  // pt->DistanceToOut( pos, dir, vecgeom::kInfinity );
  // }

  // {
  //   Vector3D<Precision> pos(-8.218764, -0.095058, -3.912914);
  //   Vector3D<Precision> dir(0.946795, -0.048742, 0.318123);

  //   pt->DistanceToOut( pos, dir, vecgeom::kInfinity );
  // }

  // {
  //   SOA3D<Precision> pos(2);
  //   SOA3D<Precision> dirs(2);
  //   pos.push_back( Vector3D<Precision>(-10.612969, 2.979640, -1.546988) );
  //   pos.push_back( Vector3D<Precision>(0.208269, 10.608797, 3.425159) );

  //   dirs.push_back( Vector3D<Precision>(0.523476, -0.326030, 0.787196));
  //   dirs.push_back( Vector3D<Precision>(-0.667455, -0.602817, -0.437167));

  //   double steps[2]={vecgeom::kInfinity, vecgeom::kInfinity};
  //   double distances[2];
  //   pt->DistanceToOut( pos, dirs, steps, distances ); 

  //   pt->DistanceToOut( Vector3D<Precision>(-10.612969, 2.979640, -1.546988), Vector3D<Precision>(0.523476, -0.326030, 0.787196), vecgeom::kInfinity );

  // }

  // return 1;


  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorld(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(3);
  tester.SetTolerance(1E-8);
  tester.SetPoolMultiplier(1);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.RunInsideBenchmark();
  tester.RunToInBenchmark();
  tester.RunToOutBenchmark();

  return 0;
}
