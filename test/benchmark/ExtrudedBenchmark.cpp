#include "VecGeom/base/Config.h"

// #ifndef VECGEOM_ENABLE_CUDA

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include "VecGeom/base/Stopwatch.h"
#include <iostream>
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Extruded.h"
#include "VecGeom/volumes/SExtru.h"
#include "VecGeom/management/GeoManager.h"

using namespace vecgeom;

// #endif

int main(int argc, char *argv[])
{
  // #ifndef VECGEOM_ENABLE_CUDA
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_INT(nvert, 8);
  OPTION_INT(nsect, 2);
  OPTION_BOOL(convex, 't');
  OPTION_BOOL(tsl, 't');

  Precision rmin = 10.;
  Precision rmax = 20.;

  vecgeom::XtruVertex2 *vertices = new vecgeom::XtruVertex2[nvert];
  vecgeom::XtruSection *sections = new vecgeom::XtruSection[nsect];
  Precision *x                   = new Precision[nvert];
  Precision *y                   = new Precision[nvert];

  Precision phi = 2. * kPi / nvert;
  Precision r;
  for (int i = 0; i < nvert; ++i) {
    r = rmax;
    if (i % 2 > 0 && !convex) r = rmin;
    vertices[i].x = r * vecCore::math::Cos(i * phi);
    vertices[i].y = r * vecCore::math::Sin(i * phi);
    x[i]          = vertices[i].x;
    y[i]          = vertices[i].y;
  }
  for (int i = 0; i < nsect; ++i) {
    sections[i].fOrigin.Set(0, 0, -20. + i * 40. / (nsect - 1));
    sections[i].fScale = 1;
  }

  Transformation3D placement(0, 0, 0);
  UnplacedBox worldUnplaced = UnplacedBox(30., 30., 30.);
  LogicalVolume world("world", &worldUnplaced);

  UnplacedSExtruVolume sxtruv(nvert, x, y, -20, 20);
  LogicalVolume sxtruVol("xtru", &sxtruv);
  if (!tsl) {
    world.PlaceDaughter("extruded", &sxtruVol, &placement);
    std::cout << "Benchmarking simple extruded polygon (SExtru) having " << nvert << " vertices and " << nsect
              << " sections\n";
  }

  // UnplacedExtruded xtru(nvert, vertices, nsect, sections);
  auto xtru = GeoManager::MakeInstance<UnplacedExtruded>(nvert, vertices, nsect, sections);
  // LogicalVolume xtruVol("xtru", &xtru);
  LogicalVolume xtruVol("xtru", xtru);

  if (tsl) {
    // world.PlaceDaughter("extruded", &xtruVol, &placement);
    world.PlaceDaughter(&xtruVol, &placement);
    std::cout << "Benchmarking extruded polygon having " << nvert << " vertices and " << nsect << " sections\n";
  }

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(2);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetToInBias(0.8);
  tester.SetPoolMultiplier(1);
  delete[] x;
  delete[] y;

  //  tester.RunToInBenchmark();
  //  tester.RunToOutBenchmark();
  return tester.RunBenchmark();
  // #else
  //   return 0;
  // #endif
}
