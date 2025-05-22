#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/Tube.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGTube        = vecgeom::SimpleTube;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat, double grazing_tolerance);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  OPTION_DOUBLE(dz, 1.);
  OPTION_DOUBLE(rmax, 6.);
  OPTION_DOUBLE(rmin, 2.);
  OPTION_DOUBLE(sphi, 0.);
  OPTION_DOUBLE(dphi, vecgeom::kTwoPi);
  OPTION_DOUBLE(grazing, 0.);

  auto tube = new VGTube("vecgeomTube", dz, rmax, rmin, sphi, dphi);
  tube->Print();
  return runTester<VPlacedVolume>(tube, npoints, debug, stat, grazing);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat, double grazing_tolerance)
{
  ShapeTester<ImplT> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetSolidTolerance(vecgeom::kHalfTolerance);
  tester.SetGrazingTolerance(grazing_tolerance);
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << "\n";
  std::cout << "=========================================================\n";
  if (shape) delete shape;
  return errcode;
}
