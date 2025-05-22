#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/Box.h"

using VPlacedVolume = vecgeom::VPlacedVolume;
using VGBox         = vecgeom::SimpleBox;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat, double grazing_tolerance);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  OPTION_DOUBLE(dx, 10.);
  OPTION_DOUBLE(dy, 15.);
  OPTION_DOUBLE(dz, 20.);
  OPTION_DOUBLE(grazing, 0.);

  auto box = new VGBox("vecgeomBox", dx, dy, dz);
  box->Print();
  return runTester<VPlacedVolume>(box, npoints, debug, stat, grazing);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat, double grazing_tolerance)
{

  ShapeTester<ImplT> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetGrazingTolerance(grazing_tolerance);
  tester.SetErrorOnZeroDoutGrazing(true);
#ifdef VECGEOM_SINGLE_PRECISION
  tester.SetSolidTolerance(1.e-4);
#endif
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << "\n";
  std::cout << "=========================================================\n";

  if (shape) delete shape;
  return errcode;
}
