#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/Hype.h"
typedef vecgeom::SimpleHype Hype_t;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);

  using vecgeom::kPi;
  auto hype = new Hype_t("test_VecGeomHype", 5., 20, kPi / 6, kPi / 3, 50);
  hype->Print();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(true);
  #ifdef VECGEOM_SINGLE_PRECISION
     tester.SetSolidTolerance(1.e-4);
     tester.SetSolidFarAway(1.e4);
  #endif
  int errCode = tester.Run(hype);

  std::cout << "Final Error count for Shape *** " << hype->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (hype) delete hype;
  return 0;
}
