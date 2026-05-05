#include "test/benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/Extruded.h"
#include "test/core/TessellatedOrb.h"

using namespace vecgeom;

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_BOOL(convex, false);

  LogicalVolume vol("xtru", ExtrudedMultiLayer(convex));
  VPlacedVolume *placed = vol.Place();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(false);
  tester.SetCheckConventions(false);
#ifdef VECGEOM_SINGLE_PRECISION
  tester.SetSolidTolerance(1.e-4);
#endif
  int errCode = tester.Run(placed);

  std::cout << "Final Error count for Shape *** " << placed->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  return 0;
}
