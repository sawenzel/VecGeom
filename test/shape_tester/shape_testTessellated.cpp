#ifndef VECGEOM_ENABLE_CUDA

#include "test/benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/Tessellated.h"
#include "test/core/TessellatedOrb.h"

using namespace vecgeom;
using Tessellated_t = vecgeom::SimpleTessellated;
#endif

int main(int argc, char *argv[])
{
#ifndef VECGEOM_ENABLE_CUDA
  if (argc == 1) {
    std::cout << "Usage: shape_testTessellated <-ngrid N>\n"
                 "       N - number of theta/phi segments for the sphere\n";
  }
  OPTION_INT(npoints, 1000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(ngrid, 50);
  constexpr double r = 10.;

  Tessellated_t *solid     = 0;
  solid                    = new Tessellated_t("test_VecGeomTessellated");
  UnplacedTessellated *tsl = (UnplacedTessellated *)solid->GetUnplacedVolume();
  size_t nfacets           = TessellatedOrb(r, ngrid, *tsl);
  std::cout << "Testing tessellated sphere with ngrid = " << ngrid << " (nfacets=" << nfacets << ")\n";
  tsl->Close();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetTestBoundaryErrors(false);
  #ifdef VECGEOM_SINGLE_PRECISION
    tester.SetSolidTolerance(1e-5);
  #endif
  int errCode = tester.Run(solid);

  std::cout << "Final Error count for Shape *** " << solid->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================" << std::endl;

  if (solid) delete solid;
#endif
  return 0;
}
