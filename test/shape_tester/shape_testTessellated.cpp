#include "test/benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/Tessellated.h"
#include "test/core/TessellatedOrb.h"

using namespace vecgeom;
using Tessellated_t = vecgeom::SimpleTessellated;

int main(int argc, char *argv[])
{
  if (argc == 1) {
    std::cout << "Usage: shape_testTessellated <-ngrid N>\n"
                 "       N - number of theta/phi segments for the sphere\n";
  }
  OPTION_INT(npoints, 1000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_STRING(obj, {""}); // tessellated from object file
  OPTION_INT(ngrid, 50);
  constexpr double r = 10.;

  Tessellated_t *solid     = 0;
  solid                    = new Tessellated_t("test_VecGeomTessellated");
  UnplacedTessellated *tsl = (UnplacedTessellated *)solid->GetUnplacedVolume();
  if (obj.size() > 0) {
    // fill from OBJ file and close
    size_t nfacets = tsl->FillFromObjFile(obj, true);
    if (!nfacets) {
      std::cerr << "No facets found in obj file " << obj << "\n";
      return 1;
    }
    std::cout << "Testing tessellated from OBJ file " << obj << " (nfacets=" << nfacets << ")\n";
  } else {
    size_t nfacets = TessellatedOrb(r, ngrid, *tsl);
    std::cout << "Testing tessellated sphere with ngrid = " << ngrid << " (nfacets=" << nfacets << ")\n";
    tsl->Close();
  }

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.SetCheckConventions(false); // detecting wrong side is difficult/costly for safety --> not followed on purpose
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
  return 0;
}
