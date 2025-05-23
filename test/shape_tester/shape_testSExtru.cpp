#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/SExtru.h"

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_DOUBLE(grazing, 0.);

  constexpr size_t N = 20;
  vecgeom::Precision x[N], y[N];
  for (size_t i = 0; i < N; ++i) {
    x[i] = 4 * std::sin(i * (2. * M_PI) / N);
    y[i] = 4 * std::cos(i * (2. * M_PI) / N);
  }

  auto volume = new vecgeom::SimpleSExtru("test_VecGeomSExtru", N, x, y, -5., 10.);
  volume->Print();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  tester.SetGrazingTolerance(grazing);
  tester.SetErrorOnZeroDoutGrazing(true);

  int errCode = tester.Run(volume);

  std::cout << "Final Error count for Shape *** " << volume->GetName() << "*** = " << errCode << "\n";
  std::cout << "=========================================================\n";

  if (volume) delete volume;
  return 0;
}
