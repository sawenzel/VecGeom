// A program to create a very simple (double nested) assembly
#include "TGeoVolume.h"
#include "TGeoTube.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "VecGeom/base/Assert.h"
#include <iostream>

// global constants defining the arrangement
int gNumberOfModulesPerRow = 4;
int gNumberOfTubesPerRow   = 4;

TGeoVolumeAssembly *MakeElementaryAssembly(int tubesperrow, double Lx, double Lz)
{
  // Lx is half length of "virtual" assembly box
  double tuberadius     = Lx / tubesperrow;
  double tubehalflength = 0.99 * Lz;

  TGeoVolumeAssembly *ass = new TGeoVolumeAssembly("INNERASS");
  TGeoTube *t             = new TGeoTube(0., tuberadius, tubehalflength);
  TGeoVolume *v           = new TGeoVolume("WIRE", t);

  size_t counter = 0;
  for (size_t x = 0; x < (size_t)tubesperrow; ++x) {
    double cx = -Lx + tuberadius + x * (2 * tuberadius);
    for (size_t y = 0; y < (size_t)tubesperrow; ++y) {
      double cy = -Lx + tuberadius + y * (2 * tuberadius);
      ass->AddNode(v, counter, new TGeoTranslation(cx, cy, 0));
      counter++;
    }
  }
  return ass;
}

TGeoVolume *MakeTopAssembly(int modulesperrow, double worldL)
{
  if (modulesperrow < 1) return nullptr;
  double Delta = worldL / (modulesperrow * 10.); // small space between modules
  double BoxL  = (worldL * 2 - (modulesperrow + 1) * Delta) / (2. * modulesperrow);

  TGeoVolumeAssembly *ass        = new TGeoVolumeAssembly("OUTERASS");
  TGeoVolumeAssembly *elementary = MakeElementaryAssembly(gNumberOfTubesPerRow, BoxL, worldL);

  size_t counter = 0;
  double cx(0.);
  double cy(0.);
  for (size_t x = 0; x < (size_t)modulesperrow; ++x) {
    cx = -worldL + (Delta + BoxL) + x * (2 * BoxL + Delta);
    for (size_t y = 0; y < (size_t)modulesperrow; ++y) {
      cy = -worldL + (Delta + BoxL) + y * (2 * BoxL + Delta);
      ass->AddNode(elementary, counter, new TGeoTranslation(cx, cy, 0));
      counter++;
    }
  }
  VECGEOM_ASSERT(std::abs(cx + BoxL + Delta - worldL) < 1E-6);
  VECGEOM_ASSERT(std::abs(cy + BoxL + Delta - worldL) < 1E-6);

  return ass;
}

// check a couple of simple things
__attribute__((noinline)) void test1(TGeoVolume *v)
{
  double testpoint[3] = {0, 0, 0};
  std::cerr << v->Contains(testpoint) << "\n";
}

// check a couple of simple things
__attribute__((noinline)) void test2()
{
  double testpoint[3] = {0, 0, 0};
  gGeoManager->FindNode(testpoint[0], testpoint[1], testpoint[2]);
  std::cerr << gGeoManager->GetPath() << "\n";

  double testpoint2[3] = {-30, -30, 0};
  gGeoManager->FindNode(testpoint2[0], testpoint2[1], testpoint2[2]);
  std::cerr << gGeoManager->GetPath() << "\n";
}

int main(int argc, char *argv[])
{
  if (argc >= 3) {
    gNumberOfModulesPerRow = atoi(argv[1]);
    gNumberOfTubesPerRow   = atoi(argv[2]);
  }

  std::cout << "CREATING ASSEMBLY " << gNumberOfModulesPerRow << " x " << gNumberOfTubesPerRow << "\n";

  double worldL     = 100.;
  auto *topAssembly = MakeTopAssembly(gNumberOfModulesPerRow, worldL);
  TGeoVolume *top   = gGeoManager->MakeBox("TOP", nullptr, worldL, worldL, worldL);
  top->AddNode(topAssembly, 0, new TGeoIdentity());
  gGeoManager->SetTopVolume(top);
  gGeoManager->CloseGeometry();
  gGeoManager->Export("assemblytest.root");

  // this crashes unfortunately
  // gGeoManager->Export("assemblytest.gdml");

  // test1(topAssembly);
  //  test1(gGeoManager->FindVolumeFast("INNERASS"));
  test2();
  return 0;
}
