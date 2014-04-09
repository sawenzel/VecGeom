#include "cpugpu.h"

#include <string>
#include "base/stopwatch.h"
#include "management/cuda_manager.h"
#include "navigation/navigationstate.h"
#include "navigation/simple_navigator.h"
#include "volumes/box.h"
#include "volumes/utilities/volume_utilities.h"
#include "TGeoManager.h"

using namespace vecgeom;

void CreateRootGeometry();
// void LocatePointsGpu(Precision *const x, Precision *const y,
//                      Precision *const z, const unsigned size, const int depth,
//                      int *const output);

int main(const int argc, char const *const *const argv) {

  int n = 0;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    int value = 0;
    try {
      value = std::stoi(arg);
      n = value;
    } catch (std::invalid_argument err) {}
  }
  if (n == 0) {
    std::cerr << "No particles argument received. Exiting.\n";
    return -1;
  }

  CreateRootGeometry();

  RootGeoManager::Instance().LoadRootGeometry();
  CudaManager::Instance().LoadGeometry();
  CudaManager::Instance().Synchronize();
  CudaManager::Instance().PrintGeometry();

  const int depth = 4;

  SOA3D<Precision> points(n);
  volumeutilities::FillRandomPoints(*GeoManager::Instance().world(), points);
  int *const results = new int[n]; 
  int *const results_gpu = new int[n]; 

  std::cout << "Running for " << n << " points...\n";

  SimpleNavigator navigator;
  Stopwatch sw;
  sw.Start();
  for (int i = 0; i < n; ++i) {
    NavigationState path(depth);
    results[i] =
        navigator.LocatePoint(GeoManager::Instance().world(), points[i], path,
                              true)->id();
  }
  const double cpu = sw.Stop();
  std::cout << "Points located on CPU in " << cpu << "s.\n";

  sw.Start();
  // LocatePointsGpu(&points.x(0), &points.y(0), &points.z(0), n, depth,
  //                 results_gpu);
  const double gpu = sw.Stop();
  std::cout << "Points located on GPU in " << gpu
            << "s (including memory transfer).\n";

  // Compare output
  for (int i = 0; i < n; ++i) {
    std::cout << results[i] << " vs. " << results_gpu[i] << std::endl;
    assert(results[i] == results_gpu[i]);
  }
  std::cout << "All points located within same volume on CPU and GPU.\n";

  return 0;
}

void CreateRootGeometry() {
  double L = 10.;
  double Lz = 10.;
  const double Sqrt2 = sqrt(2.);
  TGeoVolume * world =  ::gGeoManager->MakeBox("worldl",0, L, L, Lz );
  TGeoVolume * boxlevel2 = ::gGeoManager->MakeBox("b2l",0, Sqrt2*L/2./2., Sqrt2*L/2./2., Lz );
  TGeoVolume * boxlevel3 = ::gGeoManager->MakeBox("b3l",0, L/2./2., L/2./2., Lz);
  TGeoVolume * boxlevel1 = ::gGeoManager->MakeBox("b1l",0, L/2., L/2., Lz );
  // TGeoVolume * test = ::gGeoManager->MakeCone("unimplemented test", NULL, .1,
                                              // .2, 5, 2, 4);

  boxlevel2->AddNode( boxlevel3, 0, new TGeoRotation("rot1",0,0,45));
  boxlevel1->AddNode( boxlevel2, 0, new TGeoRotation("rot2",0,0,-45));
  world->AddNode(boxlevel1, 0, new TGeoTranslation(-L/2.,0,0));
  world->AddNode(boxlevel1, 1, new TGeoTranslation(+L/2.,0,0));
  // world->AddNode(test, 2, new TGeoTranslation(0,0,0));
  ::gGeoManager->SetTopVolume(world);
  ::gGeoManager->CloseGeometry();
}