#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "benchmarking/distance_to_in.h"
#include "management/rootgeo_manager.h"
#include "management/geo_manager.h"
#include "TGeoManager.h"

using namespace vecgeom;

void CreateRootGeometry();

int main() {

  CreateRootGeometry();
  RootGeoManager::Instance().LoadRootGeometry();

  DistanceToInBenchmarker tester(GeoManager::Instance().world());
  tester.set_verbose(2);
  tester.set_repetitions(4096);
  tester.set_n_points(1<<13);
  tester.BenchmarkAll();

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
  boxlevel2->AddNode( boxlevel3, 0, new TGeoRotation("rot1",0,0,45));
  boxlevel1->AddNode( boxlevel2, 0, new TGeoRotation("rot2",0,0,-45));
  world->AddNode(boxlevel1, 0, new TGeoTranslation(-L/2.,0,0));
  world->AddNode(boxlevel1, 1, new TGeoTranslation(+L/2.,0,0));
  ::gGeoManager->SetTopVolume(world);
  ::gGeoManager->CloseGeometry();
}