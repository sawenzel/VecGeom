#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/navigation/LoopNavigator.h>
#include <VecGeom/navigation/SimpleSafetyEstimator.h>
#include <VecGeom/surfaces/BrepHelper.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/Navigator.h>
#include <VecGeom/volumes/Box.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/Tube.h>
#include <VecGeom/volumes/Trd.h>

#include "test/benchmark/ArgParser.h"
#ifdef VECGEOM_GDML
#include <persistency/gdml/source/include/Frontend.h>
#endif

using namespace vecgeom;
using Vec3D      = vecgeom::Vector3D<vecgeom::Precision>;
using BrepHelper = vgbrep::BrepHelper<vecgeom::Precision>;
using SurfData   = vgbrep::SurfData<vecgeom::Precision>;

static void CreateVecGeomWorld()
{
  static constexpr double WorldSize = 10;
  auto worldSolid                   = new UnplacedBox(WorldSize, WorldSize, WorldSize);
  auto worldLogic                   = new LogicalVolume("World", worldSolid);
  auto worldPlaced                  = worldLogic->Place();

  static constexpr double BoxSize = 1;
  auto boxSolid                   = new UnplacedBox(BoxSize, BoxSize, BoxSize);
  auto boxLogic                   = new LogicalVolume("Box", boxSolid);
  Transformation3D boxPlacement(5, 5, 5);
  worldLogic->PlaceDaughter(boxLogic, &boxPlacement);

  static constexpr double TubeRmin = 1;
  static constexpr double TubeRmax = 2;
  static constexpr double TubeZ    = 2;
  static constexpr double TubeSPhi = kPi / 4;
  static constexpr double TubeDPhi = 7 * kPi / 4;
  auto tubeSolid                   = new GenericUnplacedTube(TubeRmin, TubeRmax, TubeZ, TubeSPhi, TubeDPhi);
  auto tubeLogic                   = new LogicalVolume("Tube", tubeSolid);
  Transformation3D tubePlacement(5, 5, -5);
  worldLogic->PlaceDaughter(tubeLogic, &tubePlacement);

  static constexpr double TrdX1 = 1;
  static constexpr double TrdX2 = 2;
  static constexpr double TrdY1 = 2;
  static constexpr double TrdY2 = 1;
  static constexpr double TrdZ  = 1.5;
  auto trdSolid                 = new GenericUnplacedTrd(TrdX1, TrdX2, TrdY1, TrdY2, TrdZ);
  auto trdLogic                 = new LogicalVolume("Trd", trdSolid);
  Transformation3D trdPlacement(5, -5, 5);
  worldLogic->PlaceDaughter(trdLogic, &trdPlacement);

  // add a tetrahedron
  Vector3D<double> v0(0, 0, 2);
  Vector3D<double> v1(-1, 1, -2);
  Vector3D<double> v2(-1, -1, -2);
  Vector3D<double> v3(1, -1, -2);
  auto tetSolid = new UnplacedTet(v0, v1, v2, v3);
  auto tetLogic = new LogicalVolume("Tet", tetSolid);
  Transformation3D tetPlacement(-5, 5, 5);
  worldLogic->PlaceDaughter(tetLogic, &tetPlacement);

  GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

[[maybe_unused]] static void CreateVecGeomWorldFromGDML()
{
  GeoManager::Instance().SetMinPerScene(1000);
  auto load = vgdml::Frontend::Load("default.gdml", false, 1.0);
  if (!load) {
    std::cerr << "Error loading GDML file" << std::endl;
    return;
  }

  auto world = GeoManager::Instance().GetWorld();
  if (!world) {
    std::cerr << "Error getting world from GDML" << std::endl;
    return;
  }
  vecgeom::cxx::BVHManager::Init();
}

NavigationState Locate(Precision x, Precision y, Precision z)
{
  Vector3D<Precision> pos(x, y, z);
  NavigationState state;
  vgbrep::protonav::LocatePointIn(GeoManager::Instance().GetWorld(), pos, state, true);
  return state;
}

static void TestHost(Vector3D<Precision> pos, Vector3D<Precision> dir)
{
  dir.Normalize();

  NavigationState state = Locate(pos.x(), pos.y(), pos.z());
  NavigationState out;

  vecgeom::Precision distance, dist1, safety;
  auto kPush = 1000 * vecgeom::kToleranceDist<vecgeom::Precision>;
  auto *nav  = NewSimpleNavigator<>::Instance();
  nav->FindNextBoundaryAndStep(pos, dir, state, out, kInfLength, distance);
  dist1  = LoopNavigator::ComputeStepAndPropagatedState(pos, dir, kInfLength, state, out, kPush);
  safety = SimpleSafetyEstimator::Instance()->ComputeSafety(pos, state);
  printf("VecGeom (NewSimp, LoopNav): dists = %f %f, safety = %f\n", distance, dist1, safety);

  ExitSurfState exit;
  distance = vgbrep::protonav::ComputeStepAndHit(pos, dir, state, out, exit);
  safety   = vgbrep::protonav::ComputeSafety(pos, state, exit.common_id);
  printf("surf@HOST: distance = %f, safety = %f\n", distance, safety);
}

// In testCUDA.cu
void TestCUDA(const SurfData &surfData, Precision px, Precision py, Precision pz, Precision dx, Precision dy,
              Precision dz);

int main(int argc, char *argv[])
{
  std::vector<double> zero = {1, 1, 1};
  OPTION_VECTOR(pos, zero);
  OPTION_VECTOR(dir, zero);
  OPTION_STRING(gdml_name, "");
  assert(pos.size() == 3 && dir.size() == 3);
  // transform to Vec3D for further handling
  Vec3D vpos = {pos[0], pos[1], pos[2]};
  Vec3D vdir = {dir[0], dir[1], dir[2]};
  GeoManager::Instance().SetMinPerScene(1000);

  if (gdml_name.length() > 0) {
    printf("Using GDML file: %s\n", gdml_name.c_str());
    CreateVecGeomWorldFromGDML();
  } else {
    printf("Using programmatic geometry...\n");
    CreateVecGeomWorld();
  }
  if (!BrepHelper::Instance().CreateLocalSurfaces()) return 1;
  if (!BrepHelper::Instance().CreateCommonSurfacesScenes()) return 2;

  const SurfData &surfData = BrepHelper::Instance().GetSurfData();
  TestHost(vpos, vdir);

  // Transfer geometry, needed to get the NavStateIndices...
  CudaAssertError(CudaDeviceSetStackLimit(8192));
  auto &cudaManager = vecgeom::CudaManager::Instance();
  cudaManager.LoadGeometry(GeoManager::Instance().GetWorld());
  cudaManager.Synchronize();

  TestCUDA(surfData, vpos.x(), vpos.y(), vpos.z(), vdir.x(), vdir.y(), vdir.z());

  BrepHelper::Instance().ClearData();
  return 0;
}
