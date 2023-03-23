#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/navigation/SimpleSafetyEstimator.h>
#include <VecGeom/surfaces/BrepHelper.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/Navigator.h>
#include <VecGeom/volumes/Box.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/Tube.h>
#include <VecGeom/volumes/Trd.h>

using namespace vecgeom;
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

  GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

NavStateIndex Locate(Precision x, Precision y, Precision z)
{
  Vector3D<Precision> pos(x, y, z);
  NavStateIndex state;
  vgbrep::protonav::LocatePointIn(GeoManager::Instance().GetWorld(), pos, state, true);
  return state;
}

static void TestHost()
{
  Vector3D<Precision> pos(0, 0, 0);
  Vector3D<Precision> dir(1, 1, 1);
  dir.Normalize();

  NavStateIndex state = Locate(pos.x(), pos.y(), pos.z());
  NavStateIndex out;

  auto *nav = NewSimpleNavigator<>::Instance();
  vecgeom::Precision distance, safety;
  nav->FindNextBoundaryAndStep(pos, dir, state, out, kInfLength, distance);
  safety = SimpleSafetyEstimator::Instance()->ComputeSafety(pos, state);
  printf("VecGeom: distance = %f, safety = %f\n", distance, safety);

  int exit = 0;
  distance = vgbrep::protonav::ComputeStepAndHit(pos, dir, state, out, exit);
  safety   = vgbrep::protonav::ComputeSafety(pos, state, exit);
  printf("HOST: distance = %f, safety = %f\n", distance, safety);
}

// In testCUDA.cu
void TestCUDA(const SurfData &surfData);

int main(int argc, char *argv[])
{
  CreateVecGeomWorld();

  if (!BrepHelper::Instance().CreateLocalSurfaces()) return 1;
  if (!BrepHelper::Instance().CreateCommonSurfacesFlatTop()) return 2;

  const SurfData &surfData = BrepHelper::Instance().GetSurfData();

  // Transfer geometry, needed to get the NavStateIndices...
  auto &cudaManager = vecgeom::CudaManager::Instance();
  cudaManager.LoadGeometry(GeoManager::Instance().GetWorld());
  cudaManager.Synchronize();

  TestHost();
  TestCUDA(surfData);

  BrepHelper::Instance().ClearData();

  return 0;
}
