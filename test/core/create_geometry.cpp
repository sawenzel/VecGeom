#include <stdio.h>

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Version.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/management/CudaManager.h"
#endif
#include "VecGeom/navigation/GlobalLocator.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Box.h"

using namespace vecgeom;

int main()
{
  std::cout << "VecGeom version " << vecgeom_version << std::endl;
  static_assert(VECGEOM_VERSION >= 0x020000, "Version is too old");

  // Vector3D<Precision> vec1(5, 3, 1);
  // Vector3D<Precision> vec2(2, 8, 0);
  // std::cout << (vec1 < vec2) << "\n";

  UnplacedBox world_params    = UnplacedBox(4., 4., 4.);
  UnplacedBox largebox_params = UnplacedBox(1.5, 1.5, 1.5);
  UnplacedBox smallbox_params = UnplacedBox(0.5, 0.5, 0.5);

  LogicalVolume worldl(&world_params);

  LogicalVolume largebox("Large box", &largebox_params);
  LogicalVolume smallbox("Small box", &smallbox_params);

  Transformation3D origin     = Transformation3D();
  Transformation3D placement1 = Transformation3D(2, 2, 2);
  Transformation3D placement2 = Transformation3D(-2, 2, 2);
  Transformation3D placement3 = Transformation3D(2, -2, 2);
  Transformation3D placement4 = Transformation3D(2, 2, -2);
  Transformation3D placement5 = Transformation3D(-2, -2, 2);
  Transformation3D placement6 = Transformation3D(-2, 2, -2);
  Transformation3D placement7 = Transformation3D(2, -2, -2);
  Transformation3D placement8 = Transformation3D(-2, -2, -2);

  largebox.PlaceDaughter(&smallbox, &origin);
  worldl.PlaceDaughter(&largebox, &placement1);
  worldl.PlaceDaughter(&largebox, &placement2);
  worldl.PlaceDaughter(&largebox, &placement3);
  worldl.PlaceDaughter(&largebox, &placement4);
  worldl.PlaceDaughter("Hello the world!", &largebox, &placement5);
  worldl.PlaceDaughter(&largebox, &placement6);
  worldl.PlaceDaughter(&largebox, &placement7);
  worldl.PlaceDaughter(&largebox, &placement8);

  VPlacedVolume *world_placed = worldl.Place();
  GeoManager::Instance().SetWorld(world_placed);
  GeoManager::Instance().CloseGeometry();

  std::cerr << "Printing world content:\n";
  world_placed->PrintContent();

  Vector3D<Precision> point(2, 2, 2);
  NavigationState *path = NavigationState::MakeInstance(4);
  GlobalLocator::LocateGlobalPoint(world_placed, point, *path, true);
  path->Print();

  GeoManager::Instance().FindLogicalVolume("Large box");
  GeoManager::Instance().FindPlacedVolume("Large box");

  NavigationState::ReleaseInstance(path);

  return 0;
}
