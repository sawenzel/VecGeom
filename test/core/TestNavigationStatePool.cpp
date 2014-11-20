#include "navigation/NavigationState.h"
#include "navigation/NavStatePool.h"
#include "base/Global.h"
#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#include "management/CudaManager.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "navigation/SimpleNavigator.h"

#include <iostream>
using namespace vecgeom;


#ifdef VECGEOM_CUDA
// declaration of some external "user-space" kernel
extern void LaunchNavigationKernel( void* gpu_ptr, int depth, int );
#endif



int main()
{
  // Load a geometry
  RootGeoManager::Instance().LoadRootGeometry("Ex03.root");
  CudaManager::Instance().set_verbose(3);
  CudaManager::Instance().LoadGeometry();

  // why to I have to do this??
  CudaManager::Instance().Synchronize();

  CudaManager::Instance().PrintGeometry();

  std::cout << std::flush;
  
   // generate some points
  int npoints=3;
  SOA3D<Precision> testpoints(npoints);
  //testpoints.reserve(npoints)
  //testpoints.resize(npoints);

  // generate some points in the world
  volumeUtilities::FillContainedPoints(*GeoManager::Instance().GetWorld(), testpoints, false);

  // generat
  SimpleNavigator nav;

  NavStatePool pool(npoints, GeoManager::Instance().getMaxDepth() );
  pool.Print();
  std::cerr << "#################" << std::endl;
  
  // fill states
  for(int i=0;i<testpoints.size();++i){
   //     std::cerr << testpoints[i] << "\n";
      nav.LocatePoint(GeoManager::Instance().GetWorld(), testpoints[i], *pool[i], true);
  }
  pool.Print();

  pool.CopyToGpu();

  // launch some kernel on GPU using the
  std::cerr << "sizeof navigation state on CPU " << sizeof(vecgeom::NavigationState) << "\n";
#ifdef VECGEOM_CUDA
  LaunchNavigationKernel( pool.GetGPUPointer(), GeoManager::Instance().getMaxDepth(), npoints );
#endif

  pool.CopyFromGpu();

  pool.Print();
}


