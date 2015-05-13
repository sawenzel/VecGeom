#include "navigation/NavigationState.h"
#include "navigation/NavStatePool.h"
#include "base/Global.h"
#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#ifdef VECGEOM_NVCC
#include "management/CudaManager.h"
#endif
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
  RootGeoManager::Instance().LoadRootGeometry("ExN03.root");
#ifdef VECGEOM_CUDA
  CudaManager::Instance().set_verbose(3);
  CudaManager::Instance().LoadGeometry();

  // why do I have to do this here??
  CudaManager::Instance().Synchronize();

  CudaManager::Instance().PrintGeometry();

  std::cout << std::flush;
#endif

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
  for(unsigned int i=0;i<testpoints.size();++i){
   //     std::cerr << testpoints[i] << "\n";
      nav.LocatePoint(GeoManager::Instance().GetWorld(), testpoints[i], *pool[i], true);
  }
  pool.Print();
  std::cerr << "sizeof navigation state on CPU: " << sizeof(vecgeom::cxx::NavigationState) <<" and "<< sizeof(vecgeom::NavigationState) << " bytes/state\n";

#ifdef VECGEOM_CUDA
   pool.CopyToGpu();
#endif

  // launch some kernel on GPU using the
#ifdef VECGEOM_CUDA
  printf("TestNavStatePool: calling LaunchNavigationKernel()...\n");
  LaunchNavigationKernel( pool.GetGPUPointer(), GeoManager::Instance().getMaxDepth(), npoints );
  printf("TestNavStatePool: waiting for CudaDeviceSynchronize()...\n");
  cudaDeviceSynchronize();
  printf("TestNavStatePool: synchronized!\n");
#endif

// #ifdef VECGEOM_CUDA
//   pool.CopyFromGpu();
// #endif

  pool.Print();
}


