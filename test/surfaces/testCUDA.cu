#include <VecGeom/surfaces/cuda/BrepCudaManager.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/Navigator.h>

using namespace vecgeom;
using BrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using SurfData = vgbrep::SurfData<vecgeom::Precision>;

static __global__ void Test(Vector3D<Precision> pos, Vector3D<Precision> dir, NavigationState const state)
{
  int exit = 0;
  NavigationState out;
  vecgeom::Precision distance = vgbrep::protonav::ComputeStepAndHit(pos, dir, state, out, exit);
  vecgeom::Precision safety = vgbrep::protonav::ComputeSafety(pos, state, exit);

  printf("DEVICE: distance = %f, safety = %f\n", distance, safety);
}

// In testCUDA.cpp
NavigationState Locate(Precision x, Precision y, Precision z);

void TestCUDA(const SurfData &surfData)
{
  BrepCudaManager::Instance().TransferSurfData(surfData);

  Vector3D<Precision> pos(0, 0, 0);
  Vector3D<Precision> dir(1, 1, 1);
  dir.Normalize();

  // Locate the point on the host; has to be in testCUDA.cpp because we
  // need the world volume and use the vecgeom::cxx namespace...
  NavigationState state = Locate(pos.x(), pos.y(), pos.z());

  Test<<<1, 1>>>(pos, dir, state);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());

  BrepCudaManager::Instance().Cleanup();
}
