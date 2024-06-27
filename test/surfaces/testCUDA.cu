#include <VecGeom/surfaces/cuda/BrepCudaManager.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/Navigator.h>
#ifdef VECGEOM_CUDA_INTERFACE
#include <VecGeom/management/CudaManager.h>
#endif

using namespace vecgeom;
using BrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using SurfData        = vgbrep::SurfData<vecgeom::Precision>;

static __global__ void Test(Vector3D<Precision> pos, Vector3D<Precision> dir, NavigationState const state)
{
  vgbrep::CrossedSurface crossed_surf;
  NavigationState out;
  vecgeom::Precision distance = vgbrep::protonav::ComputeStepAndHit(pos, dir, state, out, crossed_surf);
  int common_id               = crossed_surf.hit_surf.GetCSindex();
  vecgeom::Precision safety   = vgbrep::protonav::ComputeSafety(pos, state, common_id);
  printf("Surf@DEVICE: distance = %f, safety = %f\n", distance, safety);
}

// In testCUDA.cpp
NavigationState Locate(Precision x, Precision y, Precision z);

void TestCUDA(const SurfData &surfData, Precision px, Precision py, Precision pz, Precision dx, Precision dy,
              Precision dz)
{
  Vector3D<Precision> pos(px, py, pz);
  Vector3D<Precision> dir(dx, dy, dz);
  dir.Normalize();

  BrepCudaManager::Instance().TransferSurfData(surfData);

  // Locate the point on the host; has to be in testCUDA.cpp because we
  // need the world volume and use the vecgeom::cxx namespace...
  NavigationState state = Locate(pos.x(), pos.y(), pos.z());

  Test<<<1, 1>>>(pos, dir, state);
  BREP_CUDA_CHECK(cudaDeviceSynchronize());

  BrepCudaManager::Instance().Cleanup();
}
