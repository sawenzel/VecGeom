#include "VecGeom/base/Global.h"

#ifdef VECGEOM_GDML
#include "Frontend.h" // VecGeom/gdml/Frontend.h
#endif

#include "VecGeom/management/BVHManager.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/CudaManager.h"

#include <cstdio>
#include <cstdlib>

#include <err.h>

using namespace vecgeom;

void check_host_bvh(int id)
{
  if (auto bvh = BVHManager::GetBVH(id)) bvh->Print();
}

void check_device_bvh(int id);

int main(int argc, char **argv)
{
#ifdef VECGEOM_GDML
  bool verbose   = false;
  bool validate  = false;
  double mm_unit = 0.1;

  if (argc == 1) errx(ENOENT, "No input GDML file");

  const char *filename = argv[1];

  if (!filename || !vgdml::Frontend::Load(filename, validate, mm_unit, verbose))
    errx(EBADF, "Cannot open file '%s'", filename);

  CudaAssertError(CudaDeviceSetStackLimit(8192));
  auto &geoManager  = GeoManager::Instance();
  auto &cudaManager = CudaManager::Instance();

  if (!geoManager.IsClosed()) errx(1, "Geometry not closed");

  BVHManager::Init();

  cudaManager.LoadGeometry(geoManager.GetWorld());
  cudaManager.Synchronize();

  auto gpu_world = cudaManager.world_gpu();

  if (!gpu_world) errx(EFAULT, "Invalid world pointer on GPU: %p", gpu_world);

  BVHManager::DeviceInit();

  for (auto item : geoManager.GetLogicalVolumesMap()) {
    if (item.second->GetDaughters().size() > 0) {
      printf("Host:   ");
      check_host_bvh(item.first);
      printf("Device: ");
      check_device_bvh(item.first);
      printf("\n");
    }
  }
#endif
  return EXIT_SUCCESS;
}
