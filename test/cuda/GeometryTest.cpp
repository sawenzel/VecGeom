// Author: Stephan Hageboeck, CERN, 2021

/** \file GeometryTest.cpp
 * This test validates a given geometry on the GPU.
 * - A geometry is read from a GDML file passed as argument.
 * - This geometry is constructed both for CPU and GPU.
 * - It is subsequently visited on the GPU, while information like volume IDs and transformations are recorded in a
 * large array.
 * - This array is copied to the host, and the host geometry is visited, comparing the data coming from the GPU.
 *
 * Which data are recorded and how they are compared is completely controlled by the struct GeometryInfo.
 */

#include "GeometryTest.h"

#include "VecGeom/base/Assert.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/CudaManager.h"
#include "test/benchmark/ArgParser.h"

#ifdef VECGEOM_GDML
#include "Frontend.h" // VecGeom/gdml/Frontend.h
#endif

#include <algorithm>
#include <err.h>
#include <cstring>
#include <exception>

using namespace vecgeom;

void visitVolumes(const VPlacedVolume *volume, GeometryInfo *geoData, std::size_t &volCounter,
                  const std::size_t nGeoData, unsigned int depth)
{
  VECGEOM_ASSERT(volCounter < nGeoData);
  geoData[volCounter++] = GeometryInfo{depth, *volume};

  for (const VPlacedVolume *daughter : volume->GetDaughters()) {
    visitVolumes(daughter, geoData, volCounter, nGeoData, depth + 1);
  }
}

void compareGeometries(const cxx::VPlacedVolume *hostVolume, std::size_t &volumeCounter,
                       const std::vector<GeometryInfo> &deviceGeometry, unsigned int depth)
{
  if (volumeCounter >= deviceGeometry.size()) {
    errx(3, "No device volume corresponds to volume %zu", volumeCounter);
  }

  GeometryInfo host{depth, *hostVolume};
  GeometryInfo const &device = deviceGeometry[volumeCounter];
  if (!(host == device)) {
    printf("\n** CPU:\n");
    host.print();
    printf("** GPU:\n");
    device.print();

    errx(4, "Volume #%zu (id=%d label=%s logicalId=%d) differs from GPU volume (id=%d logicalId=%d)\n", volumeCounter,
         hostVolume->id(), hostVolume->GetLabel().c_str(), hostVolume->GetLogicalVolume()->id(), device.id,
         device.logicalId);
  }
  volumeCounter++;

  for (const cxx::VPlacedVolume *daughter : hostVolume->GetDaughters()) {
    compareGeometries(daughter, volumeCounter, deviceGeometry, depth + 1);
  }
}

std::size_t MaxDeviceVisitStackCapacity(const cxx::VPlacedVolume *hostVolume)
{
  std::vector<const cxx::VPlacedVolume *> stack;
  stack.reserve(1024);
  stack.push_back(hostVolume);

  std::size_t maxStackSize = stack.size();
  while (!stack.empty()) {
    auto const *current = stack.back();
    stack.pop_back();

    auto const &daughters = current->GetDaughters();
    for (int i = static_cast<int>(daughters.size()) - 1; i >= 0; --i) {
      stack.push_back(daughters[i]);
    }
    maxStackSize = std::max(maxStackSize, stack.size());
  }
  return maxStackSize;
}

int main(int argc, char **argv)
{
#ifdef VECGEOM_GDML
  OPTION_INT(verbosity, 0);
  OPTION_INT(stacksize, 8192);
  OPTION_INT(heapsize, 8388608);
  OPTION_BOOL(validate, false);
  OPTION_BOOL(timing, false);
  double mm_unit = 0.1;

  if (argc == 1)
    errx(ENOENT, "No input GDML file. \n\tUsage: ./GeometryTest <gdml> [-verbosity N] [-stacksize SS] [-heapsize HS]");

  const char *filename = argv[1];

  Stopwatch totalTimer, initTimer, loadTimer, cudaTimer, visitTimer, compareTimer;
  totalTimer.Start();
  initTimer.Start();
  loadTimer.Start();
  if (!filename || !vgdml::Frontend::Load(filename, validate, mm_unit, verbosity > 0))
    errx(EBADF, "Cannot open file '%s'", filename);
  const auto loadTime = loadTimer.Stop();

  auto &geoManager  = GeoManager::Instance();
  auto &cudaManager = CudaManager::Instance();

  // larger CUDA stack size needed for cms2018 or run3 geometries
  if (std::strstr(filename, "cms-run3.gdml") || std::strstr(filename, "cms2018.gdml")) {
    printf("Setting stack size to 8704\n");
    VECGEOM_DEVICE_API_CALL(DeviceSetLimit(VECGEOM_DEVICE_API_SYMBOL(LimitStackSize), 8 * 1024 + 512)); // default=8KB
  }

  if (std::strstr(filename, "cms-hllhc.gdml")) {
    printf("Setting heap size to 9MB\n");
    VECGEOM_DEVICE_API_CALL(
        DeviceSetLimit(VECGEOM_DEVICE_API_SYMBOL(LimitMallocHeapSize), 9 * 1024 * 1024)); // default=8MB
  }

  // user-requested stack or heap size
  if (stacksize != 8 * 1024) { // default=8KB
    printf("Setting stack size to %i\n", stacksize);
    VECGEOM_DEVICE_API_CALL(DeviceSetLimit(VECGEOM_DEVICE_API_SYMBOL(LimitStackSize), stacksize));
  }
  if (heapsize != 8 * 1024 * 1024) { // default=8MB
    printf("Setting heap size to %i\n", heapsize);
    VECGEOM_DEVICE_API_CALL(DeviceSetLimit(VECGEOM_DEVICE_API_SYMBOL(LimitMallocHeapSize), heapsize));
  }

  if (!geoManager.IsClosed()) errx(1, "Geometry not closed");

  double cudaTime = 0.;
  double initTime = 0.;
  try {
    cudaManager.set_verbose(verbosity);
    cudaTimer.Start();
    cudaManager.LoadGeometry(geoManager.GetWorld());
    cudaManager.Synchronize();
    cudaTime = cudaTimer.Stop();
    initTime = initTimer.Stop();
  } catch (std::exception const &) {
    cudaTime = cudaTimer.Stop();
    initTime = initTimer.Stop();
    if (timing) {
      printf("GEOMETRYTEST_TIMING load_close_geometry_s %.6f\n", loadTime);
      printf("GEOMETRYTEST_TIMING cuda_load_sync_s %.6f\n", cudaTime);
      printf("GEOMETRYTEST_TIMING init_including_cuda_s %.6f\n", initTime);
      printf("GEOMETRYTEST_TIMING total_s %.6f\n", totalTimer.Stop());
      fflush(stdout);
    }
    throw;
  }

  if (verbosity > 0) {
    printf("#PV known to GeoManager: %li\n", geoManager.GetPlacedVolumesCount());
    printf("#LV known to GeoManager: %li\n", geoManager.GetRegisteredVolumesCount());
    printf("# unique navig states: %li\n", geoManager.GetTotalNodeCount());
  }

  printf("Visiting device geometry ... ");
  const std::size_t numVols = geoManager.GetTotalNodeCount();
  const auto stackCapacity  = MaxDeviceVisitStackCapacity(geoManager.GetWorld());
  visitTimer.Start();
  auto deviceGeometry  = visitDeviceGeometry(cudaManager.world_gpu(), numVols, stackCapacity);
  const auto visitTime = visitTimer.Stop();

  printf("Comparing to host geometry ... ");
  std::size_t volumeCounter = 0;
  compareTimer.Start();
  compareGeometries(geoManager.GetWorld(), volumeCounter, deviceGeometry, 0);
  const auto compareTime = compareTimer.Stop();
  printf("%zu volumes. Done.\n", volumeCounter);
  const auto totalTime = totalTimer.Stop();
  if (timing) {
    printf("GEOMETRYTEST_TIMING load_close_geometry_s %.6f\n", loadTime);
    printf("GEOMETRYTEST_TIMING cuda_load_sync_s %.6f\n", cudaTime);
    printf("GEOMETRYTEST_TIMING init_including_cuda_s %.6f\n", initTime);
    printf("GEOMETRYTEST_TIMING device_visit_s %.6f\n", visitTime);
    printf("GEOMETRYTEST_TIMING host_compare_s %.6f\n", compareTime);
    printf("GEOMETRYTEST_TIMING total_s %.6f\n", totalTime);
  }
#endif
  return EXIT_SUCCESS;
}
