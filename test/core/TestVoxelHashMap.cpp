#include "VecGeom/base/FlatVoxelHashMap.h"
#include "VecGeom/base/Vector3D.h"
#undef NDEBUG
#include "VecGeom/base/Assert.h"

using namespace vecgeom;

void testGeneralVersion()
{
  int Nx = 10;
  int Ny = 10;
  int Nz = 10;
  Vector3D<float> lower(-5, -5, -5);
  Vector3D<float> dim(10, 10, 10);

  // a voxel structure mapping to vector of ints per voxel
  FlatVoxelHashMap<int> voxels(lower, dim, Nx, Ny, Nz);

  Vector3D<float> p1(-4.9f, -4.9f, -4.9f);
  VECGEOM_ASSERT(voxels.getVoxelKey(p1) == 0);
  Vector3D<float> p3(4.99f, 4.99f, 4.99f);
  VECGEOM_ASSERT(voxels.getVoxelKey(p3) == Nx * Ny * Nz - 1);

  int length{0};
  VECGEOM_ASSERT(voxels.isOccupied(p1) == false);
  VECGEOM_ASSERT(voxels.getProperties(p1, length) == nullptr);
  VECGEOM_ASSERT(length == 0);

  voxels.addProperty(p1, 111);
  voxels.addProperty(p1, 112);
  VECGEOM_ASSERT(voxels.isOccupied(p1) == true);
  VECGEOM_ASSERT(voxels.getProperties(p1, length) != nullptr);
  VECGEOM_ASSERT(length == 2);
  auto props = voxels.getProperties(p1, length);
  VECGEOM_ASSERT(props[0] == 111);
  VECGEOM_ASSERT(props[1] == 112);

  // nearby point in same voxel
  Vector3D<float> p2(-4.85f, -4.85f, -4.85f);
  VECGEOM_ASSERT(voxels.isOccupied(p2) == true);
  VECGEOM_ASSERT(voxels.getProperties(p2, length) != nullptr);
}

void testScalarVersion()
{
  int Nx = 10;
  int Ny = 10;
  int Nz = 10;
  Vector3D<float> lower(-5, -5, -5);
  Vector3D<float> dim(10, 10, 10);

  // a voxel structure mapping to a single int per voxel
  FlatVoxelHashMap<int, true> voxels(lower, dim, Nx, Ny, Nz);

  Vector3D<float> p1(-4.9f, -4.9f, -4.9f);
  VECGEOM_ASSERT(voxels.getVoxelKey(p1) == 0);
  Vector3D<float> p3(4.99f, 4.99f, 4.99f);
  VECGEOM_ASSERT(voxels.getVoxelKey(p3) == Nx * Ny * Nz - 1);

  int length{0};
  VECGEOM_ASSERT(voxels.isOccupied(p1) == false);
  VECGEOM_ASSERT(voxels.getProperties(p1, length) == nullptr);
  VECGEOM_ASSERT(length == 0);

  voxels.addProperty(p1, 111);
  VECGEOM_ASSERT(voxels.isOccupied(p1) == true);
  VECGEOM_ASSERT(voxels.getProperties(p1, length) != nullptr);
  VECGEOM_ASSERT(length == 1);
  auto props = voxels.getProperties(p1, length);
  VECGEOM_ASSERT(props[0] == 111);

  // nearby point in same voxel
  Vector3D<float> p2(-4.85f, -4.85f, -4.85f);
  VECGEOM_ASSERT(voxels.isOccupied(p2) == true);
  VECGEOM_ASSERT(voxels.getProperties(p2, length) != nullptr);
}

int main()
{
  testGeneralVersion();
  testScalarVersion();
  std::cout << "test passed\n";
  return 0;
}
