#include "VecGeom/base/FlatVoxelHashMap.h"
#include "VecGeom/base/Vector3D.h"
#undef NDEBUG
#include <cassert>

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
  assert(voxels.getVoxelKey(p1) == 0);
  Vector3D<float> p3(4.99f, 4.99f, 4.99f);
  assert(voxels.getVoxelKey(p3) == Nx * Ny * Nz - 1);

  int length{0};
  assert(voxels.isOccupied(p1) == false);
  assert(voxels.getProperties(p1, length) == nullptr);
  assert(length == 0);

  voxels.addProperty(p1, 111);
  voxels.addProperty(p1, 112);
  assert(voxels.isOccupied(p1) == true);
  assert(voxels.getProperties(p1, length) != nullptr);
  assert(length == 2);
  auto props = voxels.getProperties(p1, length);
  assert(props[0] == 111);
  assert(props[1] == 112);

  // nearby point in same voxel
  Vector3D<float> p2(-4.85f, -4.85f, -4.85f);
  assert(voxels.isOccupied(p2) == true);
  assert(voxels.getProperties(p2, length) != nullptr);
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
  assert(voxels.getVoxelKey(p1) == 0);
  Vector3D<float> p3(4.99f, 4.99f, 4.99f);
  assert(voxels.getVoxelKey(p3) == Nx * Ny * Nz - 1);

  int length{0};
  assert(voxels.isOccupied(p1) == false);
  assert(voxels.getProperties(p1, length) == nullptr);
  assert(length == 0);

  voxels.addProperty(p1, 111);
  assert(voxels.isOccupied(p1) == true);
  assert(voxels.getProperties(p1, length) != nullptr);
  assert(length == 1);
  auto props = voxels.getProperties(p1, length);
  assert(props[0] == 111);

  // nearby point in same voxel
  Vector3D<float> p2(-4.85f, -4.85f, -4.85f);
  assert(voxels.isOccupied(p2) == true);
  assert(voxels.getProperties(p2, length) != nullptr);
}

int main()
{
  testGeneralVersion();
  testScalarVersion();
  std::cout << "test passed\n";
  return 0;
}
