#include "VecGeom/base/FlatVoxelHashMap.h"
#include "VecGeom/base/Vector3D.h"

#include <gtest/gtest.h>

using namespace vecgeom;

class VoxelHashMapTest : public ::testing::Test {
protected:
  static constexpr int Nx = 10, Ny = 10, Nz = 10;
  const Vector3D<float> lower{-5, -5, -5};
  const Vector3D<float> dim{10, 10, 10};
  const Vector3D<float> p1{-4.9f, -4.9f, -4.9f};
  const Vector3D<float> p2{-4.85f, -4.85f, -4.85f};
  const Vector3D<float> p3{4.99f, 4.99f, 4.99f};
};

TEST_F(VoxelHashMapTest, GeneralVersion)
{
  // a voxel structure mapping to vector of ints per voxel
  FlatVoxelHashMap<int> voxels(lower, dim, Nx, Ny, Nz);

  EXPECT_EQ(voxels.getVoxelKey(p1), 0);
  EXPECT_EQ(voxels.getVoxelKey(p3), Nx * Ny * Nz - 1);

  int length{0};
  EXPECT_FALSE(voxels.isOccupied(p1));
  EXPECT_EQ(voxels.getProperties(p1, length), nullptr);
  EXPECT_EQ(length, 0);

  voxels.addProperty(p1, 111);
  voxels.addProperty(p1, 112);
  EXPECT_TRUE(voxels.isOccupied(p1));
  EXPECT_NE(voxels.getProperties(p1, length), nullptr);
  EXPECT_EQ(length, 2);
  auto props = voxels.getProperties(p1, length);
  EXPECT_EQ(props[0], 111);
  EXPECT_EQ(props[1], 112);

  // nearby point in same voxel
  EXPECT_TRUE(voxels.isOccupied(p2));
  EXPECT_NE(voxels.getProperties(p2, length), nullptr);
}

TEST_F(VoxelHashMapTest, ScalarVersion)
{
  // a voxel structure mapping to a single int per voxel
  FlatVoxelHashMap<int, true> voxels(lower, dim, Nx, Ny, Nz);

  EXPECT_EQ(voxels.getVoxelKey(p1), 0);
  EXPECT_EQ(voxels.getVoxelKey(p3), Nx * Ny * Nz - 1);

  int length{0};
  EXPECT_FALSE(voxels.isOccupied(p1));
  EXPECT_EQ(voxels.getProperties(p1, length), nullptr);
  EXPECT_EQ(length, 0);

  voxels.addProperty(p1, 111);
  EXPECT_TRUE(voxels.isOccupied(p1));
  EXPECT_NE(voxels.getProperties(p1, length), nullptr);
  EXPECT_EQ(length, 1);
  auto props = voxels.getProperties(p1, length);
  EXPECT_EQ(props[0], 111);

  // nearby point in same voxel
  EXPECT_TRUE(voxels.isOccupied(p2));
  EXPECT_NE(voxels.getProperties(p2, length), nullptr);
}
