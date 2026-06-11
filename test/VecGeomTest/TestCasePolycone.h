#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEPOLYCONE_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEPOLYCONE_HH

#include "VecGeom/volumes/GenericPolycone.h"
#include "VecGeom/volumes/Polycone.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

inline std::unique_ptr<vecgeom::VPlacedVolume> MakePolyconeCmsLikeTestSolid()
{
  const vecgeom::Precision z[15]    = {-1520.,  -804., -804., -515.345, -515.345, -177.,  -177., 149.561,
                                       149.561, 575.,  575.,  982.812,  982.812,  1166.7, 1524.};
  const vecgeom::Precision rmin[15] = {1238., 1238., 1238., 1238., 1238., 1238., 1238.,  1238.,
                                       1238., 1238., 1238., 1238., 1238., 1238., 1455.22};
  const vecgeom::Precision rmax[15] = {1555.01, 1555.01, 1538.05, 1538.05, 1523.26, 1523.26, 1506.24, 1506.24,
                                       1488.52, 1488.52, 1471.28, 1471.28, 1455.22, 1455.22, 1455.22};
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimplePolycone("test-polycone-cms-like", 0., vecgeom::kTwoPi, 15, z, rmin, rmax));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakePolyconeHecLiquidArgonTestSolid()
{
  const vecgeom::Precision z[4]    = {0., 280.5, 280.7, 816.7};
  const vecgeom::Precision rmin[4] = {371., 371., 474., 474.};
  const vecgeom::Precision rmax[4] = {2130., 2130., 2130., 2130.};
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimplePolycone("test-polycone-hec-liquid-argon", 0., vecgeom::kTwoPi, 4, z, rmin, rmax));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeGenericPolyconeIrregularTestSolid()
{
  const int num_rz                   = 10;
  const vecgeom::Precision r[num_rz] = {1., 5., 3., 4., 9., 9., 3., 3., 2., 1.};
  const vecgeom::Precision z[num_rz] = {0., 1., 2., 3., 0., 5., 4., 3., 2., 1.};
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleGenericPolycone("test-generic-polycone-irregular", 0., vecgeom::kTwoPi, num_rz, r, z));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakePolyconeTwoSectionSharpJumpTestSolid()
{
  const vecgeom::Precision z[4]    = {-120., -20., -20., 120.};
  const vecgeom::Precision rmin[4] = {0., 0., 0., 0.};
  const vecgeom::Precision rmax[4] = {15., 15., 70., 70.};
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimplePolycone("test-polycone-two-section-sharp-jump", 0., vecgeom::kTwoPi, 4, z, rmin, rmax));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakePolyconeManySectionAlternatingTestSolid()
{
  const vecgeom::Precision z[10]    = {-150., -90., -90., -30., -30., 30., 30., 90., 90., 150.};
  const vecgeom::Precision rmin[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  const vecgeom::Precision rmax[10] = {20., 20., 65., 65., 18., 18., 72., 72., 25., 25.};
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimplePolycone("test-polycone-many-section-alternating", 0., vecgeom::kTwoPi, 10, z, rmin, rmax));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakePolyconeNearlyRepeatedZTestSolid()
{
  const vecgeom::Precision z[6]    = {-100., -5., -4.999, 40., 40.001, 100.};
  const vecgeom::Precision rmin[6] = {0., 0., 0., 0., 0., 0.};
  const vecgeom::Precision rmax[6] = {30., 30., 50., 50., 20., 20.};
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimplePolycone("test-polycone-nearly-repeated-z", 0., vecgeom::kTwoPi, 6, z, rmin, rmax));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeGenericPolyconeZigZagProfileTestSolid()
{
  const int num_rz = 8;
  // Keep the closing segment at a strictly smaller radius than every interior
  // zig-zag excursion so the RZ contour stays valid without introducing axis
  // vertices that make surface sampling noisy.
  const vecgeom::Precision r[num_rz] = {10., 20., 40., 15., 45., 18., 50., 10.};
  const vecgeom::Precision z[num_rz] = {-140., -120., -60., -20., 10., 40., 80., 140.};
  return std::unique_ptr<vecgeom::VPlacedVolume>(
      new vecgeom::SimpleGenericPolycone("test-generic-polycone-zigzag-profile", 0., vecgeom::kTwoPi, num_rz, r, z));
}

} // namespace test
} // namespace vecgeom

#endif
