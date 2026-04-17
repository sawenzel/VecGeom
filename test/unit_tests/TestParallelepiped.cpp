// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Unit test for Paralleliped.
/// @file test/unit_tests/TestParallelepiped.cpp
/// @author Created by Mihaela Gheata
/// @author Revised by Evgueni Tcherniaev

// ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Parallelepiped.h"
#include "ApproxEqual.h"
#include <cmath>

template <class Parallelepiped_t>
bool TestParallelepiped()
{
  using namespace vecgeom::VECGEOM_IMPL_NAMESPACE;
  using vecgeom::Precision;
  using Vec_t = vecgeom::Vector3D<vecgeom::Precision>;
  EnumInside inside;
  const Precision dx    = 20;
  const Precision dy    = 30;
  const Precision dz    = 40;
  const Precision alpha = kDegToRad * 30;
  const Precision theta = kDegToRad * 30;
  const Precision phi   = kDegToRad * 45;

  Parallelepiped_t para("Test Parallelepiped", dx, dy, dz, alpha, theta, phi);

  // Points on faces
  Vec_t pzero(0, 0, 0);
  Vec_t ponxside(dx, 0, 0), ponyside(0, dy, 0),
      ponzside(dz * para.GetTanThetaCosPhi(), dz * para.GetTanThetaSinPhi(), dz);
  Vec_t ponmxside(-dx, 0, 0), ponmyside(0, -dy, 0),
      ponmzside(-dz * para.GetTanThetaCosPhi(), -dz * para.GetTanThetaSinPhi(), -dz);

  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);

  Precision Dist;
  Vec_t normal, norm;
  bool valid;

  // check Cubic volume

  int Npoints = 10000000;
  std::cout << "=== Check Capacity()" << std::endl;
  Precision vol      = para.Capacity();
  Precision volCheck = para.GetUnplacedVolume()->EstimateCapacity(Npoints);
  std::cout << " vol = " << vol << "   mc_estimated = " << volCheck << std::endl;
  VECGEOM_ASSERT(std::abs(vol - volCheck) < 0.01 * vol);

  // Check Surface area

  std::cout << "=== Check SurfaceArea()" << std::endl;
  Precision surf      = para.SurfaceArea();
  Precision surfCheck = para.GetUnplacedVolume()->EstimateSurfaceArea(Npoints);
  std::cout << " surf = " << surf << "   mc_estimated = " << surfCheck << std::endl;
  VECGEOM_ASSERT(std::abs(surf - surfCheck) < 0.01 * surf);

  // Check Extent and cached BBox

  std::cout << "=== Check Extent() and cached BBox" << std::endl;
  Vec_t minExtent, maxExtent;
  Vec_t minBBox, maxBBox;
  Vec_t minCheck(kInfLength, kInfLength, kInfLength);
  Vec_t maxCheck(-kInfLength, -kInfLength, -kInfLength);
  para.Extent(minExtent, maxExtent);
  para.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  for (int i = 0; i < Npoints; ++i) {
    Vec_t p = para.GetUnplacedVolume()->SamplePointOnSurface();
    minCheck.Set(std::min(p.x(), minCheck.x()), std::min(p.y(), minCheck.y()), std::min(p.z(), minCheck.z()));
    maxCheck.Set(std::max(p.x(), maxCheck.x()), std::max(p.y(), maxCheck.y()), std::max(p.z(), maxCheck.z()));
  }
  std::cout << " calculated: min = " << minExtent << " max = " << maxExtent << std::endl;
  std::cout << " estimated:  min = " << minCheck << " max = " << maxCheck << std::endl;

  VECGEOM_ASSERT(std::abs(minExtent.x() - minCheck.x()) < 0.001 * std::abs(minExtent.x()));
  VECGEOM_ASSERT(std::abs(minExtent.y() - minCheck.y()) < 0.001 * std::abs(minExtent.y()));
  VECGEOM_ASSERT(minExtent.z() == minCheck.z());
  VECGEOM_ASSERT(std::abs(maxExtent.x() - maxCheck.x()) < 0.001 * std::abs(maxExtent.x()));
  VECGEOM_ASSERT(std::abs(maxExtent.y() - maxCheck.y()) < 0.001 * std::abs(maxExtent.y()));
  VECGEOM_ASSERT(maxExtent.z() == maxCheck.z());
  VECGEOM_ASSERT(ApproxEqual<Precision>(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual<Precision>(maxExtent, maxBBox));

  // Check Inside

  std::cout << "=== Check Inside()" << std::endl;
  VECGEOM_ASSERT(para.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(para.Inside(pbigz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(para.Inside(ponxside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(para.Inside(ponyside) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(para.Inside(ponzside) == vecgeom::EInside::kSurface);

  inside = para.Inside(ponzside + ponxside);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponzside + ponmxside);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponzside + ponyside);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponzside + ponmyside);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponmzside + ponxside);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponmzside + ponmxside);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponmzside + ponyside);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  inside = para.Inside(ponmzside + ponmyside);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);

  // Check Surface Normal

  std::cout << "=== Check Normal()" << std::endl;
  Vec_t pp[8], nn[3], ptest;
  pp[0].Set(-dx, -dy, -dz);
  pp[1].Set(dx, -dy, -dz);
  pp[2].Set(dx, dy, -dz);
  pp[3].Set(-dx, dy, -dz);
  pp[4].Set(-dx, -dy, dz);
  pp[5].Set(dx, -dy, dz);
  pp[6].Set(dx, dy, dz);
  pp[7].Set(-dx, dy, dz);
  for (int i = 0; i < 8; ++i) {
    pp[i].x() += para.GetTanAlpha() * pp[i].y() + para.GetTanThetaCosPhi() * pp[i].z();
    pp[i].y() += para.GetTanThetaSinPhi() * pp[i].z();
  }
  nn[0] = para.GetUnplacedVolume()->GetNormal(0);
  nn[1] = para.GetUnplacedVolume()->GetNormal(1);
  nn[2] = Vec_t(0, 0, 1);

  // check facets
  ptest = Vec_t(dx, 0, 0);
  valid = para.Normal(ptest, normal);
  VECGEOM_ASSERT(valid && normal == nn[0]);
  valid = para.Normal(1.1 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == nn[0]);
  valid = para.Normal(0.9 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == nn[0]);

  ptest = Vec_t(-dx, 0, 0);
  valid = para.Normal(ptest, normal);
  VECGEOM_ASSERT(valid && normal == -nn[0]);
  valid = para.Normal(1.1 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == -nn[0]);
  valid = para.Normal(0.9 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == -nn[0]);

  ptest = Vec_t(0, dy, 0);
  valid = para.Normal(ptest, normal);
  VECGEOM_ASSERT(valid && normal == nn[1]);
  valid = para.Normal(1.1 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == nn[1]);
  valid = para.Normal(0.9 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == nn[1]);

  ptest = Vec_t(0, -dy, 0);
  valid = para.Normal(ptest, normal);
  VECGEOM_ASSERT(valid && normal == -nn[1]);
  valid = para.Normal(1.1 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == -nn[1]);
  valid = para.Normal(0.9 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == -nn[1]);

  ptest = Vec_t(0, 0, dz);
  valid = para.Normal(ptest, normal);
  VECGEOM_ASSERT(valid && normal == nn[2]);
  valid = para.Normal(1.1 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == nn[2]);
  valid = para.Normal(0.9 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == nn[2]);

  ptest = Vec_t(0, 0, -dz);
  valid = para.Normal(ptest, normal);
  VECGEOM_ASSERT(valid && normal == -nn[2]);
  valid = para.Normal(1.1 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == -nn[2]);
  valid = para.Normal(0.9 * ptest, normal);
  VECGEOM_ASSERT(!valid && normal == -nn[2]);

  // check edges
  valid = para.Normal((pp[0] + pp[1]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (-nn[2] - nn[1]).Unit());
  valid = para.Normal((pp[1] + pp[2]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (-nn[2] + nn[0]).Unit());
  valid = para.Normal((pp[2] + pp[3]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (-nn[2] + nn[1]).Unit());
  valid = para.Normal((pp[3] + pp[0]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (-nn[2] - nn[0]).Unit());

  valid = para.Normal((pp[4] + pp[5]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (nn[2] - nn[1]).Unit());
  valid = para.Normal((pp[5] + pp[6]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (nn[2] + nn[0]).Unit());
  valid = para.Normal((pp[6] + pp[7]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (nn[2] + nn[1]).Unit());
  valid = para.Normal((pp[7] + pp[4]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (nn[2] - nn[0]).Unit());

  valid = para.Normal((pp[0] + pp[4]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (-nn[0] - nn[1]).Unit());
  valid = para.Normal((pp[1] + pp[5]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (nn[0] - nn[1]).Unit());
  valid = para.Normal((pp[2] + pp[6]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (nn[0] + nn[1]).Unit());
  valid = para.Normal((pp[3] + pp[7]) / 2, normal);
  VECGEOM_ASSERT(valid && normal == (-nn[0] + nn[1]).Unit());

  // check nodes
  valid = para.Normal(pp[0], normal);
  VECGEOM_ASSERT(valid && normal == (-nn[2] - nn[1] - nn[0]).Unit());
  valid = para.Normal(pp[1], normal);
  VECGEOM_ASSERT(valid && normal == (-nn[2] - nn[1] + nn[0]).Unit());
  valid = para.Normal(pp[2], normal);
  VECGEOM_ASSERT(valid && normal == (-nn[2] + nn[1] + nn[0]).Unit());
  valid = para.Normal(pp[3], normal);
  VECGEOM_ASSERT(valid && normal == (-nn[2] + nn[1] - nn[0]).Unit());

  valid = para.Normal(pp[4], normal);
  VECGEOM_ASSERT(valid && normal == (nn[2] - nn[1] - nn[0]).Unit());
  valid = para.Normal(pp[5], normal);
  VECGEOM_ASSERT(valid && normal == (nn[2] - nn[1] + nn[0]).Unit());
  valid = para.Normal(pp[6], normal);
  VECGEOM_ASSERT(valid && normal == (nn[2] + nn[1] + nn[0]).Unit());
  valid = para.Normal(pp[7], normal);
  VECGEOM_ASSERT(valid && normal == (nn[2] + nn[1] - nn[0]).Unit());

  // Check SafetyToOut

  std::cout << "=== Check SafetyToOut()" << std::endl;
  Dist = para.SafetyToOut(pzero);
  VECGEOM_ASSERT(Dist < dx);

  Dist = para.SafetyToOut(ponxside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToOut(ponyside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToOut(ponzside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToOut(ponmxside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToOut(ponmyside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToOut(ponmzside);
  VECGEOM_ASSERT(Dist == 0.);

  // Check SafetyToIn

  std::cout << "=== Check SafetyToIn()" << std::endl;
  Dist = para.SafetyToIn(pzero);
  VECGEOM_ASSERT(Dist < dx);

  Dist = para.SafetyToIn(ponxside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToIn(ponyside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToIn(ponzside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToIn(ponmxside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToIn(ponmyside);
  VECGEOM_ASSERT(Dist == 0.);

  Dist = para.SafetyToIn(ponmzside);
  VECGEOM_ASSERT(Dist == 0.);

  VECGEOM_ASSERT(ApproxEqual<Precision>(para.SafetyToIn(pbigx), para.SafetyToIn(pbigmx)));
  VECGEOM_ASSERT(ApproxEqual<Precision>(para.SafetyToIn(pbigy), para.SafetyToIn(pbigmy)));
  VECGEOM_ASSERT(ApproxEqual<Precision>(para.SafetyToIn(pbigz), para.SafetyToIn(pbigmz)));

  // DistanceToOut(P,V)

  std::cout << "=== Check DistanceToOut()" << std::endl;
  Dist = para.DistanceToOut(pzero, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, dx));
  Dist = para.DistanceToOut(pzero, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, dx));
  Dist = para.DistanceToOut(pzero, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, dy));
  Dist = para.DistanceToOut(pzero, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, dy));
  Dist = para.DistanceToOut(pzero, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, dz));
  Dist = para.DistanceToOut(pzero, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, dz));

  Dist = para.DistanceToOut(ponxside, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = para.DistanceToOut(ponmxside, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = para.DistanceToOut(ponyside, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = para.DistanceToOut(ponmyside, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = para.DistanceToOut(ponzside, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = para.DistanceToOut(ponmzside, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));

  // DistanceToIn(P,V)

  std::cout << "=== Check DistanceToIn()" << std::endl;
  Dist = para.DistanceToIn(pbigx, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 - dx));
  Dist = para.DistanceToIn(pbigmx, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 - dx));
  Dist = para.DistanceToIn(pbigy, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 - dy));
  Dist = para.DistanceToIn(pbigmy, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 - dy));
  Dist = para.DistanceToIn(pbigz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 - dz));
  Dist = para.DistanceToIn(pbigmz, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 - dz));
  Dist = para.DistanceToIn(pbigx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = para.DistanceToIn(pbigmx, vxy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));

  // Check SamplePointOnSurface()

  std::cout << "=== Check SamplePointOnSurface()" << std::endl;
  Vec_t Vx(1., 0., 0.);
  Vec_t Vy(para.GetTanAlpha(), 1., 0.);
  Vec_t Vz(para.GetTanThetaCosPhi(), para.GetTanThetaSinPhi(), 1.);
  Vec_t Nx = Vy.Cross(Vz);
  Vec_t Ny = Vz.Cross(Vx);
  Vec_t Nz(0., 0., 1.);
  Precision sx = 4. * para.GetY() * para.GetZ() * Nx.Mag();
  Precision sy = 4. * para.GetZ() * para.GetX() * Ny.Mag();
  Precision sz = 4. * para.GetX() * para.GetY();
  Nx.Normalize();
  Ny.Normalize();
  Precision Dx = -Nx.x() * para.GetX();
  Precision Dy = -Ny.y() * para.GetY();
  int nxneg = 0, nxpos = 0, nyneg = 0, nypos = 0, nzneg = 0, nzpos = 0;
  int nfactor = 100, ntot = 2. * (sx + sy + sz) * nfactor;
  for (int i = 0; i < ntot; i++) {
    Vec_t p = para.GetUnplacedVolume()->SamplePointOnSurface();
    VECGEOM_ASSERT(para.Inside(p) == vecgeom::kSurface);
    if (std::abs(Nx.Dot(p) - Dx) < kHalfTolerance) {
      ++nxneg;
    } else if (std::abs(Nx.Dot(p) + Dx) < kHalfTolerance) {
      ++nxpos;
    } else if (std::abs(Ny.Dot(p) - Dy) < kHalfTolerance) {
      ++nyneg;
    } else if (std::abs(Ny.Dot(p) + Dy) < kHalfTolerance) {
      ++nypos;
    } else if (p.z() == -para.GetZ()) {
      ++nzneg;
    } else {
      ++nzpos;
    }
  }
  std::cout << "facet surface -/+x, -/+y, -/+z: " << "\t" << sx << ", \t" << sx << ", \t" << sy << ", \t" << sy
            << ", \t" << sz << ", \t" << sz << std::endl;
  std::cout << "n. of samples -/+x, -/+y, -/+z: " << "\t" << nxneg << ", \t" << nxpos << ", \t" << nyneg << ", \t"
            << nypos << ", \t" << nzneg << ", \t" << nzpos << std::endl;
  VECGEOM_ASSERT(std::abs(nxneg - sx * nfactor) < 0.01 * sx * nfactor);
  VECGEOM_ASSERT(std::abs(nxpos - sx * nfactor) < 0.01 * sx * nfactor);
  VECGEOM_ASSERT(std::abs(nyneg - sy * nfactor) < 0.01 * sy * nfactor);
  VECGEOM_ASSERT(std::abs(nypos - sy * nfactor) < 0.01 * sy * nfactor);
  VECGEOM_ASSERT(std::abs(nzneg - sz * nfactor) < 0.01 * sz * nfactor);
  VECGEOM_ASSERT(std::abs(nzpos - sz * nfactor) < 0.01 * sz * nfactor);

  return true;
}

int main(int argc, char *argv[])
{

  TestParallelepiped<vecgeom::SimpleParallelepiped>();
  std::cout << "\nVecGeom Parallelepiped passed\n" << std::endl;
  return 0;
}
