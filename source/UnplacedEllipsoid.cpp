// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/UnplacedEllipsoid.cpp
/// @author Evgueni Tcherniaev

#include "VecGeom/volumes/EllipticUtilities.h"
#include "VecGeom/volumes/UnplacedEllipsoid.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/SpecializedEllipsoid.h"
#include "VecGeom/base/RNG.h"
#include <stdio.h>
#include <cmath>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
UnplacedEllipsoid::UnplacedEllipsoid()
{
  fGlobalConvexity = true;
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
UnplacedEllipsoid::UnplacedEllipsoid(Precision dx, Precision dy, Precision dz, Precision zBottomCut, Precision zTopCut)
{
  fGlobalConvexity       = true;
  fEllipsoid.fDx         = dx;
  fEllipsoid.fDy         = dy;
  fEllipsoid.fDz         = dz;
  fEllipsoid.fZBottomCut = zBottomCut;
  fEllipsoid.fZTopCut    = zTopCut;
  CheckParameters();
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
void UnplacedEllipsoid::CheckParameters()
{
  // Check semi-axes
  Precision tol = 2. * kTolerance;
  if (fEllipsoid.fDx < tol || fEllipsoid.fDy < tol || fEllipsoid.fDy < tol) {
    VECGEOM_LOG(error) << "Invalid semi-axes of Ellipsoid { " << fEllipsoid.fDx << ", " << fEllipsoid.fDy << ", "
              << fEllipsoid.fDz << " }";
    fEllipsoid.fDx = fEllipsoid.fDy = fEllipsoid.fDz = tol;
  }
  Precision A = fEllipsoid.fDx;
  Precision B = fEllipsoid.fDy;
  Precision C = fEllipsoid.fDz;

  // Check cuts
  if (fEllipsoid.fZBottomCut == 0 && fEllipsoid.fZTopCut == 0) {
    fEllipsoid.fZBottomCut = -C;
    fEllipsoid.fZTopCut    = C;
  }
  if ((fEllipsoid.fZBottomCut >= C) || (fEllipsoid.fZTopCut <= -C) || (fEllipsoid.fZBottomCut >= fEllipsoid.fZTopCut)) {
    VECGEOM_LOG(error) << "Invalid cut planes { " << fEllipsoid.fZBottomCut << ", " << fEllipsoid.fZTopCut
                       << " } of Ellipsoid { " << A << ", " << B << ", " << C << " }";
    fEllipsoid.fZBottomCut = -C;
    fEllipsoid.fZTopCut    = C;
  }
  fEllipsoid.fZBottomCut = vecCore::math::Max(fEllipsoid.fZBottomCut, -fEllipsoid.fDz);
  fEllipsoid.fZTopCut    = vecCore::math::Min(fEllipsoid.fZTopCut, fEllipsoid.fDz);

  // Compute volume and surface area
  Precision piAB          = kPi * A * B;
  Precision piAB_3        = piAB / 3.;
  fEllipsoid.fCubicVolume = 4. * piAB_3 * C;
  fEllipsoid.fSurfaceArea = LateralSurfaceArea();
  if (fEllipsoid.fZBottomCut > -C) {
    Precision hbot = 1. + fEllipsoid.fZBottomCut / C;
    fEllipsoid.fCubicVolume -= piAB_3 * hbot * hbot * (2. * C - fEllipsoid.fZBottomCut);
    fEllipsoid.fSurfaceArea += piAB * hbot * (2. - hbot);
  }
  if (fEllipsoid.fZTopCut < C) {
    Precision htop = 1. - fEllipsoid.fZTopCut / C;
    fEllipsoid.fCubicVolume -= piAB_3 * htop * htop * (2. * C + fEllipsoid.fZTopCut);
    fEllipsoid.fSurfaceArea += piAB * htop * (2. - htop);
  }

  // Set extent in x and y
  fEllipsoid.fXmax = A;
  fEllipsoid.fYmax = B;
  if (fEllipsoid.fZBottomCut > 0.) {
    Precision ratio  = fEllipsoid.fZBottomCut / C;
    Precision scale  = vecCore::math::Sqrt((1. - ratio) * (1 + ratio));
    fEllipsoid.fXmax = A * scale;
    fEllipsoid.fYmax = B * scale;
  }
  if (fEllipsoid.fZTopCut < 0.) {
    Precision ratio  = fEllipsoid.fZTopCut / C;
    Precision scale  = vecCore::math::Sqrt((1. - ratio) * (1 + ratio));
    fEllipsoid.fXmax = A * scale;
    fEllipsoid.fYmax = B * scale;
  }

  // Precalculated values
  fEllipsoid.fRsph = vecCore::math::Max(A, B, C); // bounding sphere
  fEllipsoid.fR    = vecCore::math::Min(A, B, C); // radius of sphere after scaling
  fEllipsoid.fSx   = fEllipsoid.fR / A;           // scale factor in x
  fEllipsoid.fSy   = fEllipsoid.fR / B;           // scale factor in y
  fEllipsoid.fSz   = fEllipsoid.fR / C;           // scale factor in z

  // Scaled cuts
  fEllipsoid.fScZBottomCut = fEllipsoid.fZBottomCut * fEllipsoid.fSz;                  // scaled bottom cut
  fEllipsoid.fScZTopCut    = fEllipsoid.fZTopCut * fEllipsoid.fSz;                     // scaled top cut
  fEllipsoid.fScZMidCut    = 0.5 * (fEllipsoid.fScZTopCut + fEllipsoid.fScZBottomCut); // middle position
  fEllipsoid.fScZDimCut    = 0.5 * (fEllipsoid.fScZTopCut - fEllipsoid.fScZBottomCut); // half dimension

  // Coefficients for approximation of distance : Q1 * (x^2 + y^2 + z^2) - Q2
  fEllipsoid.fQ1 = 0.5 / fEllipsoid.fR;
  fEllipsoid.fQ2 = 0.5 * (fEllipsoid.fR + kHalfTolerance * kHalfTolerance / fEllipsoid.fR);
}; // namespace VECGEOM_IMPL_NAMESPACE

VECCORE_ATT_HOST_DEVICE
Precision UnplacedEllipsoid::LateralSurfaceArea() const
{
  const int Nphi = 100;
  const int Nz   = 200;
  Precision rho[Nz + 1];

  // Set array of rho
  Precision zbot = fEllipsoid.fZBottomCut / fEllipsoid.fDz;
  Precision ztop = fEllipsoid.fZTopCut / fEllipsoid.fDz;
  Precision dz   = (ztop - zbot) / Nz;
  for (int iz = 0; iz < Nz; ++iz) {
    Precision z = zbot + iz * dz;
    rho[iz]     = Sqrt((1. + z) * (1. - z));
  }
  rho[Nz] = Sqrt((1. + ztop) * (1. - ztop));

  // Compute area
  zbot           = fEllipsoid.fZBottomCut;
  ztop           = fEllipsoid.fZTopCut;
  dz             = (ztop - zbot) / Nz;
  Precision area = 0.;
  Precision dphi = kHalfPi / Nphi;
  for (int iphi = 0; iphi < Nphi; ++iphi) {
    Precision phi1 = iphi * dphi;
    Precision phi2 = (iphi == Nphi - 1) ? kHalfPi : phi1 + dphi;
    Precision cos1 = Cos(phi1) * fEllipsoid.fDx;
    Precision cos2 = Cos(phi2) * fEllipsoid.fDx;
    Precision sin1 = Sin(phi1) * fEllipsoid.fDy;
    Precision sin2 = Sin(phi2) * fEllipsoid.fDy;
    for (int iz = 0; iz < Nz; ++iz) {
      Precision z1   = zbot + iz * dz;
      Precision z2   = (iz == Nz - 1) ? ztop : z1 + dz;
      Precision rho1 = rho[iz];
      Precision rho2 = rho[iz + 1];
      Vector3D<Precision> p1(rho1 * cos1, rho1 * sin1, z1);
      Vector3D<Precision> p2(rho1 * cos2, rho1 * sin2, z1);
      Vector3D<Precision> p3(rho2 * cos1, rho2 * sin1, z2);
      Vector3D<Precision> p4(rho2 * cos2, rho2 * sin2, z2);
      area += ((p4 - p1).Cross(p3 - p2)).Mag();
    }
  }
  return 2. * area;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedEllipsoid::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  aMin.Set(-fEllipsoid.fXmax, -fEllipsoid.fYmax, fEllipsoid.fZBottomCut);
  aMax.Set(fEllipsoid.fXmax, fEllipsoid.fYmax, fEllipsoid.fZTopCut);
}

Vector3D<Precision> UnplacedEllipsoid::SamplePointOnSurface() const
{
  Precision A    = GetDx();
  Precision B    = GetDy();
  Precision C    = GetDz();
  Precision Zbot = GetZBottomCut();
  Precision Ztop = GetZTopCut();

  // Calculate cut areas
  Precision Hbot = 1. + Zbot / C;
  Precision Htop = 1. - Ztop / C;
  Precision piAB = kPi * A * B;
  Precision Sbot = piAB * Hbot * (2. - Hbot);
  Precision Stop = piAB * Htop * (2. - Htop);

  // Select surface (0 - bottom cut, 1 - lateral surface, 2 - top cut)
  Precision S      = SurfaceArea();
  Precision select = S * RNG::Instance().uniform();
  int k            = 0;
  if (select > Sbot) k = 1;
  if (select > S - Stop) k = 2;

  // Pick random point on selected surface (rejection sampling)
  Vector3D<Precision> p;
  switch (k) {
  case 0: // bootom z-cut
  {
    Precision scale         = vecCore::math::Sqrt(Hbot * (2. - Hbot));
    Vector2D<Precision> rho = EllipticUtilities::RandomPointInEllipse(A * scale, B * scale);
    p.Set(rho.x(), rho.y(), Zbot);
    break;
  }
  case 1: // lateral surface
  {
    Precision x, y, z;
    Precision mu_max = std::max(std::max(A * B, A * C), B * C);
    for (int i = 0; i < 1000; ++i) {
      // generate random point on unit sphere
      z             = (Zbot + (Ztop - Zbot) * RNG::Instance().uniform()) / C;
      Precision rho = vecCore::math::Sqrt((1. + z) * (1. - z));
      Precision phi = kTwoPi * RNG::Instance().uniform();
      x             = rho * vecCore::math::Cos(phi);
      y             = rho * vecCore::math::Sin(phi);
      // check  acceptance
      Precision xbc = x * B * C;
      Precision yac = y * A * C;
      Precision zab = z * A * B;
      Precision mu  = std::sqrt(xbc * xbc + yac * yac + zab * zab);
      if (mu_max * RNG::Instance().uniform() <= mu) break;
    }
    p.Set(A * x, B * y, C * z);
    break;
  }
  case 2: // top z-cut
  {
    Precision scale         = vecCore::math::Sqrt(Htop * (2. - Htop));
    Vector2D<Precision> rho = EllipticUtilities::RandomPointInEllipse(A * scale, B * scale);
    p.Set(rho.x(), rho.y(), Ztop);
    break;
  }
  }
  return p;
}

// VECCORE_ATT_HOST_DEVICE
std::ostream &UnplacedEllipsoid::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Ellipsoid: (x/dx)^2 + (y/dy)^2 + (z/dz)^2 = 1"
     << " Parameters: \n"
     << "   dx        : " << fEllipsoid.fDx << "\n"
     << "   dy        : " << fEllipsoid.fDy << "\n"
     << "   dz        : " << fEllipsoid.fDz << "\n"
     << "   BottomCut : " << fEllipsoid.fZBottomCut << "\n"
     << "   TopCut    : " << fEllipsoid.fZTopCut << "\n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);

  return os;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedEllipsoid::Print() const
{
  printf("Ellipsoid {%.2f, %.2f, %.2f, %.2f, %.2f}", fEllipsoid.fDx, fEllipsoid.fDy, fEllipsoid.fDz,
         fEllipsoid.fZBottomCut, fEllipsoid.fZTopCut);
}

void UnplacedEllipsoid::Print(std::ostream &os) const
{
  os << "Ellipsoid {" << fEllipsoid.fDx << ", " << fEllipsoid.fDy << ", " << fEllipsoid.fDz << ", "
     << fEllipsoid.fZBottomCut << ", " << fEllipsoid.fZTopCut << "}";
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedEllipsoid::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{
  typedef Vector3D<Precision> Vec_t;
  SolidMesh *sm = new SolidMesh();

  sm->ResetMesh((nSegments + 1) * (nSegments + 1), nSegments * nSegments + 2);

  // fill vertex array
  Vec_t *vertices = new Vec_t[(nSegments + 1) * (nSegments + 1)];

  Precision phi_step = 2 * M_PI / nSegments;
  Precision phi      = 0.;

  Precision theta      = (GetDz() - GetZTopCut()) * kPi / (2 * GetDz());
  Precision theta_step = (GetZTopCut() - GetZBottomCut()) / (2 * GetDz()) * kPi / nSegments;

  size_t idx = 0;
  for (size_t i = 0; i <= nSegments; ++i, theta += theta_step, phi = 0.) {
    for (size_t j = 0; j <= nSegments; ++j, phi += phi_step) {
      vertices[idx++] = Vec_t(GetDx() * std::sin(theta) * std::cos(phi), GetDy() * std::sin(theta) * std::sin(phi),
                              GetDz() * std::cos(theta));
    }
  }
  sm->SetVertices(vertices, (nSegments + 1) * (nSegments + 1));
  delete[] vertices;
  sm->TransformVertices(trans);

  // add polygons
  for (size_t j = 0, k = 0; j < nSegments; j++, k++) {
    for (size_t i = 0, l = k + nSegments + 1; i < nSegments; i++, k++, l++) {
      sm->AddPolygon(4, {l + 1, k + 1, k, l}, true);
    }
  }

  Utils3D::vector_t<size_t> indices;
  indices.reserve(nSegments);

  if (GetZTopCut() != GetDz()) {
    // upper surface
    for (size_t i = 0; i < nSegments; i++) {
      indices.push_back(i);
    }
    sm->AddPolygon(nSegments, indices, true);
  }

  if (GetZBottomCut() != -GetDz()) {
    // lower surface
    indices.clear();
    for (size_t i = 0, k = (nSegments + 1) * (nSegments + 1); i < nSegments; i++, k--) {
      indices.push_back(k - 1);
    }
    sm->AddPolygon(nSegments, indices, true);
  }

  return sm;
}
#endif

#ifndef VECCORE_CUDA
VPlacedVolume *UnplacedEllipsoid::Create(LogicalVolume const *const logical_volume,
                                         Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedEllipsoid(logical_volume, transformation);
    return placement;
  }
  return new SpecializedEllipsoid(logical_volume, transformation);
}

VPlacedVolume *UnplacedEllipsoid::SpecializedVolume(LogicalVolume const *const volume,
                                                    Transformation3D const *const transformation,
                                                    VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedEllipsoid>(volume, transformation, placement);
}
#else

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedEllipsoid::Create(LogicalVolume const *const logical_volume,
                                         Transformation3D const *const transformation, const int id, const int copy_no,
                                         const int child_id, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedEllipsoid(logical_volume, transformation, id, copy_no, child_id);
    return placement;
  }
  return new SpecializedEllipsoid(logical_volume, transformation, id, copy_no, child_id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedEllipsoid::SpecializedVolume(LogicalVolume const *const volume,
                                                    Transformation3D const *const transformation, const int id,
                                                    const int copy_no, const int child_id,
                                                    VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedEllipsoid>(volume, transformation, id, copy_no, child_id,
                                                                  placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedEllipsoid::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedEllipsoid>(in_gpu_ptr, GetDx(), GetDy(), GetDz(), GetZBottomCut(), GetZTopCut());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedEllipsoid::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedEllipsoid>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedEllipsoid>::SizeOf();
template void DevicePtr<cuda::UnplacedEllipsoid>::Construct(const Precision dx, const Precision dy, const Precision dz,
                                                            const Precision zBottomCut, const Precision zTopCut) const;

} // namespace cxx

#endif
} // namespace vecgeom
