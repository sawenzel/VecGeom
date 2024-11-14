// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/UnplacedEllipticalCone.cpp
/// @author Raman Sehgal, Evgueni Tcherniaev

#include "VecGeom/volumes/EllipticUtilities.h"
#include "VecGeom/volumes/UnplacedEllipticalCone.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/SpecializedEllipticalCone.h"
#include "VecGeom/base/RNG.h"
#include <stdio.h>
#include <cmath>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
UnplacedEllipticalCone::UnplacedEllipticalCone(Precision a, Precision b, Precision h, Precision zcut)
{
  SetParameters(a, b, h, zcut);
  fGlobalConvexity = true;
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
void UnplacedEllipticalCone::CheckParameters()
{
  Precision tol = 2. * kTolerance;
  if (fEllipticalCone.fDx < tol || fEllipticalCone.fDy < tol || fEllipticalCone.fDz < tol ||
      fEllipticalCone.fZCut < tol) {
    VECGEOM_LOG(error) << "Wrong parameteres EllipticalCone { " << fEllipticalCone.fDx << ", " << fEllipticalCone.fDy << ", "
              << fEllipticalCone.fDz << ", " << fEllipticalCone.fZCut << " }";
    if (fEllipticalCone.fDx < tol) fEllipticalCone.fDx = tol;
    if (fEllipticalCone.fDy < tol) fEllipticalCone.fDy = tol;
    if (fEllipticalCone.fDz < tol) fEllipticalCone.fDz = tol;
    if (fEllipticalCone.fZCut < tol) fEllipticalCone.fZCut = tol;
  }
  fEllipticalCone.fZCut = vecCore::math::Min(fEllipticalCone.fDz, fEllipticalCone.fZCut);
  Precision dx          = fEllipticalCone.fDx;
  Precision dy          = fEllipticalCone.fDy;
  Precision h           = fEllipticalCone.fDz;
  Precision zcut        = fEllipticalCone.fZCut;

  // Set surface area and volume
  Precision h1                 = h - zcut;
  Precision h2                 = h + zcut;
  fEllipticalCone.fSurfaceArea = EllipticUtilities::EllipticalConeLateralArea(dx, dy, 1.) * (h2 + h1) * (h2 - h1) +
                                 kPi * dx * dy * (h2 * h2 + h1 * h1);
  fEllipticalCone.fCubicVolume = kPi * dx * dy * (h2 * h2 * h2 - h1 * h1 * h1) / 3.;

  // Set precalculated values
  Precision xmax             = dx * h2;
  Precision ymax             = dy * h2;
  Precision dxymin           = vecCore::math::Min(dx, dy);
  fEllipticalCone.fRsph      = vecCore::math::Sqrt(xmax * xmax + ymax * ymax + zcut * zcut);
  fEllipticalCone.invDx      = 1. / dx;
  fEllipticalCone.invDy      = 1. / dy;
  fEllipticalCone.cosAxisMin = dxymin / vecCore::math::Sqrt(1. + dxymin * dxymin);
  fEllipticalCone.dApex      = kHalfTolerance / fEllipticalCone.cosAxisMin;
};

VECCORE_ATT_HOST_DEVICE
void UnplacedEllipticalCone::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  Precision h    = GetZMax();
  Precision zcut = GetZTopCut();
  Precision xmax = GetSemiAxisX() * (h + zcut);
  Precision ymax = GetSemiAxisY() * (h + zcut);
  aMin.Set(-xmax, -ymax, -zcut);
  aMax.Set(xmax, ymax, zcut);
}

Vector3D<Precision> UnplacedEllipticalCone::SamplePointOnLateralSurface() const
{
  // Get parameters
  Precision dx   = GetSemiAxisX();
  Precision dy   = GetSemiAxisY();
  Precision h    = GetZMax();
  Precision zcut = GetZTopCut();

  // Select random radial direction
  Precision a      = dx * h; // x semi axis at z = 0
  Precision b      = dy * h; // y semi axis at z = 0
  Precision aa     = a * a;
  Precision bb     = b * b;
  Precision hh     = h * h;
  Precision R      = vecCore::math::Max(a, b);
  Precision mu_max = R * vecCore::math::Sqrt(hh + R * R);
  Precision x, y;
  for (int i = 0; i < 1000; ++i) {
    Precision phi = kTwoPi * RNG::Instance().uniform();
    x             = vecCore::math::Cos(phi);
    y             = vecCore::math::Sin(phi);
    Precision xx  = x * x;
    Precision yy  = y * y;
    Precision E   = hh + aa * xx + bb * yy;
    Precision F   = (aa - bb) * x * y;
    Precision G   = aa * yy + bb * xx;
    Precision mu  = vecCore::math::Sqrt(E * G - F * F);
    if (mu_max * RNG::Instance().uniform() <= mu) break;
  }

  // Select random distance from apex
  Precision zmin = h - zcut;
  Precision zmax = h + zcut;
  Precision rnd  = RNG::Instance().uniform();
  Precision zh   = vecCore::math::Sqrt(rnd * zmax * zmax + (1. - rnd) * zmin * zmin);

  // Sample point
  return Vector3D<Precision>(zh * dx * x, zh * dy * y, h - zh);
}

Vector3D<Precision> UnplacedEllipticalCone::SamplePointOnSurface() const
{
  // Get parameters
  Precision dx   = GetSemiAxisX();
  Precision dy   = GetSemiAxisY();
  Precision h    = GetZMax();
  Precision zcut = GetZTopCut();

  Precision x0   = dx * h; // x semi axis at z=0
  Precision y0   = dy * h; // y semi axis at z=0
  Precision s0   = EllipticUtilities::EllipticalConeLateralArea(x0, y0, h);
  Precision kmin = 1. - zcut / h;
  Precision kmax = 1. + zcut / h;

  // Set areas (base at -Z, side surface, base at +Z)
  Precision szmin    = kPi * x0 * y0 * kmax * kmax;
  Precision szmax    = kPi * x0 * y0 * kmin * kmin;
  Precision sside    = s0 * (kmax * kmax - kmin * kmin);
  Precision ssurf[3] = {szmin, szmin + sside, szmin + sside + szmax};

  // Select surface
  Precision select = ssurf[2] * RNG::Instance().uniform();
  int k            = 2;
  if (select <= ssurf[1]) k = 1;
  if (select <= ssurf[0]) k = 0;

  // Pick random point on selected surface
  //
  Vector3D<Precision> p;
  switch (k) {
  case 0: // base at -Z, uniform distribution, rejection sampling
  {
    Precision zh            = h + zcut;
    Vector2D<Precision> rho = EllipticUtilities::RandomPointInEllipse(zh * dx, zh * dy);
    p.Set(rho.x(), rho.y(), -zcut);
    break;
  }
  case 1: // lateral surface, uniform distribution, rejection sampling
  {
    p = SamplePointOnLateralSurface();
    break;
  }
  case 2: // base at +Z, uniform distribution, rejection sampling
  {
    Precision zh            = h - zcut;
    Vector2D<Precision> rho = EllipticUtilities::RandomPointInEllipse(zh * dx, zh * dy);
    p.Set(rho.x(), rho.y(), zcut);
    break;
  }
  }
  return p;
}

// VECCORE_ATT_HOST_DEVICE
std::ostream &UnplacedEllipticalCone::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: EllipticalCone: (x/a)^2 + (y/b)^2 = (z-h)^2\n"
     << " Parameters: \n"
     << "   a    : " << fEllipticalCone.fDx << "\n"
     << "   b    : " << fEllipticalCone.fDy << "\n"
     << "   h    : " << fEllipticalCone.fDz << "\n"
     << "   zcut : " << fEllipticalCone.fZCut << "\n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedEllipticalCone::Print() const
{
  printf("EllipticalCone {%.2f, %.2f, %.2f, %.2f}", fEllipticalCone.fDx, fEllipticalCone.fDy, fEllipticalCone.fDz,
         fEllipticalCone.fZCut);
}

void UnplacedEllipticalCone::Print(std::ostream &os) const
{
  os << "EllipticalCone {" << fEllipticalCone.fDx << ", " << fEllipticalCone.fDy << ", " << fEllipticalCone.fDz << ", "
     << fEllipticalCone.fZCut << "}";
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedEllipticalCone::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{

  Precision a       = GetSemiAxisX();
  Precision b       = GetSemiAxisY();
  Precision h       = GetZMax();
  Precision zTopCut = GetZTopCut();
  SolidMesh *sm     = new SolidMesh();
  typedef Vector3D<Precision> Vec_t;

  sm->ResetMesh(2 * (nSegments + 1), nSegments + 2);
  Vec_t *const vertices = new Vec_t[2 * (nSegments + 1)];
  Precision cos, sin;
  for (size_t i = 0; i <= nSegments; i++) {
    cos = std::cos(2 * i * M_PI / nSegments);
    sin = std::sin(2 * i * M_PI / nSegments);

    vertices[i]                 = Vec_t((h + zTopCut) * a * cos, (h + zTopCut) * b * sin, -zTopCut); // lower vertices
    vertices[i + nSegments + 1] = Vec_t((h - zTopCut) * a * cos, (h - zTopCut) * b * sin, zTopCut);  //  upper vertices
  }
  sm->SetVertices(vertices, 2 * (nSegments + 1));
  delete[] vertices;

  sm->TransformVertices(trans);

  Utils3D::vector_t<size_t> indices;
  indices.reserve(nSegments);
  // upper surface
  for (size_t i = 0; i < nSegments; i++) {
    indices.push_back(i + nSegments + 1);
  }

  sm->AddPolygon(nSegments, indices, true);
  indices.clear();

  // lower surface
  for (size_t i = nSegments; i > 0; i--) {
    indices.push_back(i - 1);
  }

  sm->AddPolygon(nSegments, indices, true);

  // lateral surfaces
  for (size_t i = 0; i < nSegments; i++) {
    sm->AddPolygon(4, {i, i + 1, i + 1 + nSegments + 1, i + nSegments + 1}, true);
  }

  return sm;
}
#endif

#ifndef VECCORE_CUDA
VPlacedVolume *UnplacedEllipticalCone::Create(LogicalVolume const *const logical_volume,
                                              Transformation3D const *const transformation,
                                              VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedEllipticalCone(logical_volume, transformation);
    return placement;
  }
  return new SpecializedEllipticalCone(logical_volume, transformation);
}

VPlacedVolume *UnplacedEllipticalCone::SpecializedVolume(LogicalVolume const *const volume,
                                                         Transformation3D const *const transformation,
                                                         VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedEllipticalCone>(volume, transformation, placement);
}
#else

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedEllipticalCone::Create(LogicalVolume const *const logical_volume,
                                              Transformation3D const *const transformation, const int id,
                                              const int copy_no, const int child_id, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedEllipticalCone(logical_volume, transformation, id, copy_no, child_id);
    return placement;
  }
  return new SpecializedEllipticalCone(logical_volume, transformation, id, copy_no, child_id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedEllipticalCone::SpecializedVolume(LogicalVolume const *const volume,
                                                         Transformation3D const *const transformation, const int id,
                                                         const int copy_no, const int child_id,
                                                         VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedEllipticalCone>(volume, transformation, id, copy_no, child_id,
                                                                       placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedEllipticalCone::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedEllipticalCone>(in_gpu_ptr, GetSemiAxisX(), GetSemiAxisY(), GetZMax(), GetZTopCut());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedEllipticalCone::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedEllipticalCone>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedEllipticalCone>::SizeOf();
template void DevicePtr<cuda::UnplacedEllipticalCone>::Construct(const Precision a, const Precision b,
                                                                 const Precision h, const Precision zcut) const;

} // namespace cxx

#endif
} // namespace vecgeom
