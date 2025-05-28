// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/UnplacedTrd.cpp
/// @author Georgios Bitzes

#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/volumes/UnplacedTrd.h"
#include "VecGeom/volumes/SpecializedTrd.h"
#include "VecGeom/volumes/utilities/GenerationUtilities.h"
#include "VecGeom/base/RNG.h"
#ifdef VECGEOM_ROOT
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Trd.hh"
#endif

#ifndef VECCORE_CUDA
#include "VecGeom/volumes/UnplacedImplAs.h"
#endif

#include "VecGeom/management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
TGeoShape const *UnplacedTrd::ConvertToRoot(char const *label) const
{
  if (dy1() == dy2()) {
    return new TGeoTrd1(label, dx1(), dx2(), dy1(), dz());
  }
  return new TGeoTrd2(label, dx1(), dx2(), dy1(), dy2(), dz());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *UnplacedTrd::ConvertToGeant4(char const *label) const
{
  return new G4Trd(label, dx1(), dx2(), dy1(), dy2(), dz());
}
#endif
#endif

template <>
UnplacedTrd *Maker<UnplacedTrd>::MakeInstance(const Precision x1, const Precision x2, const Precision y1,
                                              const Precision y2, const Precision z)
{

#if !defined(VECGEOM_NO_SPECIALIZATION) && !defined(VECCORE_CUDA)
  // Trd becomes a box, hence returning a box from the factory
  if ((x1 == x2) && (y1 == y2)) {
    return new SUnplacedImplAs<SUnplacedTrd<TrdTypes::Trd2>, UnplacedBox>(x1, y1, z);
  }

  // Specialized Trd of type Trd1
  if (y1 == y2) {
    return new SUnplacedTrd<TrdTypes::Trd1>(x1, x2, y1, z);
  } else {
    return new SUnplacedTrd<TrdTypes::Trd2>(x1, x2, y1, y2, z);
  }
#else
  return new SUnplacedTrd<TrdTypes::UniversalTrd>(x1, x2, y1, y2, z);
#endif
}

// special case Trd1 when dY1 == dY2
template <>
UnplacedTrd *Maker<UnplacedTrd>::MakeInstance(const Precision x1, const Precision x2, const Precision y1,
                                              const Precision z)
{

#if !defined(VECGEOM_NO_SPECIALIZATION) && !defined(VECCORE_CUDA)
  // Trd1 becomes a box, hence returning a box from the factory
  if (x1 == x2) {
    return new SUnplacedImplAs<SUnplacedTrd<TrdTypes::Trd1>, UnplacedBox>(x1, y1, z);
  } else {
    return new SUnplacedTrd<TrdTypes::Trd1>(x1, x2, y1, z);
  }
#else
  return new SUnplacedTrd<TrdTypes::UniversalTrd>(x1, x2, y1, z);
#endif
}

VECCORE_ATT_HOST_DEVICE
void UnplacedTrd::Print() const
{
  printf("UnplacedTrd {%.2f, %.2f, %.2f, %.2f, %.2f}", dx1(), dx2(), dy1(), dy2(), dz());
}

void UnplacedTrd::Print(std::ostream &os) const
{
  os << "UnplacedTrd {" << dx1() << ", " << dx2() << ", " << dy1() << ", " << dy2() << ", " << dz();
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedTrd::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{
  SolidMesh *sm = new SolidMesh();
  sm->ResetMesh(8, 6);

  //-z
  Precision x1 = fTrd.fDX1;
  Precision y1 = fTrd.fDY1;

  //+z
  Precision x2 = fTrd.fDX2;
  Precision y2 = fTrd.fDY2;

  Precision z = fTrd.fDZ;

  typedef Vector3D<Precision> Vec_t;
  const Vec_t vertices[] = {Vec_t(-x1, -y1, -z), Vec_t(-x1, y1, -z), Vec_t(x1, y1, -z), Vec_t(x1, -y1, -z),
                            Vec_t(-x2, -y2, z),  Vec_t(-x2, y2, z),  Vec_t(x2, y2, z),  Vec_t(x2, -y2, z)};

  sm->SetVertices(vertices, 8);
  sm->TransformVertices(trans);

  sm->AddPolygon(4, {0, 1, 2, 3}, true);
  sm->AddPolygon(4, {4, 7, 6, 5}, true);
  sm->AddPolygon(4, {0, 4, 5, 1}, true);
  sm->AddPolygon(4, {1, 5, 6, 2}, true);
  sm->AddPolygon(4, {2, 6, 7, 3}, true);
  sm->AddPolygon(4, {3, 7, 4, 0}, true);

  return sm;
}
#endif

Precision UnplacedTrd::Capacity() const
{
  return 2 * (fTrd.fDX1 + fTrd.fDX2) * (fTrd.fDY1 + fTrd.fDY2) * fTrd.fDZ +
         (2. / 3.) * (fTrd.fDX1 - fTrd.fDX2) * (fTrd.fDY1 - fTrd.fDY2) * fTrd.fDZ;
}

Precision UnplacedTrd::SurfaceArea() const
{
  Precision dz = 2 * fTrd.fDZ;
  bool xvert   = (fTrd.fDX1 == fTrd.fDX2) ? true : false;
  Precision SA = 0.0;

  // Sum of area for planes Perp. to +X and -X
  Precision ht = (xvert) ? dz : Sqrt((fTrd.fDX1 - fTrd.fDX2) * (fTrd.fDX1 - fTrd.fDX2) + dz * dz);
  SA += 2.0 * 2.0 * 0.5 * (fTrd.fDY1 + fTrd.fDY2) * ht;

  // Sum of area for planes Perp. to +Y and -Y
  SA += 2.0 * 2.0 * 0.5 * (fTrd.fDX1 + fTrd.fDX2) * ht; // if xvert then topology forces to become yvert for closing

  // Sum of area for top and bottom planes +Z and -Z
  SA += 4. * (fTrd.fDX1 * fTrd.fDY1) + 4. * (fTrd.fDX2 * fTrd.fDY2);

  return SA;
}

int UnplacedTrd::ChooseSurface() const
{
  int choice = 0; // 0-1 = zm, 2-3 = zp, 4-5 = ym, 6-7 = yp, 8-9 = xm, 10-11 = xp
  Precision S[12], Stotal = 0.0;

  S[0] = S[1] = GetMinusZArea() / 2.0;
  S[2] = S[3] = GetPlusZArea() / 2.0;

  Precision yarea = GetPlusYArea() / (fTrd.fDX1 + fTrd.fDX2);
  S[4] = S[6] = yarea * fTrd.fDX1;
  S[5] = S[7] = yarea * fTrd.fDX2;

  Precision xarea = GetPlusXArea() / (fTrd.fDY1 + fTrd.fDY2);
  S[8] = S[10] = xarea * fTrd.fDY1;
  S[9] = S[11] = xarea * fTrd.fDY2;

  for (int i = 0; i < 12; ++i)
    Stotal += S[i];

  // random value to choose triangle to place the point
  Precision rand = RNG::Instance().uniform() * Stotal;

  while (rand > S[choice])
    rand -= S[choice], choice++;

  VECGEOM_ASSERT(choice < 12);
  return choice;
}

Vector3D<Precision> UnplacedTrd::SamplePointOnSurface() const
//                                             /
//                 p6-------p7                 /
//                / |       | \                /
//               / p4-------p5 \               /
//              /  /         \  \              /
//            p2--/-----------\--p3            /
//             | /             \ |             /
//             |/               \|             /
//            p0-----------------p1            /
//                                             /
{
  Vector3D<Precision> A, B, C;

  int surface = ChooseSurface();
  switch (surface) {
  case 0: // -Z face, 1st triangle (p0-p1-p2)
    A.Set(-fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    B.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    C.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    break;
  case 1: // -Z face, 2nd triangle (p3-p1-p2)
    A.Set(fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    B.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    C.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    break;
  case 2: // +Z face, 1st triangle (p4-p5-p6)
    A.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    B.Set(fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    C.Set(-fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    break;
  case 3: // +Z face, 2nd triangle (p7-p5-p6)
    A.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    B.Set(fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    C.Set(-fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    break;
  case 4: // -Y face, 1st triangle (p0-p1-p4)
    A.Set(-fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    B.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    C.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    break;
  case 5: // -Y face, 2nd triangle (p5-p1-p4)
    A.Set(fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    B.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    C.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    break;
  case 6: // +Y face, 1st triangle (p3-p2-p7)
    A.Set(fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    B.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    C.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    break;
  case 7: // +Y face, 2nd triangle (p6-p2-p7)
    A.Set(-fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    B.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    C.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    break;
  case 8: // -X face, 1st triangle (p0-p4-p2)
    A.Set(-fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    B.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    C.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    break;
  case 9: // -X face, 2nd triangle (p6-p4-p2)
    A.Set(-fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    B.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    C.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    break;
  case 10: // +X face, 1st triangle (p3-p7-p1)
    A.Set(fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    B.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    C.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    break;
  case 11: // +X face, 2nd triangle (p3-p7-p1)
    A.Set(fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    B.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    C.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    break;
  }

  Precision r1 = RNG::Instance().uniform();
  Precision r2 = RNG::Instance().uniform();
  if (r1 + r2 > 1.0) {
    r1 = 1.0 - r1;
    r2 = 1.0 - r2;
  }

  return A + r1 * (B - A) + r2 * (C - A);
}

VECCORE_ATT_HOST_DEVICE
bool UnplacedTrd::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{
  using vecCore::math::Abs;
  using vecCore::math::Min;

  int noSurfaces = 0;
  Vector3D<Precision> sumnorm(0., 0., 0.), vecnorm(0., 0., 0.);
  Precision distz;

  distz = Abs(Abs(point[2]) - fTrd.fDZ);

  Precision xnorm = 1.0 / sqrt(4 * fTrd.fDZ * fTrd.fDZ + (fTrd.fDX2 - fTrd.fDX1) * (fTrd.fDX2 - fTrd.fDX1));
  Precision ynorm = 1.0 / sqrt(4 * fTrd.fDZ * fTrd.fDZ + (fTrd.fDY2 - fTrd.fDY1) * (fTrd.fDY2 - fTrd.fDY1));

  Precision distmx =
      -2.0 * fTrd.fDZ * point[0] - (fTrd.fDX2 - fTrd.fDX1) * point[2] - fTrd.fDZ * (fTrd.fDX1 + fTrd.fDX2);
  distmx *= xnorm;

  Precision distpx =
      2.0 * fTrd.fDZ * point[0] - (fTrd.fDX2 - fTrd.fDX1) * point[2] - fTrd.fDZ * (fTrd.fDX1 + fTrd.fDX2);
  distpx *= xnorm;

  Precision distmy =
      -2.0 * fTrd.fDZ * point[1] - (fTrd.fDY2 - fTrd.fDY1) * point[2] - fTrd.fDZ * (fTrd.fDY1 + fTrd.fDY2);
  distmy *= ynorm;

  Precision distpy =
      2.0 * fTrd.fDZ * point[1] - (fTrd.fDY2 - fTrd.fDY1) * point[2] - fTrd.fDZ * (fTrd.fDY1 + fTrd.fDY2);
  distpy *= ynorm;

  Precision safmin = Min(Abs(distmx), Abs(distpx));
  safmin           = Min(safmin, Min(Abs(distmy), Abs(distpy)));
  safmin           = Min(safmin, Abs(distz));
  bool valid       = safmin <= kHalfTolerance;

  if (Abs(distmx) - safmin <= kHalfTolerance) {
    noSurfaces++;
    sumnorm += Vector3D<Precision>(-2.0 * fTrd.fDZ, 0.0, -(fTrd.fDX2 - fTrd.fDX1)) * xnorm;
  }
  if (Abs(distpx) - safmin <= kHalfTolerance) {
    noSurfaces++;
    sumnorm += Vector3D<Precision>(2.0 * fTrd.fDZ, 0.0, -(fTrd.fDX2 - fTrd.fDX1)) * xnorm;
  }
  if (Abs(distpy) - safmin <= kHalfTolerance) {
    noSurfaces++;
    sumnorm += Vector3D<Precision>(0.0, 2.0 * fTrd.fDZ, -(fTrd.fDY2 - fTrd.fDY1)) * ynorm;
  }
  if (Abs(distmy) - safmin <= kHalfTolerance) {
    noSurfaces++;
    sumnorm += Vector3D<Precision>(0.0, -2.0 * fTrd.fDZ, -(fTrd.fDY2 - fTrd.fDY1)) * ynorm;
  }

  if (Abs(distz) - safmin <= kHalfTolerance) {
    noSurfaces++;
    if (point[2] >= 0.)
      sumnorm += Vector3D<Precision>(0., 0., 1.);
    else
      sumnorm += Vector3D<Precision>(0., 0., -1.);
  }
  if (noSurfaces == 0) {
#ifdef UDEBUG
    UUtils::Exception("UnplacedTrapezoid::SurfaceNormal(point)", "GeomSolids1002", Warning, 1,
                      "Point is not on surface.");
#endif
    // vecnorm = ApproxSurfaceNormal( Vector3D<Precision>(point[0],point[1],point[2]) );
    vecnorm =
        Vector3D<Precision>(0., 0., 1.); // any plane will do it, since false is returned, so save the CPU cycles...
  } else if (noSurfaces == 1)
    vecnorm = sumnorm;
  else
    vecnorm = sumnorm.Unit();

  norm[0] = vecnorm[0];
  norm[1] = vecnorm[1];
  norm[2] = vecnorm[2];

  return valid;
}

std::ostream &UnplacedTrd::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Trd\n"
     << " Parameters: \n"
     << "     half lengths X1,X2: " << fTrd.fDX1 << "mm, " << fTrd.fDX2 << "mm \n"
     << "     half lengths Y1,Y2: " << fTrd.fDY1 << "mm, " << fTrd.fDY2 << "mm \n"
     << "     half length Z: " << fTrd.fDZ << "mm \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTrd::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<SUnplacedTrd<TrdTypes::UniversalTrd>>(in_gpu_ptr, dx1(), dx2(), dy1(), dy2(), dz());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTrd::CopyToGpu() const
{
  return CopyToGpuImpl<SUnplacedTrd<TrdTypes::UniversalTrd>>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::SUnplacedTrd<cuda::TrdTypes::UniversalTrd>>::SizeOf();
template void DevicePtr<cuda::SUnplacedTrd<cuda::TrdTypes::UniversalTrd>>::Construct(
    const Precision dx1, const Precision dx2, const Precision dy1, const Precision dy2, const Precision z) const;
template void DevicePtr<cuda::SUnplacedTrd<cuda::TrdTypes::UniversalTrd>>::Construct(const Precision dx1,
                                                                                     const Precision dx2,
                                                                                     const Precision dy1,
                                                                                     const Precision z) const;

} // namespace cxx

#endif

} // namespace vecgeom
