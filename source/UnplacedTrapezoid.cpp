/// \file UnplacedTrapezoid.cpp
/// \author Guilherme Lima (lima@fnal.gov)
//
// 140407 G. Lima    - based on equivalent box code
// 160722 G. Lima    Revision + moving to new backend structure

#include "VecGeom/volumes/UnplacedTrapezoid.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/volumes/UnplacedTrd.h"
#include "VecGeom/volumes/UnplacedParallelepiped.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/SpecializedTrapezoid.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/volumes/kernel/shapetypes/TrdTypes.h"
#include <cstdio>
#include <iostream>

#ifdef VECGEOM_ROOT
#include "TGeoArb8.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Trap.hh"
#endif

#ifndef VECCORE_CUDA
#include "VecGeom/volumes/UnplacedImplAs.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using Vec3D = Vector3D<Precision>;

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
TGeoShape const *UnplacedTrapezoid::ConvertToRoot(char const *label) const
{
  return new TGeoTrap(label, dz(), theta() * kRadToDeg, phi() * kRadToDeg, dy1(), dx1(), dx2(), alpha1() * kRadToDeg,
                      dy2(), dx3(), dx4(), alpha2() * kRadToDeg);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *UnplacedTrapezoid::ConvertToGeant4(char const *label) const
{
  return new G4Trap(label, dz(), theta(), phi(), dy1(), dx1(), dx2(), alpha1(), dy2(), dx3(), dx4(), alpha2());
}
#endif
#endif

template <>
UnplacedTrapezoid *Maker<UnplacedTrapezoid>::MakeInstance(const Precision dz, const Precision theta,
                                                          const Precision phi, const Precision dy1, const Precision dx1,
                                                          const Precision dx2, const Precision Alpha1,
                                                          const Precision dy2, const Precision dx3, const Precision dx4,
                                                          const Precision Alpha2)
{

#if !defined(VECGEOM_NO_SPECIALIZATION) && !defined(VECCORE_CUDA)
  return GetSpecialized(dz, theta, phi, dy1, dx1, dx2, Alpha1, dy2, dx3, dx4, Alpha2);
#else
  return new UnplacedTrapezoid(dz, theta, phi, dy1, dx1, dx2, Alpha1, dy2, dx3, dx4, Alpha2);
#endif
}

template <>
UnplacedTrapezoid *Maker<UnplacedTrapezoid>::MakeInstance(TrapCorners const pt)
{
#if !defined(VECGEOM_NO_SPECIALIZATION) && !defined(VECCORE_CUDA)

  Precision dz      = pt[7].z();
  Precision DzRecip = 1.0 / dz;

  Precision dy1       = 0.50 * (pt[2].y() - pt[0].y());
  Precision dx1       = 0.50 * (pt[1].x() - pt[0].x());
  Precision dx2       = 0.50 * (pt[3].x() - pt[2].x());
  Precision tanAlpha1 = 0.25 * (pt[2].x() + pt[3].x() - pt[1].x() - pt[0].x()) / dy1;
  Precision Alpha1    = atan(tanAlpha1);

  Precision dy2       = 0.50 * (pt[6].y() - pt[4].y());
  Precision dx3       = 0.50 * (pt[5].x() - pt[4].x());
  Precision dx4       = 0.50 * (pt[7].x() - pt[6].x());
  Precision tanAlpha2 = 0.25 * (pt[6].x() + pt[7].x() - pt[5].x() - pt[4].x()) / dy2;
  Precision Alpha2    = atan(tanAlpha2);

  Precision TthetaCphi = (pt[4].x() + dy2 * tanAlpha2 + dx3) * DzRecip;
  Precision TthetaSphi = (pt[4].y() + dy2) * DzRecip;

  Precision theta = atan(sqrt(TthetaSphi * TthetaSphi + TthetaCphi * TthetaCphi));
  Precision phi   = atan2(TthetaSphi, TthetaCphi);

  return GetSpecialized(dz, theta, phi, dy1, dx1, dx2, Alpha1, dy2, dx3, dx4, Alpha2);
#else
  return new UnplacedTrapezoid(pt);
#endif
}

#if !defined(VECGEOM_NO_SPECIALIZATION) && !defined(VECCORE_CUDA)
UnplacedTrapezoid *GetSpecialized(const Precision dz, const Precision theta, const Precision phi, const Precision dy1,
                                  const Precision dx1, const Precision dx2, const Precision Alpha1, const Precision dy2,
                                  const Precision dx3, const Precision dx4, const Precision Alpha2)
{

  // Box Like Trapezoid
  if (theta == 0. && phi == 0. && Alpha1 == 0. && Alpha2 == 0 && dx1 == dx2 && dx2 == dx3 && dx3 == dx4 && dy1 == dy2) {
    return new SUnplacedImplAs<UnplacedTrapezoid, UnplacedBox>(dx1, dy1, dz);
  }
  // Trd1 Like Trapezoid
  if (theta == 0. && phi == 0. && Alpha1 == 0. && Alpha2 == 0 && dx1 == dx2 && dx3 == dx4 && dx2 != dx3 && dy1 == dy2) {
    return new SUnplacedImplAs<UnplacedTrapezoid, SUnplacedTrd<TrdTypes::Trd1>>(dx1, dx3, dy1, dz);
  }
  // Trd2 Like Trapezoid
  if (theta == 0. && phi == 0. && Alpha1 == 0. && Alpha2 == 0 && dx1 == dx2 && dx3 == dx4 && dx2 != dx3 && dy1 != dy2) {
    return new SUnplacedImplAs<UnplacedTrapezoid, SUnplacedTrd<TrdTypes::Trd2>>(dx1, dx3, dy1, dy2, dz);
  }

  // Parallelepiped Like Trapezoid
  if (Alpha1 == Alpha2 && dx1 == dx2 && dx2 == dx3 && dx3 == dx4 && dy1 == dy2) {
    return new SUnplacedImplAs<UnplacedTrapezoid, UnplacedParallelepiped>(dx1, dy1, dz, Alpha1, theta, phi);
  }

  // if none of the above then return the full Trapezoid.
  return new UnplacedTrapezoid(dz, theta, phi, dy1, dx1, dx2, Alpha1, dy2, dx3, dx4, Alpha2);
}
#endif

VECCORE_ATT_HOST_DEVICE
UnplacedTrapezoid::UnplacedTrapezoid(TrapCorners const corners) : fTrap()
{
  // fill data members
  fromCornersToParameters(corners);
  ComputeBBox();
}

/*VECCORE_ATT_HOST_DEVICE
UnplacedTrapezoid::UnplacedTrapezoid(Precision dx, Precision dy, Precision dz, Precision)
    : fTrap(dz, 0., 0., dy, dx, dx, 0., dy, dx, dx, 0.)
{
// TODO: this needs a proper logger treatment as per geantv conventions
#ifndef VECCORE_CUDA
  fprintf(stderr, "*** ERROR: STEP-based trapezoid constructor called, but not implemented ***");
#endif
  VECGEOM_ASSERT(false);
}*/

VECCORE_ATT_HOST_DEVICE
UnplacedTrapezoid::UnplacedTrapezoid(Precision dx1, Precision dx2, Precision dy, Precision dz)
    : fTrap(dz, 0., 0., dy, dx1, dx1, 0., dy, dx2, dx2, 0.)
{
  MakePlanes();
  fGlobalConvexity = true;
  ComputeBBox();
}

UnplacedTrapezoid::UnplacedTrapezoid(Precision dx1, Precision dx2, Precision dy1, Precision dy2, Precision dz)
    : UnplacedTrapezoid(dz, 0., 0., dy1, dx1, dx1, 0., dy2, dx2, dx2, 0.)
{
  MakePlanes();
  fGlobalConvexity = true;
  ComputeBBox();
}

UnplacedTrapezoid::UnplacedTrapezoid(Precision dx, Precision dy, Precision dz, Precision alpha, Precision theta,
                                     Precision phi)
    : fTrap(dz, theta, phi, dy, dx, dx, alpha, dy, dx, dx, alpha)
{
  // TODO: validate alpha usage here
  fTrap.fTanAlpha1 = std::tan(alpha);
  fTrap.fTanAlpha2 = fTrap.fTanAlpha1;
  MakePlanes();
  fGlobalConvexity = true;
  ComputeBBox();
}
VECCORE_ATT_HOST_DEVICE
UnplacedTrapezoid::UnplacedTrapezoid(Precision xbox, Precision ybox, Precision zbox)
    : fTrap(zbox, 0., 0., ybox, xbox, xbox, 0., ybox, xbox, xbox, 0.)
{
  // validity check
  // TODO: this needs a proper logger treatment as per geantv conventions
#ifndef VECCORE_CUDA
  if (xbox <= 0 || ybox <= 0 || zbox <= 0) {
    VECGEOM_LOG(warning) << "Invalid input length parameters for Solid: "
                 "UnplacedTrapezoid X="
              << xbox << ", Y=" << ybox << ", Z=" << zbox;
  }
#endif

  MakePlanes();
  fGlobalConvexity = true;
  ComputeBBox();
}

Precision UnplacedTrapezoid::Capacity() const
{
  // cubic approximation used in Geant4
  Precision vol = fTrap.fDz * ((fTrap.fDx1 + fTrap.fDx2 + fTrap.fDx3 + fTrap.fDx4) * (fTrap.fDy1 + fTrap.fDy2) +
                               (fTrap.fDx4 + fTrap.fDx3 - fTrap.fDx2 - fTrap.fDx1) * (fTrap.fDy2 - fTrap.fDy1) / 3.0);

  /*
  // GL: leaving this here for future reference
    // accurate volume calculation
    TrapCorners pt;
    this->fromPlanesToCorners(pt);

    // more precise, hopefully correct version (to be checked)
    Precision BmZm = pt[1].x() - pt[0].x();
    Precision BpZm = pt[3].x() - pt[2].x();
    Precision BmZp = pt[5].x() - pt[4].x();
    Precision BpZp = pt[7].x() - pt[6].x();
    Precision xCorr = (BpZp-BpZm + BmZp-BmZm) / (BpZm+BmZm);

    Precision ymZm = pt[0].y();
    Precision ypZm = pt[2].y();
    Precision ymZp = pt[4].y();
    Precision ypZp = pt[6].y();
    Precision yCorr = (ypZp-ypZm - (ymZp-ymZm)) / (ypZm-ymZm);

    Precision volume = 4*fDz*fDy1*(fDx1+fDx2) * ( 1.0 + (xCorr + yCorr)/2.0 + xCorr*yCorr/3.0 );
  */

  return vol;
}

/*
Precision UnplacedTrapezoid::SurfaceArea() const
{

  Vec3D ba(fDx1 - fDx2 + fTanAlpha1 * 2 * fDy1, 2 * fDy1, 0);
  Vec3D bc(2 * fDz * fTthetaCphi - (fDx4 - fDx2) + fTanAlpha2 * fDy2 - fTanAlpha1 * fDy1,
           2 * fDz * fTthetaSphi + fDy2 - fDy1, 2 * fDz);
  Vec3D dc(-fDx4 + fDx3 + 2 * fTanAlpha2 * fDy2, 2 * fDy2, 0);
  Vec3D da(-2 * fDz * fTthetaCphi - (fDx1 - fDx3) - fTanAlpha1 * fDy1 + fTanAlpha2 * fDy2,
           -2 * fDz * fTthetaSphi - fDy1 + fDy2, -2 * fDz);

  Vec3D ef(fDx2 - fDx1 + 2 * fTanAlpha1 * fDy1, 2 * fDy1, 0);
  Vec3D eh(2 * fDz * fTthetaCphi + fDx3 - fDx1 + fTanAlpha1 * fDy1 - fTanAlpha2 * fDy2,
           2 * fDz * fTthetaSphi - fDy2 + fDy1, 2 * fDz);
  Vec3D gh(fDx3 - fDx4 - 2 * fTanAlpha2 * fDy2, -2 * fDy2, 0);
  Vec3D gf(-2 * fDz * fTthetaCphi + fDx2 - fDx4 + fTanAlpha1 * fDy1 - fTanAlpha2 * fDy2,
           -2 * fDz * fTthetaSphi + fDy1 - fDy2, -2 * fDz);

  Vec3D cr;
  cr             = ba.Cross(bc);
  Precision babc = cr.Mag();
  cr             = dc.Cross(da);
  Precision dcda = cr.Mag();
  cr             = ef.Cross(eh);
  Precision efeh = cr.Mag();
  cr             = gh.Cross(gf);
  Precision ghgf = cr.Mag();

  Precision surfArea = 2 * fDy1 * (fDx1 + fDx2) + 2 * fDy2 * (fDx3 + fDx4) +
                       (fDx1 + fDx3) * Sqrt(4 * fDz * fDz + Pow(fDy2 - fDy1 - 2 * fDz * fTthetaSphi, 2.0)) +
                       (fDx2 + fDx4) * Sqrt(4 * fDz * fDz + Pow(fDy2 - fDy1 + 2 * fDz * fTthetaSphi, 2.0)) +
                       0.5 * (babc + dcda + efeh + ghgf);

  return surfArea;
}
*/
Precision UnplacedTrapezoid::SurfaceArea() const
{
  const TrapezoidStruct<Precision> &t = fTrap;
  Vec3D ba(t.fDx1 - t.fDx2 + t.fTanAlpha1 * 2 * t.fDy1, 2 * t.fDy1, 0);
  Vec3D bc(2 * t.fDz * t.fTthetaCphi - (t.fDx4 - t.fDx2) + t.fTanAlpha2 * t.fDy2 - t.fTanAlpha1 * t.fDy1,
           2 * t.fDz * t.fTthetaSphi + t.fDy2 - t.fDy1, 2 * t.fDz);
  Vec3D dc(-t.fDx4 + t.fDx3 + 2 * t.fTanAlpha2 * t.fDy2, 2 * t.fDy2, 0);
  Vec3D da(-2 * t.fDz * t.fTthetaCphi - (t.fDx1 - t.fDx3) - t.fTanAlpha1 * t.fDy1 + t.fTanAlpha2 * t.fDy2,
           -2 * t.fDz * t.fTthetaSphi - t.fDy1 + t.fDy2, -2 * t.fDz);

  Vec3D ef(t.fDx2 - t.fDx1 + 2 * t.fTanAlpha1 * t.fDy1, 2 * t.fDy1, 0);
  Vec3D eh(2 * t.fDz * t.fTthetaCphi + t.fDx3 - t.fDx1 + t.fTanAlpha1 * t.fDy1 - t.fTanAlpha2 * t.fDy2,
           2 * t.fDz * t.fTthetaSphi - t.fDy2 + t.fDy1, 2 * t.fDz);
  Vec3D gh(t.fDx3 - t.fDx4 - 2 * t.fTanAlpha2 * t.fDy2, -2 * t.fDy2, 0);
  Vec3D gf(-2 * t.fDz * t.fTthetaCphi + t.fDx2 - t.fDx4 + t.fTanAlpha1 * t.fDy1 - t.fTanAlpha2 * t.fDy2,
           -2 * t.fDz * t.fTthetaSphi + t.fDy1 - t.fDy2, -2 * t.fDz);

  Vec3D cr;
  cr             = ba.Cross(bc);
  Precision babc = cr.Mag();
  cr             = dc.Cross(da);
  Precision dcda = cr.Mag();
  cr             = ef.Cross(eh);
  Precision efeh = cr.Mag();
  cr             = gh.Cross(gf);
  Precision ghgf = cr.Mag();

  Precision surfArea =
      2 * t.fDy1 * (t.fDx1 + t.fDx2) + 2 * t.fDy2 * (t.fDx3 + t.fDx4) +
      (t.fDx1 + t.fDx3) *
          std::sqrt(4 * t.fDz * t.fDz + Pow(t.fDy2 - t.fDy1 - 2 * t.fDz * t.fTthetaSphi, Precision(2.0))) +
      (t.fDx2 + t.fDx4) *
          std::sqrt(4 * t.fDz * t.fDz + Pow(t.fDy2 - t.fDy1 + 2 * t.fDz * t.fTthetaSphi, Precision(2.0))) +
      0.5 * (babc + dcda + efeh + ghgf);

  return surfArea;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedTrapezoid::Extent(Vec3D &aMin, Vec3D &aMax) const
{
  aMin.z() = -fTrap.fDz;
  aMax.z() = fTrap.fDz;

  TrapCorners pt;
  this->fromPlanesToCorners(pt);

  Precision ext01 = Max(pt[0].x(), pt[1].x());
  Precision ext23 = Max(pt[2].x(), pt[3].x());
  Precision ext45 = Max(pt[4].x(), pt[5].x());
  Precision ext67 = Max(pt[6].x(), pt[7].x());
  Precision extA  = Max(ext01, ext23);
  Precision extB  = Max(ext45, ext67);
  aMax.x()        = Max(extA, extB);

  ext01    = Min(pt[0].x(), pt[1].x());
  ext23    = Min(pt[2].x(), pt[3].x());
  ext45    = Min(pt[4].x(), pt[5].x());
  ext67    = Min(pt[6].x(), pt[7].x());
  extA     = Min(ext01, ext23);
  extB     = Min(ext45, ext67);
  aMin.x() = Min(extA, extB);

  ext01    = Max(pt[0].y(), pt[1].y());
  ext23    = Max(pt[2].y(), pt[3].y());
  ext45    = Max(pt[4].y(), pt[5].y());
  ext67    = Max(pt[6].y(), pt[7].y());
  extA     = Max(ext01, ext23);
  extB     = Max(ext45, ext67);
  aMax.y() = Max(extA, extB);

  ext01    = Min(pt[0].y(), pt[1].y());
  ext23    = Min(pt[2].y(), pt[3].y());
  ext45    = Min(pt[4].y(), pt[5].y());
  ext67    = Min(pt[6].y(), pt[7].y());
  extA     = Min(ext01, ext23);
  extB     = Min(ext45, ext67);
  aMin.y() = Min(extA, extB);
}

Vector3D<Precision> UnplacedTrapezoid::SamplePointOnSurface() const
{
  TrapCorners pt;
  this->fromPlanesToCorners(pt);

  // make sure we provide the points in a clockwise fashion
  Precision chose = RNG::Instance().uniform() * SurfaceArea();

  Precision sumArea = 0.0;
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[0])) {
    return GetPointOnPlane(pt[0], pt[1], pt[5], pt[4]);
  }

  sumArea += fTrap.sideAreas[0];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[1])) {
    return GetPointOnPlane(pt[2], pt[6], pt[7], pt[3]);
  }

  sumArea += fTrap.sideAreas[1];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[2])) {
    return GetPointOnPlane(pt[0], pt[4], pt[6], pt[2]);
  }

  sumArea += fTrap.sideAreas[2];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[3])) {
    return GetPointOnPlane(pt[1], pt[3], pt[7], pt[5]);
  }

  sumArea += fTrap.sideAreas[3];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[4])) {
    return GetPointOnPlane(pt[0], pt[2], pt[3], pt[1]);
  }

  sumArea += fTrap.sideAreas[4];
  if ((chose >= sumArea) && (chose < sumArea + fTrap.sideAreas[5])) {
    return GetPointOnPlane(pt[4], pt[5], pt[7], pt[6]);
  }

  // should never get here...
  return Vec3D(0., 0., 0.);
}

Vec3D UnplacedTrapezoid::GetPointOnPlane(Vec3D const &p0, Vec3D const &p1, Vec3D const &p2, Vec3D const &p3) const
{
  Precision lambda1, lambda2, chose, aOne, aTwo;
  Vec3D t, u, v, w, Area, normal;

  t = p1 - p0;
  u = p2 - p1;
  v = p3 - p2;
  w = p0 - p3;

  Area = Vec3D(w.y() * v.z() - w.z() * v.y(), w.z() * v.x() - w.x() * v.z(), w.x() * v.y() - w.y() * v.x());

  aOne = 0.5 * Area.Mag();

  Area = Vec3D(t.y() * u.z() - t.z() * u.y(), t.z() * u.x() - t.x() * u.z(), t.x() * u.y() - t.y() * u.x());

  aTwo = 0.5 * Area.Mag();

  chose = RNG::Instance().uniform(0., aOne + aTwo);

  if ((chose >= 0.) && (chose < aOne)) {
    lambda1 = RNG::Instance().uniform(0., 1.);
    lambda2 = RNG::Instance().uniform(0., lambda1);
    return (p2 + lambda1 * v + lambda2 * w);
  }

  // else

  lambda1 = RNG::Instance().uniform(0., 1.);
  lambda2 = RNG::Instance().uniform(0., lambda1);
  return (p0 + lambda1 * t + lambda2 * u);
}

/*
VECCORE_ATT_HOST_DEVICE
void UnplacedTrapezoid::GetParametersList(int, Precision *aArray) const
{
  aArray[0] = GetRadius();
}
*/

VECCORE_ATT_HOST_DEVICE
UnplacedTrapezoid *UnplacedTrapezoid::Clone() const
{
  return new UnplacedTrapezoid(*this);
}

VECCORE_ATT_HOST_DEVICE
void UnplacedTrapezoid::Print() const
{
  // Note: units printed out chosen such that same numbers can be used as arguments to full constructor
  printf("UnplacedTrapezoid {%.3fmm, %.3frad, %.3frad, %.3fmm, %.3fmm, %.3fmm, %.3frad, %.3fmm, %.3fmm, %.3fmm, "
         "%.3frad}\n",
         fTrap.fDz, this->theta(), this->phi(), fTrap.fDy1, fTrap.fDx1, fTrap.fDx2, this->alpha1(), fTrap.fDy2,
         fTrap.fDx3, fTrap.fDx4, this->alpha2());
}

void UnplacedTrapezoid::Print(std::ostream &os) const
{
  // Note: units printed out chosen such that same numbers can be used as arguments to full constructor
  os << "UnplacedTrapezoid { " << fTrap.fDz << "mm, " << this->theta() << "rad, " << this->phi() << "rad, "
     << fTrap.fDy1 << "mm, " << fTrap.fDx1 << "mm, " << fTrap.fDx2 << "mm, " << this->alpha1() << "rad, " << fTrap.fDy2
     << "mm, " << fTrap.fDx3 << "mm, " << fTrap.fDx4 << "mm, " << this->alpha2() << "rad }\n";
}

VECCORE_ATT_HOST_DEVICE
void UnplacedTrapezoid::FromParametersToCorners(TrapCorners pt) const
{
  const TrapezoidStruct<Precision> &t = fTrap;

  // hopefully the compiler will optimize the repeated multiplications ... to be checked!
  Precision dxdyDy1 = t.fTanAlpha1 * t.fDy1;
  Precision dxdyDy2 = t.fTanAlpha2 * t.fDy2;
  Precision dxdzDz  = t.fTthetaCphi * t.fDz;
  Precision dydzDz  = t.fTthetaSphi * t.fDz;

  pt[0] = Vec3D(-dxdzDz - dxdyDy1 - t.fDx1, -dydzDz - t.fDy1, -t.fDz);
  pt[1] = Vec3D(-dxdzDz - dxdyDy1 + t.fDx1, -dydzDz - t.fDy1, -t.fDz);
  pt[2] = Vec3D(-dxdzDz + dxdyDy1 - t.fDx2, -dydzDz + t.fDy1, -t.fDz);
  pt[3] = Vec3D(-dxdzDz + dxdyDy1 + t.fDx2, -dydzDz + t.fDy1, -t.fDz);
  pt[4] = Vec3D(+dxdzDz - dxdyDy2 - t.fDx3, +dydzDz - t.fDy2, +t.fDz);
  pt[5] = Vec3D(+dxdzDz - dxdyDy2 + t.fDx3, +dydzDz - t.fDy2, +t.fDz);
  pt[6] = Vec3D(+dxdzDz + dxdyDy2 - t.fDx4, +dydzDz + t.fDy2, +t.fDz);
  pt[7] = Vec3D(+dxdzDz + dxdyDy2 + t.fDx4, +dydzDz + t.fDy2, +t.fDz);
}

VECCORE_ATT_HOST_DEVICE
void UnplacedTrapezoid::fromPlanesToCorners(TrapCorners pt) const
{
  const TrapezoidStruct<Precision> &t = fTrap;
  using TrapSidePlane                 = TrapezoidStruct<Precision>::TrapSidePlane;
  TrapSidePlane pl0                   = t.GetPlane(0); // -Y
  TrapSidePlane pl1                   = t.GetPlane(1); // +Y
  TrapSidePlane pl2                   = t.GetPlane(2); // -X
  TrapSidePlane pl3                   = t.GetPlane(3); // +X

  pt[0].z() = pt[1].z() = pt[2].z() = pt[3].z() = -t.fDz;
  pt[4].z() = pt[5].z() = pt[6].z() = pt[7].z() = t.fDz;

  pt[0].y() = pt[1].y() = -(pl0.fD - t.fDz * pl0.fC) / pl0.fB; // intersect -Y ; z=-fDz
  pt[2].y() = pt[3].y() = -(pl1.fD - t.fDz * pl1.fC) / pl1.fB; // intersect +Y ; z=-fDz
  pt[4].y() = pt[5].y() = -(pl0.fD + t.fDz * pl0.fC) / pl0.fB; // intersect -Y ; z=+fDz
  pt[6].y() = pt[7].y() = -(pl1.fD + t.fDz * pl1.fC) / pl1.fB; // intersect +Y ; z=+fDz

  pt[0].x() = ((pl2.fB / pl0.fB) * (pl0.fD - t.fDz * pl0.fC) - (pl2.fD - t.fDz * pl2.fC)) / pl2.fA; // int. -X ; -Y ; -Z
  pt[1].x() = ((pl3.fB / pl0.fB) * (pl0.fD - t.fDz * pl0.fC) - (pl3.fD - t.fDz * pl3.fC)) / pl3.fA; // int. +X ; -Y ; -Z
  pt[2].x() = ((pl2.fB / pl1.fB) * (pl1.fD - t.fDz * pl1.fC) - (pl2.fD - t.fDz * pl2.fC)) / pl2.fA; // int. -X ; +Y ; -Z
  pt[3].x() = ((pl3.fB / pl1.fB) * (pl1.fD - t.fDz * pl1.fC) - (pl3.fD - t.fDz * pl3.fC)) / pl3.fA; // int. +X ; +Y ; -Z
  pt[4].x() = ((pl2.fB / pl0.fB) * (pl0.fD + t.fDz * pl0.fC) - (pl2.fD + t.fDz * pl2.fC)) / pl2.fA; // int. -X ; -Y ; +Z
  pt[5].x() = ((pl3.fB / pl0.fB) * (pl0.fD + t.fDz * pl0.fC) - (pl3.fD + t.fDz * pl3.fC)) / pl3.fA; // int. +X ; -Y ; +Z
  pt[6].x() = ((pl2.fB / pl1.fB) * (pl1.fD + t.fDz * pl1.fC) - (pl2.fD + t.fDz * pl2.fC)) / pl2.fA; // int. -X ; +Y ; +Z
  pt[7].x() = ((pl3.fB / pl1.fB) * (pl1.fD + t.fDz * pl1.fC) - (pl3.fD + t.fDz * pl3.fC)) / pl3.fA; // int. +X ; +Y ; +Z
}

VECCORE_ATT_HOST_DEVICE
void UnplacedTrapezoid::fromCornersToParameters(TrapCorners const pt)
{
  fTrap.fDz         = pt[7].z();
  Precision DzRecip = 1.0 / fTrap.fDz;

  fTrap.fDy1       = 0.50 * (pt[2].y() - pt[0].y());
  fTrap.fDx1       = 0.50 * (pt[1].x() - pt[0].x());
  fTrap.fDx2       = 0.50 * (pt[3].x() - pt[2].x());
  fTrap.fTanAlpha1 = 0.25 * (pt[2].x() + pt[3].x() - pt[1].x() - pt[0].x()) / fTrap.fDy1;

  fTrap.fDy2       = 0.50 * (pt[6].y() - pt[4].y());
  fTrap.fDx3       = 0.50 * (pt[5].x() - pt[4].x());
  fTrap.fDx4       = 0.50 * (pt[7].x() - pt[6].x());
  fTrap.fTanAlpha2 = 0.25 * (pt[6].x() + pt[7].x() - pt[5].x() - pt[4].x()) / fTrap.fDy2;

  fTrap.fTthetaCphi = (pt[4].x() + fTrap.fDy2 * fTrap.fTanAlpha2 + fTrap.fDx3) * DzRecip;
  fTrap.fTthetaSphi = (pt[4].y() + fTrap.fDy2) * DzRecip;

  fTrap.fTheta = atan(sqrt(fTrap.fTthetaSphi * fTrap.fTthetaSphi + fTrap.fTthetaCphi * fTrap.fTthetaCphi));
  fTrap.fPhi   = atan2(fTrap.fTthetaSphi, fTrap.fTthetaCphi);

#ifndef VECCORE_CUDA
  // check planarity of all four sides
  bool good = MakePlanes(pt);
  if (!good) VECGEOM_LOG(warning) << "Corners provided fail coplanarity tests";
#endif

  fGlobalConvexity = true;
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedTrapezoid::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{
  SolidMesh *sm = new SolidMesh();
  sm->ResetMesh(8, 6);
  TrapCorners pts;

  FromParametersToCorners(pts);

  sm->SetVertices(pts, 8);
  sm->TransformVertices(trans);

  sm->AddPolygon(4, {1, 0, 2, 3}, true); // bottom
  sm->AddPolygon(4, {5, 7, 6, 4}, true); // top
  sm->AddPolygon(4,
                 {
                     1,
                     5,
                     4,
                     0,
                 },
                 true);
  sm->AddPolygon(4, {0, 4, 6, 2}, true);
  sm->AddPolygon(4, {2, 6, 7, 3}, true);
  sm->AddPolygon(4, {3, 7, 5, 1}, true);

  return sm;
}
#endif

//////////////////////////////////////////////////////////////////////////////
//
// Calculate the coefficients of the plane p1->p2->p3->p4->p1
// where the ThreeVectors 1-4 are in clockwise order when viewed from
// "inside" of the plane (i.e. opposite to normal vector, which points outwards).
//
// Return true if the ThreeVectors are coplanar + set coefficients
//        false if ThreeVectors are not coplanar
//
#ifndef VECGEOM_PLANESHELL
using TrapSidePlane = TrapezoidStruct<Precision>::TrapSidePlane;
#endif

VECCORE_ATT_HOST_DEVICE
bool UnplacedTrapezoid::MakeAPlane(const Vec3D &p1, const Vec3D &p2, const Vec3D &p3, const Vec3D &p4,
#ifdef VECGEOM_PLANESHELL
                                   unsigned int iplane)
#else
                                   TrapSidePlane &plane)
#endif
{
  // Let create diagonals 3-1 and 4-2 than (3-1)x(4-2) provides
  // vector perpendicular to the plane directed to outside !!!
  Vec3D normalVector = (p3 - p1).Cross(p4 - p2);
  normalVector.Normalize();

  // Calculate fD: a centroid is in the best plane, so fD = -n.Dot(centr)
  Vec3D centr = 0.25 * (p1 + p2 + p3 + p4);
  Precision d = -normalVector.Dot(centr);

  // check coplanarity
  bool good = true;
#if !defined(NDEBUG) && !defined(VECCORE_CUDA_DEVICE_COMPILATION)
  Precision resid1 = normalVector.Dot(p1 - centr);
  Precision resid2 = normalVector.Dot(p2 - centr);
  Precision resid3 = normalVector.Dot(p3 - centr);
  Precision resid4 = normalVector.Dot(p4 - centr);
  Precision resid  = Max(Max(fabs(resid1), fabs(resid2)), Max(fabs(resid3), fabs(resid4)));
  if (resid > 1000 * kTolerance) {
    VECGEOM_LOG(warning) << "Coplanarity test fails by residual = " << resid
                         << ".\n"
                            "\tcorner 1: ("
                         << p1.x() << ", " << p1.y() << ", " << p1.z()
                         << ")\n"
                            "\tcorner 2: ("
                         << p2.x() << ", " << p2.y() << ", " << p2.z()
                         << ")\n"
                            "\tcorner 3: ("
                         << p3.x() << ", " << p3.y() << ", " << p3.z()
                         << ")\n"
                            "\tcorner 4: ("
                         << p4.x() << ", " << p4.y() << ", " << p4.z() << ")";

    // We can be very loose here, because we will take a real plane, to replace
    // a non-planar face suggested by input points, up to a maximum residual below
    if (resid > 1000 * kTolerance) good = false;
  }
#endif

#ifdef VECGEOM_PLANESHELL
  fTrap.fPlanes.Set(iplane, normalVector.x(), normalVector.y(), normalVector.z(), d);
#else
  plane.fA = normalVector.x();
  plane.fB = normalVector.y();
  plane.fC = normalVector.z();
  plane.fD = d;

  unsigned int iplane = (&plane - fTrap.fPlanes); // pointer arithmetics used here
#endif

  fTrap.sideAreas[iplane] = 0.5 * ((p2 - p1).Cross(p3 - p1).Mag() + (p3 - p1).Cross(p4 - p1).Mag());
  fTrap.normals[iplane]   = normalVector;
  return good;
}

VECCORE_ATT_HOST_DEVICE
bool UnplacedTrapezoid::MakePlanes()
{
  TrapCorners pt;
  FromParametersToCorners(pt);
  return MakePlanes(pt);
}

VECCORE_ATT_HOST_DEVICE
bool UnplacedTrapezoid::MakePlanes(TrapCorners const pt)
{

  // Checking coplanarity of all four side faces
  bool good = true;

// Bottom side with normal approx. -Y
#ifdef VECGEOM_PLANESHELL
  good = MakeAPlane(pt[0], pt[1], pt[5], pt[4], 0);
#else
  good                = MakeAPlane(pt[0], pt[1], pt[5], pt[4], fTrap.fPlanes[0]);
#endif

#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  if (!good) VECGEOM_LOG(error) << "Face at ~-Y not planar for Solid: UnplacedTrapezoid";
#endif

// Top side with normal approx. +Y
#ifdef VECGEOM_PLANESHELL
  good = MakeAPlane(pt[2], pt[6], pt[7], pt[3], 1);
#else
  good                = MakeAPlane(pt[2], pt[6], pt[7], pt[3], fTrap.fPlanes[1]);
#endif

#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  if (!good) VECGEOM_LOG(error) << "Face at ~+Y not planar for Solid: UnplacedTrapezoid";
#endif

// Front side with normal approx. -X
#ifdef VECGEOM_PLANESHELL
  good = MakeAPlane(pt[0], pt[4], pt[6], pt[2], 2);
#else
  good                = MakeAPlane(pt[0], pt[4], pt[6], pt[2], fTrap.fPlanes[2]);
#endif

#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  if (!good) VECGEOM_LOG(error) << "Face at ~-X not planar for Solid: UnplacedTrapezoid";
#endif

// Back side with normal approx. +X
#ifdef VECGEOM_PLANESHELL
  good = MakeAPlane(pt[1], pt[3], pt[7], pt[5], 3);
#else
  good                = MakeAPlane(pt[1], pt[3], pt[7], pt[5], fTrap.fPlanes[3]);
#endif

#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  if (!good) VECGEOM_LOG(error) << "Face at ~+X not planar for Solid: UnplacedTrapezoid";
#endif

  // include areas for -Z,+Z surfaces
  fTrap.sideAreas[4] = 2 * (fTrap.fDx1 + fTrap.fDx2) * fTrap.fDy1;
  fTrap.sideAreas[5] = 2 * (fTrap.fDx3 + fTrap.fDx4) * fTrap.fDy2;
  fTrap.normals[4]   = Vec3D(0, 0, -1);
  fTrap.normals[5]   = Vec3D(0, 0, 1);

  return good;
}

//===================== specialization stuff
#ifndef VECCORE_CUDA

VPlacedVolume *UnplacedTrapezoid::Create(LogicalVolume const *const logical_volume,
                                         Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedTrapezoid(logical_volume, transformation);
    return placement;
  }
  return new SpecializedTrapezoid(logical_volume, transformation);
}

VPlacedVolume *UnplacedTrapezoid::SpecializedVolume(LogicalVolume const *const volume,
                                                    Transformation3D const *const transformation,
                                                    VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTrapezoid>(volume, transformation, placement);
}

#else

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedTrapezoid::Create(LogicalVolume const *const logical_volume,
                                         Transformation3D const *const transformation, const int id, const int copy_no,
                                         const int child_id, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedTrapezoid(logical_volume, transformation, id, copy_no, child_id);
    return placement;
  }
  return new SpecializedTrapezoid(logical_volume, transformation, id, copy_no, child_id);
}

VECCORE_ATT_DEVICE VPlacedVolume *UnplacedTrapezoid::SpecializedVolume(LogicalVolume const *const volume,
                                                                       Transformation3D const *const transformation,
                                                                       const int id, const int copy_no,
                                                                       const int child_id,
                                                                       VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTrapezoid>(volume, transformation, id, copy_no, child_id,
                                                                  placement);
}

#endif

//========== CUDA stuff
#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTrapezoid::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedTrapezoid>(in_gpu_ptr, fTrap.fDz, fTrap.fTheta, fTrap.fPhi, fTrap.fDy1, fTrap.fDx1,
                                          fTrap.fDx2, this->alpha1(), fTrap.fDy2, fTrap.fDx3, fTrap.fDx4,
                                          this->alpha2());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTrapezoid::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedTrapezoid>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {
template size_t DevicePtr<cuda::UnplacedTrapezoid>::SizeOf();

template void DevicePtr<cuda::UnplacedTrapezoid>::Construct(const Precision dz, const Precision theta,
                                                            const Precision phi, const Precision dy1,
                                                            const Precision dx1, const Precision dx2,
                                                            const Precision tanAlpha1, const Precision dy2,
                                                            const Precision dx3, const Precision dx4,
                                                            const Precision tanAlpha2) const;

} // namespace cxx

#endif

} // namespace vecgeom
