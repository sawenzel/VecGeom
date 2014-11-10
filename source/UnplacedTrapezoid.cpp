/**
 * @file   source/UnplacedTrapezoid.cpp
 * @author Guilherme Lima (lima 'at' fnal 'dot' gov)
 *
 * 140407 G.Lima - based on equivalent box code
 */

#include "volumes/UnplacedTrapezoid.h"

#include "management/VolumeFactory.h"
#include "volumes/SpecializedTrapezoid.h"
#include "volumes/utilities/GenerationUtilities.h"

#include <stdio.h>

#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
  #include <cmath>
#endif

namespace VECGEOM_NAMESPACE {

typedef Vector3D<Precision> Vec3D;

/// angles theta,phi are in radians
VECGEOM_CUDA_HEADER_BOTH
UnplacedTrapezoid::UnplacedTrapezoid(Precision pDz, Precision pTheta, Precision pPhi,
                           Precision pDy1, Precision pDx1, Precision pDx2, Precision pTanAlpha1,
                           Precision pDy2, Precision pDx3, Precision pDx4, Precision pTanAlpha2 )
  : fDz(pDz), fTheta(pTheta), fPhi(pPhi)
  , fDy1(pDy1), fDx1(pDx1), fDx2(pDx2), fTanAlpha1(pTanAlpha1)
  , fDy2(pDy2), fDx3(pDx3), fDx4(pDx4), fTanAlpha2(pTanAlpha2)
  , fTthetaCphi(0.f), fTthetaSphi(0.f), fbbx(0.f), fbby(0.f), fbbz(0.f), fPlanes()
{
    // validity check
    if( pDz <= 0 || pDy1 <= 0 || pDx1 <= 0 ||
        pDx2 <= 0 || pDy2 <= 0 || pDx3 <= 0 || pDx4 <= 0 ) {

      printf("UnplacedTrapezoid(pDz,...) - GeomSolids0002, Fatal Exception: Invalid input length parameters for Solid: UnplacedTrapezoid\n");
      printf("\t X=%f, %f, %f, %f", pDx1, pDx2, pDx3, pDx4);
      printf("\t Y=%f, %f", pDy1, pDy2);
      printf("\t Z=%f\n", pDz);

      // force a crash in a CPU/GPU portable way -- any better (graceful) way to do this?
      Assert(true);  // -- Fatal Exception: Invalid input length parameters for Solid: UnplacedTrapezoid
    }

    fTthetaSphi = tan(pTheta)*sin(pPhi);
    fTthetaCphi = tan(pTheta)*cos(pPhi);
    MakePlanes();
}

  UnplacedTrapezoid::UnplacedTrapezoid(Precision xbox, Precision ybox, Precision zbox)
    : fDz(zbox), fTheta(0.f), fPhi(0.f)
    , fDy1(ybox), fDx1(xbox), fDx2(xbox), fTanAlpha1(0.f)
    , fDy2(ybox), fDx3(xbox), fDx4(xbox), fTanAlpha2(0.f)
    , fTthetaCphi(0.f), fTthetaSphi(0.f), fbbx(xbox), fbby(ybox), fbbz(zbox), fPlanes()
{
    // validity check
  if( xbox <= 0 || ybox <= 0 || zbox <= 0 ) {
      printf("UnplacedTrapezoid(xbox,...) - GeomSolids0002, Fatal Exception: Invalid input length parameters for Solid: UnplacedTrapezoid\n");
      printf("\t X=%f, Y=%f, Z=%f", xbox, ybox, zbox);

      // force a crash in a CPU/GPU portable way -- any better (graceful) way to do this?
      Assert(true);  // -- Fatal Exception: Invalid input length parameters for Solid: UnplacedTrapezoid
    }

    MakePlanes();
}

UnplacedTrapezoid::UnplacedTrapezoid(Precision const* params )
  : fDz( params[0] )
  , fTheta( params[1] )
  , fPhi( params[2] )
  , fDy1( params[3] )
  , fDx1( params[4] )
  , fDx2( params[5] )
  , fTanAlpha1(params[6] )
  , fDy2( params[7] )
  , fDx3( params[8] )
  , fDx4( params[9] )
  , fTanAlpha2(params[10] )
  , fTthetaCphi(0)
  , fTthetaSphi(0)
  , fbbx(0.f), fbby(0.f), fbbz(0.f), fPlanes()
{
  Precision const& theta = params[1];
  Precision const& phi   = params[2];
  fTthetaSphi = tan(theta)*sin(phi);
  fTthetaCphi = tan(theta)*cos(phi);
  MakePlanes();
}

UnplacedTrapezoid::UnplacedTrapezoid( TrapCorners_t const& corners )
  : fDz(0.f), fTheta(0.f), fPhi(0.f)
  , fDy1(0.f), fDx1(0.f), fDx2(0.f), fTanAlpha1(0.f)
  , fDy2(0.f), fDx3(0.f), fDx4(0.f), fTanAlpha2(0.f)
  , fTthetaCphi(0.f), fTthetaSphi(0.f), fbbx(0.f), fbby(0.f), fbbz(0.f), fPlanes()
{
  // check planarity of all four sides
  bool good = MakePlanes(corners);
  Assert( good ); // ERROR: corners provided fail coplanarity tests.

  // fill data members
  fromCornersToParameters(corners);
}

UnplacedTrapezoid::UnplacedTrapezoid( UnplacedTrapezoid const& other )
  : fDz(other.fDz)
  , fTheta(other.fTheta)
  , fPhi(other.fPhi)
  , fDy1(other.fDy1)
  , fDx1(other.fDx1)
  , fDx2(other.fDx2)
  , fTanAlpha1(other.fTanAlpha1)
  , fDy2(other.fDy2)
  , fDx3(other.fDx3)
  , fDx4(other.fDx4)
  , fTanAlpha2(other.fTanAlpha2)
  , fTthetaCphi(other.fTthetaCphi)
  , fTthetaSphi(other.fTthetaSphi)
  , fbbx(0.f), fbby(0.f), fbbz(0.f), fPlanes()
{
  MakePlanes();
}

UnplacedTrapezoid& UnplacedTrapezoid::operator=( UnplacedTrapezoid const& other ) {

  fDz =other.fDz;
  fTheta = other.fTheta;
  fPhi = other.fPhi;
  fDy1 = other.fDy1;
  fDx1 = other.fDx1;
  fDx2 = other.fDx2;
  fTanAlpha1 = other.fTanAlpha1;
  fDy2 = other.fDy2;
  fDx3 = other.fDx3;
  fDx4 = other.fDx4;
  fTanAlpha2 = other.fTanAlpha2;
  fTthetaSphi = other.fTthetaSphi;
  fTthetaCphi = other.fTthetaCphi;

  MakePlanes();

  return *this;
}

UnplacedTrapezoid::~UnplacedTrapezoid() {};

void UnplacedTrapezoid::Print() const {
  printf("UnplacedTrapezoid {%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f}\n",
         GetDz(), GetDy1(), GetDx1(), GetDx2(), GetTanAlpha1(), GetDy2(), GetDx3(), GetDx4(), GetTanAlpha2(),
         GetTanThetaSinPhi(), GetTanThetaCosPhi() );
}

void UnplacedTrapezoid::Print(std::ostream &os) const {
  os << "UnplacedTrapezoid {"
     <<' '<< GetDz()
     <<' '<< GetDy1()
     <<' '<< GetDx1()
     <<' '<< GetDx2()
     <<' '<< GetTanAlpha1()
     <<' '<< GetDy2()
     <<' '<< GetDx3()
     <<' '<< GetDx4()
     <<' '<< GetTanAlpha2()
     <<' '<< GetTanThetaSinPhi()
     <<' '<< GetTanThetaCosPhi()
     <<"}\n";
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTrapezoid::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

  // return new(placement) SpecializedTrapezoid<transCodeT, rotCodeT>(
  return CreateSpecializedWithPlacement<SpecializedTrapezoid<transCodeT, rotCodeT> >(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, id, placement); // TODO: add bounding box?
#else
      logical_volume, transformation, placement);
#endif
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTrapezoid::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<UnplacedTrapezoid>(
    volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
    id,
#endif
    placement);
}

bool UnplacedTrapezoid::MakePlanes() {
  TrapCorners_t pt;
  fromParametersToCorners(pt);
  return MakePlanes(pt);
}

bool UnplacedTrapezoid::MakePlanes(TrapCorners_t const & pt) {

  // Checking coplanarity of all four side faces
  bool good = true;

  // Bottom side with normal approx. -Y
#ifndef VECGEOM_PLANESHELL_DISABLE
  good = MakePlane(pt[0],pt[1],pt[5],pt[4],0);
#else
  good = MakePlane(pt[0],pt[1],pt[5],pt[4],fPlanes[0]);
#endif
  Assert( good ); // GeomSolids0002 - Face at ~-Y not planar for Solid: UnplacedTrapezoid

  // Top side with normal approx. +Y
#ifndef VECGEOM_PLANESHELL_DISABLE
  good = MakePlane(pt[2],pt[6],pt[7],pt[3],1);
#else
  good = MakePlane(pt[2],pt[6],pt[7],pt[3],fPlanes[1]);
#endif
  Assert( good ); // GeomSolids0002 - Face at ~+Y not planar for Solid: UnplacedTrapezoid

  // Front side with normal approx. -X
#ifndef VECGEOM_PLANESHELL_DISABLE
  good = MakePlane(pt[0],pt[4],pt[6],pt[2],2);
#else
  good = MakePlane(pt[0],pt[4],pt[6],pt[2],fPlanes[2]);
#endif
  Assert( good ); // GeomSolids0002 - Face at ~-X not planar for Solid: UnplacedTrapezoid

  // Back side with normal approx. +X
#ifndef VECGEOM_PLANESHELL_DISABLE
  good = MakePlane(pt[1],pt[3],pt[7],pt[5],3);
#else
  good = MakePlane(pt[1],pt[3],pt[7],pt[5],fPlanes[3]);
#endif
  Assert( good ); // GeomSolids0002 - Face at ~+X not planar for Solid: UnplacedTrapezoid

  // include areas for -Z,+Z surfaces
  sideAreas[4] = 2*(fDx1+fDx2)*fDy1;
  sideAreas[5] = 2*(fDx3+fDx4)*fDy2;

  return good;
}

//////////////////////////////////////////////////////////////////////////////
//
// Calculate the coef's of the plane p1->p2->p3->p4->p1
// where the ThreeVectors 1-4 are in clockwise order when viewed from
// "inside" of the plane (i.e. opposite to normal vector, which points outwards).
//
// Return true if the ThreeVectors are coplanar + set coef;s
//        false if ThreeVectors are not coplanar

bool UnplacedTrapezoid::MakePlane(
    const Vec3D& p1,
    const Vec3D& p2,
    const Vec3D& p3,
    const Vec3D& p4,
#ifndef VECGEOM_PLANESHELL_DISABLE
    unsigned int iplane)
#else
    TrapSidePlane& plane )
#endif
{
  bool good;
  Precision a, b, c, norm;
  Vec3D v12, v13, v14, Vcross;

  v12    = p2 - p1;
  v13    = p3 - p1;
  v14    = p4 - p1;
  Vcross = v12.Cross(v13);

  // check coplanarity
  if (std::fabs( v14.Dot(Vcross)/(Vcross.Mag()*v14.Mag()) ) > kTolerance)  {
    Assert( false ); // "UnplacedTrapezoid: ERROR: Coplanarity test failure!
    good = false;
  }
  else {
    // a,b,c correspond to the x/y/z components of the
    // normal vector to the plane

    // Let create diagonals 3-1 and 4-2 than (3-1)x(4-2) provides
    // vector perpendicular to the plane directed to outside !!!
    // and a,b,c, = f(1,2,3,4) external relative to trapezoid normal

    //??? can these be optimized?
    a = +(p3.y() - p1.y())*(p4.z() - p2.z())
       - (p4.y() - p2.y())*(p3.z() - p1.z());

    b = -(p3.x() - p1.x())*(p4.z() - p2.z())
       + (p4.x() - p2.x())*(p3.z() - p1.z());

    c = +(p3.x() - p1.x())*(p4.y() - p2.y())
       - (p4.x() - p2.x())*(p3.y() - p1.y());

    norm = 1.0 / std::sqrt( a*a + b*b + c*c ); // normalization factor, always positive

#ifndef VECGEOM_PLANESHELL_DISABLE
    a *= norm;
    b *= norm;
    c *= norm;

    // Calculate fD: p1 is in plane so fD = -n.p1.Vect()
    Precision d = -( a*p1.x() + b*p1.y() + c*p1.z() );

    fPlanes.Set( iplane, a, b, c, d );
#else
    plane.fA = a*norm;
    plane.fB = b*norm;
    plane.fC = c*norm;

    // Calculate fD: p1 is in plane so fD = -n.p1.Vect()
    plane.fD = -( plane.fA*p1.x() + plane.fB*p1.y() + plane.fC*p1.z() );

    unsigned int iplane = (&plane - fPlanes);  // pointer arithmetics used here
#endif

    sideAreas[iplane] = 0.5* ( Vcross.Mag() + v13.Cross(v14).Mag() );
    good = true;
  }
  return good;
}

#ifdef VECGEOM_USOLIDS
VECGEOM_CUDA_HEADER_BOTH
bool UnplacedTrapezoid::Normal(Vector3D<Precision> const& point, Vector3D<Precision>& norm) const {

  int noSurfaces = 0;
  Vec3D sumnorm(0., 0., 0.), vecnorm(0.,0.,0.);
  Precision distz;

#ifndef VECGEOM_PLANESHELL_DISABLE
  Precision distances[4];
  fPlanes.DistanceToPoint( point, distances );
  for(unsigned int i=0; i<4; ++i) {
    if ( std::fabs(distances[i]) <= kHalfTolerance) {
      noSurfaces ++;
      sumnorm += Vec3D( fPlanes.fA[i], fPlanes.fB[i], fPlanes.fC[i] );
    }
  }

  distz = std::fabs(point[2]) - fDz;
#else
  distz = std::fabs(std::fabs(point[2]) - fDz);

  Precision distx, distmx, disty, distmy;
  distmy = std::fabs(fPlanes[0].fA * point[0] + fPlanes[0].fB * point[1]
                     + fPlanes[0].fC * point[2] + fPlanes[0].fD);

  disty = std::fabs(fPlanes[1].fA * point[0] + fPlanes[1].fB * point[1]
                    + fPlanes[1].fC * point[2] + fPlanes[1].fD);

  distmx = std::fabs(fPlanes[2].fA * point[0] + fPlanes[2].fB * point[1]
                     + fPlanes[2].fC * point[2] + fPlanes[2].fD);

  distx = std::fabs(fPlanes[3].fA * point[0] + fPlanes[3].fB * point[1]
                    + fPlanes[3].fC * point[2] + fPlanes[3].fD);


  if (distx <= kHalfTolerance) {
    noSurfaces ++;
    sumnorm += Vec3D( fPlanes[3].fA, fPlanes[3].fB, fPlanes[3].fC );
  }
  if (distmx <= kHalfTolerance) {
    noSurfaces ++;
    sumnorm += Vec3D( fPlanes[2].fA, fPlanes[2].fB, fPlanes[2].fC );
  }
  if (disty <= kHalfTolerance) {
    noSurfaces ++;
    sumnorm += Vec3D( fPlanes[1].fA, fPlanes[1].fB, fPlanes[1].fC );
  }
  if (distmy <= kHalfTolerance) {
    noSurfaces ++;
    sumnorm += Vec3D( fPlanes[0].fA, fPlanes[0].fB, fPlanes[0].fC );
  }
#endif

  if ( std::fabs(distz) <= kHalfTolerance) {
    noSurfaces ++;
    if (point[2] >= 0.)  sumnorm += Vec3D(0.,0.,1.);
    else                 sumnorm -= Vec3D(0.,0.,1.);
  }
  if (noSurfaces == 0) {
#ifdef UDEBUG
    UUtils::Exception("UnplacedTrapezoid::SurfaceNormal(point)", "GeomSolids1002",
                      Warning, 1, "Point is not on surface.");
#endif
    vecnorm = ApproxSurfaceNormal( Vec3D(point[0],point[1],point[2]) );
    // vecnorm = Vec3D(0,0,1);  // any plane will do it, since false is returned, so save the CPU cycles...
  }
  else if (noSurfaces == 1) vecnorm = sumnorm;
  else                      vecnorm = sumnorm.Unit();

  norm[0] = vecnorm[0];
  norm[1] = vecnorm[1];
  norm[2] = vecnorm[2];

  return noSurfaces != 0;
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedTrapezoid::Extent(Vec3D& aMin, Vec3D& aMax) const {
  aMin.z() = -fDz;
  aMax.z() = fDz;

  TrapCorners_t pt;
  this->fromParametersToCorners(pt);

  Precision ext01 = std::max(pt[0].x(),pt[1].x());
  Precision ext23 = std::max(pt[2].x(),pt[3].x());
  Precision ext45 = std::max(pt[4].x(),pt[5].x());
  Precision ext67 = std::max(pt[6].x(),pt[7].x());
  Precision extA = ext01>ext23 ? ext01 : ext23;
  Precision extB = ext45>ext67 ? ext45 : ext67;
  aMax.x() = (extA > extB) ? extA : extB;

  ext01 = std::min(pt[0].x(),pt[1].x());
  ext23 = std::min(pt[2].x(),pt[3].x());
  ext45 = std::min(pt[4].x(),pt[5].x());
  ext67 = std::min(pt[6].x(),pt[7].x());
  extA = ext01<ext23 ? ext01 : ext23;
  extB = ext45<ext67 ? ext45 : ext67;
  aMin.x() = (extA < extB) ? extA : extB;

  ext01 = std::max(pt[0].y(),pt[1].y());
  ext23 = std::max(pt[2].y(),pt[3].y());
  ext45 = std::max(pt[4].y(),pt[5].y());
  ext67 = std::max(pt[6].y(),pt[7].y());
  extA = ext01>ext23 ? ext01 : ext23;
  extB = ext45>ext67 ? ext45 : ext67;
  aMax.y() = (extA > extB) ? extA : extB;

  ext01 = std::min(pt[0].y(),pt[1].y());
  ext23 = std::min(pt[2].y(),pt[3].y());
  ext45 = std::min(pt[4].y(),pt[5].y());
  ext67 = std::min(pt[6].y(),pt[7].y());
  extA = ext01<ext23 ? ext01 : ext23;
  extB = ext45<ext67 ? ext45 : ext67;
  aMin.y() = (extA < extB) ? extA : extB;
}
#endif

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedTrapezoid::SurfaceArea() const {

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
  cr = ba.Cross(bc);
  Precision babc = cr.Mag();
  cr = dc.Cross(da);
  Precision dcda = cr.Mag();
  cr = ef.Cross(eh);
  Precision efeh = cr.Mag();
  cr = gh.Cross(gf);
  Precision ghgf = cr.Mag();

  Precision surfArea = 2 * fDy1 * (fDx1 + fDx2) + 2 * fDy2 * (fDx3 + fDx4)
    + (fDx1 + fDx3) * std::sqrt(4 * fDz * fDz + std::pow(fDy2 - fDy1 - 2 * fDz * fTthetaSphi, 2))
    + (fDx2 + fDx4) * std::sqrt(4 * fDz * fDz + std::pow(fDy2 - fDy1 + 2 * fDz * fTthetaSphi, 2))
    + 0.5 * (babc + dcda + efeh + ghgf);

  return surfArea;
}

#ifdef VECGEOM_USOLIDS
Vec3D UnplacedTrapezoid::GetPointOnSurface() const {

  TrapCorners_t pt;
  this->fromParametersToCorners(pt);

  // make sure we provide the points in a clockwise fashion

  Precision chose = RNG::Instance().uniform() * SurfaceArea();

  Precision sumArea = 0.0;
  if ((chose >= sumArea) && (chose < sumArea+sideAreas[0])) {
    return GetPointOnPlane(pt[0], pt[4], pt[5], pt[1]);
  }

  sumArea += sideAreas[0];
  if ((chose >= sumArea) && (chose < sumArea+sideAreas[1])) {
    return GetPointOnPlane(pt[2], pt[3], pt[7], pt[6]);
  }

  sumArea += sideAreas[1];
  if ((chose >= sumArea) && (chose < sumArea+sideAreas[2])) {
    return GetPointOnPlane(pt[0], pt[2], pt[6], pt[4]);
  }

  sumArea += sideAreas[2];
  if ((chose >= sumArea) && (chose < sumArea+sideAreas[3])) {
    return GetPointOnPlane(pt[1], pt[5], pt[7], pt[3]);
  }

  sumArea += sideAreas[3];
  if ((chose >= sumArea) && (chose < sumArea+sideAreas[4])) {
    return GetPointOnPlane(pt[0], pt[1], pt[3], pt[2]);
  }

  sumArea += sideAreas[4];
  if ((chose >= sumArea) && (chose < sumArea+sideAreas[5])) {
    return GetPointOnPlane(pt[4], pt[6], pt[7], pt[5]);
  }

  // should never get here...
  return Vec3D(0.,0.,0.);
}

VECGEOM_CUDA_HEADER_BOTH
Vec3D UnplacedTrapezoid::GetPointOnPlane(Vec3D p0, Vec3D p1, Vec3D p2, Vec3D p3) const {

  Precision lambda1, lambda2, chose, aOne, aTwo;
  Vec3D t, u, v, w, Area, normal;

  t = p1 - p0;
  u = p2 - p1;
  v = p3 - p2;
  w = p0 - p3;

  Area = Vec3D(w.y() * v.z() - w.z() * v.y(),
               w.z() * v.x() - w.x() * v.z(),
               w.x() * v.y() - w.y() * v.x());

  aOne = 0.5 * Area.Mag();

  Area = Vec3D(t.y() * u.z() - t.z() * u.y(),
               t.z() * u.x() - t.x() * u.z(),
               t.x() * u.y() - t.y() * u.x());

  aTwo = 0.5 * Area.Mag();

  chose = UUtils::Random(0., aOne + aTwo);

  if ((chose >= 0.) && (chose < aOne)) {
    lambda1 = UUtils::Random(0., 1.);
    lambda2 = UUtils::Random(0., lambda1);
    return (p2 + lambda1 * v + lambda2 * w);
  }

  // else

  lambda1 = UUtils::Random(0., 1.);
  lambda2 = UUtils::Random(0., lambda1);

  return (p0 + lambda1 * t + lambda2 * u);
}
#endif

VECGEOM_CUDA_HEADER_BOTH
Vec3D UnplacedTrapezoid::ApproxSurfaceNormal(const Vec3D& point) const {
  Precision safe = kInfinity, Dist, safez;
  int i, imin = 0;
  for (i = 0; i < 4; i++) {
#ifndef VECGEOM_PLANESHELL_DISABLE
    Dist = std::fabs( fPlanes.fA[i] * point.x() + fPlanes.fB[i] * point.y()
                    + fPlanes.fC[i] * point.z() + fPlanes.fD[i] );
#else
    Dist = std::fabs( fPlanes[i].fA * point.x() + fPlanes[i].fB * point.y()
                    + fPlanes[i].fC * point.z() + fPlanes[i].fD );
#endif
    if (Dist < safe) {
      safe = Dist;
      imin = i;
    }
  }
  safez = std::fabs(std::fabs(point.z()) - fDz);
  if (safe < safez) {
#ifndef VECGEOM_PLANESHELL_DISABLE
    return Vec3D(fPlanes.fA[imin], fPlanes.fB[imin], fPlanes.fC[imin]);
#else
    return Vec3D(fPlanes[imin].fA, fPlanes[imin].fB, fPlanes[imin].fC);
#endif
  }
  else {
    if (point.z() > 0) {
      return Vec3D(0, 0, 1);
    }
    else {
      return Vec3D(0, 0, -1);
    }
  }
  // should never reach this point
  return Vec3D(0.,0.,0.);
}

VECGEOM_CUDA_HEADER_BOTH
Precision UnplacedTrapezoid::Volume() const {

  // cubic approximation used in Geant4
  Precision cubicVolume = fDz*( (fDx1+fDx2+fDx3+fDx4)*(fDy1 + fDy2)
                                + (fDx4+fDx3-fDx2-fDx1)*(fDy2 - fDy1)/3.0 );

  /*
  // GL: leaving this
    // accurate volume calculation
    TrapCorners_t pt;
    this->fromParametersToCorners(pt);

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

    return cubicVolume;
}

void UnplacedTrapezoid::fromCornersToParameters( TrapCorners_t const& pt) {

    fDz = pt[7].z();
    Precision DzRecip = 1.0 / fDz;

    fDy1 = 0.50*( pt[2].y() - pt[0].y() );
    fDx1 = 0.50*( pt[1].x() - pt[0].x() );
    fDx2 = 0.50*( pt[3].x() - pt[2].x() );
    fTanAlpha1 = 0.25*( pt[2].x() + pt[3].x() - pt[1].x() - pt[0].x() ) / fDy1;

    fDy2 = 0.50*( pt[6].y() - pt[4].y() );
    fDx3 = 0.50*( pt[5].x() - pt[4].x() );
    fDx4 = 0.50*( pt[7].x() - pt[6].x() );
    fTanAlpha2 = 0.25*( pt[6].x() + pt[7].x() - pt[5].x() - pt[4].x() ) / fDy2;

    fTthetaCphi = ( pt[4].x() + fDy2*fTanAlpha2 + fDx3) * DzRecip;
    fTthetaSphi = ( pt[4].y() + fDy2) * DzRecip;

    fTheta = atan(sqrt(fTthetaSphi*fTthetaSphi+fTthetaCphi*fTthetaCphi));
    fPhi   = atan2(fTthetaSphi, fTthetaCphi);
  }

  void UnplacedTrapezoid::fromParametersToCorners( TrapCorners_t& pt ) const {

      // hopefully the compiler will optimize the repeated multiplications ... to be checked!
      pt[0] = Vec3D(-fDz*fTthetaCphi-fDy1*fTanAlpha1-fDx1, -fDz*fTthetaSphi-fDy1, -fDz);
      pt[1] = Vec3D(-fDz*fTthetaCphi-fDy1*fTanAlpha1+fDx1, -fDz*fTthetaSphi-fDy1, -fDz);
      pt[2] = Vec3D(-fDz*fTthetaCphi+fDy1*fTanAlpha1-fDx2, -fDz*fTthetaSphi+fDy1, -fDz);
      pt[3] = Vec3D(-fDz*fTthetaCphi+fDy1*fTanAlpha1+fDx2, -fDz*fTthetaSphi+fDy1, -fDz);
      pt[4] = Vec3D(+fDz*fTthetaCphi-fDy2*fTanAlpha2-fDx3, +fDz*fTthetaSphi-fDy2, +fDz);
      pt[5] = Vec3D(+fDz*fTthetaCphi-fDy2*fTanAlpha2+fDx3, +fDz*fTthetaSphi-fDy2, +fDz);
      pt[6] = Vec3D(+fDz*fTthetaCphi+fDy2*fTanAlpha2-fDx4, +fDz*fTthetaSphi+fDy2, +fDz);
      pt[7] = Vec3D(+fDz*fTthetaCphi+fDy2*fTanAlpha2+fDx4, +fDz*fTthetaSphi+fDy2, +fDz);

  }

  VECGEOM_CUDA_HEADER_BOTH
  std::ostream& UnplacedTrapezoid::StreamInfo(std::ostream &os) const {
    Assert( 0 ); // Not implemented yet
    return os;
  }

} // End global namespace


namespace vecgeom {
// only the GPU-related methods should go inside this namespace
#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedTrapezoid_CopyToGpu(
    Precision dz, Precision theta, Precision phi,
    Precision dy1, Precision dx1, Precision dx2, Precision pTanAlpha1,
    Precision dy2, Precision dx3, Precision dx4, Precision pTanAlpha2,
    VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedTrapezoid::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedTrapezoid_CopyToGpu(GetDz(), GetTheta(), GetPhi(),
                              GetDy1(), GetDx1(), GetDx2(), GetTanAlpha1(),
                              GetDy2(), GetDx3(), GetDx4(), GetTanAlpha2(),
                              gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedTrapezoid::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedTrapezoid>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

  __global__
  void UnplacedTrapezoid_ConstructOnGpu(
    const Precision dz, const Precision theta, const Precision phi,
    const Precision dy1, const Precision dx1, const Precision dx2, const Precision tanAlpha1,
    const Precision dy2, const Precision dx3, const Precision dx4, const Precision tanAlpha2,
    VUnplacedVolume *const gpu_ptr)
  {
    new(gpu_ptr) vecgeom_cuda::UnplacedTrapezoid(dz, theta, phi,
                                                 dy1, dx1, dx2, tanAlpha1,
                                                 dy2, dx3, dx4, tanAlpha2 );
  }

  void UnplacedTrapezoid_CopyToGpu(
    const Precision dz, const Precision theta, const Precision phi,
    const Precision dy1, const Precision dx1, const Precision dx2, const Precision tanAlpha1,
    const Precision dy2, const Precision dx3, const Precision dx4, const Precision tanAlpha2,
    VUnplacedVolume *const gpu_ptr)
  {
    UnplacedTrapezoid_ConstructOnGpu<<<1, 1>>>(dz, theta, phi,
                                               dy1, dx1, dx2, tanAlpha1,
                                               dy2, dx3, dx4, tanAlpha2,
                                               gpu_ptr);
  }
#endif // VECGEOM_NVCC

} // End namespace vecgeom

