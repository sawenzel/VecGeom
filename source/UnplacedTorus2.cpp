/// \file UnplacedTorus2.cpp

#include "volumes/UnplacedTorus2.h"
#include "volumes/SpecializedTorus2.h"
#include "base/RNG.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTorus2::Print() const {
  printf("UnplacedTorus2 {%.2f, %.2f, %.2f, %.2f, %.2f}",
         rmin(), rmax(), rtor(), sphi(), dphi() );
}

void UnplacedTorus2::Print(std::ostream &os) const {
  os << "UnplacedTorus2 {" << rmin() << ", " << rmax() << ", " << rtor()
     << ", " << sphi() << ", " << dphi();
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTorus2::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedTorus2<transCodeT, rotCodeT>(logical_volume,
      transformation
#ifdef VECGEOM_NVCC
      , NULL, id
#endif
);
    return placement;
  }
  return new SpecializedTorus2<transCodeT, rotCodeT>(logical_volume,
                                                  transformation
#ifdef VECGEOM_NVCC
        , NULL, id
#endif
        );
}


VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTorus2::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  return VolumeFactory::CreateByTransformation<
      UnplacedTorus2>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}


Vector3D<Precision> UnplacedTorus2::GetPointOnSurface() const {
    // taken from Geant4
    Precision cosu, sinu,cosv, sinv, aOut, aIn, aSide, chose, phi, theta, rRand;


    phi   = RNG::Instance().uniform( fSphi, fSphi + fDphi );
    theta = RNG::Instance().uniform( 0., vecgeom::kTwoPi );

    cosu   = std::cos(phi);    sinu = std::sin(phi);
    cosv   = std::cos(theta);  sinv = std::sin(theta);

    // compute the areas

    aOut   = (fDphi)* vecgeom::kTwoPi *fRtor*fRmax;
    aIn    = (fDphi)* vecgeom::kTwoPi *fRtor*fRmin;
    aSide  = vecgeom::kPi * (fRmax*fRmax-fRmin*fRmin);

     if ((fSphi == 0.) && (fDphi == vecgeom::kTwoPi )){ aSide = 0; }
     chose = RNG::Instance().uniform(0.,aOut + aIn + 2.*aSide);

     if(chose < aOut)
     {
       return Vector3D<Precision> ((fRtor+fRmax*cosv)*cosu,
                             (fRtor+fRmax*cosv)*sinu, fRmax*sinv);
     }
     else if( (chose >= aOut) && (chose < aOut + aIn) )
     {
       return Vector3D<Precision> ((fRtor+fRmin*cosv)*cosu,
                             (fRtor+fRmin*cosv)*sinu, fRmin*sinv);
     }
     else if( (chose >= aOut + aIn) && (chose < aOut + aIn + aSide) )
     {
       rRand = volumeUtilities::GetRadiusInRing(fRmin,fRmax);
       return Vector3D<Precision> ((fRtor+rRand*cosv)*std::cos(fSphi),
                             (fRtor+rRand*cosv)*std::sin(fSphi), rRand*sinv);
     }
     else
     {
       rRand = volumeUtilities::GetRadiusInRing(fRmin,fRmax);
       return Vector3D<Precision> ((fRtor+rRand*cosv)*std::cos(fSphi+fDphi),
                             (fRtor+rRand*cosv)*std::sin(fSphi+fDphi),
                             rRand*sinv);
     }
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTorus2::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedTorus2>(in_gpu_ptr, fRmin, fRmax, fRtor, fSphi, fDphi);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTorus2::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedTorus2>();
}

#endif // VECGEOM_CUDA_INTERFACE

// Return unit normal of surface closest to p
// - note if point on z axis, ignore phi divided sides
// - unsafe if point close to z axis a rmin=0 - no explicit checks
bool UnplacedTorus2::Normal(Vector3D<Precision> const& point, Vector3D<Precision>& norm) const {

  int noSurfaces = 0;
  bool valid = true;
 
  Precision rho2, rho, pt2, pt, pPhi;
  Precision distRMin = kInfinity;
  Precision distSPhi = kInfinity, distEPhi = kInfinity;

  // To cope with precision loss
  //
  const Precision delta = Max(10.0*kTolerance,
                                  1.0e-8*(fRtor+fRmax));
  const Precision dAngle = 10.0*kTolerance;

  Vector3D<Precision> nR, nPs, nPe;
  Vector3D<Precision>  sumnorm(0.,0.,0.);

  rho2 = point.x()*point.x() + point.y()*point.y();
  rho = Sqrt(rho2);
  pt2 = rho2+point.z()*point.z() +fRtor * (fRtor-2*rho);
  pt2 = Max(pt2, 0.0); // std::fabs(pt2);
  pt = Sqrt(pt2) ;

  Precision distRMax = Abs(pt - fRmax);
  if(fRmin) distRMin = Abs(pt - fRmin);

  if( rho > delta && pt != 0.0 )
  {
    Precision redFactor= (rho-fRtor)/rho;
    nR = Vector3D<Precision>( point.x()*redFactor,  // p.x()*(1.-fRtor/rho),
                        point.y()*redFactor,  // p.y()*(1.-fRtor/rho),
                        point.z()          );
    nR *= 1.0/pt;
  }

  if ( fDphi < kTwoPi ) // && rho ) // old limitation against (0,0,z)
  {
    if ( rho )
    {
      pPhi = std::atan2(point.y(),point.x());

      if(pPhi < fSphi-delta)            { pPhi += kTwoPi; }
      else if(pPhi > fSphi+fDphi+delta) { pPhi -= kTwoPi; }

      distSPhi = Abs( pPhi - fSphi );
      distEPhi = Abs(pPhi-fSphi-fDphi);
    }
    nPs = Vector3D<Precision>( sin(fSphi),-cos(fSphi),0);
    nPe = Vector3D<Precision>(-sin(fSphi+fDphi),cos(fSphi+fDphi),0);
  } 
  if( distRMax <= delta )
  {
    noSurfaces ++;
    sumnorm += nR;
  }
  else if( fRmin && (distRMin <= delta) ) // Must not be on both Outer and Inner
  {
    noSurfaces ++;
    sumnorm -= nR;
  }

  //  To be on one of the 'phi' surfaces,
  //  it must be within the 'tube' - with tolerance

  if( (fDphi < kTwoPi) && (fRmin-delta <= pt) && (pt <= (fRmax+delta)) )
  {
    if (distSPhi <= dAngle)
    {
      noSurfaces ++;
      sumnorm += nPs;
    }
    if (distEPhi <= dAngle) 
    {
      noSurfaces ++;
      sumnorm += nPe;
    }
  }
  if ( noSurfaces == 0 )
  {

    valid = false;
  }
  else if ( noSurfaces == 1 )  { norm = sumnorm; }
  else                         { norm = sumnorm.Unit(); }

 
  return valid ;
}


} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTorus2>::SizeOf();
template void DevicePtr<cuda::UnplacedTorus2>::Construct(
    const Precision rmin, const Precision rmax, const Precision rtor,
    const Precision sphi, const Precision dphi) const;

} // End cxx namespace

#endif

} // End global namespace

