/// \file UnplacedTube.cpp
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/UnplacedTube.h"
#include "volumes/SpecializedTube.h"
#include "backend/Backend.h"
#ifndef VECGEOM_NVCC
#include "base/RNG.h"
#include <cassert>
#include <cmath>
#include <iostream>
#endif

#include "volumes/utilities/GenerationUtilities.h"
#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTube::Print() const {
  printf("UnplacedTube {%.2f, %.2f, %.2f, %.2f, %.2f}",
         rmin(), rmax(), z(), sphi(), dphi() );
}

void UnplacedTube::Print(std::ostream &os) const {
  os << "UnplacedTube {" << rmin() << ", " << rmax() << ", " << z()
     << ", " << sphi() << ", " << dphi() << "}";
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTube::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

      using namespace TubeTypes;
      __attribute__((unused)) const UnplacedTube &tube = static_cast<const UnplacedTube&>( *(logical_volume->GetUnplacedVolume()) );

      #ifdef VECGEOM_NVCC
        #define RETURN_SPECIALIZATION(tubeTypeT) return CreateSpecializedWithPlacement< \
            SpecializedTube<transCodeT, rotCodeT, tubeTypeT> >(logical_volume, transformation, id, placement)
      #else
        #define RETURN_SPECIALIZATION(tubeTypeT) return CreateSpecializedWithPlacement< \
            SpecializedTube<transCodeT, rotCodeT, tubeTypeT> >(logical_volume, transformation, placement)
      #endif

#ifdef GENERATE_TUBE_SPECIALIZATIONS
      if(tube.rmin() <= 0) {
        if(tube.dphi() >= 2*M_PI)  RETURN_SPECIALIZATION(NonHollowTube);
        if(tube.dphi() == M_PI)    RETURN_SPECIALIZATION(NonHollowTubeWithPiSector); // == M_PI ???

        if(tube.dphi() < M_PI)     RETURN_SPECIALIZATION(NonHollowTubeWithSmallerThanPiSector);
        if(tube.dphi() > M_PI)     RETURN_SPECIALIZATION(NonHollowTubeWithBiggerThanPiSector);
      }
      else if(tube.rmin() > 0) {
        if(tube.dphi() >= 2*M_PI)  RETURN_SPECIALIZATION(HollowTube);
        if(tube.dphi() == M_PI)    RETURN_SPECIALIZATION(HollowTubeWithPiSector); // == M_PI ???

        if(tube.dphi() < M_PI)     RETURN_SPECIALIZATION(HollowTubeWithSmallerThanPiSector);
        if(tube.dphi() > M_PI)     RETURN_SPECIALIZATION(HollowTubeWithBiggerThanPiSector);
      }
#endif

      RETURN_SPECIALIZATION(UniversalTube);

      #undef RETURN_SPECIALIZATION
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTube::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  return VolumeFactory::CreateByTransformation<
      UnplacedTube>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}


#ifndef VECGEOM_NVCC

int UnplacedTube::ChooseSurface() const
{
    int choice = 0; // 0 = rTop, 1 = rBot, 2 = phiLeft, 3 = phiRight, 4 = zIn, 5 = zOut
    Precision S[6], Stotal = 0.0;

    S[0] = S[1] = GetTopArea();        // 50% divide into top and bottom
    S[2] = S[3] = GetLateralPhiArea(); // 50% divide into left and right
    S[4] = GetLateralRInArea();        // inner tube surface area
    S[5] = GetLateralROutArea();       // outer tube surface area

    for (int i = 0; i < 6; ++i)
        Stotal += S[i];

    /* random value to choose surface to place the point */
    Precision rand = RNG::Instance().uniform() * Stotal;

    while (rand > S[choice])
        rand -= S[choice], choice++;

    assert(choice < 6);

    return choice;
}

Vector3D<Precision> UnplacedTube::GetPointOnSurface() const
{
    int surface = ChooseSurface();
    Precision rVal   = RNG::Instance().uniform(rmin(), rmax());
    Precision phiVal = RNG::Instance().uniform(sphi(), sphi() + dphi());
    Precision zVal   = RNG::Instance().uniform() * 2.0 * z() - z();

    switch (surface) {
        case 0: zVal =  z(); break;
        case 1: zVal = -z(); break;
        case 2: phiVal = sphi(); break;
        case 3: phiVal = sphi() + dphi(); break;
        case 4: rVal = rmin(); break;
        case 5: rVal = rmax(); break;
    }

    Precision xVal = rVal * cos(phiVal);
    Precision yVal = rVal * sin(phiVal);

    return Vector3D<Precision>(xVal, yVal, zVal);
}

bool UnplacedTube::Normal(Vector3D<Precision> const& point, Vector3D<Precision>& norm) const {
    int nosurface = 0;  // idea from trapezoid;; change nomenclature as confusing

    Precision x2y2 = Sqrt(point.x()*point.x() + point.y()*point.y());
    bool inZ = ((point.z() < fZ + kTolerance ) && (point.z() > -fZ-kTolerance)); // in right z range
    bool inR = ((x2y2 >= fRmin) && (x2y2 <= fRmax)); // in right r range
    // bool inPhi = fWedge.Contains(point);

    // is the point on the surface?
    // Do WE REALLY NEED TO CHECK THIS?? THE CALLEE CAN CHECK IT IF HE LIKES TO...
    // The contract should be that the input to this function is on the surfaces
    // hence:: assert( IsOnSurface ... might be appropriate )
    //onSurf =  ((Abs(x2y2 - fRmin) <= kTolerance) && inZ);  // inner
    //onSurf |= ((Abs(x2y2 - fRmax) <= kTolerance) && inZ); // outer
    //onSurf |= (annul && (Abs(point.z() - fZ) <= kTolerance)); // top
    //onSurf |= (annul && (Abs(point.z() + fZ) <= kTolerance)); // bottom
    //onSurf |= (annul && (Abs(fiVal - fi1) <= kTolerance));  // left i.e. sphi()
    //onSurf |= (annul && (Abs(fiVal - fi2) <= kTolerance));  // right i.e. sphi() + dphi()

    // can we combine these two into one??
    if ( inR && (Abs(point.z() - fZ) <= kTolerance))  { // top lid, normal along +Z
        norm[0] = 0.;
        norm[1] = 0.;
        norm[2] = 1.;
        nosurface++;
    }
    if ( inR && (Abs(point.z() + fZ) <= kTolerance))  {  // bottom base, normal along -Z
        if( nosurface > 0){
        // norm exists already; just add to it
        norm[2] += -1;
        }
        else{
            norm[0] = 0.0;
            norm[1] = 0.0;
            norm[2] = -1;
        }
        nosurface++;
    }
    if( fRmin > 0. ){
        if ( inZ && (Abs(x2y2 - fRmin) <= kTolerance)) { // inner tube wall, normal  towards center
            Precision invx2y2 = 1./x2y2;
            if(nosurface == 0){
                norm[0] = -point[0]*invx2y2;
                norm[1] = -point[1]*invx2y2;   // -ve due to inwards
                norm[2] = 0.0;
            }
            else {
                norm[0] += -point[0]*invx2y2;
                norm[1] += -point[1]*invx2y2;
            }
            nosurface++;
        }
    }
    if ( inZ && (Abs(x2y2 - fRmax) <= kTolerance)) { // outer tube wall, normal outwards
        Precision invx2y2 = 1./x2y2;
        if(nosurface > 0){
            norm[0] += point[0]*invx2y2;
            norm[1] += point[1]*invx2y2;
        }
        else
        {
            norm[0] = point[0]*invx2y2;
            norm[1] = point[1]*invx2y2;
            norm[2] = 0.0;
        }
        nosurface++;
    }

    // otherwise we get a normal from the wedge
    if( fDphi < vecgeom::kTwoPi ){
        if (inR && fPhiWedge.IsOnSurface1(point)){
            if( nosurface == 0)
                norm = -fPhiWedge.GetNormal1();
            else
                norm += -fPhiWedge.GetNormal1();
             nosurface++;
        }
        if (inR && fPhiWedge.IsOnSurface2(point)){
            if ( nosurface == 0)
                norm = -fPhiWedge.GetNormal2();
            else
                norm += -fPhiWedge.GetNormal2();
            nosurface++;
        }
    }
    if( nosurface > 1) norm=norm/std::sqrt(nosurface);
    return nosurface != 0; // this is for testing only
}

/*
  VECGEOM_CUDA_HEADER_BOTH
  Precision UnplacedTube::SurfaceArea () const {
    Precision area = fDphi * (fRmin + fRmax) * (2 * fZ + fRmax - fRmin);
    if (fDphi<kTwoPi) {
      area += 4 * fZ * (fRmax - fRmin);
    }
    return area;
  }

  */
#endif

  void UnplacedTube::Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const {
    // most general case
    aMin = Vector3D<Precision>(-fRmax,-fRmax,-fZ);
    aMax = Vector3D<Precision>( fRmax, fRmax, fZ);

    if(fDphi==kTwoPi) return;

    // check how many of phi=90, 180, 270, 360deg are outside this tube
    auto Rin = 0.5*(fRmax+fRmin);
    bool phi0out   = ! GetWedge().Contains<kScalar>( Vector3D<Precision>( Rin, 0, 0) );
    bool phi90out  = ! GetWedge().Contains<kScalar>( Vector3D<Precision>( 0, Rin, 0) );
    bool phi180out = ! GetWedge().Contains<kScalar>( Vector3D<Precision>(-Rin, 0, 0) );
    bool phi270out = ! GetWedge().Contains<kScalar>( Vector3D<Precision>( 0,-Rin, 0) );

    // if none of those 4 phis is outside, largest box still required
    if( !(phi0out || phi90out || phi180out || phi270out) ) return;

    // some extent(s) of box will be reduced
    // --> think of 4 points A,B,C,D such that A,B are at Rmin, C,D at Rmax
    //     and A,C at startPhi (fSphi), B,D at endPhi (fSphi+fDphi)
    auto Cx = fRmax * cos(fSphi);
    auto Dx = fRmax * cos(fSphi+fDphi);
    auto Cy = fRmax * sin(fSphi);
    auto Dy = fRmax * sin(fSphi+fDphi);

    // then rewrite box sides whenever each one of those phis are not contained in the tube section
    if( phi0out )   aMax.x() = Max( Cx, Dx );
    if( phi90out )  aMax.y() = Max( Cy, Dy );
    if( phi180out ) aMin.x() = Min( Cx, Dx );
    if( phi270out ) aMin.y() = Min( Cy, Dy );

    if(fDphi>=kPi) return;

    auto Ax = fRmin * cos(fSphi);
    auto Bx = fRmin * cos(fSphi+fDphi);
    auto Ay = fRmin * sin(fSphi);
    auto By = fRmin * sin(fSphi+fDphi);

    Precision temp;
    temp = Max(Ax, Bx);
    aMax.x() =  temp > aMax.x() ? temp : aMax.x();

    temp = Max(Ay, By);
    aMax.y() =  temp > aMax.y() ? temp : aMax.y();

    temp = Min(Ax, Bx);
    aMin.x() =  temp < aMin.x() ? temp : aMin.x();

    temp = Min(Ay, By);
    aMin.y() =  temp < aMin.y() ? temp : aMin.y();

    return;
  }


#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTube::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedTube>(in_gpu_ptr, rmin(), rmax(), z(), sphi(), dphi());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTube::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedTube>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTube>::SizeOf();
template void DevicePtr<cuda::UnplacedTube>::Construct(
    const Precision rmin, const Precision rmax, const Precision z,
    const Precision sphi, const Precision dphi) const;

} // End cxx namespace

#endif

} // End global namespace
