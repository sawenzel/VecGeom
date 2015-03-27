/// \file UnplacedTube.cpp
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/UnplacedTube.h"
#include "volumes/SpecializedTube.h"
#include "backend/Backend.h"
#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
#include <cassert>
#include <cmath>
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
      __attribute__((unused)) const UnplacedTube &tube = static_cast<const UnplacedTube&>( *(logical_volume->unplaced_volume()) );

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


// VECGEOM_CUDA_HEADER_BOTH
#if !defined(VECGEOM_NVCC)
Vector3D<Precision> UnplacedTube::GetPointOnSurface() const {
  // select on which surface to create the point; should be done in proportion to the area of that surface

  // once decided create a point
	Precision rArea = 0.0, phiArea = 0.0, zInArea = 0.0, zOutArea = 0.0;
	Precision rVal, phiVal, zVal;

    int choice = ChooseSurface(rArea, phiArea, zInArea, zOutArea); 
	Precision RRND = RNG::Instance().uniform() * (rmax() - rmin()) + rmin();
	Precision FRND = RNG::Instance().uniform(sphi(), sphi() + dphi());
	Precision ZRND = RNG::Instance().uniform() * z();
	// 0::rTop, 1::rBot, 2::phiLeft, 3::phiRight, 4::zIn, 5::zOut
		
	if (choice == 0) {   // TOP
		//rVal = RNG::Instance().uniform() * (rmax() - rmin()) + rmin();
		rVal = RRND;
		phiVal = FRND;
		zVal = z();
		//phiVal = RNG::Instance().uniform(sphi(), sphi() + dphi());
	} else if (choice == 1) {
		//rVal = RNG::Instance().uniform() * (rmax() - rmin()) + rmin();
		rVal = RRND;
		phiVal = FRND;
		zVal = -1.0 * z();
		//phiVal = RNG::Instance().uniform(sphi(), sphi() + dphi());
	} else if (choice == 2) {
		//rVal = RNG::Instance().uniform() * (rmax() - rmin()) + rmin();
		//zVal = RNG::Instance().uniform() * z();
		rVal = RRND;
		phiVal = sphi();
		zVal = ZRND;
	} else if (choice == 3) {
		//rVal = RNG::Instance().uniform() * (rmax() - rmin()) + rmin();
		//zVal = RNG::Instance().uniform() * z();
		rVal = RRND;
		phiVal = sphi() + dphi(); std::cout << phiVal << "  ";
		zVal = ZRND;
	} else if (choice == 4) {
		rVal = rmin();
		phiVal = FRND;
		zVal = ZRND;
		//zVal = RNG::Instance().uniform() * z();
		//phiVal = RNG::Instance().uniform(sphi(), sphi() + dphi());
	} else if (choice == 5) {
		rVal = rmax();
		phiVal = FRND;
		zVal = ZRND;
		//zVal = RNG::Instance().uniform() * z();
		//phiVal = RNG::Instance().uniform(sphi(), sphi() + dphi());
	}
	 Precision xVal = rVal * cos(phiVal);
	 Precision yVal = rVal * sin(phiVal);
	return Vector3D<Precision>(xVal, yVal, zVal);
}
#endif

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

  VECGEOM_CUDA_HEADER_BOTH
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



  //  This computes where the random point would be placed
  // 1::rTop, 2::rBot, 3::phiLeft, 4::phiRight, 5::zIn, 6::zOut
  VECGEOM_CUDA_HEADER_BOTH
  int UnplacedTube::ChooseSurface(Precision &rArea, Precision &phiArea, Precision &zInArea, Precision &zOutArea) const {
      int i, j, nChoice = 6; 
	  Precision totArea, sumWeight = 0.0, minArea = 9999999.9;
	  
	  rArea = GetTopArea();  		// 50% divide into top and bottom
	  phiArea = GetLateralPhiArea();	// 50% divide into Left and Right
	  zInArea = GetLateralRInArea();	// Inner tube surface
	  zOutArea = GetLateralROutArea();	// Outer tube surface
	  totArea = 2.0*rArea + 2.0*phiArea + zInArea + zOutArea;
	  Array<Precision>prob;
	  Array<int>iprob;

	  // prob contains the value of weightage while iprobe contains the serial number denoting surface

	  prob[0] = rArea/totArea;   iprob[0]=0; minArea = Min(minArea, prob[0]);   sumWeight += prob[0];   // circular top
	  prob[1] = prob[0];	     iprob[1]=1;                                    sumWeight += prob[1];	// circular bottom
	  prob[2] = phiArea/totArea; iprob[2]=2; minArea = Min(minArea, prob[2]);   sumWeight += prob[2];	// phi left
	  prob[3] = prob[2];	     iprob[3]=3;                                    sumWeight += prob[3];	// phi right
	  prob[4] = zInArea/totArea; iprob[4]=4; minArea = Min(minArea, prob[4]);   sumWeight += prob[4];	// Tube Inner Surface
	  prob[5] = zOutArea/totArea;iprob[5]=5; minArea = Min(minArea, prob[5]);   sumWeight += prob[5];	// Tube Outer Surface

	  //  Sort the array
	  Precision tmp1, tmp2;
	  for (i = 0; i < nChoice - 1; i++) {
		  for (j = 0; j < nChoice - 1; j++) {
			  if (prob[j] > prob[j+1]) {
				  tmp1 = prob[j];        tmp2 = iprob[j];
				  prob[j] = prob[j+1];   iprob[j] = iprob[j+1];
				  prob[j+1] = tmp1;      iprob[j+1] = tmp2; // sort iprob also depending on prob value
			  }
		  }
	  }

      // Precision firstRnd = RNG::Instance().uniform() * (sumWeight - minArea) + minArea;
      Precision firstRnd = RNG::Instance().uniform() * sumWeight ;
	 
	  // total number of choices = 6 as rtop, rbot, phileft, phiright, zin, zout
      for(i = 0; i < nChoice; i++) {
		if(firstRnd < prob[i]) return i;
		firstRnd -= prob[i]; 
	  }
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
