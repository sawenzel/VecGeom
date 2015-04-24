/// \file UnplacedHype.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/UnplacedHype.h"

#include "management/VolumeFactory.h"
#include "volumes/SpecializedHype.h"
//#include "volumes/kernel/shapetypes/HypeTypes.h"
#include "volumes/utilities/GenerationUtilities.h"

#include <stdio.h>
#include "base/RNG.h"
#include "base/Global.h"

namespace VECGEOM_NAMESPACE {

    
    VECGEOM_CUDA_HEADER_BOTH
    void UnplacedHype::SetParameters(const Precision rMin, const Precision stIn,
                                     const Precision rMax, const Precision stOut,
                                     const Precision dz){
        
        //TODO: add eventual check
        fRmin=rMin;
        fStIn=stIn;
        fRmax=rMax;
        fStOut=stOut;
        fDz=dz;
    }
    
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedHype::UnplacedHype(const Precision rMin, const Precision stIn,
                               const Precision rMax, const Precision stOut,
                               const Precision dz){
        
        SetParameters(rMin, stIn, rMax, stOut, dz);
        
        fTIn=tan(fStIn*kDegToRad);          //Tangent of the Inner stereo angle
        fTOut=tan(fStOut*kDegToRad);        //Tangent of the Outer stereo angle
        fTIn2=fTIn*fTIn;                    //squared value of fTIn
        fTOut2=fTOut*fTOut;                 //squared value of fTOut
        
        fTIn2Inv=1./fTIn2;
        fTOut2Inv=1./fTOut2;
        
        fRmin2=fRmin*fRmin;
        fRmax2=fRmax*fRmax;
        fDz2=fDz*fDz;
        
        fEndInnerRadius2=fTIn2*fDz2+fRmin2;
        fEndOuterRadius2=fTOut2*fDz2+fRmax2;
        fEndInnerRadius=Sqrt(fEndInnerRadius2);
        fEndOuterRadius=Sqrt(fEndOuterRadius2);
        fInSqSide=Sqrt(2)*fRmin;

        CalcCapacity();
		CalcSurfaceArea(); 
        
    }

//__________________New Added function_______________________________________________

VECGEOM_CUDA_HEADER_BOTH
bool UnplacedHype::InnerSurfaceExists() const{
 return (fRmin > 0.) || (fStIn != 0.);
}

void UnplacedHype::CalcCapacity()
  {
      if (fCubicVolume != 0.)
        {
         ;
        }
      else
        {
             fCubicVolume = Volume(true)-Volume(false);
        }
      
  }

Precision UnplacedHype::Volume(bool outer)
  {
	 if (outer)
		return 2*kPi*fDz* ((fRmax)*(fRmax) + (fDz2*fTOut2/3.));
	 else
		return 2*kPi*fDz* ((fRmin)*(fRmin) + (fDz2*fTIn2/3.));
  }


void UnplacedHype::CalcSurfaceArea()
  {
      
      if (fSurfaceArea != 0.)
        {
          ;
        }
      else
         {
			  fSurfaceArea = Area(true);
			//fSurfaceArea = ( Area(true)+Area(false) )/ 2 ; //this logic needs to be checked
														   // For Sphere It is actually addition of surface area of outer and inner shell
		}
  }

Precision UnplacedHype::Area(bool outer)
  {
	Precision fT=0.,fR=0.;
	if(outer)
	{
  		fT = fTOut;
		fR = fRmax;	
	}
	else
	{
  		fT = fTIn;
		fR = fRmin;	
	}

    Precision p = fT*std::sqrt(fT*fT);
	Precision q = p*fDz*std::sqrt(fR*fR + (std::pow(fT,2)+std::pow(fT,4))*std::pow(fDz,2) );
	Precision r = fR*fR*std::asinh(p*fDz/fR);
	Precision ar =  ((q+r)/(2*p))*4*kPi;
	return ar;
  }



VECGEOM_CUDA_HEADER_BOTH  //This line is not there in UnplacedBox.cpp
void UnplacedHype::Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const
  {
    // Returns the full 3D cartesian extent of the solid.
      aMin.Set(-fRmax,-fRmax,-fDz);
      aMax.Set(fRmax,fRmax,fDz);
  }

//VECGEOM_CUDA_HEADER_BOTH
std::string UnplacedHype::GetEntityType() const
  {
      return "Hyperboloid\n";
  }

VECGEOM_CUDA_HEADER_BOTH
void UnplacedHype::GetParametersList(int, double* aArray)const
  {
      aArray[0] = GetRmin();
      aArray[1] = GetStIn();
      aArray[2] = GetRmax();
      aArray[3] = GetStOut();
      aArray[4] = GetDz();

  }
  
  #ifdef VECGEOM_NVCC
  Vector3D<Precision> UnplacedHype::GetPointOnSurface() const{}
  #else 
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> UnplacedHype::GetPointOnSurface() const
  {
  }
  #endif

VECGEOM_CUDA_HEADER_BOTH
UnplacedHype* UnplacedHype::Clone() const
  {
      return new UnplacedHype(fRmin,fStIn,fRmax,fStOut,fDz);
  }

#ifdef VECGEOM_NVCC
  std::ostream& UnplacedHype::StreamInfo(std::ostream& os) const{}
#else
  VECGEOM_CUDA_HEADER_BOTH
  std::ostream& UnplacedHype::StreamInfo(std::ostream& os) const
  //Definition taken from 
  {
      
   int oldprc = os.precision(16);
   os << "-----------------------------------------------------------\n"
   //  << "		*** Dump for solid - " << GetName() << " ***\n"
   //  << "		===================================================\n"
   
   << " Solid type: VecGeomHype\n"
     << " Parameters: \n"

     << "				Inner radius: " << fRmin << " mm \n"
	 << "				Inner Stereo Angle " << fStIn << " rad \n"
     << "               Outer radius: " <<fRmax <<"mm\n"    
     << "               Outer Stereo Angle " << fStOut << " rad \n"
     << "               Half Height: "<<fDz<<" mm \n"
     << "-----------------------------------------------------------\n";
   os.precision(oldprc);

   return os;
  }
#endif
//_________________________________________________________________


//__________________________________________________________________
    
    void UnplacedHype::Print() const {
        
    }
//__________________________________________________________________
    
    void UnplacedHype::Print(std::ostream &os) const {
        
    }
    
template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedHype::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
    if (placement) {
    return new(placement) SpecializedHype<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
        logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
        logical_volume, transformation);
#endif
  }
  return new SpecializedHype<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
      logical_volume, transformation);
#endif
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedHype::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<
      UnplacedHype>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void UnplacedHype_CopyToGpu(VUnplacedVolume *const gpu_ptr);

VUnplacedVolume* UnplacedHype::CopyToGpu(
    VUnplacedVolume *const gpu_ptr) const {
  UnplacedHype_CopyToGpu(gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VUnplacedVolume* UnplacedHype::CopyToGpu() const {
  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedHype>();
  return this->CopyToGpu(gpu_ptr);
}

#endif

#ifdef VECGEOM_NVCC

class VUnplacedVolume;

__global__
void UnplacedHype_ConstructOnGpu(VUnplacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::UnplacedHype();
}

void UnplacedHype_CopyToGpu(VUnplacedVolume *const gpu_ptr) {
  UnplacedHype_ConstructOnGpu<<<1, 1>>>(gpu_ptr);
}

#endif

} // End namespace vecgeom
