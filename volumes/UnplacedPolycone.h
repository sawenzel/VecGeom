/*
 * UnplacedPolycone.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */


#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedCone.h"
#include "base/Vector.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedPolycone; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedPolycone );

VECGEOM_DEVICE_FORWARD_DECLARE( class PolyconeSection; )
VECGEOM_DEVICE_DECLARE_CONV( PolyconeSection );


inline namespace VECGEOM_IMPL_NAMESPACE {

// helper structure to encapsulate a section
struct PolyconeSection
{
   VECGEOM_CUDA_HEADER_BOTH
   PolyconeSection()
      : fSolid(0), fShift(0.0), fTubular(0), fConvex(0)
   {}

   VECGEOM_CUDA_HEADER_BOTH
   ~PolyconeSection() = default;
   
   UnplacedCone *fSolid;
   double fShift;
   bool fTubular;
   bool fConvex; // TRUE if all points in section are concave in regards to whole polycone, will be determined
};


//typedef std::vector<PolyconeSection> vec;
//typedef Vector<PolyconeSection> vec;

class UnplacedPolycone : public VUnplacedVolume, public AlignedBase {

public:
    // the members
    // for the phi section --> will be replaced by a wedge
    Precision fStartPhi;
    Precision fDeltaPhi;
    Precision fEndPhi;

    int fNz;
    //Precision * fRmin;
    //Precision * fRmax;
    //Precision * fZ;

    // actual internal storage
    Vector<PolyconeSection> fSections;
    Vector<double> fZs;

public:
    VECGEOM_CUDA_HEADER_BOTH
    void Init(
         double phiStart,        // initial phi starting angle
         double phiTotal,        // total phi angle
         int numZPlanes,         // number of z planes
         const double zPlane[],  // position of z planes
         const double rInner[],  // tangent distance to inner surface
         const double rOuter[]);

    // the constructor
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedPolycone( Precision phistart, Precision deltaphi,
            int Nz, Precision * rmin,
            Precision * rmax, Precision * z ) :
                fStartPhi(phistart),
                fDeltaPhi(deltaphi),
                fNz(Nz),
      //          fRmin(rmin),
                //fRmax(rmax),
                //fZ(z),
                fSections(),
                fZs()
                {

        // init internal members
        Init(phistart, deltaphi, Nz, z, rmin, rmax);
    }

    VECGEOM_CUDA_HEADER_BOTH
    int GetNz() const {return fNz;}

    VECGEOM_CUDA_HEADER_BOTH
    int GetNSections() const {return fSections.size();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetStartPhi() const {return fStartPhi;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetDeltaPhi() const {return fDeltaPhi;}
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetEndPhi() const {return fStartPhi+fDeltaPhi;}

    VECGEOM_CUDA_HEADER_BOTH
    int GetSectionIndex( Precision zposition ) const {
     //TODO: consider bindary search
     if( zposition < fZs[0] ) return -1;
     for(int i=0;i<GetNSections();++i)
       {
           if( zposition >= fZs[i] && zposition <= fZs[i+1] )
                return i;
       }
     return -2;
    }

    VECGEOM_CUDA_HEADER_BOTH
    PolyconeSection const & GetSection( Precision zposition ) const {
        //TODO: consider bindary search
        int i = GetSectionIndex(zposition);
        return fSections[i];
    }

    VECGEOM_CUDA_HEADER_BOTH
    // GetSection if index is known
    PolyconeSection const & GetSection( int index ) const {
      return fSections[index];
    }

    //#ifdef VECGEOM_USOLIDS
    VECGEOM_CUDA_HEADER_BOTH
    Precision Capacity() const
    {
        Precision cubicVolume = 0.;
        for (int i = 0; i < GetNSections(); i++)
        {
            PolyconeSection const& section = fSections[i];
            cubicVolume += section.fSolid->Capacity();
        }
        return cubicVolume;
    }
    VECGEOM_CUDA_HEADER_BOTH
    Precision SurfaceArea() const;

    VECGEOM_CUDA_HEADER_BOTH
    bool Normal(Vector3D<Precision> const& point, Vector3D<Precision>& norm) const;
    VECGEOM_CUDA_HEADER_BOTH
    void Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const;

#ifndef VECGEOM_NVCC
    Vector3D<Precision> GetPointOnSurface() const;

 // Methods for random point generation
    VECGEOM_CUDA_HEADER_BOTH
    Vector3D<Precision> GetPointOnCone(Precision fRmin1, Precision fRmax1,
                            Precision fRmin2, Precision fRmax2,
                            Precision zOne,   Precision zTwo,
                            Precision& totArea) const;
    VECGEOM_CUDA_HEADER_BOTH
    Vector3D<Precision> GetPointOnTubs(Precision fRMin, Precision fRMax,
                            Precision zOne,  Precision zTwo,
                            Precision& totArea) const;
    VECGEOM_CUDA_HEADER_BOTH
    Vector3D<Precision> GetPointOnCut(Precision fRMin1, Precision fRMax1,
                           Precision fRMin2, Precision fRMax2,
                           Precision zOne,   Precision zTwo,
                           Precision& totArea) const;
    VECGEOM_CUDA_HEADER_BOTH
    Vector3D<Precision> GetPointOnRing(Precision fRMin, Precision fRMax,
                            Precision fRMin2, Precision fRMax2,
                            Precision zOne) const;
#endif
    //#endif

    // a method to reconstruct "plane" section arrays for z, rmin and rmax
    template<typename PushableContainer>
    void ReconstructSectionArrays(PushableContainer & z,
            PushableContainer & rmin,
            PushableContainer & rmax) const;


    // these methods are required by VUnplacedVolume
    //
public:
    virtual int memory_size() const { return sizeof(*this); }

    VECGEOM_CUDA_HEADER_BOTH
    virtual void Print() const;

    template <TranslationCode transCodeT, RotationCode rotCodeT>
    VECGEOM_CUDA_HEADER_DEVICE
    static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                                 Transformation3D const *const transformation,
    #ifdef VECGEOM_NVCC
                                 const int id,
    #endif
                                 VPlacedVolume *const placement = NULL);


    #ifdef VECGEOM_CUDA_INTERFACE
      virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedPolycone>::SizeOf(); }
      virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
      virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
    #endif

    private:

      virtual void Print(std::ostream &os) const;

      VECGEOM_CUDA_HEADER_DEVICE
      virtual VPlacedVolume* SpecializedVolume(
          LogicalVolume const *const volume,
          Transformation3D const *const transformation,
          const TranslationCode trans_code, const RotationCode rot_code,
    #ifdef VECGEOM_NVCC
          const int id,
    #endif
          VPlacedVolume *const placement = NULL) const;


}; // end class UnplacedPolycone

template<typename PushableContainer>
void UnplacedPolycone::ReconstructSectionArrays(PushableContainer & z,
        PushableContainer & rmin,
        PushableContainer & rmax) const {

      double prevrmin, prevrmax;
      bool putlowersection=true;
      for(int i=0;i< GetNSections();++i){
          UnplacedCone const * cone = GetSection(i).fSolid;
          if( putlowersection ){
            rmin.push_back(cone->GetRmin1());
            rmax.push_back(cone->GetRmax1());
            z.push_back(-cone->GetDz() + GetSection(i).fShift);
          }
          rmin.push_back(cone->GetRmin2());
          rmax.push_back(cone->GetRmax2());
          z.push_back(cone->GetDz() + GetSection(i).fShift);

          prevrmin = cone->GetRmin2();
          prevrmax = cone->GetRmax2();

          // take care of a possible discontinuity
          if( i < GetNSections()-1 && ( prevrmin != GetSection(i+1).fSolid->GetRmin1()
             || prevrmax != GetSection(i+1).fSolid->GetRmax1() ) ) {
             putlowersection = true;
          }
          else{
             putlowersection = false;
          }
      }
}

} // end inline namespace

} // end vecgeom namespace

#endif /* VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_ */
