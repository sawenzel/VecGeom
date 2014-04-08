/*
 * PhysicalPolycone.h
 *
 *  Created on: Jan 8, 2014
 *      Author: swenzel
 */

#ifndef PHYSICALPOLYCONE_H_
#define PHYSICALPOLYCONE_H_

#include <vector>
#include <algorithm>
#include <map>

#include "UPolycone.hh"
#include "TGeoPcon.h"

#include "ShapeFactories.h"
#include "GlobalDefs.h"

#include "UVoxelizer.hh"

// class encapsulating the defining properties of a Polycone
template<typename Float_t=double>
class
PolyconeParameters
{
public:
   enum PconSectionType {kTube, kCone};

private:
   bool has_rmin; // this is true if a least one section has an inner radius
   bool is_full_phi; // true if DPhi =

   Float_t fSPhi; // start angle
   Float_t fDPhi; // total phi angle
   Float_t fMaxR; //
   Float_t fMaxRSqr; //
   int numZPlanes; // number of z planes ( minimum 2 )
   int halfnumZPlanes;
   std::vector<Float_t> zPlanes; // position of z planes
   std::vector<Float_t> rInner; // inner radii at position of z planes
   std::vector<Float_t> rOuter; // outer radii at position of z planes
   std::vector<PconSectionType> pconsectype;
   std::vector<PhysicalVolume const *> pconsection; // maps a section to a concrete tube or cone; mapping is to PhysicalVolume pointer, concrete information is stored in pconsectype )

   void DetermineAndSetRminProperty( int numZPlanes, Float_t const rinner[] );
   PconSectionType DetermineSectionType( Float_t rinner1, Float_t rinner2, Float_t router1, Float_t router2 ) const;
   void Init(Float_t sphi, Float_t dphi, int numZPlanes, Float_t const zplanes[], Float_t const rinner[], Float_t const router[]); // see USolids

public:
   // a constructor
   PolyconeParameters(Float_t sphi,
                  Float_t dphi,
                  int numzplanes,
                  Float_t const zplanes[],
                  Float_t const rinner[],
                  Float_t const router[] ) : zPlanes(numZPlanes), rInner(numZPlanes), rOuter(numZPlanes),
                  halfnumZPlanes(numzplanes/2), numZPlanes(numzplanes), pconsectype(numzplanes-1), pconsection(numzplanes-1), fSPhi(sphi), fDPhi(dphi) {
      Init(sphi, dphi, numzplanes, zplanes, rinner, router);
   }

   // the following methods will be used by the factory to determine the polyconetype
   bool HasRmin() const {return has_rmin;}
   bool IsFullPhi() const {return is_full_phi;}
   Float_t GetDPhi() const {return fDPhi;}
   Float_t GetSPhi() const {return fSPhi;}
   Float_t GetDZ() const {return (zPlanes[numZPlanes-1] - zPlanes[0])/2.;}
   Float_t GetMaxR() const { return fMaxR; }
   Float_t GetMaxRSqr() const { return fMaxRSqr; }

   std::vector<Float_t> const & GetZPlanes() const {return zPlanes;}
   std::vector<Float_t> const & GetInnerRs() const {return rInner;}
   std::vector<Float_t> const & GetOuterRs() const {return rOuter;}
   Float_t GetZPlane(int i) const {return zPlanes[i];}
   Float_t GetInnerR(int i) const {return rInner[i];}
   Float_t GetOuterR(int i) const {return rOuter[i];}

   PconSectionType GetPConSectionType( int i ) const { return pconsectype[i]; }

   int GetNZ() const { return numZPlanes; }
   int GetNumSections() const { return numZPlanes-1; }
   PhysicalVolume const * GetSectionVolume(int i) const { return pconsection[i]; }

   inline
   int FindZSection( Float_t zcoord ) const {
      for(auto i=0;i<numZPlanes;i++){
         if( zPlanes[i] <= zcoord && zcoord < zPlanes[i+1] ) return i;
      }
      return -1;
   };

   __attribute__((always_inline))
   inline
   int FindZSectionDaniel( Float_t zcoord ) const
   {
      // we can get rid of this divison;
      // int l = zPlanes.size()/2;
      int l = halfnumZPlanes;

      int p = l;
      while(! (zPlanes[p] <= zcoord && zcoord < zPlanes[p+1]) && l > 0 )
      {
         l = l/2; // >> 2;
         p+=(1 - 2*(zPlanes[p] >= zcoord)) * l; // this determines direction to go for p
      }

      std::cerr << ((zPlanes[p] <= zcoord && zcoord < zPlanes[p+1]) ? "FOUND" : "NOT FOUND") << std::endl;

      return p;
   }

   inline
   int FindZSectionBS( Float_t zcoord ) const
   {
      return UVoxelizer::BinarySearch( zPlanes, zcoord );
   }


   void Print() const {};

};

// this might be placed into a separate header later
template<typename Float_t>
void PolyconeParameters<Float_t>::Init(  Float_t sphi, Float_t dphi, int numZPlanes, Float_t const zplanes[], Float_t const rinner[], Float_t const router[]  )
{
   DetermineAndSetRminProperty( numZPlanes, rinner );

   // check validity !!
   // TOBEDONE

    // copy data ( might not be necessary )
   for(auto i=0; i<numZPlanes; i++)
   {
      zPlanes[i]=zplanes[i];
      rInner[i]=rinner[i];
      rOuter[i]=router[i];
   }

   for(auto i=0; i<numZPlanes-1; i++ )
   {
      // analyse rmin / rmax at planes i and i+1
      pconsectype[i] =  DetermineSectionType( rinner[i], rinner[i+1], router[i], router[i+1] );

      if( pconsectype[i] == kTube )
      {
         // creates a tube as part of the polycone which is correctly placed ( translated )
         pconsection[i] = TubeFactory::template Create<1, 1296>(
               new TubeParameters<Float_t>(rinner[i], router[i], (zplanes[i+1] - zplanes[i])/2., sphi, dphi),
               new TransformationMatrix(0,0, (zplanes[i+1] + zplanes[i])/2., 0, 0, 0),
               !has_rmin);
      }
      if( pconsectype[i] == kCone )
      {
         // creates a cone as part of the polycone which is correctly places ( translated )
         pconsection[i] = ConeFactory::template Create<1, 1296>(
               new ConeParameters<Float_t>(rinner[i], router[i], rinner[i+1], router[i+1], (zplanes[i+1] - zplanes[i])/2., sphi, dphi),
               new TransformationMatrix(0, 0, (zplanes[i+1] + zplanes[i])/2., 0, 0, 0),
               !has_rmin);
      }
   }

   fMaxR = *std::max_element(rOuter.begin(), rOuter.end());
   fMaxRSqr = fMaxR*fMaxR;
}

template<typename Float_t>
typename PolyconeParameters<Float_t>::PconSectionType PolyconeParameters<Float_t>::DetermineSectionType( Float_t rinner1, Float_t rinner2, Float_t router1, Float_t router2 ) const
{
   bool istube=true;
   if( ! Utils::IsSameWithinTolerance(rinner1, rinner2) ) istube=false;
   if( ! Utils::IsSameWithinTolerance(router1, router2) ) istube=false;
   return (istube)? kTube : kCone;
}

template<typename Float_t>
void PolyconeParameters<Float_t>::DetermineAndSetRminProperty( int numZPlanes, Float_t const rinner[] )
{
   has_rmin = false;
   for(auto i=0; i<numZPlanes; i++)
   {
      if( rinner[i] > Utils::frHalfTolerance ) has_rmin = true;
   }
}



template<TranslationIdType tid, RotationIdType rid, typename ConeType=ConeTraits::HollowConeWithPhi, typename Float_t=double>
class
PlacedPolycone : public PhysicalVolume
{
private:
   PolyconeParameters<Float_t> const * pconparams;

public:
   PlacedPolycone( PolyconeParameters<Float_t> const * pcp, TransformationMatrix const * tm ) : PhysicalVolume(tm), pconparams(pcp)
   {
      this->bbox = new PlacedBox<1,1296>(
            new BoxParameters( pconparams->GetMaxR(), pconparams->GetMaxR(), pconparams->GetDZ() ),
            IdentityTransformationMatrix );

      analogoususolid = new UPolycone( "internal_usolid", pconparams->GetSPhi(),
            pconparams->GetDPhi(),
            pconparams->GetNZ(),
            &pconparams->GetZPlanes()[0],
            &pconparams->GetInnerRs()[0],
            &pconparams->GetOuterRs()[0]);

      analogousrootsolid = new TGeoPcon("internal_tgeopcon",
                                 pconparams->GetSPhi()*360/(2.*M_PI),
                                 pconparams->GetSPhi()+360*pconparams->GetDPhi()/(2.*M_PI),
                                 pconparams->GetNZ());
      // have to fill information for ROOT Polycone
      for( auto j=0; j < pconparams->GetNZ(); j++ )
      {
         ((TGeoPcon *) analogousrootsolid)->DefineSection(j,
                  pconparams->GetZPlane(j),
                  pconparams->GetInnerR(j),
                  pconparams->GetOuterR(j));
      }

      /*
      if(! (tid==0 && rid==1296) )
      {
         unplacedtube = new PlacedUSolidsTube<0,1296,TubeType,T>( _tb, m );
      }
       */
   }


   virtual double DistanceToIn( Vector3D const &, Vector3D const &, double ) const {return 0.;}
   virtual double DistanceToOut( Vector3D const &, Vector3D const &, double ) const {return 0.;}

   inline
   virtual bool   Contains( Vector3D const & x) const {Vector3D xprime;matrix->MasterToLocal<tid,rid>(x,xprime); return
   this->PlacedPolycone<tid,rid,ConeType,Float_t>::UnplacedContains(xprime);}

   __attribute__((always_inline))
   inline
   virtual bool   UnplacedContains( Vector3D const & x) const {return false;}

   // the contains function knowing about the surface
   __attribute__((always_inline))
   inline
   virtual GlobalTypes::SurfaceEnumType   UnplacedContains_WithSurface( Vector3D const & ) const;
   virtual GlobalTypes::SurfaceEnumType   Contains_WithSurface( Vector3D const & ) const;

   virtual double SafetyToIn( Vector3D const & ) const {return 0.;}
   virtual double SafetyToOut( Vector3D const & ) const {return 0.;}

   // for basket treatment (supposed to be dispatched to particle parallel case)
   virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {};
   virtual void DistanceToOut( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {};


   // for basket treatment (supposed to be dispatched to loop case over (SIMD) optimized 1-particle function)
   virtual void DistanceToInIL( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {};
   //   virtual void DistanceToInIL( std::vector<Vector3D> const &, std::vector<Vector3D> const &, double const * /*steps*/, double * /*result*/ ) const;
   virtual void DistanceToInIL( Vector3D const *, Vector3D const *, double const * /*steps*/, double * /*result*/, int /*size*/ ) const {};

   virtual PhysicalVolume const * GetAsUnplacedVolume() const { std::cerr << "NOT IMPLEMENTED" << std::endl; return 0;};
};


template<TranslationIdType tid, RotationIdType rid, typename ConeType, typename Float_t>
GlobalTypes::SurfaceEnumType PlacedPolycone<tid,rid,ConeType,Float_t>::Contains_WithSurface(Vector3D const & x) const
{
   Vector3D xp;
   matrix->MasterToLocal<tid,rid>(x,xp);
   return this->PlacedPolycone<tid,rid,ConeType,Float_t>::UnplacedContains_WithSurface( xp );
}


template<TranslationIdType tid, RotationIdType rid, typename ConeType, typename Float_t>
GlobalTypes::SurfaceEnumType PlacedPolycone<tid,rid,ConeType,Float_t>::UnplacedContains_WithSurface(Vector3D const & x) const
{
   // algorithm:
    // check if point is outsize z range
   // find z section of point
   // call Contains function of that section ( no virtual call!! will be inlined )
   // if point found on z-surface of that section, need to check neighbouring sections as well

   // this should be replaced by bounding box ( or bounding tube check )
   // if( reinterpret_cast<PlacedBox<1,1296> const *>(bbox)->PlacedBox<1,1296>::Contains(x) == false )
   //   return GlobalTypes::kOutside;

   if( x.z < pconparams->GetZPlane(0)-Utils::frHalfTolerance
         || x.z > pconparams->GetZPlane(pconparams->GetNZ()-1) + Utils::frHalfTolerance )
      return GlobalTypes::kOutside;

   if ( x.x*x.x + x.y*x.y > pconparams->GetMaxRSqr() )
      return GlobalTypes::kOutside;


   // TODO: this should do a binary search or linear search as a function of the number of segments
   // may achieve this by template-classifying a polycone as large or small
   int zsection = pconparams->FindZSection( x.z );

   GlobalTypes::SurfaceEnumType result;

   // section is a tube
   PhysicalVolume const * section = pconparams->GetSectionVolume( zsection );
   if( pconparams->GetPConSectionType( zsection ) == PolyconeParameters<Float_t>::kTube )
   {
      // now comes a bit of template magic to achieve inlining of optimized function / ( no virtual function call )
      // we basically know that the segments are of the following type
      typedef PlacedUSolidsTube<1, 1296, typename ConeTraits::ConeTypeToTubeType<ConeType>::type, Float_t> TargetTube_t;
      TargetTube_t const * tube;
      tube = reinterpret_cast< TargetTube_t const *> (section);
      result=tube->TargetTube_t::Contains_WithSurface(x); // specifying the type after -> get rids of the virtual call and inlines this method
   }
   // section is a cone
   if( pconparams->GetPConSectionType( zsection ) == PolyconeParameters<Float_t>::kCone )
   {
      // now comes a bit of template magic to achieve inlining of optimized function / ( no virtual function call )
      PlacedCone< 1, 1296, ConeType, Float_t> const * cone;
      cone = reinterpret_cast< PlacedCone< 1, 1296, ConeType, Float_t> const *> (section);
      result=cone->PlacedCone<1,1296, ConeType,Float_t>::Contains_WithSurface( x );
   }

   // TODO: case where point is on a z-plane surface

   return result;
}


// I am not quite sure if I need this stuff
//
struct PolyconeFactory
{
   template<int tid, int rid>
   static
   PhysicalVolume * Create( PolyconeParameters<> const * tp, TransformationMatrix const * tm )
   {
      if( !tp->HasRmin() )
      {
         if ( Utils::IsSameWithinTolerance ( tp->GetDPhi(), Utils::kTwoPi ) )
         {
            return new PlacedPolycone<tid,rid,ConeTraits::NonHollowCone>(tp, tm);
         }
         else if ( Utils::IsSameWithinTolerance ( tp->GetDPhi(), Utils::kPi ) )
         {
            return new PlacedPolycone<tid,rid,ConeTraits::NonHollowConeWithPhiEqualsPi>(tp, tm);
         }
         else
         {
            return new PlacedPolycone<tid,rid,ConeTraits::NonHollowConeWithPhi>(tp, tm);
         }
      }
      else
      {
         if ( Utils::IsSameWithinTolerance ( tp->GetDPhi(), Utils::kTwoPi ) )
         {
            return new PlacedPolycone<tid,rid,ConeTraits::HollowCone>(tp, tm);
         }
         else if ( Utils::IsSameWithinTolerance ( tp->GetDPhi(), Utils::kPi ) )
         {
            return new PlacedPolycone<tid,rid,ConeTraits::HollowConeWithPhiEqualsPi>(tp, tm);
         }
         else
         {
            return new PlacedPolycone<tid,rid,ConeTraits::HollowConeWithPhi>(tp,tm);
         }
      }
   }
};


#endif /* PHYSICALPOLYCONE_H_ */
