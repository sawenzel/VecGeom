/*
 * PlacedPolycone.cpp
 *
 *  Created on: Dec 9, 2014
 *      Author: swenzel
 */

#include "volumes/SpecializedPolycone.h"
#include <iostream>

#ifdef VECGEOM_ROOT
#include "TGeoPcon.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Polycone.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {


#ifndef VECGEOM_NVCC
  VPlacedVolume const* PlacedPolycone::ConvertToUnspecialized() const
  {
      return new SimplePolycone(GetLabel().c_str(), logical_volume(), transformation());
  }
#ifdef VECGEOM_ROOT
  TGeoShape const* PlacedPolycone::ConvertToRoot() const
  {
      UnplacedPolycone const * unplaced = GetUnplacedVolume();
      TGeoPcon* rootshape = new TGeoPcon(
              unplaced->fStartPhi*kRadToDeg,
              unplaced->fDeltaPhi*kRadToDeg,
              unplaced->fNz );


      std::vector<double> rmin;
      std::vector<double> rmax;
      std::vector<double> z;

      double prevrmin, prevrmax;
      bool putlowersection=true;
      for(int i=0;i< unplaced->GetNSections();++i){
          UnplacedCone const * cone = unplaced->GetSection(i).solid;
          if( putlowersection ){
            rmin.push_back(cone->GetRmin1());
            rmax.push_back(cone->GetRmax1());
            z.push_back(-cone->GetDz() + unplaced->GetSection(i).shift);
          }
          rmin.push_back(cone->GetRmin2());
          rmax.push_back(cone->GetRmax2());
          z.push_back(cone->GetDz() + unplaced->GetSection(i).shift);

          prevrmin = cone->GetRmin2();
          prevrmax = cone->GetRmax2();

          // take care of a possible discontinuity
          if( i < unplaced->GetNSections()-1 && ( prevrmin != unplaced->GetSection(i+1).solid->GetRmin1()
             || prevrmax != unplaced->GetSection(i+1).solid->GetRmax1() ) ) {
             putlowersection = true;
          }
          else{
             putlowersection = false;
          }
      }

      // now transfer the parameter to the ROOT polycone
      for(int i=0;i<unplaced->fNz;++i)
      {
          rootshape->DefineSection(i,z[i],rmin[i],rmax[i]);
      }
      rootshape->InspectShape();

      return rootshape;
  }


#endif
#ifdef VECGEOM_USOLIDS
  ::VUSolid const* PlacedPolycone::ConvertToUSolids() const
  {
      return NULL;
  }
#endif
#ifdef VECGEOM_GEANT4
  G4VSolid const* PlacedPolycone::ConvertToGeant4() const
  {
      UnplacedPolycone const * unplaced = GetUnplacedVolume();
      double * zvector = new double[unplaced->GetNz()];
      double * rminvector = new double[unplaced->GetNz()];
      double * rmaxvector = new double[unplaced->GetNz()];

      double prevrmin, prevrmax;
            bool putlowersection=true;
            for(int i=0, counter=0; i<unplaced->GetNSections();++i){

                UnplacedCone const * cone = unplaced->GetSection(i).solid;
                if( putlowersection ){
                  rminvector[counter] = cone->GetRmin1();
                  rmaxvector[counter] = cone->GetRmax1();
                  zvector[counter] = -cone->GetDz() + unplaced->GetSection(i).shift;
		  std::cerr << "###" << counter << " " << zvector[counter] << "\n";
		  counter++;
                }
                rminvector[counter] = cone->GetRmin2();
                rmaxvector[counter] = cone->GetRmax2();
                zvector[counter]   =  cone->GetDz() + unplaced->GetSection(i).shift;
		std::cerr << "###" << counter << " " << zvector[counter] << "\n";                
		counter++;

                prevrmin = cone->GetRmin2();
                prevrmax = cone->GetRmax2();

		putlowersection = false;
                // take care of a possible discontinuity
                if( i < unplaced->GetNSections()-1 && ( prevrmin != unplaced->GetSection(i+1).solid->GetRmin1()
                   || prevrmax != unplaced->GetSection(i+1).solid->GetRmax1() ) ) {
                   putlowersection = true;
                }
                else{
                   putlowersection = false;
                }
            }

     G4Polycone * g4shape = new G4Polycone("",unplaced->fStartPhi,
             unplaced->fDeltaPhi,
             unplaced->fNz,
             zvector,
             rminvector,
             rmaxvector
     );

     g4shape->StreamInfo( std::cout );
     
     return g4shape;
  }
#endif
#endif // VECGEOM_NVCC

}} // end namespace
