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
              unplaced->fStartPhi,
              unplaced->fDeltaPhi,
              unplaced->fNz );

      // fill Rmin
      double * rmin = rootshape->GetRmin();
      double * rmax = rootshape->GetRmax();
      double * z = rootshape->GetZ();

      double prevrmin, prevrmax;
      bool putlowersection=true;
      for(int i=0, counter=0;i< unplaced->GetNSections();++i){
          UnplacedCone const * cone = unplaced->GetSection(i).solid;
          if( putlowersection ){
            rmin[counter] = cone->GetRmin1();
            rmax[counter] = cone->GetRmax1();
            z[counter] = -cone->GetDz() + unplaced->GetSection(i).shift;
            counter++;
          }
          rmin[counter] = cone->GetRmin2();
          rmax[counter] = cone->GetRmax2();
          z[counter]   =  cone->GetDz() + unplaced->GetSection(i).shift;;
          counter++;

          prevrmin = cone->GetRmin2();
          prevrmax = cone->GetRmax2();

          // take care of a possible discontinuity
          if( i < unplaced->GetNSections()-1 && ( prevrmin != unplaced->GetSection(i+1).solid->GetRmin1()
             || prevrmax != unplaced->GetSection(i+1).solid->GetRmax2() ) ) {
             putlowersection = true;
          }
          else{
             putlowersection = false;
          }
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
		  counter++;
                }
                rminvector[counter] = cone->GetRmin2();
                rmaxvector[counter] = cone->GetRmax2();
                zvector[counter]   =  cone->GetDz() + unplaced->GetSection(i).shift;
                counter++;

                prevrmin = cone->GetRmin2();
                prevrmax = cone->GetRmax2();

                // take care of a possible discontinuity
                if( i < unplaced->GetNSections()-1 && ( prevrmin != unplaced->GetSection(i+1).solid->GetRmin1()
                   || prevrmax != unplaced->GetSection(i+1).solid->GetRmax2() ) ) {
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

     return g4shape;
  }
#endif
#endif // VECGEOM_NVCC

}} // end namespace
