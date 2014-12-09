/*
 * PlacedPolycone.cpp
 *
 *  Created on: Dec 9, 2014
 *      Author: swenzel
 */

#include "volumes/SpecializedPolycone.h"

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
      return NULL;
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
      return NULL;
  }
#endif
#endif // VECGEOM_NVCC

}} // end namespace
