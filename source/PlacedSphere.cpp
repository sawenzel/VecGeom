/// @file PlacedSphere.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/PlacedSphere.h"
#include "volumes/Sphere.h"
#include "base/Global.h"
#include "base/AOS3D.h"
#include "base/SOA3D.h"
#include "backend/Backend.h"

#ifdef VECGEOM_USOLIDS
#include "USphere.hh"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoSphere.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Sphere.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedSphere::ConvertToUnspecialized() const {
  return new SimpleSphere(GetLabel().c_str(), GetLogicalVolume(),
                          GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedSphere::ConvertToRoot() const {
  return new TGeoSphere(GetLabel().c_str(),GetInnerRadius(),GetOuterRadius(),
                        GetStartThetaAngle()*kRadToDeg,(GetStartThetaAngle()+GetDeltaThetaAngle())*kRadToDeg,
                        GetStartPhiAngle()*kRadToDeg, (GetStartPhiAngle()+GetDeltaPhiAngle())*kRadToDeg);
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedSphere::ConvertToUSolids() const {

   return new USphere(GetLabel().c_str(),GetInnerRadius(),GetOuterRadius(),
                      GetStartPhiAngle(), GetDeltaPhiAngle(),
                      GetStartThetaAngle(),GetDeltaThetaAngle());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedSphere::ConvertToGeant4() const {
   return new G4Sphere(GetLabel().c_str(),GetInnerRadius(),GetOuterRadius(),
                       GetStartPhiAngle(), GetDeltaPhiAngle(),
                       GetStartThetaAngle(),GetDeltaThetaAngle());
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC( SpecializedSphere )

#endif // VECGEOM_NVCC

} // End global namespace

