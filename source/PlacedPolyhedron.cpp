/// \file PlacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedPolyhedron.h"

#include "volumes/kernel/GenericKernels.h"
#include "volumes/SpecializedPolyhedron.h"

#ifdef VECGEOM_ROOT
#include "TGeoPgon.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UPolyhedra.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Polyhedra.hh"
#endif

namespace VECGEOM_NAMESPACE {

#ifndef VECGEOM_NVCC



VPlacedVolume const* PlacedPolyhedron::ConvertToUnspecialized() const {
  return new SimplePolyhedron(GetLabel().c_str(), logical_volume(),
                              transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedPolyhedron::ConvertToRoot() const {

  const int zPlaneCount = GetZSegmentCount()+1;

  TGeoPgon *pgon = new TGeoPgon(
      GetLabel().c_str(), GetPhiStart(), GetPhiDelta(), GetSideCount(),
      zPlaneCount);

  // Define sections of TGeoPgon. It takes care of the rest internally once the
  // last section is set.
  for (int i = 0; i < zPlaneCount; ++i) {
    pgon->DefineSection(i, GetZPlanes()[i], GetRMin()[i], GetRMax()[i]);
  }

  return pgon;
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedPolyhedron::ConvertToUSolids() const {
  
  return new UPolyhedra(
      GetLabel().c_str(),
      GetPhiStart(),
      GetPhiDelta(),
      GetSideCount(),
      GetZSegmentCount()+1,
      &GetZPlanes()[0],
      &GetRMin()[0],
      &GetRMax()[0]);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedPolyhedron::ConvertToGeant4() const {

  return new G4Polyhedra(
      GetLabel().c_str(),
      GetPhiStart(),
      GetPhiDelta(),
      GetSideCount(),
      GetZSegmentCount()+1,
      &GetZPlanes()[0],
      &GetRMin()[0],
      &GetRMax()[0]);
}
#endif

#endif // VECGEOM_NVCC

#ifdef VECGEOM_CUDA_INTERFACE
VPlacedVolume* PlacedPolyhedron::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  assert(0);
  return NULL;
}
VPlacedVolume* PlacedPolyhedron::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  assert(0);
  return NULL;
}
#endif

} // End global namespace
