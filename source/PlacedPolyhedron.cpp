/// \file PlacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedPolyhedron.h"

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

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedPolyhedron::ConvertToUnspecialized() const {
  return new SimplePolyhedron(GetLabel().c_str(), logical_volume(),
                              transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedPolyhedron::ConvertToRoot() const {

  const int zPlaneCount = GetSegmentCount()+1;

  TGeoPgon *pgon = new TGeoPgon(GetLabel().c_str(), 0, 360, GetSideCount(),
                                zPlaneCount);

  Precision *z = new Precision[zPlaneCount];
  Precision *rMin = new Precision[zPlaneCount];
  Precision *rMax = new Precision[zPlaneCount];

  GetUnplacedVolume()->ExtractZPlanes(z, rMin, rMax);

  // Define sections of TGeoPgon. It takes care of the rest internally once the
  // last section is set.
  for (int i = 0; i < zPlaneCount; ++i) {
    pgon->DefineSection(i, z[i], rMin[i], rMax[i]);
  }

  delete[] z;
  delete[] rMin;
  delete[] rMax;

  return pgon;
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedPolyhedron::ConvertToUSolids() const {

  const int zPlaneCount = GetSegmentCount()+1;

  Precision *z = new Precision[zPlaneCount];
  Precision *rMin = new Precision[zPlaneCount];
  Precision *rMax = new Precision[zPlaneCount];

  GetUnplacedVolume()->ExtractZPlanes(z, rMin, rMax);

UPolyhedra *polyhedra = new UPolyhedra(
      GetLabel().c_str(),
      0, // Phi start
      360, // Phi change
      GetSideCount(),
      zPlaneCount,
      z,
      rMin,
      rMax);

  delete[] z;
  delete[] rMin;
  delete[] rMax;

  return polyhedra;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedPolyhedron::ConvertToGeant4() const {

  const int zPlaneCount = GetSegmentCount()+1;

  Precision *z = new Precision[zPlaneCount];
  Precision *rMin = new Precision[zPlaneCount];
  Precision *rMax = new Precision[zPlaneCount];

  GetUnplacedVolume()->ExtractZPlanes(z, rMin, rMax);

  G4Polyhedra *polyhedra = new G4Polyhedra(GetLabel().c_str(), 0, 360,
                                           GetSideCount(), zPlaneCount,
                                           z, rMin, rMax);

  delete[] z;
  delete[] rMin;
  delete[] rMax;

  return polyhedra;
}
#endif

#endif // VECGEOM_BENCHMARK

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
