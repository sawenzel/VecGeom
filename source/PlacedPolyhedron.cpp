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

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedPolyhedron::ConvertToUnspecialized() const {
  return new SimplePolyhedron(GetLabel().c_str(), logical_volume(),
                              transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedPolyhedron::ConvertToRoot() const {
  assert(0);
  return NULL;
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedPolyhedron::ConvertToUSolids() const {
  assert(0);
  return NULL;
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