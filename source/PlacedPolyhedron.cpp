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
  UnplacedPolyhedron const &unplaced = *GetUnplacedVolume();
  Polygon const &corners = *unplaced.GetCorners();
  TGeoPgon *pgon = new TGeoPgon(GetLabel().c_str(), unplaced.GetPhiStart(),
                                unplaced.GetPhiDelta(), unplaced.GetSideCount(),
                                corners.GetVertixCount());
  // Set the corners somehow...
  return pgon;
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedPolyhedron::ConvertToUSolids() const {
  UnplacedPolyhedron const &unplaced = *GetUnplacedVolume();
  Polygon const &corners = *unplaced.GetCorners();
  double *r = new double[corners.GetVertixCount()];
  double *z = new double[corners.GetVertixCount()];
  for (int i = 0, iEnd = corners.GetVertixCount(); i < iEnd; ++i) {
    Vector2D<Precision> vertix = corners[i];
    r[i] = vertix[0];
    z[i] = vertix[1];
  }
  return new UPolyhedra(GetLabel(), unplaced.GetPhiStart(),
                        unplaced.GetPhiDelta(), unplaced.GetSideCount(),
                        corners.GetVertixCount(), r, z);
}
#endif

#endif // VECGEOM_BENCHMARK

} // End global namespace