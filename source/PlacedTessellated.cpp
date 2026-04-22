/// @file PlacedTessellated.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch), Sandro Wenzel (sandro.wenzel@cern.ch)

#include "VecGeom/volumes/Tessellated.h"
#include "VecGeom/volumes/PlacedVolume.h"

#ifndef VECCORE_CUDA
#ifdef VECGEOM_GEANT4
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#endif
#ifdef VECGEOM_ROOT
#include "TGeoTessellated.h"
#endif

#endif // VECCORE_CUDA

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedTessellated::PrintType() const
{
  printf("PlacedTessellated");
}

void PlacedTessellated::PrintType(std::ostream &s) const
{
  s << "PlacedTessellated";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedTessellated::ConvertToUnspecialized() const
{
  return new SimpleTessellated(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedTessellated::ConvertToRoot() const
{
  using Vertex_t = Tessellated::Vertex_t;
  auto tsl       = new TGeoTessellated();
  for (size_t ifacet = 0; ifacet < GetUnplacedVolume()->GetNFacets(); ++ifacet) {
    TriangleFacet<double> *facet = GetUnplacedVolume()->GetFacet(ifacet);
    tsl->AddFacet(Vertex_t(facet->fVertices[0].x(), facet->fVertices[0].y(), facet->fVertices[0].z()),
                  Vertex_t(facet->fVertices[1].x(), facet->fVertices[1].y(), facet->fVertices[1].z()),
                  Vertex_t(facet->fVertices[2].x(), facet->fVertices[2].y(), facet->fVertices[2].z()));
  }
  tsl->CloseShape();
  return tsl;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTessellated::ConvertToGeant4() const
{
  G4TessellatedSolid *tsl = new G4TessellatedSolid("");
  for (size_t ifacet = 0; ifacet < GetUnplacedVolume()->GetNFacets(); ++ifacet) {
    TriangleFacet<double> *facet = GetUnplacedVolume()->GetFacet(ifacet);
    tsl->AddFacet(new G4TriangularFacet(
        G4ThreeVector(facet->fVertices[0].x(), facet->fVertices[0].y(), facet->fVertices[0].z()),
        G4ThreeVector(facet->fVertices[1].x(), facet->fVertices[1].y(), facet->fVertices[1].z()),
        G4ThreeVector(facet->fVertices[2].x(), facet->fVertices[2].y(), facet->fVertices[2].z()), ABSOLUTE));
  }
  tsl->SetSolidClosed(true);
  return tsl;
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

// VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedTessellated)

namespace cxx {
template size_t DevicePtr<cuda::SpecializedTessellated>::SizeOf();
template void DevicePtr<cuda::SpecializedTessellated>::Construct(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                                 DevicePtr<cuda::Transformation3D> const transform,
                                                                 const unsigned int id, const int copy_no,
                                                                 const int child_id) const;
// template void ConstructManyOnGpu<cuda::SpecializedTessellated>( std::size_t nElement, DevicePtr<cuda::VPlacedVolume>
// const *gpu_ptrs, DevicePtr<cuda::LogicalVolume> const *logical, DevicePtr<cuda::Transformation3D> const *trafo,
// decltype(std::declval<VPlacedVolume>().id()) const *ids, decltype(std::declval<VPlacedVolume>().GetCopyNo()) const
// *copyNos, decltype(std::declval<VPlacedVolume>().GetChildId()) const *childIds); }
} // namespace cxx

#endif

} // End global namespace
