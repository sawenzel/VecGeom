#include "VecGeom/base/Assert.h"
#include "VecGeom/volumes/Extruded.h"

#include <vector>

using namespace vecgeom;

namespace {

UnplacedExtruded MakeExtrudedMultiLayer(bool convex)
{
  constexpr size_t nvert = 8;
  constexpr size_t nsect = 4;

  XtruVertex2 vertices[nvert];
  XtruSection sections[nsect];

  vertices[0] = {-3., -3.};
  vertices[1] = {-3., 3.};
  vertices[2] = {3., 3.};
  vertices[3] = {3., -3.};
  if (convex) {
    vertices[4] = {1.5, -3.5};
    vertices[5] = {0.5, -3.6};
    vertices[6] = {-0.5, -3.6};
    vertices[7] = {-1.5, -3.5};
  } else {
    vertices[4] = {1.5, -3.};
    vertices[5] = {1.5, 1.5};
    vertices[6] = {-1.5, 1.5};
    vertices[7] = {-1.5, -3.};
  }

  sections[0].fOrigin.Set(-2., 1., -4.);
  sections[0].fScale = 1.5;
  sections[1].fOrigin.Set(0., 0., 1.);
  sections[1].fScale = 0.5;
  sections[2].fOrigin.Set(0., 0., 1.5);
  sections[2].fScale = 0.7;
  sections[3].fOrigin.Set(2., 2., 4.);
  sections[3].fScale = 0.9;

  return UnplacedExtruded(nvert, vertices, nsect, sections);
}

template <typename MeshHelper>
void ValidateGeant4StyleMesh(MeshHelper const &mesh, size_t expectedVertices, size_t expectedFacets)
{
  VECGEOM_ASSERT(mesh.GetNvertices() == expectedVertices);
  VECGEOM_ASSERT(mesh.GetNfacets() == expectedFacets);

  std::vector<Vector3D<Precision>> vertices;
  vertices.reserve(mesh.GetNvertices());
  for (size_t i = 0; i < mesh.GetNvertices(); ++i)
    vertices.push_back(mesh.GetVertex(i));

  std::vector<bool> referenced(vertices.size(), false);
  for (size_t ifacet = 0; ifacet < mesh.GetNfacets(); ++ifacet) {
    size_t indices[3];
    mesh.GetFacetVertices(ifacet, indices);

    // This reproduces the contract needed by G4Polyhedron::SetFacet: all facet
    // indices must refer to valid, distinct vertices in the exported vertex table.
    for (size_t i = 0; i < 3; ++i) {
      VECGEOM_ASSERT(indices[i] < vertices.size());
      referenced[indices[i]] = true;
    }
    VECGEOM_ASSERT(indices[0] != indices[1]);
    VECGEOM_ASSERT(indices[1] != indices[2]);
    VECGEOM_ASSERT(indices[2] != indices[0]);

    const auto e1 = vertices[indices[1]] - vertices[indices[0]];
    const auto e2 = vertices[indices[2]] - vertices[indices[0]];
    VECGEOM_ASSERT(e1.Cross(e2).Mag2() > Precision(1.e-20));
  }

  for (bool used : referenced)
    VECGEOM_ASSERT(used);
}

void TestExtrudedMeshHelper(bool convex)
{
  auto xtru = MakeExtrudedMultiLayer(convex);
  auto mesh = xtru.GetMeshHelper();

  const size_t nvertices     = xtru.GetNVertices();
  const size_t nsections     = xtru.GetNSections();
  const size_t expectedVerts = nvertices * nsections;
  const size_t expectedFaces = 2 * (nvertices - 2) + 2 * nvertices * (nsections - 1);

  ValidateGeant4StyleMesh(mesh, expectedVerts, expectedFaces);

  const auto &legacyTsl = xtru.GetStruct().fTslHelper;
  VECGEOM_ASSERT(legacyTsl.fVertices.size() == expectedVerts);
  VECGEOM_ASSERT(legacyTsl.fFacets.size() == expectedFaces);
  for (size_t ifacet = 0; ifacet < legacyTsl.fFacets.size(); ++ifacet) {
    // Temporary Geant4 compatibility layout: old callers expect
    // fTslHelper.fFacets[i]->fIndices to index fTslHelper.fVertices.
    const auto *facet = legacyTsl.fFacets[ifacet];
    VECGEOM_ASSERT(facet != nullptr);
    for (size_t i = 0; i < 3; ++i)
      VECGEOM_ASSERT(facet->fIndices[i] < legacyTsl.fVertices.size());
    VECGEOM_ASSERT(facet->fIndices[0] != facet->fIndices[1]);
    VECGEOM_ASSERT(facet->fIndices[1] != facet->fIndices[2]);
    VECGEOM_ASSERT(facet->fIndices[2] != facet->fIndices[0]);
  }
}

} // namespace

int main()
{
  TestExtrudedMeshHelper(true);
  TestExtrudedMeshHelper(false);
  return 0;
}
