#include "VecGeom/base/Assert.h"
#include "VecGeom/volumes/Tessellated.h"

#include <vector>

using namespace vecgeom;

namespace {

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

void TestTessellatedMeshHelper()
{
  const Vector3D<Precision> a{0., 0., 0.};
  const Vector3D<Precision> b{1., 0., 0.};
  const Vector3D<Precision> c{0., 1., 0.};
  const Vector3D<Precision> d{0., 0., 1.};

  UnplacedTessellated tessellated;
  tessellated.AddTriangularFacet(a, c, b);
  tessellated.AddTriangularFacet(a, b, d);
  tessellated.AddTriangularFacet(a, d, c);
  tessellated.AddTriangularFacet(b, c, d);
  tessellated.Close();

  auto mesh = tessellated.GetMeshHelper();
  ValidateGeant4StyleMesh(mesh, 4, 4);
}

} // namespace

int main()
{
  TestTessellatedMeshHelper();
  return 0;
}
