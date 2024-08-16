#ifndef VECGEOM_SURFACE_TETCONVERTER_H_
#define VECGEOM_SURFACE_TETCONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Tet.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Tet, a tetrahedron
/// @tparam Real_t Precision type
/// @param tet Tet solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateTetSurfaces(vecgeom::UnplacedTet const &tet, int logical_id, bool intersection = false)
{
  using Vector3D = vecgeom::Vector3D<Real_t>;

  const auto &tetstr = tet.GetStruct();
  int isurf;
  LogicExpressionCPU logic; // AND logic: 0 & 1 & 2 & 3
  vecgeom::Transformation3D transf;
  std::vector<Vector3D>
      vert; // Stores coordinates of the four corners that define a tet. It is initialised with 3D coordinates.

  // corners represented as vectors
  auto const *corners = tetstr.fVertex;
  auto assertFace     = [](int isurf) {
    assert(isurf >= 0 && CPUsurfData<Real_t>::Instance().fLocalSurfaces[isurf].fFrame.type == FrameType::kTriangle);
  };

  // surface 1:
  vert  = {corners[0], corners[1], corners[2]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  assertFace(isurf);
  logic.push_back(isurf);

  // surface 2:
  vert  = {corners[0], corners[2], corners[3]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  assertFace(isurf);
  logic.push_back(land);
  logic.push_back(isurf);

  // surface 3:
  vert  = {corners[0], corners[3], corners[1]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  assertFace(isurf);
  logic.push_back(land);
  logic.push_back(isurf);

  // surface 4:
  vert  = {corners[1], corners[3], corners[2]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  assertFace(isurf);
  logic.push_back(land);
  logic.push_back(isurf);

  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
