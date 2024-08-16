#ifndef VECGEOM_SURFACE_GENTRAPCONVERTER_H_
#define VECGEOM_SURFACE_GENTRAPCONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/GenTrap.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for a generic trapezoid (Arb8)
/// @tparam Real_t Precision type
/// @param gtrap generalized trapezoid solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateGenTrapSurfaces(vecgeom::UnplacedGenTrap const &gtrap, int logical_id, bool intersection = false)
{
  using Vector3 = vecgeom::Vector3D<Real_t>;
  int isurf;
  LogicExpressionCPU logic; // AND logic: 0 & 1 & 2 & 3 & 4 & 5

  auto vertices = gtrap.GetVertices();

  // corners represented as vectors (-x, -y, -z), (+x, -y, -z), (+x, +y, -z), (-x, +y, -z), same for +z
  std::vector<Vector3> corners = {vertices[0], vertices[3], vertices[2], vertices[1],
                                  vertices[4], vertices[7], vertices[6], vertices[5]};

  std::vector<Vector3> vert;

  // surface at -dx:
  vert  = {corners[3], corners[0], corners[4], corners[7]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(isurf);

  // surface at +dx:
  vert  = {corners[1], corners[2], corners[6], corners[5]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(land);
  logic.push_back(isurf);

  // surface at -dy:
  vert  = {corners[0], corners[1], corners[5], corners[4]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(land);
  logic.push_back(isurf);

  // surface at +dy:
  vert  = {corners[2], corners[3], corners[7], corners[6]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(land);
  logic.push_back(isurf);

  // surface at -dz:
  vert  = {corners[0], corners[3], corners[2], corners[1]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(land);
  logic.push_back(isurf);

  // surface at +dz:
  vert  = {corners[4], corners[5], corners[6], corners[7]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(land);
  logic.push_back(isurf);

  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
