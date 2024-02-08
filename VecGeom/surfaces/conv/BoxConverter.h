#ifndef VECGEOM_SURFACE_BOXCONVERTER_H_
#define VECGEOM_SURFACE_BOXCONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Box.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Box
/// @tparam Real_t Precision type
/// @param box Box solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateBoxSurfaces(vecgeom::UnplacedBox const &box, int logical_id)
{
  using Vector3              = vecgeom::Vector3D<Real_t>;
  const bool use_surf_safety = true;
  int isurf;
  LogicExpressionCPU logic; // AND logic: 0 & 1 & 2 & 3 & 4 & 5

  auto dx = box.x();
  auto dy = box.y();
  auto dz = box.z();
  // corners represented as vectors (-x, -y, -z), (+x, -y, -z), (+x, +y, -z), (-x, +y, -z), same for +z
  std::vector<Vector3> corners = {{-dx, -dy, -dz}, {dx, -dy, -dz}, {dx, dy, -dz}, {-dx, dy, -dz},
                                  {-dx, -dy, dz},  {dx, -dy, dz},  {dx, dy, dz},  {-dx, dy, dz}};
  auto assertWindow            = [](int isurf) {
    assert(isurf >= 0 && CPUsurfData<Real_t>::Instance().fLocalSurfaces[isurf].fFrame.type == kWindow);
  };
  std::vector<Vector3> vert;
  // surface at -dx:
  vert  = {corners[3], corners[0], corners[4], corners[7]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id, use_surf_safety);
  assertWindow(isurf);
  logic.push_back(isurf);
  // surface at +dx:
  vert  = {corners[1], corners[2], corners[6], corners[5]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id, use_surf_safety);
  assertWindow(isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at -dy:
  vert  = {corners[0], corners[1], corners[5], corners[4]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id, use_surf_safety);
  assertWindow(isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at +dy:
  vert  = {corners[2], corners[3], corners[7], corners[6]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id, use_surf_safety);
  assertWindow(isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at -dz:
  vert  = {corners[0], corners[3], corners[2], corners[1]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id, use_surf_safety);
  assertWindow(isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at +dz:
  vert  = {corners[4], corners[5], corners[6], corners[7]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id, use_surf_safety);
  assertWindow(isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
